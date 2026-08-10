"""Run the Phase-3 LLM enrichment on a small, deterministic sample or pilot.

Inputs are the corrected nvBench Large V2 train and val files only. ``test.jsonl``
and the human-evaluation items are never read and never written -- the loader
refuses any path containing ``test``.

Modes:
  ``--mode sample``  10 records; final status TECHNICAL_SAMPLE_PASS / TECHNICAL_SAMPLE_FAIL.
  ``--mode pilot``   30 records disjoint from the sample; writes the blank human
                     audit template; final status WAITING_FOR_ENRICHMENT_R1.

Every candidate is validated (strict JSON -> Pydantic -> immutable fingerprint ->
content rules) before a trusted merge writes the six presentation fields. Nothing
is written into ``data/frozen`` or into the v1/v2 source trees.

Usage:
    python experiments/scripts/run_enrichment_sample.py --mode sample
    python experiments/scripts/run_enrichment_sample.py --mode pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.enrichment import (  # noqa: E402
    ENRICHABLE_FIELDS,
    ENRICHMENT_SPEC_VERSION,
    build_messages,
    enrichment_provenance,
    immutable_diff,
    immutable_fingerprint,
    merge_enrichment,
    parse_payload,
    prompt_sha256,
    select_records,
    selection_summary,
    validate_payload,
)
from src.data_pipeline.enrichment_provider import (  # noqa: E402
    api_key_present,
    build_client,
    create_completion,
    load_provider_config,
    parse_json_object,
    provider_manifest_entry,
    sanitize_error,
    validate_config,
)
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402

DEFAULT_SOURCE_DIR = "data/staging/dashboard_v3/nvbench_large_v2"
DEFAULT_OUT_ROOT = "data/staging/enrichment"

MODE_SETTINGS = {
    "sample": {"n": 10, "out": "sample_10", "seed": 42},
    "pilot": {"n": 30, "out": "pilot_30", "seed": 42},
}

# A sample fails materially if fewer than this share of candidates is accepted,
# or if any immutable source field was touched at all.
MIN_ACCEPT_RATE = 0.8

# HTTP statuses that will not improve by retrying the next record: stop the run
# instead of spending one call per record on a dead credential or missing model.
FATAL_STATUS_CODES = {401: "MISSING_API_CREDENTIALS", 403: "MISSING_API_CREDENTIALS",
                      404: "MODEL_NOT_ACCESSIBLE"}

AUDIT_HUMAN_COLUMNS = (
    "reviewer_id", "users_ok", "context_summary_ok", "layout_ok", "styling_ok",
    "interactions_ok", "rationales_ok", "invented_content_found",
    "immutable_violation_found", "overall_accept", "notes",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-3 enrichment sample / pilot runner")
    p.add_argument("--mode", choices=tuple(MODE_SETTINGS), default="sample")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--seed", type=int, default=None, help="override the mode default seed")
    p.add_argument("--n", type=int, default=None, help="override the mode record count")
    p.add_argument("--max-tokens", type=int, default=3000,
                   help="output budget per call (xhigh reasoning tokens count against it)")
    p.add_argument("--exclude-from", default=None,
                   help="JSONL of already-enriched inputs whose item_ids are skipped "
                        "(default for --mode pilot: the sample's input_records.jsonl)")
    p.add_argument("--dry-run", action="store_true", help="select and prompt only; no API call")
    return p.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _PROJECT_ROOT / candidate


def load_split(source_dir: Path, name: str) -> List[Dict[str, Any]]:
    """Read one split file, refusing held-out test data outright."""
    if "test" in name.lower():
        raise ValueError(f"refusing to read held-out data: {name}")
    path = source_dir / name
    if not path.exists():
        raise FileNotFoundError(f"input split not found: {path}")
    return read_jsonl(path)


def _usage_totals(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    return {
        "calls": len(rows),
        "input_tokens_total": sum(r.get("input_tokens") or 0 for r in rows),
        "output_tokens_total": sum(r.get("output_tokens") or 0 for r in rows),
        "latency_ms_total": sum(latencies),
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else None,
        "latency_ms_max": max(latencies) if latencies else None,
    }


AUDIT_VALIDATION_COLUMNS = ("automatic_validation_status", "automatic_reason_codes",
                            "automatic_validation_details")

AUDIT_HEADER = [
    "item_id", "split", "chart_type", "task_type", "kpi", "encoding_x", "encoding_y",
    "users", "context_summary", "layout", "styling", "interactions", "rationales",
    *AUDIT_VALIDATION_COLUMNS, *AUDIT_HUMAN_COLUMNS,
]


def build_audit_rows(selected: Sequence[Mapping[str, Any]],
                     accepted: Sequence[Mapping[str, Any]],
                     rejected: Sequence[Mapping[str, Any]],
                     raw_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One audit row per *selected* record — accepted and rejected alike.

    Human review needs the automatically rejected candidates too, so the template
    covers every selected record. Rows follow the deterministic selection order,
    never acceptance status. Enrichment content comes from the merged record when
    accepted, and from the raw model reply when rejected (so a reviewer sees what
    the model actually produced); nothing is re-validated or re-generated here.
    """
    accepted_by_id = {str(r.get("item_id")): r for r in accepted}
    rejected_by_id = {str(r.get("item_id")): r for r in rejected}
    raw_by_id = {str(r.get("item_id")): r for r in raw_rows}

    rows: List[Dict[str, Any]] = []
    for record in selected:
        item_id = str(record.get("item_id"))
        enriched = accepted_by_id.get(item_id)
        rejection = rejected_by_id.get(item_id)
        source = enriched or record

        mapping = (source.get("recommendation") or {}).get("kpi_chart_mapping") or [{}]
        mapping0 = mapping[0] if mapping else {}
        encoding = mapping0.get("encoding") or {}
        recommendation = source.get("recommendation") or {}

        if enriched is not None:
            users = (enriched.get("brief") or {}).get("users")
            content = {
                "context_summary": recommendation.get("context_summary"),
                "layout": recommendation.get("layout"),
                "styling": recommendation.get("styling"),
                "interactions": recommendation.get("interactions"),
                "rationales": recommendation.get("rationales"),
            }
        else:
            # Rejected: show the unmerged model reply, parsed only for readability.
            reply = parse_json_object((raw_by_id.get(item_id) or {}).get("response_text") or "") or {}
            users = reply.get("users")
            content = {key: reply.get(key) for key in
                       ("context_summary", "layout", "styling", "interactions", "rationales")}

        row = {
            "item_id": item_id,
            "split": record.get("split"),
            "chart_type": mapping0.get("chart_type"),
            "task_type": mapping0.get("task_type"),
            "kpi": mapping0.get("kpi"),
            "encoding_x": encoding.get("x"),
            "encoding_y": encoding.get("y"),
            "users": users,
            **{key: json.dumps(value, ensure_ascii=False) for key, value in content.items()},
            "automatic_validation_status": "rejected" if rejection is not None else "accepted",
            "automatic_reason_codes": "; ".join(rejection.get("reason_codes") or []) if rejection else "",
            "automatic_validation_details": "; ".join(rejection.get("details") or []) if rejection else "",
        }
        row.update({column: "" for column in AUDIT_HUMAN_COLUMNS})
        rows.append(row)
    return rows


def _write_audit_template(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the audit CSV; human-review columns stay blank by construction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report_markdown(mode: str, manifest: Mapping[str, Any]) -> str:
    selection = manifest["selection"]
    usage = manifest["usage"]
    lines = [
        f"# Phase-3 enrichment — {mode} run",
        "",
        f"- spec version: `{ENRICHMENT_SPEC_VERSION}`",
        f"- status: **{manifest['status']}**",
        f"- timestamp (UTC): {manifest['timestamp_utc']}",
        f"- model: `{manifest['provider']['model']}`",
        f"- requested reasoning effort: `{manifest['provider']['requested_reasoning_effort']}`",
        f"- applied reasoning mode: `{manifest['provider']['applied_reasoning_mode']}`",
        f"- temperature: {manifest['provider']['temperature']}",
        "",
        "## Selection",
        "",
        f"- records: {selection['n']} (unique source groups: {selection['unique_source_groups']})",
        f"- split mix: {selection['split']}",
        f"- chart types: {selection['chart_type']}",
        f"- source: `{manifest['inputs']['source_dir']}` (train + val only; test never read)",
        "",
        "## Outcome",
        "",
        f"- accepted: {manifest['counts']['accepted']}/{manifest['counts']['candidates']}",
        f"- rejected: {manifest['counts']['rejected']}",
        f"- immutable-field violations: {manifest['counts']['immutable_violations']}",
        f"- rejection reasons: {manifest['counts']['reason_codes'] or 'none'}",
        "",
        "## Cost",
        "",
        f"- calls: {usage['calls']}",
        f"- tokens: in={usage['input_tokens_total']} out={usage['output_tokens_total']}",
        f"- latency: mean={usage['latency_ms_mean']} ms, max={usage['latency_ms_max']} ms, "
        f"total={usage['latency_ms_total']} ms",
        "",
        "## Guarantees",
        "",
        f"- LLM-written fields: {', '.join(ENRICHABLE_FIELDS)}",
        "- immutable fingerprint compared before/after every merge; any change rejects the "
        "candidate with `immutable_source_field_changed`",
        "- merge performed by trusted code only; the model reply is data, never applied directly",
        "- interaction fields checked against the record's source columns",
        "- numbers not present in the source record reject the candidate",
        "- rationales must name the given chart type, task and an encoded field",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:  # noqa: C901 - linear pipeline script
    args = parse_args()
    settings = MODE_SETTINGS[args.mode]
    n = args.n or settings["n"]
    seed = args.seed if args.seed is not None else settings["seed"]
    out_dir = _resolve(args.out_root) / settings["out"]
    source_dir = _resolve(args.source_dir)

    cfg = load_provider_config()
    problems = validate_config(cfg)
    if problems:
        print("config invalid: " + "; ".join(problems))
        print("status: CONNECTION_TEST_FAILED")
        return 1
    if not args.dry_run and not api_key_present(cfg):
        print(f"{cfg.api_key_env} is not set (see docs/project/ENRICHMENT_PROVIDER_SETUP.md)")
        print("status: MISSING_API_CREDENTIALS")
        return 1

    train = load_split(source_dir, "train.jsonl")
    val = load_split(source_dir, "val.jsonl")

    exclude: List[str] = []
    exclude_path = args.exclude_from
    if exclude_path is None and args.mode == "pilot":
        default_exclude = _resolve(args.out_root) / MODE_SETTINGS["sample"]["out"] / "input_records.jsonl"
        exclude_path = str(default_exclude) if default_exclude.exists() else None
    if exclude_path:
        exclude = [str(r.get("item_id")) for r in read_jsonl(_resolve(exclude_path))]

    selected = select_records(train, val, n=n, seed=seed, exclude_item_ids=exclude)
    if len(selected) < n:
        print(f"only {len(selected)} of {n} records available after exclusions")
        return 1

    print("=" * 60)
    print(f"PHASE-3 ENRICHMENT - {args.mode.upper()} ({len(selected)} records)")
    print("=" * 60)
    print(f"  source        : {source_dir} (train+val only)")
    print(f"  selection     : {selection_summary(selected)}")
    print(f"  excluded ids  : {len(exclude)}")

    if args.dry_run:
        write_jsonl(selected, out_dir / "input_records.jsonl")
        print(f"  DRY-RUN       : prompts built, no API call. inputs -> {out_dir}")
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    client = build_client(cfg)

    # Preflight: one tiny call proves credential + model before spending the batch.
    try:
        preflight = create_completion(
            client, cfg,
            [{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=256,
        )
        print(f"  preflight     : ok ({preflight.latency_ms} ms, "
              f"reasoning={preflight.applied_reasoning_mode})")
    except Exception as exc:  # noqa: BLE001 - reported sanitized, run never starts
        status_code = getattr(exc, "status_code", None)
        status = FATAL_STATUS_CODES.get(int(status_code)) if status_code is not None else None
        print(f"  preflight     : FAILED - {sanitize_error(exc, cfg)}")
        print(f"  status        : {status or 'CONNECTION_TEST_FAILED'}")
        print("=" * 60)
        return 1

    raw_rows: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    reason_totals: Dict[str, int] = {}
    immutable_violations = 0
    applied_modes: List[str] = []
    provider_notes: List[str] = []
    last_result = None
    abort_status: Optional[str] = None
    abort_detail: Optional[str] = None

    for index, record in enumerate(selected, 1):
        item_id = str(record.get("item_id"))
        messages = build_messages(record)
        fingerprint_before = immutable_fingerprint(record)
        budget = args.max_tokens
        codes: List[str] = []
        details: List[str] = []
        result = None
        parsed_obj: Optional[Dict[str, Any]] = None

        try:
            result = create_completion(client, cfg, messages, json_object=True, max_tokens=budget)
            parsed_obj = parse_json_object(result.text)
            if parsed_obj is None and result.truncated:
                budget *= 2
                result = create_completion(client, cfg, messages, json_object=True, max_tokens=budget)
                parsed_obj = parse_json_object(result.text)
        except KeyboardInterrupt:
            abort_status, abort_detail = "RUN_INTERRUPTED", "interrupted by user"
            print(f"  [{index:>2}/{len(selected)}] interrupted - writing partial artifacts")
            break
        except Exception as exc:  # noqa: BLE001 - recorded per record; fatal codes stop the run
            codes, details = ["api_error"], [sanitize_error(exc, cfg)]
            status_code = getattr(exc, "status_code", None)
            if status_code is not None and int(status_code) in FATAL_STATUS_CODES:
                abort_status = FATAL_STATUS_CODES[int(status_code)]
                abort_detail = details[0]

        if result is not None:
            last_result = result
            applied_modes.append(result.applied_reasoning_mode)
            provider_notes.extend(result.notes)
            raw_rows.append({
                "item_id": item_id,
                "response_text": result.text,
                "finish_reason": result.finish_reason,
                "max_tokens": budget,
                "latency_ms": result.latency_ms,
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.completion_tokens,
                "applied_reasoning_mode": result.applied_reasoning_mode,
                "structured_output_applied": result.structured_output_applied,
                "notes": result.notes,
            })

        merged: Optional[Dict[str, Any]] = None
        if not codes:
            if parsed_obj is None:
                codes, details = ["response_not_json"], [
                    f"reply not a JSON object (finish_reason={getattr(result, 'finish_reason', None)})"
                ]
            else:
                payload, schema_codes, schema_details = parse_payload(parsed_obj)
                if payload is None:
                    codes, details = schema_codes, schema_details
                else:
                    codes, details = validate_payload(record, payload)
                    provenance = enrichment_provenance(
                        model=result.model, applied_reasoning_mode=result.applied_reasoning_mode,
                        requested_effort=result.requested_reasoning_effort, temperature=cfg.temperature,
                        prompt_sha256=prompt_sha256(messages), run_id=run_id,
                        fingerprint_before=fingerprint_before,
                    )
                    candidate = merge_enrichment(record, payload, provenance)
                    if immutable_fingerprint(candidate) != fingerprint_before:
                        codes = list(dict.fromkeys(codes + ["immutable_source_field_changed"]))
                        details.append("changed immutable keys: " + ", ".join(immutable_diff(record, candidate)))
                    elif not codes:
                        merged = candidate

        if "immutable_source_field_changed" in codes:
            immutable_violations += 1
        for code in codes:
            reason_totals[code] = reason_totals.get(code, 0) + 1

        if merged is not None:
            accepted.append(merged)
            print(f"  [{index:>2}/{len(selected)}] accepted  {item_id}")
        else:
            rejected.append({
                "item_id": item_id,
                "split": record.get("split"),
                "reason_codes": codes,
                "details": details,
                "fingerprint_before": fingerprint_before,
            })
            print(f"  [{index:>2}/{len(selected)}] REJECTED  {item_id}  {codes}")

        if abort_status:
            print(f"  aborting after record {index}: {abort_status} - no further API calls")
            print(f"  cause: {abort_detail}")
            break

    attempted = len(accepted) + len(rejected)
    accept_rate = len(accepted) / len(selected) if selected else 0.0
    if abort_status:
        status = abort_status
    elif args.mode == "sample":
        passed = accept_rate >= MIN_ACCEPT_RATE and immutable_violations == 0
        status = "TECHNICAL_SAMPLE_PASS" if passed else "TECHNICAL_SAMPLE_FAIL"
    else:
        status = "WAITING_FOR_ENRICHMENT_R1"

    manifest = {
        "stage": f"phase3_enrichment_{args.mode}",
        "spec_version": ENRICHMENT_SPEC_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "seed": seed,
        "inputs": {
            "source_dir": str(source_dir.relative_to(_PROJECT_ROOT)) if source_dir.is_relative_to(_PROJECT_ROOT) else str(source_dir),
            "files": ["train.jsonl", "val.jsonl"],
            "test_split_used": False,
            "human_eval_items_used": False,
            "excluded_item_ids": len(exclude),
        },
        "provider": {
            **provider_manifest_entry(cfg, last_result),
            "applied_reasoning_modes_observed": sorted(set(applied_modes)),
            "provider_notes": sorted(set(provider_notes)),
        },
        "enrichable_fields": list(ENRICHABLE_FIELDS),
        "selection": selection_summary(selected),
        "counts": {
            "candidates": len(selected),
            "attempted": attempted,
            "aborted": abort_status is not None,
            "abort_reason": abort_detail,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "accept_rate": round(accept_rate, 3),
            "immutable_violations": immutable_violations,
            "reason_codes": dict(sorted(reason_totals.items())),
            "min_accept_rate_required": MIN_ACCEPT_RATE if args.mode == "sample" else None,
        },
        "usage": _usage_totals(raw_rows),
        "outputs": {
            "input_records": "input_records.jsonl",
            "raw_responses": "raw_responses.jsonl",
            "accepted_records": "accepted_records.jsonl",
            "rejected_records": "rejected_records.jsonl",
            "report": "report.md",
        },
    }

    write_jsonl(selected, out_dir / "input_records.jsonl")
    write_jsonl(raw_rows, out_dir / "raw_responses.jsonl")
    write_jsonl(accepted, out_dir / "accepted_records.jsonl")
    write_jsonl(rejected, out_dir / "rejected_records.jsonl")

    if args.mode == "pilot" and not abort_status:
        audit_path = out_dir / "manual_enrichment_audit_template_30.csv"
        audit_rows = build_audit_rows(selected, accepted, rejected, raw_rows)
        _write_audit_template(audit_path, audit_rows)
        manifest["outputs"]["audit_template"] = audit_path.name
        manifest["audit_template"] = {
            "path": audit_path.name,
            "rows": len(audit_rows),
            "selected_pilot_records": len(selected),
            "automatically_accepted": len(accepted),
            "automatically_rejected": len(rejected),
            "human_audit_rows": len(audit_rows),
            "row_order": "deterministic selection order (input_records.jsonl)",
            "validation_columns": list(AUDIT_VALIDATION_COLUMNS),
            "human_columns_blank": list(AUDIT_HUMAN_COLUMNS),
        }

    write_json(manifest, out_dir / "manifest.json")
    (out_dir / "report.md").write_text(_report_markdown(args.mode, manifest), encoding="utf-8")

    usage = manifest["usage"]
    print("-" * 60)
    print(f"  accepted      : {len(accepted)}/{len(selected)} (rate {accept_rate:.2f})")
    print(f"  rejected      : {len(rejected)} {dict(sorted(reason_totals.items()))}")
    print(f"  immutable     : {immutable_violations} violations")
    print(f"  reasoning     : requested={cfg.reasoning_effort} applied={sorted(set(applied_modes))}")
    print(f"  tokens        : in={usage['input_tokens_total']} out={usage['output_tokens_total']}")
    print(f"  latency       : mean={usage['latency_ms_mean']} ms max={usage['latency_ms_max']} ms")
    print(f"  outputs       : {out_dir}")
    print(f"  status        : {status}")
    print("=" * 60)
    return 0 if status in ("TECHNICAL_SAMPLE_PASS", "WAITING_FOR_ENRICHMENT_R1") else 1


if __name__ == "__main__":
    raise SystemExit(main())
