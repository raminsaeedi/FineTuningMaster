"""Resolve only the rejected records of the completed full enrichment run.

Two stages, in order:
  1. offline revalidation of every rejected record against the current validators
     (no API call) -- resolves records that failed on a narrow validator defect;
  2. at most **one** new API call per record that is still rejected, using the same
     model, prompt, schema and validator versions as the full run.

Accepted records are never re-called and never regenerated. Previous raw responses
are preserved; retries are appended to a separate file with attempt number and
timestamp. Records that stay invalid are permanently rejected and reported.

Outputs under ``<run-dir>/targeted_retry/`` plus the reconciled final accepted sets
under ``<run-dir>/final/``.

Usage:
    python experiments/scripts/run_enrichment_targeted_retry.py --offline-only
    python experiments/scripts/run_enrichment_targeted_retry.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.enrichment import (  # noqa: E402
    ENRICHMENT_SPEC_VERSION,
    build_messages,
    enrichment_provenance,
    prompt_sha256,
)
from src.data_pipeline.enrichment_full import (  # noqa: E402
    FULL_RUN_SPEC_VERSION,
    validate_and_merge,
    write_json_atomic,
    write_jsonl_atomic,
)
from src.data_pipeline.enrichment_provider import (  # noqa: E402
    api_key_present,
    build_client,
    create_completion,
    load_provider_config,
    parse_json_object,
    sanitize_error,
)
from src.utils.io import read_jsonl  # noqa: E402

DEFAULT_RUN_DIR = "data/staging/enrichment/full_train_val_v1"
SPLITS = ("train", "val")
RETRY_MAX_TOKENS = 8192  # generous: these are the hardest records of the corpus
RETRY_TEMPERATURE = 0.0
MAX_RETRIES_PER_ITEM = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Targeted resolution of rejected enrichment records")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--offline-only", action="store_true",
                   help="revalidate only; make no API call")
    return p.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _PROJECT_ROOT / candidate


def main() -> int:
    args = parse_args()
    run_dir = _resolve(args.run_dir)
    retry_dir = run_dir / "targeted_retry"
    final_dir = run_dir / "final"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    cfg = load_provider_config()
    client = None

    print("=" * 64)
    print("PHASE-3 TARGETED RETRY - REJECTED RECORDS ONLY")
    print("=" * 64)

    resolved_offline: List[Dict[str, Any]] = []
    resolved_after_retry: List[Dict[str, Any]] = []
    permanently_rejected: List[Dict[str, Any]] = []
    retry_rows: List[Dict[str, Any]] = []
    accepted_final: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLITS}
    input_totals: Dict[str, int] = {}
    audit: List[Dict[str, Any]] = []

    for split in SPLITS:
        split_dir = run_dir / split
        records = {str(r["item_id"]): r for r in read_jsonl(split_dir / "input_records.jsonl")}
        order = [str(r["item_id"]) for r in read_jsonl(split_dir / "input_records.jsonl")]
        raw = {str(r["item_id"]): r for r in read_jsonl(split_dir / "raw_responses.jsonl")}
        results = {str(r["item_id"]): r for r in read_jsonl(split_dir / "validation_results.jsonl")}
        accepted_existing = {str(r["item_id"]): r
                             for r in read_jsonl(split_dir / "accepted_enriched.jsonl")}
        input_totals[split] = len(order)

        rejected_ids = [i for i in order if not results.get(i, {}).get("accepted")]
        print(f"  {split:<5}: {len(order)} inputs | accepted {len(accepted_existing)} | "
              f"rejected {len(rejected_ids)}")

        for item_id in rejected_ids:
            record = records[item_id]
            original = results.get(item_id, {})
            reply = parse_json_object(raw.get(item_id, {}).get("response_text") or "")
            provenance = enrichment_provenance(
                model=cfg.model,
                applied_reasoning_mode=raw.get(item_id, {}).get("applied_reasoning_mode") or "unknown",
                requested_effort=cfg.reasoning_effort, temperature=RETRY_TEMPERATURE,
                prompt_sha256=prompt_sha256(build_messages(record)), run_id=run_id,
                fingerprint_before=original.get("immutable_fingerprint_before", ""),
            )
            merged, revalidated = validate_and_merge(record, reply, provenance)
            entry = {
                "item_id": item_id, "split": split,
                "original_reason_codes": original.get("reason_codes"),
                "original_details": original.get("details"),
                "resolution": None,
                "retry_attempts": 0,
                "final_reason_codes": revalidated["reason_codes"],
            }

            if merged is not None:
                entry["resolution"] = "resolved_offline_revalidation"
                accepted_final[split].append(merged)
                resolved_offline.append(entry)
                audit.append(entry)
                print(f"     offline OK   {item_id}")
                continue

            if args.offline_only:
                entry["resolution"] = "pending_retry"
                permanently_rejected.append({**entry, "reason": "retry not attempted (--offline-only)"})
                audit.append(entry)
                print(f"     still bad    {item_id} {revalidated['reason_codes']} (no retry)")
                continue

            if client is None:
                if not api_key_present(cfg):
                    print(f"  {cfg.api_key_env} is not set; cannot retry")
                    return 1
                client = build_client(cfg)

            entry["retry_attempts"] = MAX_RETRIES_PER_ITEM
            try:
                result = create_completion(client, cfg, build_messages(record), json_object=True,
                                           max_tokens=RETRY_MAX_TOKENS, temperature=RETRY_TEMPERATURE)
                retry_row = {
                    "item_id": item_id, "split": split, "attempt_number": 2,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                    "response_text": result.text, "finish_reason": result.finish_reason,
                    "max_tokens": RETRY_MAX_TOKENS, "temperature": RETRY_TEMPERATURE,
                    "latency_ms": result.latency_ms, "input_tokens": result.prompt_tokens,
                    "output_tokens": result.completion_tokens,
                    "applied_reasoning_mode": result.applied_reasoning_mode,
                    "notes": result.notes,
                }
                retry_rows.append(retry_row)
                merged, revalidated = validate_and_merge(
                    record, parse_json_object(result.text), provenance)
                entry["final_reason_codes"] = revalidated["reason_codes"]
            except Exception as exc:  # noqa: BLE001 - recorded, never retried again
                entry["final_reason_codes"] = ["api_error"]
                entry["retry_error"] = sanitize_error(exc, cfg)
                merged = None

            if merged is not None:
                entry["resolution"] = "resolved_after_one_retry"
                accepted_final[split].append(merged)
                resolved_after_retry.append(entry)
                print(f"     retry OK     {item_id}")
            else:
                entry["resolution"] = "permanently_rejected"
                permanently_rejected.append({
                    **entry,
                    "reason": "still invalid after one retry; accepting it would require "
                              "weakening the encoding-agreement rule",
                })
                print(f"     permanent    {item_id} {entry['final_reason_codes']}")
            audit.append(entry)

        # Final accepted set for the split, in deterministic input order.
        resolved_by_id = {r["item_id"]: r for r in accepted_final[split]}
        accepted_final[split] = [accepted_existing[i] if i in accepted_existing else resolved_by_id[i]
                                 for i in order
                                 if i in accepted_existing or i in resolved_by_id]
        write_jsonl_atomic(accepted_final[split], final_dir / f"{split}_accepted_final.jsonl")

    write_jsonl_atomic(resolved_offline + resolved_after_retry, retry_dir / "resolved_after_retry.jsonl")
    write_jsonl_atomic(permanently_rejected, retry_dir / "permanently_rejected.jsonl")
    write_jsonl_atomic(permanently_rejected, final_dir / "permanently_rejected.jsonl")
    if retry_rows:
        existing = read_jsonl(retry_dir / "retry_responses.jsonl") if (
            retry_dir / "retry_responses.jsonl").exists() else []
        write_jsonl_atomic(existing + retry_rows, retry_dir / "retry_responses.jsonl")

    input_total = sum(input_totals.values())
    accepted_total = sum(len(v) for v in accepted_final.values())
    report = {
        "stage": "phase3_targeted_retry",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "full_run_spec_version": FULL_RUN_SPEC_VERSION,
        "enrichment_spec_version": ENRICHMENT_SPEC_VERSION,
        "configuration": {"model": cfg.model, "requested_reasoning_effort": cfg.reasoning_effort,
                          "temperature": RETRY_TEMPERATURE, "max_tokens": RETRY_MAX_TOKENS,
                          "max_retries_per_item": MAX_RETRIES_PER_ITEM},
        "counts": {
            "input_total": input_total,
            "input_per_split": input_totals,
            "rejected_before": len(audit),
            "resolved_offline": len(resolved_offline),
            "resolved_after_retry": len(resolved_after_retry),
            "permanently_rejected": len(permanently_rejected),
            "accepted_final": accepted_total,
            "accepted_final_per_split": {s: len(v) for s, v in accepted_final.items()},
            "reconciles": accepted_total + len(permanently_rejected) == input_total,
            "api_calls_made": len(retry_rows),
        },
        "records": audit,
    }
    write_json_atomic(report, retry_dir / "targeted_retry_report.json")

    lines = [
        "# Phase-3 targeted retry — rejected records only",
        "",
        f"- run id: `{run_id}`",
        f"- model: `{cfg.model}`, reasoning `{cfg.reasoning_effort}`, temperature {RETRY_TEMPERATURE}",
        f"- API calls made: {len(retry_rows)} (accepted records were never re-called)",
        "",
        f"- rejected before: {len(audit)}",
        f"- resolved by offline revalidation: {len(resolved_offline)}",
        f"- resolved after one retry: {len(resolved_after_retry)}",
        f"- permanently rejected: {len(permanently_rejected)}",
        "",
        f"- accepted final: {accepted_total} "
        f"({', '.join(f'{s}={len(v)}' for s, v in accepted_final.items())})",
        f"- reconciliation: accepted_final + permanently_rejected = "
        f"{accepted_total + len(permanently_rejected)} / {input_total} "
        f"({'ok' if report['counts']['reconciles'] else 'MISMATCH'})",
        "",
        "## Per record",
        "",
        "| item_id | split | original reason | resolution | final reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in audit:
        lines.append(f"| `{entry['item_id']}` | {entry['split']} | "
                     f"{', '.join(entry['original_reason_codes'] or []) or '-'} | "
                     f"{entry['resolution']} | "
                     f"{', '.join(entry['final_reason_codes'] or []) or '-'} |")
    (retry_dir / "targeted_retry_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("-" * 64)
    print(f"  resolved offline      : {len(resolved_offline)}")
    print(f"  resolved after retry  : {len(resolved_after_retry)}")
    print(f"  permanently rejected  : {len(permanently_rejected)}")
    print(f"  accepted final        : {accepted_total} "
          f"({', '.join(f'{s}={len(v)}' for s, v in accepted_final.items())})")
    print(f"  reconciles to {input_total}: {report['counts']['reconciles']}")
    print(f"  outputs               : {retry_dir} , {final_dir}")
    print("=" * 64)
    return 0 if report["counts"]["reconciles"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
