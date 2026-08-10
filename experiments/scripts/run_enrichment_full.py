"""Full Phase-3 enrichment of the nvBench Large V2 train and val splits.

Enriches exactly the 1281 train and 264 val records with the six presentation
fields, using the prompt, schema and validators that produced the accepted
technical sample and the reviewed pilot. The held-out test split and the
human-evaluation items are never read.

Behaviour: deterministic input order, response cache keyed by configuration
fingerprint, resume after interruption, at most two extra attempts per item with
exponential backoff for transient errors, atomic writes, and a validation result
for every input record (accepted + rejected == input count).

Usage:
    python experiments/scripts/run_enrichment_full.py --dry-run
    python experiments/scripts/run_enrichment_full.py --workers 4
    python experiments/scripts/run_enrichment_full.py --split val --workers 4
    python experiments/scripts/run_enrichment_full.py --report-only
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.enrichment import (  # noqa: E402
    ENRICHABLE_FIELDS,
    ENRICHMENT_SPEC_VERSION,
    SYSTEM_PROMPT,
    build_messages,
    enrichment_provenance,
    prompt_sha256,
)
from src.data_pipeline.enrichment_full import (  # noqa: E402
    EXPECTED_COUNTS,
    FULL_RUN_SPEC_VERSION,
    MAX_RETRIES,
    backoff_seconds,
    classify_error,
    config_fingerprint,
    evaluate_quality_gates,
    load_raw_cache,
    load_split,
    pending_records,
    summarize_split,
    validate_and_merge,
    verify_human_r1,
    write_json_atomic,
    write_jsonl_atomic,
)
from src.data_pipeline.enrichment_provider import (  # noqa: E402
    DOTENV_PATH,
    api_key_present,
    build_client,
    create_completion,
    load_provider_config,
    parse_json_object,
    provider_manifest_entry,
    resolve_env,
    sanitize_error,
    validate_config,
)
from src.utils.io import read_jsonl, write_yaml  # noqa: E402

DEFAULT_SOURCE_DIR = "data/staging/dashboard_v3/nvbench_large_v2"
DEFAULT_OUT_DIR = "data/staging/enrichment/full_train_val_v1"
DEFAULT_R1_FILE = "data/staging/enrichment/pilot_30/manual_enrichment_audit_template_30_R1.csv"

# Requested by the full-run configuration; the pilot ran at 0.1 and this deviation
# is recorded in the manifest rather than hidden.
FULL_RUN_TEMPERATURE = 0.0
PILOT_TEMPERATURE = 0.1
FLUSH_EVERY = 25

# Accepted pilot/val replies peak at ~2975 output tokens with xhigh reasoning, so a
# 3000 cap truncated the tail of the distribution. 4096 clears the observed maximum
# with headroom; truncated items additionally escalate to 8192 on retry.
DEFAULT_MAX_TOKENS = 4096


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full train+val Phase-3 enrichment")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--r1-file", default=DEFAULT_R1_FILE)
    p.add_argument("--split", choices=("train", "val", "both"), default="both")
    p.add_argument("--workers", type=int, default=4, help="concurrent API calls")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="initial output budget; doubled on truncation within the retry budget")
    p.add_argument("--temperature", type=float, default=FULL_RUN_TEMPERATURE)
    p.add_argument("--limit", type=int, default=None, help="cap records per split (smoke runs)")
    p.add_argument("--dry-run", action="store_true", help="verify gates and inputs, no API call")
    p.add_argument("--report-only", action="store_true",
                   help="rebuild reports from existing outputs, no API call")
    p.add_argument("--skip-human-gate", action="store_true",
                   help="only for smoke tests; the real run must pass the gate")
    return p.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _PROJECT_ROOT / candidate


class RunLog:
    """Append-only run log; no secret ever reaches it."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()

    def write(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(f"{stamp} {message}\n")


def call_with_retries(client: Any, cfg: Any, record: Mapping[str, Any], *, max_tokens: int,
                      temperature: float, log: RunLog) -> Dict[str, Any]:
    """One item: up to 1 + MAX_RETRIES attempts, backoff on transient errors.

    Returns a raw-response row (never raises for item-level failures). A fatal
    provider error (bad credential, missing model) is re-raised so the run stops
    instead of burning the whole corpus.
    """
    item_id = str(record.get("item_id"))
    messages = build_messages(record)
    attempts = 0
    last_error: Optional[str] = None

    budget = max_tokens
    while attempts <= MAX_RETRIES:
        attempts += 1
        try:
            result = create_completion(client, cfg, messages, json_object=True,
                                       max_tokens=budget, temperature=temperature)
            unusable = not (result.text or "").strip() or (
                result.truncated and parse_json_object(result.text) is None)
            if unusable and attempts <= MAX_RETRIES:
                # Truncation is a budget problem: escalate instead of repeating.
                last_error = f"unusable reply (finish_reason={result.finish_reason}) at max_tokens={budget}"
                if result.truncated:
                    budget *= 2
                log.write(f"{item_id} {last_error}, retrying at max_tokens={budget}")
                time.sleep(backoff_seconds(attempts))
                continue
            return {
                "item_id": item_id,
                "split": record.get("split"),
                "response_text": result.text,
                "finish_reason": result.finish_reason,
                "max_tokens": budget,
                "temperature": temperature,
                "latency_ms": result.latency_ms,
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.completion_tokens,
                "applied_reasoning_mode": result.applied_reasoning_mode,
                "structured_output_applied": result.structured_output_applied,
                "attempts": attempts,
                "notes": result.notes,
                "cached": False,
                "error": last_error,
            }
        except Exception as exc:  # noqa: BLE001 - classified below
            kind = classify_error(exc)
            message = sanitize_error(exc, cfg)
            if kind == "fatal":
                raise
            last_error = message
            log.write(f"{item_id} attempt {attempts} {kind}: {message}")
            if kind == "transient" and attempts <= MAX_RETRIES:
                time.sleep(backoff_seconds(attempts))
                continue
            break

    return {
        "item_id": item_id,
        "split": record.get("split"),
        "response_text": "",
        "finish_reason": None,
        "attempts": attempts,
        "error": last_error,
        "cached": False,
        "notes": [],
    }


def process_split(split: str, records: Sequence[Mapping[str, Any]], *, out_dir: Path, cfg: Any,
                  client: Any, config_hash: str, args: argparse.Namespace, log: RunLog,
                  run_id: str) -> Dict[str, Any]:
    """Enrich one split with cache/resume, then write all per-split artifacts."""
    split_dir = out_dir / split
    # Caches written before max_tokens left the fingerprint stay valid: their hash is
    # recomputed in the legacy form from the budget each row recorded.
    raw_path = split_dir / "raw_responses.jsonl"
    legacy_budgets = sorted({int(row["max_tokens"]) for row in (read_jsonl(raw_path) if raw_path.exists() else [])
                             if str(row.get("max_tokens", "")).isdigit()})
    legacy_hashes = [config_fingerprint(cfg.model, cfg.base_url, args.temperature,
                                        cfg.reasoning_effort, max_tokens=budget)
                     for budget in legacy_budgets]
    cache, preserved_raw = load_raw_cache(raw_path, config_hash, legacy_hashes)
    todo = pending_records(records, cache)

    log.write(f"{split}: {len(records)} records, {len(cache)} cache hits, {len(todo)} to call")
    print(f"  {split:<5}: {len(records)} records | cache hits {len(cache)} | to call {len(todo)}")

    fresh: Dict[str, Dict[str, Any]] = {}
    done = 0
    lock = threading.Lock()

    def work(record: Mapping[str, Any]) -> None:
        nonlocal done
        row = call_with_retries(client, cfg, record, max_tokens=args.max_tokens,
                                temperature=args.temperature, log=log)
        row["config_hash"] = config_hash
        row["run_id"] = run_id
        with lock:
            fresh[str(record.get("item_id"))] = row
            done += 1
            if done % FLUSH_EVERY == 0 or done == len(todo):
                # Preserve every historical row; only add new ones.
                write_jsonl_atomic(preserved_raw + [fresh[k] for k in fresh],
                                   split_dir / "raw_responses.jsonl")
                write_json_atomic({"split": split, "run_id": run_id, "config_hash": config_hash,
                                   "records": len(records), "completed": len(cache) + len(fresh),
                                   "cache_hits": len(cache), "called": len(fresh),
                                   "updated_utc": datetime.now(timezone.utc).isoformat()},
                                  out_dir / f"progress_{split}.json")
                print(f"  {split:<5}: {len(cache) + len(fresh)}/{len(records)} completed")

    if todo:
        workers = max(1, min(args.workers, len(todo)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, todo))

    # Deterministic input order for every artifact.
    raw_rows: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    for record in records:
        item_id = str(record.get("item_id"))
        cached_row = cache.get(item_id)
        row = dict(cached_row) if cached_row is not None else dict(fresh.get(item_id) or {})
        row.setdefault("item_id", item_id)
        row.setdefault("split", record.get("split"))
        row["cached"] = cached_row is not None
        row["config_hash"] = config_hash
        raw_rows.append(row)

        reply = parse_json_object(row.get("response_text") or "")
        provenance = enrichment_provenance(
            model=cfg.model, applied_reasoning_mode=row.get("applied_reasoning_mode") or "unknown",
            requested_effort=cfg.reasoning_effort, temperature=args.temperature,
            prompt_sha256=prompt_sha256(build_messages(record)), run_id=run_id,
            fingerprint_before=row.get("immutable_fingerprint_before") or "",
        )
        merged, result = validate_and_merge(record, reply, provenance)
        result["attempts"] = row.get("attempts")
        result["cached"] = row["cached"]
        if row.get("error") and not result["accepted"]:
            result["reason_codes"] = list(dict.fromkeys(list(result["reason_codes"]) + ["api_error"]))
            result["details"] = list(result["details"]) + [str(row["error"])]
        results.append(result)

        if merged is not None:
            accepted.append(merged)
        else:
            rejected.append({
                "item_id": item_id,
                "split": record.get("split"),
                "reason_codes": result["reason_codes"],
                "details": result["details"],
                "attempts": row.get("attempts"),
                "cached": row["cached"],
            })

    write_jsonl_atomic(records, split_dir / "input_records.jsonl")
    write_jsonl_atomic(raw_rows, split_dir / "raw_responses.jsonl")
    write_jsonl_atomic(accepted, split_dir / "accepted_enriched.jsonl")
    write_jsonl_atomic(rejected, split_dir / "rejected_records.jsonl")
    write_jsonl_atomic(results, split_dir / "validation_results.jsonl")

    summary = summarize_split(records, results)
    summary["cache_hits"] = sum(1 for r in raw_rows if r.get("cached"))
    summary["api_calls"] = sum(int(r.get("attempts") or 0) for r in raw_rows if not r.get("cached"))
    summary["retries"] = sum(max(0, int(r.get("attempts") or 1) - 1)
                             for r in raw_rows if not r.get("cached"))
    latencies = [r["latency_ms"] for r in raw_rows if r.get("latency_ms")]
    summary["input_tokens_total"] = sum(r.get("input_tokens") or 0 for r in raw_rows)
    summary["output_tokens_total"] = sum(r.get("output_tokens") or 0 for r in raw_rows)
    summary["latency_ms_mean"] = round(sum(latencies) / len(latencies), 1) if latencies else None
    summary["latency_ms_max"] = max(latencies) if latencies else None
    summary["applied_reasoning_modes"] = sorted({str(r.get("applied_reasoning_mode"))
                                                 for r in raw_rows if r.get("applied_reasoning_mode")})
    summary["duplicate_items"] = len(raw_rows) - len({r["item_id"] for r in raw_rows})
    return summary


def write_reports(out_dir: Path, manifest: Mapping[str, Any],
                  summaries: Mapping[str, Mapping[str, Any]]) -> None:
    reports = out_dir / "reports"
    write_json_atomic(manifest, reports / "full_enrichment_report.json")

    with (reports / "rejection_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "reason_code", "records"])
        for split, summary in summaries.items():
            for code, count in (summary.get("reason_codes") or {}).items():
                writer.writerow([split, code, count])

    with (reports / "distribution_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "chart_type", "records", "accepted", "accept_rate"])
        for split, summary in summaries.items():
            for chart, stats in (summary.get("per_chart_type") or {}).items():
                writer.writerow([split, chart, stats["n"], stats["accepted"], stats["accept_rate"]])

    counts = manifest["counts"]
    lines = [
        "# Phase-3 full enrichment — train + val",
        "",
        f"- status: **{manifest['status']}**",
        f"- run id: `{manifest['run_id']}`  |  config hash: `{manifest['config_hash']}`",
        f"- spec: `{FULL_RUN_SPEC_VERSION}` / `{ENRICHMENT_SPEC_VERSION}`",
        f"- model: `{manifest['provider']['model']}`, requested reasoning "
        f"`{manifest['provider']['requested_reasoning_effort']}`, applied "
        f"`{manifest['provider']['applied_reasoning_mode']}`, temperature "
        f"{manifest['configuration']['temperature']}",
        "",
        "## Inputs",
        "",
        f"- train: {summaries.get('train', {}).get('input_records', 0)}",
        f"- val: {summaries.get('val', {}).get('input_records', 0)}",
        f"- test records processed: {manifest['exclusions']['test_records_processed']}",
        f"- human-evaluation items processed: "
        f"{manifest['exclusions']['human_eval_items_processed']}",
        "",
        "## Outcome",
        "",
        f"- accepted: {counts['accepted']} / {counts['input_total']} "
        f"(rate {counts['accept_rate']})",
        f"- schema-valid rate: {counts['schema_valid_rate']}",
        f"- rejected: {counts['rejected']}",
        f"- immutable-field violations: {counts['immutable_violations']}",
        f"- API calls: {counts['api_calls']}  |  cache hits: {counts['cache_hits']}  |  "
        f"retries: {counts['retries']}",
        f"- tokens: in={counts['input_tokens_total']} out={counts['output_tokens_total']}",
        "",
        "## Field lineage",
        "",
        f"- `llm_generated`: {', '.join(ENRICHABLE_FIELDS)}",
        "- all analytical and source-backed fields keep their existing lineage and are covered "
        "by the immutable fingerprint",
        "- these six fields are LLM-generated design annotations validated by automated checks "
        "and a reviewed pilot — not nvBench, human or expert gold",
        "",
        "## Gate failures",
        "",
    ]
    failures = manifest.get("gate_failures") or []
    lines += [f"- {failure}" for failure in failures] if failures else ["- none"]
    (out_dir / "reports" / "full_enrichment_report.md").write_text("\n".join(lines) + "\n",
                                                                   encoding="utf-8")


def main() -> int:  # noqa: C901 - linear orchestration
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    source_dir = _resolve(args.source_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = RunLog(out_dir / "logs" / "run.log")

    print("=" * 64)
    print("PHASE-3 FULL ENRICHMENT - TRAIN + VAL")
    print("=" * 64)

    # 1. human R1 gate
    r1 = verify_human_r1(_resolve(args.r1_file))
    print(f"  human R1      : {r1['accepted']}/{r1['rows']} accepted "
          f"(min {r1['min_accepted_required']}), kind={r1['review_kind']}, "
          f"reviewers={r1['reviewer_ids']}")
    if not r1["gate_passed"] and not args.skip_human_gate:
        print(f"  status        : ENRICHMENT_HUMAN_GATE_FAIL")
        print(f"  reason        : rows={r1['rows']} accepted={r1['accepted']} "
              f"incomplete={r1['rows_with_incomplete_review_fields']}")
        return 1

    # 2. provider configuration
    cfg = load_provider_config()
    problems = validate_config(cfg)
    if problems:
        print("  config        : invalid - " + "; ".join(problems))
        return 1
    config_hash = config_fingerprint(cfg.model, cfg.base_url, args.temperature,
                                     cfg.reasoning_effort)
    print(f"  config hash   : {config_hash}")

    # 3./4. inputs and exclusions
    splits = ("train", "val") if args.split == "both" else (args.split,)
    records = {split: load_split(source_dir, split) for split in splits}
    for split, rows in records.items():
        expected = EXPECTED_COUNTS[split]
        actual = len(rows)
        marker = "ok" if actual == expected or args.limit else f"EXPECTED {expected}"
        print(f"  {split:<5} input : {actual} ({marker})")
        if actual != expected and not args.limit:
            print("  status        : FULL_ENRICHMENT_INCOMPLETE")
            return 1
    if args.limit:
        records = {split: rows[:args.limit] for split, rows in records.items()}
    test_processed = sum(1 for rows in records.values() for r in rows
                         if str(r.get("split")) == "test")
    print(f"  exclusions    : test processed {test_processed}, human-eval items processed 0")

    if args.dry_run:
        print("  DRY-RUN       : gates and inputs verified, no API call made")
        print("=" * 64)
        return 0

    if not api_key_present(cfg):
        # Presence only -- variable names and a boolean, never any value.
        env = resolve_env()
        status_line = ", ".join(
            f"{name}={'set' if (env.get(name) or '').strip() else 'unset'}"
            for name in ("ENRICHMENT_BASE_URL", "ENRICHMENT_API_KEY",
                         "ENRICHMENT_MODEL", "ENRICHMENT_REASONING_EFFORT"))
        print(f"  env           : {status_line}")
        print(f"  .env present  : {DOTENV_PATH.exists()}")
        print(f"  {cfg.api_key_env} is not set in this shell or in .env "
              "(see docs/project/ENRICHMENT_PROVIDER_SETUP.md). PowerShell $env: values exist "
              "only in the window that set them.")
        print("  status        : MISSING_API_CREDENTIALS")
        return 1

    client = build_client(cfg)
    preflight = None
    if not args.report_only:
        try:
            preflight = create_completion(client, cfg,
                                          [{"role": "user", "content": "Reply with the single word: ready"}],
                                          max_tokens=256, temperature=args.temperature)
            print(f"  preflight     : ok ({preflight.latency_ms} ms, "
                  f"reasoning={preflight.applied_reasoning_mode})")
        except Exception as exc:  # noqa: BLE001
            print(f"  preflight     : FAILED - {sanitize_error(exc, cfg)}")
            return 1

    write_yaml({
        "full_run_spec_version": FULL_RUN_SPEC_VERSION,
        "enrichment_spec_version": ENRICHMENT_SPEC_VERSION,
        "prompt_sha256": prompt_sha256([{"role": "system", "content": SYSTEM_PROMPT}]),
        "config_hash": config_hash,
        "provider": "openai_compatible",
        "base_url": cfg.base_url,
        "model": cfg.model,
        "requested_reasoning_effort": cfg.reasoning_effort,
        "temperature": args.temperature,
        "pilot_temperature": PILOT_TEMPERATURE,
        "max_tokens": args.max_tokens,
        "workers": args.workers,
        "max_retries_per_item": MAX_RETRIES,
        "api_key_env": cfg.api_key_env,
        "api_key_value_recorded": False,
        "enrichable_fields": list(ENRICHABLE_FIELDS),
        "source_dir": str(source_dir),
        "splits": list(splits),
        "excluded": ["test.jsonl", "human_eval_test_items_40.csv"],
    }, out_dir / "config_snapshot.yaml")

    summaries: Dict[str, Dict[str, Any]] = {}
    for split in splits:
        summaries[split] = process_split(split, records[split], out_dir=out_dir, cfg=cfg,
                                         client=client, config_hash=config_hash, args=args,
                                         log=log, run_id=run_id)

    cache_verified = any(s.get("cache_hits", 0) > 0 for s in summaries.values())
    duplicates = sum(s.get("duplicate_items", 0) for s in summaries.values())
    status, failures = evaluate_quality_gates(
        summaries,
        expected_counts={s: EXPECTED_COUNTS[s] for s in splits} if not args.limit
        else {s: len(records[s]) for s in splits},
        test_records_processed=test_processed,
        human_eval_items_processed=0,
        duplicate_items=duplicates,
        cache_verified=cache_verified or args.limit is not None,
    )

    totals = {
        "input_total": sum(s["input_records"] for s in summaries.values()),
        "results_total": sum(s["results"] for s in summaries.values()),
        "accepted": sum(s["accepted"] for s in summaries.values()),
        "rejected": sum(s["rejected"] for s in summaries.values()),
        "schema_valid": sum(s["schema_valid"] for s in summaries.values()),
        "immutable_violations": sum(s["immutable_violations"] for s in summaries.values()),
        "api_calls": sum(s["api_calls"] for s in summaries.values()),
        "cache_hits": sum(s["cache_hits"] for s in summaries.values()),
        "retries": sum(s["retries"] for s in summaries.values()),
        "input_tokens_total": sum(s["input_tokens_total"] for s in summaries.values()),
        "output_tokens_total": sum(s["output_tokens_total"] for s in summaries.values()),
    }
    totals["accept_rate"] = round(totals["accepted"] / totals["results_total"], 4) if totals["results_total"] else 0.0
    totals["schema_valid_rate"] = round(totals["schema_valid"] / totals["results_total"], 4) if totals["results_total"] else 0.0

    manifest = {
        "stage": "phase3_full_enrichment_train_val",
        "full_run_spec_version": FULL_RUN_SPEC_VERSION,
        "enrichment_spec_version": ENRICHMENT_SPEC_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gate_failures": failures,
        "config_hash": config_hash,
        "human_r1_gate": r1,
        "configuration": {
            "temperature": args.temperature,
            "pilot_temperature": PILOT_TEMPERATURE,
            "temperature_matches_pilot": args.temperature == PILOT_TEMPERATURE,
            "max_tokens": args.max_tokens,
            "workers": args.workers,
            "max_retries_per_item": MAX_RETRIES,
            "prompt_sha256": prompt_sha256([{"role": "system", "content": SYSTEM_PROMPT}]),
        },
        "provider": provider_manifest_entry(cfg, preflight),
        "exclusions": {
            "test_records_processed": test_processed,
            "human_eval_items_processed": 0,
            "test_file_read": False,
            "human_eval_file_read": False,
        },
        "field_lineage": {
            "llm_generated": list(ENRICHABLE_FIELDS),
            "annotation_kind": "llm_generated_design_annotation",
            "not_gold": ["nvbench_gold", "human_gold", "expert_gold"],
        },
        "counts": totals,
        "per_split": summaries,
        "outputs": {split: f"{split}/" for split in splits},
    }
    write_json_atomic(manifest, out_dir / "manifest.json")
    write_reports(out_dir, manifest, summaries)

    print("-" * 64)
    for split, summary in summaries.items():
        print(f"  {split:<5}: accepted {summary['accepted']}/{summary['results']} "
              f"(rate {summary['accept_rate']}) | schema {summary['schema_valid_rate']} | "
              f"immutable {summary['immutable_violations']}")
    print(f"  totals       : accepted {totals['accepted']}/{totals['input_total']} "
          f"(rate {totals['accept_rate']})")
    print(f"  api calls    : {totals['api_calls']} | cache hits {totals['cache_hits']} | "
          f"retries {totals['retries']}")
    print(f"  tokens       : in={totals['input_tokens_total']} out={totals['output_tokens_total']}")
    for failure in failures:
        print(f"  gate failure : {failure}")
    print(f"  outputs      : {out_dir}")
    print(f"  status       : {status}")
    print("=" * 64)
    return 0 if status == "PASS_FULL_ENRICHMENT_READY_FOR_HYBRID_DATASET" else 1


if __name__ == "__main__":
    raise SystemExit(main())
