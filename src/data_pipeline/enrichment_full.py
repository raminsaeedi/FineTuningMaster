"""Full-run orchestration helpers for Phase-3 enrichment (train + val only).

Adds only what the sample/pilot runner lacked for a 1545-record run: input
loading with hard exclusion of held-out data, a configuration fingerprint for
caching, cache/resume bookkeeping, transient-error classification with bounded
retries, atomic writes, explicit field lineage, and quality-gate evaluation.

Validation, prompting and merging are reused unchanged from
:mod:`src.data_pipeline.enrichment` -- the exact versions that produced the
accepted technical sample and the reviewed pilot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.data_pipeline.enrichment import (
    ENRICHABLE_FIELDS,
    ENRICHMENT_SPEC_VERSION,
    SYSTEM_PROMPT,
    immutable_diff,
    immutable_fingerprint,
    merge_enrichment,
    parse_payload,
    validate_payload,
)
from src.utils.io import read_jsonl

FULL_RUN_SPEC_VERSION = "phase3-full-v1"

SPLIT_FILES = {"train": "train.jsonl", "val": "val.jsonl"}
EXPECTED_COUNTS = {"train": 1281, "val": 264}

# Any input path matching these is held-out and must never be read for enrichment.
FORBIDDEN_INPUT_PATTERNS = ("test", "human_eval")

# Lineage vocabulary for the full run.
LINEAGE_SOURCE_BACKED = "source_backed"
LINEAGE_DETERMINISTIC = "deterministically_derived"
LINEAGE_LLM = "llm_generated"

# Which existing lineage markers map onto which class.
_SOURCE_MARKERS = ("source-provided", "source_provided", "source-backed")
_DERIVED_MARKERS = ("rule-derived", "template-derived", "aggregate-expression", "derived")

MIN_SCHEMA_VALID_RATE = 0.95
MIN_ACCEPT_RATE = 0.95
# A chart type counts as systematically failing if it is both materially worse
# than the run and not explainable by one or two records.
SYSTEMATIC_CHART_FAIL_RATE = 0.5
SYSTEMATIC_CHART_MIN_N = 5
SYSTEMATIC_REASON_SHARE = 0.5

MAX_RETRIES = 2  # additional attempts per item, so at most 3 calls
BACKOFF_BASE_SECONDS = 2.0

_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524}
_TRANSIENT_MARKERS = ("timeout", "timed out", "temporarily", "rate limit", "overloaded",
                      "connection", "reset by peer", "bad gateway", "unavailable",
                      "internal server error", "incomplete chunked read")
FATAL_STATUS = {401: "MISSING_API_CREDENTIALS", 403: "MISSING_API_CREDENTIALS",
                404: "MODEL_NOT_ACCESSIBLE"}


# ------------------------------------------------------------------ inputs


def assert_allowed_input(path: str | Path) -> None:
    """Refuse held-out inputs (test split, human-evaluation items) outright."""
    name = Path(path).name.lower()
    for pattern in FORBIDDEN_INPUT_PATTERNS:
        if pattern in name:
            raise ValueError(f"refusing to read held-out input for enrichment: {name}")


def load_split(source_dir: str | Path, split: str) -> List[Dict[str, Any]]:
    """Load one split in file order; every record must carry that split label."""
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}, got {split!r}")
    path = Path(source_dir) / SPLIT_FILES[split]
    assert_allowed_input(path)
    records = read_jsonl(path)
    wrong = [r.get("item_id") for r in records if str(r.get("split")) != split]
    if wrong:
        raise ValueError(f"{len(wrong)} records in {path.name} are not labelled {split!r}")
    ids = [str(r.get("item_id")) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate item_id values in {path.name}")
    return records


# ------------------------------------------------------- configuration hash


def config_fingerprint(model: str, base_url: str, temperature: float,
                       reasoning_effort: Optional[str], max_tokens: Optional[int] = None) -> str:
    """Stable hash of everything that can change a response.

    A cached response is reused only when this fingerprint matches, so changing the
    model, prompt, temperature or reasoning effort invalidates the cache instead of
    silently mixing generations.

    ``max_tokens`` is deliberately **not** part of the fingerprint: the cap only
    truncates a response, it does not change how tokens are generated, so a reply
    that finished on its own (``finish_reason != "length"``) is identical under a
    larger cap. Truncated replies are excluded from the cache separately, so
    raising the budget re-calls exactly the truncated items and no others. Passing
    ``max_tokens`` reproduces the earlier fingerprint form, which keeps caches
    written before this change valid.
    """
    payload = {
        "full_run_spec": FULL_RUN_SPEC_VERSION,
        "enrichment_spec": ENRICHMENT_SPEC_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "enrichable_fields": list(ENRICHABLE_FIELDS),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------- cache/resume


def is_reusable_response(row: Mapping[str, Any]) -> bool:
    """A cached reply is reusable only if the model finished it by itself.

    A reply cut off at the token cap (``finish_reason == "length"``) is an
    incomplete artifact, never a valid generation, so it is re-called under the
    larger budget instead of being replayed.
    """
    if not (row.get("response_text") or "").strip():
        return False
    return row.get("finish_reason") != "length"


def load_raw_cache(path: str | Path, config_hash: str,
                   legacy_hashes: Optional[Sequence[str]] = None,
                   ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (reusable cache by item_id, all preserved raw rows).

    A row is reusable when its configuration fingerprint matches the current one
    (or a recognised legacy form, see :func:`config_fingerprint`) *and* the reply
    finished on its own. Every historical row is preserved and rewritten, so a
    successful raw response is never lost or overwritten.
    """
    path = Path(path)
    if not path.exists():
        return {}, []
    accepted_hashes = {config_hash, *(legacy_hashes or ())}
    rows = read_jsonl(path)
    cache: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("config_hash") not in accepted_hashes:
            continue
        if not is_reusable_response(row):
            continue
        cache[str(row.get("item_id"))] = row
    return cache, rows


def pending_records(records: Sequence[Mapping[str, Any]],
                    cache: Mapping[str, Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Records still needing an API call, in deterministic input order."""
    return [r for r in records if str(r.get("item_id")) not in cache]


# ------------------------------------------------------------ atomic writes


def write_jsonl_atomic(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write JSONL via a temp file + replace, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ------------------------------------------------------------ error handling


def classify_error(exc: BaseException) -> str:
    """``fatal`` (stop the run), ``transient`` (retry) or ``permanent`` (reject item)."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        status = int(status)
        if status in FATAL_STATUS:
            return "fatal"
        if status in _TRANSIENT_STATUS:
            return "transient"
        return "permanent"
    text = str(exc).lower()
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return "transient"
    return "permanent"


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff: 2 s, 8 s, 32 s ... (attempt is 1-based)."""
    return BACKOFF_BASE_SECONDS * (4 ** max(0, attempt - 1))


# ---------------------------------------------------------------- lineage


def lineage_classification(record: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Group the record's fields into the three lineage classes.

    The six enrichment fields are always ``llm_generated``; existing markers are
    mapped onto ``source_backed`` / ``deterministically_derived``. These fields are
    LLM-generated design annotations -- never nvBench, human or expert gold.
    """
    lineage = ((record.get("brief") or {}).get("extra") or {}).get("lineage") or {}
    classes: Dict[str, List[str]] = {
        LINEAGE_SOURCE_BACKED: [], LINEAGE_DETERMINISTIC: [], LINEAGE_LLM: [],
    }
    for field, marker in lineage.items():
        if field in ENRICHABLE_FIELDS:
            continue
        text = str(marker).lower()
        if any(m in text for m in _SOURCE_MARKERS):
            classes[LINEAGE_SOURCE_BACKED].append(field)
        elif any(m in text for m in _DERIVED_MARKERS):
            classes[LINEAGE_DETERMINISTIC].append(field)
    classes[LINEAGE_LLM] = list(ENRICHABLE_FIELDS)
    return {key: sorted(value) if key != LINEAGE_LLM else value for key, value in classes.items()}


def annotate_lineage(record: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the lineage classification to an already merged record (in place)."""
    extra = record.setdefault("brief", {}).setdefault("extra", {})
    enrichment = extra.setdefault("enrichment", {})
    enrichment["field_lineage"] = {field: LINEAGE_LLM for field in ENRICHABLE_FIELDS}
    enrichment["lineage_classes"] = lineage_classification(record)
    enrichment["annotation_kind"] = "llm_generated_design_annotation"
    return record


# -------------------------------------------------------------- validation


def validate_and_merge(record: Mapping[str, Any], reply_obj: Optional[Mapping[str, Any]],
                       provenance: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Run the validated pipeline for one reply: schema -> content -> merge -> hash.

    Returns (merged record or None, validation result dict). The validation result
    is written for every input record, accepted or not, so nothing is dropped.
    """
    item_id = str(record.get("item_id"))
    fingerprint_before = immutable_fingerprint(record)
    result: Dict[str, Any] = {
        "item_id": item_id,
        "split": record.get("split"),
        "schema_valid": False,
        "accepted": False,
        "reason_codes": [],
        "details": [],
        "normalizations": [],
        "immutable_fingerprint_before": fingerprint_before,
        "immutable_fingerprint_after": None,
    }
    if reply_obj is None:
        result["reason_codes"] = ["response_not_json"]
        result["details"] = ["reply was not a parsable JSON object"]
        return None, result

    payload, codes, details = parse_payload(reply_obj)
    if payload is None:
        result["reason_codes"], result["details"] = codes, details
        return None, result
    result["schema_valid"] = True
    result["normalizations"] = [d for d in details if d.startswith("normalized ")]

    codes, details = validate_payload(record, payload)
    merged = merge_enrichment(record, payload, provenance)
    fingerprint_after = immutable_fingerprint(merged)
    result["immutable_fingerprint_after"] = fingerprint_after
    if fingerprint_after != fingerprint_before:
        codes = list(dict.fromkeys(list(codes) + ["immutable_source_field_changed"]))
        details = list(details) + ["changed immutable keys: " + ", ".join(immutable_diff(record, merged))]

    result["reason_codes"], result["details"] = list(codes), list(details)
    if codes:
        return None, result

    annotate_lineage(merged)
    result["accepted"] = True
    return merged, result


# ------------------------------------------------------------ summary/gates


def _chart_of(record: Mapping[str, Any]) -> str:
    mappings = (record.get("recommendation") or {}).get("kpi_chart_mapping") or [{}]
    return str((mappings[0] if mappings else {}).get("chart_type", "unknown"))


def summarize_split(records: Sequence[Mapping[str, Any]],
                    results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-split counts, rates, reason codes and per-chart accept rates."""
    by_id = {str(r.get("item_id")): r for r in records}
    accepted = [r for r in results if r.get("accepted")]
    rejected = [r for r in results if not r.get("accepted")]
    schema_valid = [r for r in results if r.get("schema_valid")]

    reasons: Dict[str, int] = {}
    for result in rejected:
        for code in result.get("reason_codes") or []:
            reasons[code] = reasons.get(code, 0) + 1

    per_chart: Dict[str, Dict[str, int]] = {}
    for result in results:
        chart = _chart_of(by_id.get(str(result.get("item_id")), {}))
        bucket = per_chart.setdefault(chart, {"n": 0, "accepted": 0})
        bucket["n"] += 1
        bucket["accepted"] += 1 if result.get("accepted") else 0

    total = len(results)
    return {
        "input_records": len(records),
        "results": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "schema_valid": len(schema_valid),
        "schema_valid_rate": round(len(schema_valid) / total, 4) if total else 0.0,
        "accept_rate": round(len(accepted) / total, 4) if total else 0.0,
        "immutable_violations": sum(
            1 for r in results if "immutable_source_field_changed" in (r.get("reason_codes") or [])),
        "reason_codes": dict(sorted(reasons.items())),
        "per_chart_type": {c: {**v, "accept_rate": round(v["accepted"] / v["n"], 4)}
                           for c, v in sorted(per_chart.items())},
    }


def evaluate_quality_gates(summaries: Mapping[str, Mapping[str, Any]],
                           expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
                           test_records_processed: int = 0,
                           human_eval_items_processed: int = 0,
                           duplicate_items: int = 0,
                           cache_verified: bool = False) -> Tuple[str, List[str]]:
    """Apply the documented gates. Returns (status, list of failures)."""
    failures: List[str] = []
    total_input = sum(s["input_records"] for s in summaries.values())
    total_results = sum(s["results"] for s in summaries.values())
    expected_total = sum(expected_counts.values())

    for split, expected in expected_counts.items():
        actual = summaries.get(split, {}).get("input_records", 0)
        if actual != expected:
            failures.append(f"{split} input count {actual} != expected {expected}")
    if total_input != expected_total:
        failures.append(f"input total {total_input} != {expected_total}")
    if total_results != total_input:
        failures.append(f"only {total_results} of {total_input} input records accounted for")
    if test_records_processed:
        failures.append(f"{test_records_processed} test records processed")
    if human_eval_items_processed:
        failures.append(f"{human_eval_items_processed} human-evaluation items processed")
    if duplicate_items:
        failures.append(f"{duplicate_items} items processed more than once")
    if not cache_verified:
        failures.append("cache/resume behaviour not verified")

    accepted = sum(s["accepted"] for s in summaries.values())
    schema_valid = sum(s["schema_valid"] for s in summaries.values())
    immutable = sum(s["immutable_violations"] for s in summaries.values())
    accept_rate = accepted / total_results if total_results else 0.0
    schema_rate = schema_valid / total_results if total_results else 0.0

    if immutable:
        failures.append(f"{immutable} immutable-field violations")
    if schema_rate < MIN_SCHEMA_VALID_RATE:
        failures.append(f"schema-valid rate {schema_rate:.4f} < {MIN_SCHEMA_VALID_RATE}")
    if accept_rate < MIN_ACCEPT_RATE:
        failures.append(f"accept rate {accept_rate:.4f} < {MIN_ACCEPT_RATE}")

    for chart, stats in _merged_charts(summaries).items():
        if stats["n"] >= SYSTEMATIC_CHART_MIN_N and stats["accept_rate"] < SYSTEMATIC_CHART_FAIL_RATE:
            failures.append(f"systematic failure for chart type {chart} "
                            f"({stats['accepted']}/{stats['n']} accepted)")

    rejected_total = sum(s["rejected"] for s in summaries.values())
    for code, count in _merged_reasons(summaries).items():
        if rejected_total and count / rejected_total > SYSTEMATIC_REASON_SHARE and count >= 20:
            failures.append(f"systematic rejection pattern {code} ({count} records)")

    if total_results < total_input:
        return "FULL_ENRICHMENT_INCOMPLETE", failures
    if failures:
        return "FULL_ENRICHMENT_QUALITY_GATE_FAIL", failures
    return "PASS_FULL_ENRICHMENT_READY_FOR_HYBRID_DATASET", failures


def _merged_charts(summaries: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for summary in summaries.values():
        for chart, stats in (summary.get("per_chart_type") or {}).items():
            bucket = merged.setdefault(chart, {"n": 0, "accepted": 0})
            bucket["n"] += stats["n"]
            bucket["accepted"] += stats["accepted"]
    for stats in merged.values():
        stats["accept_rate"] = round(stats["accepted"] / stats["n"], 4) if stats["n"] else 0.0
    return merged


def _merged_reasons(summaries: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for summary in summaries.values():
        for code, count in (summary.get("reason_codes") or {}).items():
            merged[code] = merged.get(code, 0) + count
    return dict(sorted(merged.items()))


# --------------------------------------------------------------- human gate


def verify_human_r1(path: str | Path, expected_rows: int = 30,
                    min_accepted: int = 27) -> Dict[str, Any]:
    """Read the human R1 audit file and evaluate the gate. Never modifies it."""
    import csv

    path = Path(path)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    accepted = [r for r in rows if str(r.get("overall_accept", "")).strip() in ("1", "yes", "true")]
    reviewer_ids = sorted({(r.get("reviewer_id") or "").strip() for r in rows})
    incomplete = [r.get("item_id") for r in rows
                  if not (r.get("reviewer_id") or "").strip()
                  or not str(r.get("overall_accept", "")).strip()]
    immutable_flags = [r.get("item_id") for r in rows
                       if (r.get("immutable_violation_found") or "").strip().lower()
                       not in ("", "none found", "none", "no", "0")]
    placeholders = [r.get("item_id") for r in rows
                    if any(re.fullmatch(r"x+|\?+|tbd|todo", (v or "").strip(), re.IGNORECASE)
                           for v in r.values())]

    passed = (len(rows) == expected_rows and len(accepted) >= min_accepted
              and not incomplete and not immutable_flags)
    return {
        "path": str(path),
        "rows": len(rows),
        "accepted": len(accepted),
        "rejected": len(rows) - len(accepted),
        "min_accepted_required": min_accepted,
        "reviewer_ids": reviewer_ids,
        # An AI pre-check is recorded honestly; it is not evidence of human review.
        "review_kind": ("ai_precheck" if any("ai" in r.lower() for r in reviewer_ids if r)
                        else "human_review"),
        "rows_with_incomplete_review_fields": incomplete,
        "rows_flagging_immutable_violation": immutable_flags,
        "rows_with_placeholder_values": placeholders,
        "gate_passed": passed,
    }
