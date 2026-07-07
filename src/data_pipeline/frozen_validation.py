"""Validation logic for frozen dataset v2 (pure, testable core).

The CLI wrapper (``experiments/scripts/validate_frozen_dataset.py``) uses these
functions to check the frozen files and write ``validation_report.md`` +
``hashes.json``. Everything here is side-effect free except :func:`sha256_of_file`.

Checks implemented (see DATASET_V2_IMPLEMENTATION_PLAN.md §5):
  - valid JSON parsing (line level)
  - Pydantic schema validity + strict TaskType/ChartType enums
  - non-empty required fields
  - duplicate item_id / duplicate brief fingerprint
  - domain / task_type / chart_type distributions
  - leakage between {train, val} and {internal_test, real_briefs}
  - SHA256 hashing of frozen files
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.schemas import ChartType, DashboardBrief, GoldItem, TaskType
from src.data_pipeline.builders.leakage import fingerprint
from src.evaluation.metrics.schema_compliance import completeness_fraction, full_schema_valid

_TASK_VALUES = {t.value for t in TaskType}
_CHART_VALUES = {c.value for c in ChartType}


def read_jsonl_strict(path: str | Path) -> Tuple[List[dict], List[str]]:
    """Read a JSONL file, returning (records, parse_errors).

    Unlike ``utils.io.read_jsonl`` this does NOT silently skip malformed lines —
    it reports them, which is exactly what the validator needs to catch.
    """
    path = Path(path)
    records: List[dict] = []
    errors: List[str] = []
    if not path.exists():
        return records, [f"file not found: {path}"]
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_num}: invalid JSON ({exc})")
    return records, errors


def validate_record(record: dict) -> List[str]:
    """Return a list of problems for one gold record (empty == valid)."""
    problems: List[str] = []
    brief = record.get("brief")
    rec = record.get("recommendation")
    if not isinstance(brief, dict):
        return ["missing or non-object 'brief'"]
    if not isinstance(rec, dict):
        return ["missing or non-object 'recommendation'"]

    # Full Pydantic contract (also enforces TaskType/ChartType on the mapping).
    try:
        GoldItem(
            item_id=record.get("item_id", ""),
            brief=DashboardBrief(**brief),
            recommendation=rec,
        )
    except Exception as exc:  # noqa: BLE001 - report any validation failure
        problems.append(f"schema invalid: {exc}")

    # Strict enums on the raw object (independent of lenient repair).
    for m in rec.get("kpi_chart_mapping", []) or []:
        if isinstance(m, dict):
            if m.get("task_type") not in _TASK_VALUES:
                problems.append(f"invalid task_type: {m.get('task_type')!r}")
            if m.get("chart_type") not in _CHART_VALUES:
                problems.append(f"invalid chart_type: {m.get('chart_type')!r}")

    # Non-empty required fields on the brief.
    for field in ("users", "goals", "kpis", "columns"):
        val = brief.get(field)
        if not val:
            problems.append(f"empty required brief field: {field}")

    # DesignOutput required keys present AND non-empty.
    if completeness_fraction(rec) < 1.0:
        problems.append("recommendation has missing/empty required keys")
    if not full_schema_valid(rec):
        problems.append("recommendation fails full schema validation")

    return problems


def find_duplicate_ids(records: List[dict]) -> List[str]:
    counts = Counter(r.get("item_id", "") for r in records)
    return sorted(i for i, c in counts.items() if c > 1)


def find_duplicate_fingerprints(records: List[dict]) -> List[str]:
    """Return item_ids that share a brief fingerprint with an earlier record."""
    seen: Dict[str, str] = {}
    dups: List[str] = []
    for r in records:
        fp = fingerprint(r.get("brief", {}))
        if fp in seen:
            dups.append(r.get("item_id", ""))
        else:
            seen[fp] = r.get("item_id", "")
    return dups


def distributions(records: List[dict]) -> Dict[str, Dict[str, int]]:
    """Domain, task_type and chart_type histograms across records."""
    domain: Counter = Counter()
    task: Counter = Counter()
    chart: Counter = Counter()
    for r in records:
        rec = r.get("recommendation", {})
        cs = rec.get("context_summary", {}) or {}
        dom = cs.get("domain") or (r.get("brief", {}).get("extra", {}) or {}).get("domain")
        if dom:
            domain[dom] += 1
        for m in rec.get("kpi_chart_mapping", []) or []:
            if isinstance(m, dict):
                if m.get("task_type"):
                    task[m["task_type"]] += 1
                if m.get("chart_type"):
                    chart[m["chart_type"]] += 1
    return {"domain": dict(domain), "task_type": dict(task), "chart_type": dict(chart)}


def leakage_report(
    train_val: List[dict], eval_records: List[dict]
) -> Dict[str, List[str]]:
    """Detect any item_id or brief-fingerprint overlap between the two groups."""
    tv_ids = {r.get("item_id", "") for r in train_val}
    tv_fps = {fingerprint(r.get("brief", {})) for r in train_val}
    id_overlap: List[str] = []
    fp_overlap: List[str] = []
    for r in eval_records:
        if r.get("item_id", "") in tv_ids:
            id_overlap.append(r.get("item_id", ""))
        if fingerprint(r.get("brief", {})) in tv_fps:
            fp_overlap.append(r.get("item_id", ""))
    return {"item_id_overlap": id_overlap, "fingerprint_overlap": fp_overlap}


def sha256_of_file(path: str | Path) -> str:
    """SHA256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
