"""Validation logic for the independent benchmark (`benchmark_v1`), pure & testable.

The CLI (`experiments/scripts/validate_benchmark.py`) uses these to produce
`experiments/results/benchmark_dataset_report.md` with the ten required checks,
including source/provenance leakage vs training and label-lineage leakage vs the
synthetic generator (`TASK_CHART`).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Set

from src.core.schemas import ChartType, TaskType
from src.data_pipeline.builders.leakage import fingerprint

TASK_VALUES: Set[str] = {t.value for t in TaskType}
CHART_VALUES: Set[str] = {c.value for c in ChartType}

_REQUIRED = (
    "benchmark_id", "domain", "users", "goals", "kpis", "columns", "task_type",
    "acceptable_chart_types", "rationale", "source_name", "source_type",
    "source_reference", "license_or_usage_note", "label_source", "label_confidence",
    "suitable_for_auto_scoring", "suitable_for_human_eval",
)


def item_brief(item: dict) -> dict:
    """Brief-like view for fingerprinting/leakage checks."""
    return {
        "users": item.get("users", ""),
        "goals": item.get("goals", []),
        "kpis": item.get("kpis", []),
        "columns": item.get("columns", []),
    }


def validate_item(item: dict) -> List[str]:
    """Return schema/enum/non-empty problems for one benchmark item."""
    problems: List[str] = []
    for f in _REQUIRED:
        if f not in item:
            problems.append(f"missing field: {f}")
    for f in ("users", "goals", "kpis", "columns", "acceptable_chart_types"):
        if f in item and not item[f]:
            problems.append(f"empty required field: {f}")
    if item.get("task_type") not in TASK_VALUES:
        problems.append(f"invalid task_type: {item.get('task_type')!r}")
    for c in item.get("acceptable_chart_types", []) or []:
        if c not in CHART_VALUES:
            problems.append(f"invalid chart in acceptable_chart_types: {c!r}")
    if item.get("source_type") not in {"real_public", "realistic_manual"}:
        problems.append(f"invalid source_type: {item.get('source_type')!r}")
    if item.get("label_source") not in {"literature_L1", "manual_expert"}:
        problems.append(f"invalid label_source: {item.get('label_source')!r}")
    return problems


def distributions(items: List[dict]) -> Dict[str, Dict[str, int]]:
    domain = Counter(it.get("domain", "?") for it in items)
    task = Counter(it.get("task_type", "?") for it in items)
    chart_label = Counter(c for it in items for c in it.get("acceptable_chart_types", []) or [])
    return {"domain": dict(domain), "task_type": dict(task), "chart_label": dict(chart_label)}


def chart_type_coverage(items: List[dict]) -> Dict[str, List[str]]:
    seen = {c for it in items for c in it.get("acceptable_chart_types", []) or []}
    return {
        "covered": sorted(seen & CHART_VALUES),
        "not_covered": sorted(CHART_VALUES - seen),
    }


def evidence_split(items: List[dict]) -> Dict[str, int]:
    strong = sum(1 for it in items
                 if it.get("source_type") == "real_public" and it.get("label_source") == "literature_L1")
    return {"strong": strong, "weak": len(items) - strong}


def scoring_split(items: List[dict]) -> Dict[str, int]:
    auto = sum(1 for it in items if it.get("suitable_for_auto_scoring"))
    human = sum(1 for it in items if it.get("suitable_for_human_eval"))
    return {"auto_scorable": auto, "human_eval": human, "human_eval_only": len(items) - auto}


def source_leakage(items: List[dict], train_briefs: List[dict]) -> List[str]:
    """benchmark_ids whose brief fingerprint collides with a training brief."""
    train_fps = {fingerprint(b) for b in train_briefs}
    return [it["benchmark_id"] for it in items if fingerprint(item_brief(it)) in train_fps]


def label_lineage_check(items: List[dict], effective_sets: Dict[str, Set[str]],
                        generator_sets: Dict[str, Set[str]]) -> Dict[str, object]:
    """Verify labels are independent of the generator.

    - label_source must be literature_L1 or manual_expert (never a generator source);
    - literature_L1 items must match the independent L1 table for their task_type;
    - report (informational) how many acceptable sets are identical to the generator's
      TASK_CHART set for the same task_type.
    """
    bad_source = [it["benchmark_id"] for it in items
                  if it.get("label_source") not in {"literature_L1", "manual_expert"}]
    l1_mismatch: List[str] = []
    identical_to_generator: List[str] = []
    for it in items:
        task = it.get("task_type")
        acc = set(it.get("acceptable_chart_types", []) or [])
        if it.get("label_source") == "literature_L1":
            if acc != set(effective_sets.get(task, set())):
                l1_mismatch.append(it["benchmark_id"])
        if task in generator_sets and acc == generator_sets[task]:
            identical_to_generator.append(it["benchmark_id"])
    return {
        "label_source_ok": not bad_source,
        "bad_label_source": bad_source,
        "l1_mismatch": l1_mismatch,
        "identical_to_generator_set": identical_to_generator,
    }
