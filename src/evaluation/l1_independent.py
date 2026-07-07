"""Independent L1 chart-selection scorer (set-valued, literature-based).

Scores the model's *primary* chart per KPI against an **independent** set of
human-effective charts derived from `data/eval/l1_chart_effectiveness_v1.csv`
(Saket 2019 + Kim & Heer 2018) — a label lineage disjoint from the synthetic
generator's `TASK_CHART` rule.

Keying: the effective-chart set is looked up by the item's `task_type`, aggregated
across `data_shape` rows into one set per task_type.

Two keying cases, reported distinctly by callers:
  * cached SYNTHETIC v1 predictions — the gold `task_type` is generator-derived, so
    L1 scores on them are DIAGNOSTIC / LIMITED (they still test chart choice against
    independent literature, but the task label shares the generator lineage);
  * `benchmark_v1` — `task_type` is independently assigned, so L1 scores are
    independent (covered items only).

L1 LIMITATION: this validates only chart-selection *acceptability for covered task
types*. It does NOT validate layout, styling, interaction, rationale, or overall
dashboard-design quality. Coverage is reported honestly and uncovered items are
excluded from accuracy — never counted correct.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.evaluation.metrics.base import chart_token, index_references, predicted_charts

DEFAULT_L1_CSV = "data/eval/l1_chart_effectiveness_v1.csv"


def load_effective_sets(csv_path: str | Path) -> Dict[str, Set[str]]:
    """Load `task_type -> set(effective chart tokens)` from the L1 CSV.

    The set for a task_type is the union of `effective_charts` across all its
    `data_shape` rows.
    """
    sets: Dict[str, Set[str]] = defaultdict(set)
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tt = (row.get("task_type") or "").strip()
            if not tt:
                continue
            for c in (row.get("effective_charts") or "").split("|"):
                c = chart_token(c)
                if c:
                    sets[tt].add(c)
    return dict(sets)


def _primary_chart_by_kpi(result) -> Dict[str, str]:
    """Map each predicted KPI (lowercased) to its primary chart token."""
    out: Dict[str, str] = {}
    if result is None or result.parsed is None:
        return out
    for m in result.parsed.kpi_chart_mapping:
        kpi = str(getattr(m, "kpi", "") or "").strip().lower()
        if kpi and kpi not in out and m.chart_type is not None:
            out[kpi] = chart_token(m.chart_type)
    return out


def score_l1(results, references, effective_sets: Dict[str, Set[str]]) -> dict:
    """Independent L1 score over cached predictions vs gold references.

    For each gold KPI entry (`kpi`, `task_type`): if the task_type is covered by the
    L1 table, the item is scored — the model's primary chart for that KPI must be in
    the effective set. Parse failure or a missing KPI on a covered entry counts as
    wrong. Uncovered entries are excluded from accuracy and reported.
    """
    ref_by_id = index_references(references or [])
    pred_by_id = {r.item_id: r for r in results}

    n_gold_kpi = 0
    n_covered = 0
    covered_correct = 0
    per_task: Dict[str, Dict[str, int]] = defaultdict(lambda: {"covered": 0, "correct": 0})
    pred_accept: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "accepted": 0})
    uncovered_tasks: Dict[str, int] = defaultdict(int)

    scored_item_ids: Set[str] = set()
    for item_id, ref in ref_by_id.items():
        reco = ref.get("recommendation", {}) or {}
        gold_mapping = reco.get("kpi_chart_mapping", []) or []
        pred_primary = _primary_chart_by_kpi(pred_by_id.get(item_id))
        for entry in gold_mapping:
            if not isinstance(entry, dict):
                continue
            task = (entry.get("task_type") or "").strip()
            kpi = str(entry.get("kpi") or "").strip().lower()
            if not task:
                continue
            n_gold_kpi += 1
            if task not in effective_sets:
                uncovered_tasks[task] += 1
                continue
            # Covered: score the model's primary chart for this KPI.
            n_covered += 1
            per_task[task]["covered"] += 1
            scored_item_ids.add(item_id)
            pred_chart = pred_primary.get(kpi)
            is_correct = pred_chart is not None and pred_chart in effective_sets[task]
            if pred_chart is not None:
                pred_accept[pred_chart]["n"] += 1
                pred_accept[pred_chart]["accepted"] += int(is_correct)
            if is_correct:
                covered_correct += 1
                per_task[task]["correct"] += 1

    coverage_rate = (n_covered / n_gold_kpi) if n_gold_kpi else None
    covered_accuracy = (covered_correct / n_covered) if n_covered else None
    return {
        "n_items_scored": len(scored_item_ids),
        "n_gold_kpi": n_gold_kpi,
        "n_covered": n_covered,
        "n_uncovered": n_gold_kpi - n_covered,
        "coverage_rate": round(coverage_rate, 4) if coverage_rate is not None else None,
        "covered_accuracy": round(covered_accuracy, 4) if covered_accuracy is not None else None,
        "covered_correct": covered_correct,
        "per_task_type": {
            t: {
                "covered": v["covered"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["covered"], 4) if v["covered"] else None,
            }
            for t, v in sorted(per_task.items())
        },
        "predicted_chart_acceptance": {
            c: {
                "n": v["n"],
                "accepted": v["accepted"],
                "acceptance_rate": round(v["accepted"] / v["n"], 4) if v["n"] else None,
            }
            for c, v in sorted(pred_accept.items())
        },
        "uncovered_task_types": dict(sorted(uncovered_tasks.items())),
        "covered_task_types": sorted(effective_sets.keys()),
    }


def _evidence_strength(item: dict) -> str:
    """strong = real_public + literature_L1; otherwise weak."""
    if item.get("source_type") == "real_public" and item.get("label_source") == "literature_L1":
        return "strong"
    return "weak"


def score_benchmark(results, benchmark_items: List[dict]) -> dict:
    """Independent benchmark chart-selection score against `acceptable_chart_types`.

    Joins predictions to benchmark items by `item_id` (== `benchmark_id`). Only items
    with `suitable_for_auto_scoring == true` are **covered**; the model's primary chart
    must be in the item's `acceptable_chart_types` set. Parse failure or a missing
    primary chart on a covered item counts as WRONG. Uncovered items are excluded from
    accuracy but counted in coverage. Uses NO synthetic gold labels.
    """
    pred_by_id = {r.item_id: r for r in results}

    n_total = len(benchmark_items)
    n_covered = covered_correct = parse_failures = 0
    per_task: Dict[str, Dict[str, int]] = defaultdict(lambda: {"covered": 0, "correct": 0})
    per_domain: Dict[str, Dict[str, int]] = defaultdict(lambda: {"covered": 0, "correct": 0})
    per_evidence: Dict[str, Dict[str, int]] = defaultdict(lambda: {"covered": 0, "correct": 0})

    for item in benchmark_items:
        if not item.get("suitable_for_auto_scoring"):
            continue
        n_covered += 1
        task = item.get("task_type", "?")
        domain = item.get("domain", "?")
        evidence = _evidence_strength(item)
        acceptable = {chart_token(c) for c in item.get("acceptable_chart_types", []) or []}

        result = pred_by_id.get(item.get("benchmark_id") or item.get("item_id"))
        preds = predicted_charts(result) if result is not None else []
        primary = preds[0] if preds else None
        if primary is None:
            parse_failures += 1
        is_correct = primary is not None and primary in acceptable

        per_task[task]["covered"] += 1
        per_domain[domain]["covered"] += 1
        per_evidence[evidence]["covered"] += 1
        if is_correct:
            covered_correct += 1
            per_task[task]["correct"] += 1
            per_domain[domain]["correct"] += 1
            per_evidence[evidence]["correct"] += 1

    def _acc(d):
        return {k: {"covered": v["covered"], "correct": v["correct"],
                    "accuracy": round(v["correct"] / v["covered"], 4) if v["covered"] else None}
                for k, v in sorted(d.items())}

    return {
        "n_total": n_total,
        "n_covered": n_covered,
        "n_uncovered": n_total - n_covered,
        "coverage_rate": round(n_covered / n_total, 4) if n_total else None,
        "covered_accuracy": round(covered_correct / n_covered, 4) if n_covered else None,
        "covered_correct": covered_correct,
        "parse_failures": parse_failures,
        "per_task_type": _acc(per_task),
        "per_domain": _acc(per_domain),
        "evidence_strength": _acc(per_evidence),
    }
