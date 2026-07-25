"""Pilot v5 assembly: strict selection + corpus-level v4-vs-v5 comparison.

Selection reuses ``select_pilot_v4`` in its default **strict** mode (no
``allow_partial``): if any chart bucket cannot reach the target from corrected
Tier-A candidates, the build refuses rather than shipping a partial pilot.

The before/after comparison re-applies the full v5 quality gate (KPI/SQL
aggregation agreement, required time-grain/grouping preservation, graduated
scoring) to both corpora at the corpus level -- never by item id -- so it stays
meaningful even when the v5 selection legitimately fails on an under-supplied
chart type.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List

from src.data_pipeline.nvbench_pilot import _mapping0, _prov, duplicate_checks
from src.data_pipeline.nvbench_pilot_v4 import select_pilot_v4  # re-exported for callers
from src.data_pipeline.nvbench_quality import (
    CHART_CHECKERS,
    check_required_constraints,
    fidelity_signals,
    kpi_suitability,
    score_and_tier,
)

__all__ = ["select_pilot_v4", "before_after_v4_v5"]

_KPI_CONFLICT_RULES = frozenset({
    "kpi_sql_aggregation_conflict", "mixed_aggregate_ambiguous_kpi",
    "query_aggregation_conflict", "broad_intent_mismatch",
})


def _corpus_quality_metrics_v5(
    records: List[Dict[str, Any]], mapping: Dict[str, Any], resolver, profiler, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    fidelity_failed_map = fidelity_signals(records, mapping, resolver)
    chart_dist: collections.Counter = collections.Counter()
    db_dist: collections.Counter = collections.Counter()
    tier_dist: collections.Counter = collections.Counter()
    scores: List[int] = []
    kpi_conflict = missing_time_grain = missing_grouping = 0
    identifier_as_measure = chart_inappropriate = 0

    for rec in records:
        iid = rec["item_id"]
        m = _mapping0(rec)
        prov = _prov(rec)
        chart_type = m.get("chart_type", "")
        chart_dist[chart_type] += 1
        db_dist[prov.get("db_id", "?")] += 1

        kpi_result = kpi_suitability(rec, profiler, cfg)
        checker = CHART_CHECKERS.get(chart_type)
        chart_result = checker(rec, profiler, cfg) if checker else {
            "passed": False, "failed_rules": [f"unknown_chart_type:{chart_type}"], "warnings": [], "evidence": {},
        }
        constraint_result = check_required_constraints(rec, profiler, cfg)
        fidelity_failed = fidelity_failed_map.get(iid, [])
        quality = score_and_tier(rec, kpi_result, chart_result, fidelity_failed, cfg,
                                 constraint_result=constraint_result)
        tier_dist[quality["tier"]] += 1
        scores.append(quality["quality_score"])

        all_failed = set(quality["failed_rules"])
        if all_failed & _KPI_CONFLICT_RULES:
            kpi_conflict += 1
        if "missing_required_time_grain" in all_failed:
            missing_time_grain += 1
        if "missing_required_grouping" in all_failed:
            missing_grouping += 1
        if any(r.startswith("identifier_") or r == "meaningless_identifier_aggregation" for r in all_failed):
            identifier_as_measure += 1
        if not chart_result["passed"]:
            chart_inappropriate += 1

    dup_checks, _ = duplicate_checks(records, strict=False)
    dup_summary = {c["check"]: c["n"] for c in dup_checks if not c["passed"]}

    return {
        "n": len(records),
        "chart_distribution": dict(chart_dist),
        "database_distribution": dict(db_dist),
        "quality_tier_distribution": dict(tier_dist),
        "kpi_conflict_count": kpi_conflict,
        "missing_time_grain_count": missing_time_grain,
        "missing_grouping_count": missing_grouping,
        "identifier_as_measure_count": identifier_as_measure,
        "chart_inappropriate_count": chart_inappropriate,
        "quality_score_range": {"min": min(scores) if scores else None, "max": max(scores) if scores else None},
        "duplicate_findings": dup_summary,
    }


def before_after_v4_v5(
    v4_records: List[Dict[str, Any]],
    v5_records: List[Dict[str, Any]],
    mapping: Dict[str, Any],
    resolver,
    profiler,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Corpus-level v4-vs-v5 comparison under the corrected v5 quality gate.

    ``v5_records`` may be the v5 selected pilot (on PASS) or the v5 corrected
    Tier-A pool (on an insufficiency FAIL) -- either way the comparison is
    corpus-level and never keyed by item id.
    """
    return {
        "note": (
            "Corpus-level comparison under the v5 quality gate. v4 records are the "
            "previously-shipped 95-record partial pilot; v5 records are the corrected "
            "corpus (selected pilot on PASS, or corrected Tier-A pool on insufficiency FAIL)."
        ),
        "v4": _corpus_quality_metrics_v5(v4_records, mapping, resolver, profiler, cfg),
        "v5": _corpus_quality_metrics_v5(v5_records, mapping, resolver, profiler, cfg),
    }
