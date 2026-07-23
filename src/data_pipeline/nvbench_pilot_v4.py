"""Pilot v4 assembly: select from the Tier-A quality pool, compare to v3.

Reuses ``select_pilot_v3`` unmodified (it already implements one-per-group,
exact-goal-dedup, chart-balanced admission, near-duplicate-awareness, and
DB-cap-with-fallback) -- the only new logic is refusing to build a pilot when
the Tier-A supply cannot satisfy the target without a fallback, per the task's
explicit insufficient-candidate policy: never relax quality rules, never fall
back to Tier B, never silently accept an over-cap admission.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Tuple

from src.data_pipeline.nvbench_pilot import _mapping0, _prov, _record, duplicate_checks, select_pilot_v3
from src.data_pipeline.nvbench_quality import CHART_CHECKERS, kpi_suitability, score_and_tier
from src.data_pipeline.nvbench_source import item_chart

NORMALIZED_CHART_TYPES = ("bar", "line", "pie", "scatter", "stacked_bar")


def select_pilot_v4(
    tier_a_items: List[Any],
    *,
    seed: int = 42,
    target_per_chart: int = 20,
    db_cap: int = 10,
    near_dup_threshold: float = 0.8,
    allow_partial: bool = False,
) -> Tuple[Optional[List[Any]], Dict[str, Any]]:
    """Select up to ``target_per_chart`` Tier-A records per chart.

    By default (``allow_partial=False``), returns ``(None, report)`` with
    ``report["status"] != "ok"`` whenever any chart bucket cannot reach exactly
    ``target_per_chart`` -- never a partial/relaxed pilot.

    ``allow_partial=True`` is an explicit, user-approved exception to that
    default: a chart whose Tier-A supply genuinely tops out below the target
    *after* mandatory one-per-group + exact-goal-dedup + near-duplicate
    exclusion (never by weakening those rules) is accepted at its true maximum
    instead of blocking the whole pilot, and the shortfall is recorded in the
    report rather than silently absorbed. A database-cap fallback is a
    different kind of relaxation (over-cap admission, not "fewer records") and
    is still always refused, regardless of ``allow_partial``.
    """
    buckets: Dict[str, List[Any]] = collections.defaultdict(list)
    for it in tier_a_items:
        buckets[item_chart(it)].append(it)
    available = {c: len(buckets.get(c, [])) for c in NORMALIZED_CHART_TYPES}
    short = {c: n for c, n in available.items() if n < target_per_chart}
    if short and not allow_partial:
        return None, {
            "status": "insufficient_tier_a_candidates",
            "available_per_chart": available,
            "short_charts": sorted(short),
            "target_per_chart": target_per_chart,
        }

    selected, report = select_pilot_v3(
        tier_a_items, seed=seed, target_per_chart=target_per_chart,
        db_cap=db_cap, near_dup_threshold=near_dup_threshold,
    )
    if report["fallbacks"]:
        # A database-cap fallback is always refused, allow_partial or not: it's
        # an over-cap admission, not simply "fewer records than the target".
        over_cap_charts = sorted({f["chart"] for f in report["fallbacks"]})
        return None, {
            "status": "insufficient_tier_a_candidates_within_db_cap",
            "available_per_chart": available,
            "short_charts": over_cap_charts,
            "target_per_chart": target_per_chart,
            "db_cap": db_cap,
            "fallbacks_would_have_been": report["fallbacks"],
        }

    # Raw per-chart Tier-A supply can still fall short of the target after
    # select_pilot_v3's own one-per-group + exact-goal-dedup + near-duplicate-
    # exclusion reduction, even when no database-cap fallback fired -- the raw
    # ``available`` count above is only a pre-check, not the final word.
    short_after_selection = {c: n for c, n in report["chart_counts"].items() if n != target_per_chart}
    if short_after_selection and not allow_partial:
        return None, {
            "status": "insufficient_tier_a_candidates_after_dedup",
            "available_per_chart": available,
            "selected_per_chart": report["chart_counts"],
            "short_charts": sorted(short_after_selection),
            "target_per_chart": target_per_chart,
        }

    report["status"] = "ok" if not short_after_selection else "ok_partial_documented_shortfall"
    report["available_per_chart"] = available
    report["shortfall"] = short_after_selection or None
    return selected, report


# --------------------------------------------------------------------------- #
# before/after v3 vs v4 (corpus-level, never by item id)
# --------------------------------------------------------------------------- #
def _identifier_related(failed_rules: List[str]) -> bool:
    return any(
        r.startswith("identifier_") or r in ("meaningless_identifier_aggregation", "possible_identifier_aggregation")
        for r in failed_rules
    )


def _corpus_quality_metrics(
    records: List[Dict[str, Any]], mapping: Dict[str, Any], resolver, profiler, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Retroactively apply the Phase 4/5 quality signals to a serialized corpus."""
    from src.data_pipeline.nvbench_quality import fidelity_signals  # local import: avoid cycle at module load

    fidelity_failed_map = fidelity_signals(records, mapping, resolver)
    chart_dist: collections.Counter = collections.Counter()
    db_dist: collections.Counter = collections.Counter()
    tier_dist: collections.Counter = collections.Counter()
    identifier_as_measure = 0
    meaningless_kpi = 0
    chart_inappropriate = 0
    source_conflict = 0
    missing_constraint = 0
    uncertain_pie = invalid_scatter = invalid_line = invalid_stacked_bar = invalid_bar = 0

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
        fidelity_failed = fidelity_failed_map.get(iid, [])
        quality = score_and_tier(rec, kpi_result, chart_result, fidelity_failed, cfg)
        tier_dist[quality["tier"]] += 1

        if _identifier_related(kpi_result["failed_rules"]) or _identifier_related(chart_result["failed_rules"]):
            identifier_as_measure += 1
        if not kpi_result["suitable"]:
            meaningless_kpi += 1
        if "broad_intent_mismatch" in kpi_result["failed_rules"]:
            source_conflict += 1
        if not chart_result["passed"]:
            chart_inappropriate += 1
            if chart_type == "pie":
                uncertain_pie += 1
            elif chart_type == "scatter":
                invalid_scatter += 1
            elif chart_type == "line":
                invalid_line += 1
            elif chart_type == "stacked_bar":
                invalid_stacked_bar += 1
            elif chart_type == "bar":
                invalid_bar += 1
        if any(f in fidelity_failed for f in (
            "constraint_fields_in_columns", "recovered_group_field_in_columns",
            "stacked_bar_has_group_field", "scatter_two_numeric_axes",
        )):
            missing_constraint += 1

    dup_checks, _dup_findings = duplicate_checks(records, strict=False)
    dup_summary = {c["check"]: c["n"] for c in dup_checks if not c["passed"]}

    return {
        "n": len(records),
        "chart_distribution": dict(chart_dist),
        "database_distribution": dict(db_dist),
        "quality_tier_distribution": dict(tier_dist),
        "identifier_as_measure_count": identifier_as_measure,
        "meaningless_kpi_count": meaningless_kpi,
        "chart_inappropriate_count": chart_inappropriate,
        "source_conflict_count": source_conflict,
        "missing_constraint_count": missing_constraint,
        "uncertain_pie_count": uncertain_pie,
        "invalid_scatter_count": invalid_scatter,
        "invalid_line_count": invalid_line,
        "invalid_stacked_bar_count": invalid_stacked_bar,
        "invalid_bar_count": invalid_bar,
        "duplicate_findings": dup_summary,
    }


def before_after_v3_v4(
    v3_records: List[Dict[str, Any]],
    v4_records: List[Dict[str, Any]],
    mapping: Dict[str, Any],
    resolver,
    profiler,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Corpus-level v3-vs-v4 comparison (never compares by item id)."""
    return {
        "ai_precheck_problems_targeted": [
            "identifier fields used as measures or correlation axes",
            "inappropriate chart choices",
            "meaningless or mismatched KPI aggregations",
            "column role and dtype conflicts",
            "missing or unclear constraints",
        ],
        "v3": _corpus_quality_metrics(v3_records, mapping, resolver, profiler, cfg),
        "v4": _corpus_quality_metrics(v4_records, mapping, resolver, profiler, cfg),
    }
