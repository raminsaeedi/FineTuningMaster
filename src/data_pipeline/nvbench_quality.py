"""Dashboard-suitability quality layer for nvBench Pilot v4.

Sits between the technically-valid nvBench candidate pool (``NvBenchBuilder``
output -- source-faithful by construction) and Pilot v4 selection. Assigns
every candidate a quality tier:

- **Tier A**: high-confidence positive training candidate. Only Tier A may
  enter Pilot v4.
- **Tier B**: source-faithful but design-unsuitable or uncertain (identifier
  used as a measure, chart choice uncertain given the data shape, etc).
- **Tier C**: a new, severe contradiction discovered only by this layer
  (rare -- most outright-invalid records were already rejected by the builder).

Reuses ``nvbench_pilot.semantic_checks`` wholesale for constraint/fidelity
validation (Phase 6) rather than reimplementing it -- that function already
implements 26 corpus-verified checks. This module adds the two checks that
genuinely do not exist yet: KPI meaningfulness (Phase 4) and chart-shape
suitability from real database evidence (Phase 5).
"""

from __future__ import annotations

import collections
import statistics
from typing import Any, Dict, List, Optional, Tuple

from src.data_pipeline.nvbench_extract import detect_query_intent, extract_base_field
from src.data_pipeline.nvbench_identifier import detect_identifier
from src.data_pipeline.nvbench_pilot import _lineage, _mapping0, _prov, _record, semantic_checks
from src.data_pipeline.nvbench_profile import DbProfiler
from src.data_pipeline.nvbench_source import parse_aggregate
from src.utils.io import read_yaml

QUALITY_RULE_VERSION = "nvbench_quality_v1"

# semantic_checks names that specifically cover constraint preservation (Phase 6);
# every other failing check counts toward the general source-fidelity component.
_CONSTRAINT_CHECK_NAMES = {
    "constraint_fields_in_columns",
    "recovered_group_field_in_columns",
    "stacked_bar_has_group_field",
    "scatter_two_numeric_axes",
    "no_nested_aggregate_remains",
    "grouping_classify_preserved",
}


def load_quality_config(path: str) -> Dict[str, Any]:
    cfg = read_yaml(path)
    for key in ("identifier", "chart", "scoring", "sampling"):
        if key not in cfg:
            raise ValueError(f"invalid nvBench quality-rules config: missing {key!r} in {path}")
    return cfg


# --------------------------------------------------------------------------- #
# shared accessors
# --------------------------------------------------------------------------- #
def _sql_of(prov: Dict[str, Any]) -> str:
    return ((prov.get("vis_query") or {}).get("data_part") or {}).get("sql_part", "") or ""


def _field_profile_for_axis(
    axis_typed: Dict[str, Any], db_id: str, sql: str, profiler: DbProfiler
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Raw field name + profile backing one typed axis (None field for COUNT(*))."""
    if axis_typed.get("aggregate"):
        base = extract_base_field(axis_typed.get("name", ""))
        if not base:
            return None, None
        return base, profiler.profile_field(db_id, base, sql_context=sql)
    name = axis_typed.get("name", "")
    if not name:
        return None, None
    return name, profiler.profile_field(db_id, name, sql_context=sql)


def _identifier_flag(profile: Optional[Dict[str, Any]], field: Optional[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if profile is None or field is None:
        return {"is_identifier": False, "confidence": "none", "evidence": [], "rule_version": "n/a"}
    return detect_identifier(profile, field, cfg)


def _evidence_insufficient(profile: Optional[Dict[str, Any]]) -> bool:
    if profile is None:
        return False  # e.g. COUNT(*): no field to profile, not a failure
    if profile.get("resolution") in ("ambiguous_table", "field_not_found"):
        return True
    return not profile.get("stats_available", False)


def _chart_shape_evidence_insufficient(profile: Optional[Dict[str, Any]]) -> bool:
    """Stricter variant for checks that need real cardinality/variation evidence
    (pie category count, scatter axis variation, stacked-bar group cardinality):
    unlike KPI checks, a missing field profile (e.g. a bare ``COUNT(*)`` axis with
    no physical column) can never certify these shape requirements, so it fails
    closed instead of being treated as "nothing to check"."""
    if profile is None:
        return True
    return _evidence_insufficient(profile)


# --------------------------------------------------------------------------- #
# Phase 4 -- KPI suitability
# --------------------------------------------------------------------------- #
def kpi_suitability(record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prov = _prov(record)
    m = _mapping0(record)
    kpi_sel = prov.get("kpi_selection") or {}
    primary_kpi = kpi_sel.get("primary_kpi") or m.get("kpi") or ""
    db_id = prov.get("db_id", "")
    sql = _sql_of(prov)
    nl_queries = [prov.get("nl_query", "")]

    failed: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {"kpi": primary_kpi}

    outer = parse_aggregate(primary_kpi)
    if outer == "COUNT":
        evidence["policy"] = "count-always-allowed"
        return {"suitable": True, "failed_rules": [], "warnings": [], "evidence": evidence}

    if outer in ("SUM", "AVG", "MIN", "MAX"):
        base = extract_base_field(primary_kpi)
        if base:
            profile = profiler.profile_field(db_id, base, sql_context=sql)
            evidence["base_field_profile"] = profile
            id_flag = detect_identifier(profile, base, cfg)
            evidence["identifier"] = id_flag
            if profile.get("resolution") == "ambiguous_table":
                failed.append("field_table_ambiguous")
            elif id_flag["is_identifier"] and id_flag["confidence"] == "strong":
                failed.append("meaningless_identifier_aggregation")
            elif id_flag["is_identifier"] and id_flag["confidence"] == "ambiguous":
                failed.append("possible_identifier_aggregation")
            if profile.get("stats_available") and profile.get("normalized_dtype") not in (None, "number"):
                failed.append("aggregate_dtype_conflict")
        intent = detect_query_intent(nl_queries, allow_count_number_of=True)
        if intent and intent != outer:
            # Broader than the builder's narrow, corpus-verified rules; kept as a
            # Tier-B signal only -- never silently escalated to a rejection here.
            warnings.append(f"broad_intent_mismatch(query={intent},encoded={outer})")
            failed.append("broad_intent_mismatch")
    else:
        # Non-aggregate KPI: a bare field used directly as the recommended KPI.
        profile = profiler.profile_field(db_id, primary_kpi, sql_context=sql)
        evidence["base_field_profile"] = profile
        id_flag = detect_identifier(profile, primary_kpi, cfg)
        evidence["identifier"] = id_flag
        if profile.get("resolution") == "ambiguous_table":
            failed.append("field_table_ambiguous")
        elif id_flag["is_identifier"]:
            failed.append("identifier_as_continuous_kpi")

    return {"suitable": not failed, "failed_rules": failed, "warnings": warnings, "evidence": evidence}


# --------------------------------------------------------------------------- #
# Phase 5 -- chart suitability
# --------------------------------------------------------------------------- #
def _base_chart_evidence(record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]):
    prov = _prov(record)
    axis_typing = prov.get("axis_typing") or {}
    db_id = prov.get("db_id", "")
    sql = _sql_of(prov)
    x_field, x_profile = _field_profile_for_axis(axis_typing.get("x") or {}, db_id, sql, profiler)
    y_field, y_profile = _field_profile_for_axis(axis_typing.get("y") or {}, db_id, sql, profiler)
    x_id = _identifier_flag(x_profile, x_field, cfg)
    y_id = _identifier_flag(y_profile, y_field, cfg)
    return prov, axis_typing, db_id, sql, (x_field, x_profile, x_id), (y_field, y_profile, y_id)


def chart_bar(record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prov, axis_typing, db_id, sql, (xf, xp, xid), (yf, yp, yid) = _base_chart_evidence(record, profiler, cfg)
    failed: List[str] = []
    warnings: List[str] = []
    yt = axis_typing.get("y") or {}
    if yt.get("role") == "measure" and yid["is_identifier"] and yid["confidence"] == "strong":
        failed.append("identifier_as_measure")
    elif yt.get("role") == "measure" and yid["is_identifier"] and yid["confidence"] == "ambiguous":
        warnings.append("possible_identifier_as_measure")
    if _evidence_insufficient(yp):
        failed.append("insufficient_measure_evidence")
    evidence = {"x_profile": xp, "y_profile": yp, "x_identifier": xid, "y_identifier": yid}
    return {"passed": not failed, "failed_rules": failed, "warnings": warnings, "evidence": evidence}


def chart_line(record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prov, axis_typing, db_id, sql, (xf, xp, xid), (yf, yp, yid) = _base_chart_evidence(record, profiler, cfg)
    failed: List[str] = []
    warnings: List[str] = []
    xt = axis_typing.get("x") or {}
    yt = axis_typing.get("y") or {}
    has_time_grain = bool((prov.get("constraints") or {}).get("time_grain"))
    has_sort = bool((prov.get("constraints") or {}).get("sort"))
    is_ordered = xt.get("dtype") == "datetime" or has_time_grain or has_sort
    if xt.get("dtype") == "categorical" and not is_ordered:
        failed.append("unordered_line_dimension")
    if yt.get("role") != "measure" or yt.get("dtype") != "number":
        failed.append("non_numeric_line_measure")
    if yt.get("role") == "measure" and yid["is_identifier"] and yid["confidence"] == "strong":
        failed.append("identifier_as_measure")
    evidence = {"x_profile": xp, "y_profile": yp, "is_ordered": is_ordered}
    return {"passed": not failed, "failed_rules": failed, "warnings": warnings, "evidence": evidence}


def chart_pie(record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prov, axis_typing, db_id, sql, (xf, xp, xid), (yf, yp, yid) = _base_chart_evidence(record, profiler, cfg)
    failed: List[str] = []
    warnings: List[str] = []
    max_categories = int(cfg["chart"]["pie"]["max_categories"])
    if _chart_shape_evidence_insufficient(xp):
        failed.append("insufficient_category_evidence")
    elif xp.get("distinct_count") is not None and xp["distinct_count"] > max_categories:
        failed.append("high_cardinality_pie")
    if xid["is_identifier"]:
        failed.append("identifier_pie_category")
    if yp is not None and yp.get("stats_available") and yp.get("n_negative"):
        # The base column carries negative values; the aggregate itself could
        # still be non-negative, but a pie slice built on a base field that admits
        # negative values has no defensible part-to-whole interpretation, so this
        # is treated as a hard Tier-A blocker rather than a soft warning.
        failed.append("negative_measure_values")
    evidence = {"x_profile": xp, "y_profile": yp, "x_identifier": xid, "max_categories": max_categories}
    return {"passed": not failed, "failed_rules": failed, "warnings": warnings, "evidence": evidence}


def chart_scatter(record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prov, axis_typing, db_id, sql, (xf, xp, xid), (yf, yp, yid) = _base_chart_evidence(record, profiler, cfg)
    failed: List[str] = []
    warnings: List[str] = []
    min_distinct = int(cfg["chart"]["scatter"]["min_distinct_values"])
    for axis_name, t, profile, idf in (("x", axis_typing.get("x") or {}, xp, xid),
                                        ("y", axis_typing.get("y") or {}, yp, yid)):
        if t.get("dtype") == "categorical":
            failed.append(f"categorical_scatter_axis:{axis_name}")
        if idf["is_identifier"]:
            failed.append(f"identifier_scatter_axis:{axis_name}")
        if _chart_shape_evidence_insufficient(profile):
            failed.append(f"insufficient_axis_evidence:{axis_name}")
        elif profile.get("distinct_count") is not None and profile["distinct_count"] < min_distinct:
            failed.append(f"low_variation_axis:{axis_name}")
    evidence = {"x_profile": xp, "y_profile": yp, "x_identifier": xid, "y_identifier": yid,
                "min_distinct_values": min_distinct}
    return {"passed": not failed, "failed_rules": failed, "warnings": warnings, "evidence": evidence}


def chart_stacked_bar(record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prov, axis_typing, db_id, sql, (xf, xp, xid), (yf, yp, yid) = _base_chart_evidence(record, profiler, cfg)
    failed: List[str] = []
    warnings: List[str] = []
    m = _mapping0(record)
    group_field = (m.get("encoding") or {}).get("group_field")
    max_group = int(cfg["chart"]["stacked_bar"]["max_group_cardinality"])
    if not group_field:
        failed.append("missing_group_field")
        group_profile = None
        group_id = {"is_identifier": False, "confidence": "none", "evidence": []}
    else:
        group_profile = profiler.profile_field(db_id, group_field, sql_context=sql)
        group_id = detect_identifier(group_profile, group_field, cfg)
        if _evidence_insufficient(group_profile):
            failed.append("insufficient_group_evidence")
        elif group_profile.get("distinct_count") is not None and group_profile["distinct_count"] > max_group:
            failed.append("high_cardinality_group")
        if group_id["is_identifier"]:
            failed.append("identifier_group_field")
    yt = axis_typing.get("y") or {}
    if yt.get("role") == "measure" and yid["is_identifier"] and yid["confidence"] == "strong":
        failed.append("identifier_as_measure")
    evidence = {"group_field": group_field, "group_profile": group_profile, "group_identifier": group_id,
                "y_profile": yp, "max_group_cardinality": max_group}
    return {"passed": not failed, "failed_rules": failed, "warnings": warnings, "evidence": evidence}


CHART_CHECKERS = {
    "bar": chart_bar,
    "line": chart_line,
    "pie": chart_pie,
    "scatter": chart_scatter,
    "stacked_bar": chart_stacked_bar,
}


# --------------------------------------------------------------------------- #
# Phase 6 -- fidelity/constraint signals (reused wholesale from semantic_checks)
# --------------------------------------------------------------------------- #
def fidelity_signals(records: List[Dict[str, Any]], mapping: Dict[str, Any], resolver) -> Dict[str, List[str]]:
    """Per-item failing ``semantic_checks`` names, inverted from its batch output."""
    checks, _warnings = semantic_checks(records, mapping, resolver)
    per_item: Dict[str, List[str]] = collections.defaultdict(list)
    for c in checks:
        if not c["passed"]:
            for iid in c["item_ids"]:
                per_item[iid].append(c["check"])
    return dict(per_item)


# --------------------------------------------------------------------------- #
# Phase 7 -- explainable score + tier
# --------------------------------------------------------------------------- #
def score_and_tier(
    record: Dict[str, Any],
    kpi_result: Dict[str, Any],
    chart_result: Dict[str, Any],
    fidelity_failed: List[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    weights = cfg["scoring"]["weights"]
    tier_a_min_score = int(cfg["scoring"]["tier_a_min_score"])

    constraint_failed = [c for c in fidelity_failed if c in _CONSTRAINT_CHECK_NAMES]
    general_fidelity_failed = [c for c in fidelity_failed if c not in _CONSTRAINT_CHECK_NAMES]

    fidelity_score = 0 if general_fidelity_failed else weights["source_fidelity"]
    constraint_score = 0 if constraint_failed else weights["constraint_completeness"]
    kpi_score = 0 if not kpi_result["suitable"] else max(
        0, weights["kpi_validity"] - 2 * len(kpi_result.get("warnings", []))
    )
    chart_score = 0 if not chart_result["passed"] else max(
        0, weights["chart_suitability"] - 2 * len(chart_result.get("warnings", []))
    )

    def _profile_ok(p: Optional[Dict[str, Any]]) -> bool:
        return p is None or (p.get("stats_available") and p.get("resolution") not in ("ambiguous_table", "field_not_found"))

    db_support_ok = all(
        _profile_ok(p)
        for p in (
            (chart_result.get("evidence") or {}).get("x_profile"),
            (chart_result.get("evidence") or {}).get("y_profile"),
            (chart_result.get("evidence") or {}).get("group_profile"),
        )
    )
    db_score = weights["db_profile_support"] if db_support_ok else 0

    score = fidelity_score + constraint_score + kpi_score + chart_score + db_score

    failed_rules = list(general_fidelity_failed) + list(constraint_failed) + \
        list(kpi_result["failed_rules"]) + list(chart_result["failed_rules"])
    if not db_support_ok:
        failed_rules.append("insufficient_db_profile_support")
    warnings = list(kpi_result.get("warnings", [])) + list(chart_result.get("warnings", []))

    mandatory_failure = bool(failed_rules)
    severe_combo = (
        "meaningless_identifier_aggregation" in kpi_result["failed_rules"]
        and "broad_intent_mismatch" in kpi_result["failed_rules"]
    )

    if score >= tier_a_min_score and not mandatory_failure:
        tier = "A"
    elif severe_combo:
        tier = "C"
    else:
        tier = "B"

    return {
        "tier": tier,
        "quality_score": score,
        "component_scores": {
            "source_fidelity": fidelity_score,
            "kpi_validity": kpi_score,
            "chart_suitability": chart_score,
            "constraint_completeness": constraint_score,
            "db_profile_support": db_score,
        },
        "passed_rules": [] if mandatory_failure else ["all_mandatory_rules"],
        "failed_rules": failed_rules,
        "warnings": warnings,
        "rule_version": QUALITY_RULE_VERSION,
    }


# --------------------------------------------------------------------------- #
# Phase 8 -- full-pool quality tiering
# --------------------------------------------------------------------------- #
def build_quality_pool(
    items: List[Any],
    mapping: Dict[str, Any],
    resolver,
    profiler: DbProfiler,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Tier every candidate in ``items`` (``GoldItem`` objects).

    Returns ``{tier_a, tier_b, tier_c}`` as lists of the original ``GoldItem``
    objects (so v3's ``select_pilot_v3`` can be reused unmodified on
    ``tier_a``), plus ``quality_by_id`` (full per-item quality result, reused
    later to validate the final Pilot v4 selection without recomputation) and
    a ``summary`` report.
    """
    by_id = {it.item_id: it for it in items}
    dict_records = [_record(it) for it in items]
    fidelity_failed_map = fidelity_signals(dict_records, mapping, resolver)

    tier_a: List[Any] = []
    tier_b: List[Any] = []
    tier_c: List[Any] = []
    quality_by_id: Dict[str, Dict[str, Any]] = {}

    rule_failure_counts: collections.Counter = collections.Counter()
    tier_by_chart: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    tier_by_db: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    scores: List[int] = []

    for rec in dict_records:
        iid = rec["item_id"]
        m = _mapping0(rec)
        prov = _prov(rec)
        chart_type = m.get("chart_type", "")
        db_id = prov.get("db_id", "?")

        kpi_result = kpi_suitability(rec, profiler, cfg)
        checker = CHART_CHECKERS.get(chart_type)
        chart_result = checker(rec, profiler, cfg) if checker else {
            "passed": False, "failed_rules": [f"unknown_chart_type:{chart_type}"], "warnings": [], "evidence": {},
        }
        fidelity_failed = fidelity_failed_map.get(iid, [])
        quality = score_and_tier(rec, kpi_result, chart_result, fidelity_failed, cfg)
        quality["kpi_suitability"] = kpi_result
        quality["chart_suitability"] = chart_result
        quality["fidelity_failed"] = fidelity_failed
        quality_by_id[iid] = quality

        for rule in quality["failed_rules"]:
            rule_failure_counts[rule] += 1
        tier_by_chart[chart_type][quality["tier"]] += 1
        tier_by_db[db_id][quality["tier"]] += 1
        scores.append(quality["quality_score"])

        item = by_id[iid]
        if quality["tier"] == "A":
            tier_a.append(item)
        elif quality["tier"] == "B":
            tier_b.append(item)
        else:
            tier_c.append(item)

    summary = {
        "rule_version": QUALITY_RULE_VERSION,
        "total_candidates": len(items),
        "tier_a_count": len(tier_a),
        "tier_b_count": len(tier_b),
        "tier_c_count": len(tier_c),
        "tier_by_chart": {k: dict(v) for k, v in tier_by_chart.items()},
        "tier_by_database": {k: dict(v) for k, v in tier_by_db.items()},
        "rule_failure_counts": dict(rule_failure_counts),
        "score_distribution": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": round(statistics.fmean(scores), 2) if scores else None,
            "median": statistics.median(scores) if scores else None,
        },
    }
    return {
        "tier_a": tier_a, "tier_b": tier_b, "tier_c": tier_c,
        "quality_by_id": quality_by_id, "summary": summary,
    }
