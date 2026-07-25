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
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

from src.data_pipeline.nvbench_extract import (
    detect_query_intent,
    extract_base_field,
    extract_group_by_fields,
    extract_select_aggregates,
    extract_time_grain_signals,
)
from src.data_pipeline.nvbench_identifier import detect_identifier
from src.data_pipeline.nvbench_pilot import _lineage, _mapping0, _prov, _record, semantic_checks
from src.data_pipeline.nvbench_profile import DbProfiler
from src.data_pipeline.nvbench_source import parse_aggregate
from src.utils.io import read_yaml

QUALITY_RULE_VERSION = "nvbench_quality_v4"

# Mandatory rules that always block Tier A (a hit forces at least Tier B),
# regardless of the numeric score. Extends the implicit "any failed rule blocks
# Tier A" contract with the explicit v5 correctness rules for traceability.
_MANDATORY_TIER_A_BLOCKERS = frozenset({
    "kpi_sql_aggregation_conflict",
    "mixed_aggregate_ambiguous_kpi",
    "query_aggregation_conflict",
    "missing_required_time_grain",
    "missing_required_grouping",
    "missing_required_filter",
    "missing_required_sort",
    "meaningless_identifier_aggregation",
    "identifier_as_measure",
    "identifier_as_continuous_kpi",
    "aggregate_dtype_conflict",
    "field_table_ambiguous",
    "pie_non_additive_kpi",
})

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

    # v5: encoded/KPI aggregate vs source SQL/VQL aggregate agreement.
    agg = check_kpi_sql_aggregation(record)
    failed.extend(agg["failed_rules"])
    warnings.extend(agg["warnings"])
    evidence["aggregation_agreement"] = agg["evidence"]

    return {"suitable": not failed, "failed_rules": failed, "warnings": warnings, "evidence": evidence}


# --------------------------------------------------------------------------- #
# v5: KPI / SQL / VQL aggregation agreement (axis-aware, not naive-global)
# --------------------------------------------------------------------------- #
def _norm_agg_of(expr: str) -> Optional[str]:
    return parse_aggregate(expr or "")


def check_kpi_sql_aggregation(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compare encoded/KPI aggregate functions against the source SQL aggregates.

    Three independent signals, all mandatory Tier-A blockers when they fire:
      (a) ``kpi_sql_aggregation_conflict`` -- an encoded axis is an aggregate over
          a base field, the SQL SELECT also aggregates that same base field, but
          with a *different* function (e.g. encoded MIN(price) vs SQL MAX(price)).
      (b) ``mixed_aggregate_ambiguous_kpi`` -- both axes are aggregates with
          different functions, so the single recorded KPI cannot faithfully
          represent both (e.g. a scatter of SUM(pop) vs AVG(life)).
      (c) ``query_aggregation_conflict`` -- the natural-language query names a
          single clear aggregate intent that disagrees with the primary KPI's
          aggregate function.
    Base fields are matched alias/case-insensitively, so ``SUM(order_quantity)``,
    ``SUM(T2.order_quantity)`` and ``sum("order_quantity")`` are equivalent.
    """
    prov = _prov(record)
    m = _mapping0(record)
    enc = m.get("encoding") or {}
    sql = _sql_of(prov)
    vql = (prov.get("vis_query") or {}).get("VQL", "") or ""
    kpi_sel = prov.get("kpi_selection") or {}
    primary_kpi = kpi_sel.get("primary_kpi") or m.get("kpi") or ""
    nl_queries = [prov.get("nl_query", "")]

    failed: List[str] = []
    warnings: List[str] = []

    sql_aggs = extract_select_aggregates(sql)
    # Prefer SQL; fall back to VQL's embedded SELECT when SQL has no aggregates.
    if not sql_aggs and vql:
        sql_aggs = extract_select_aggregates(vql)
    by_base: Dict[str, set] = collections.defaultdict(set)
    for a in sql_aggs:
        if a["base_field"]:
            by_base[a["base_field"].lower()].add(a["func"])

    evidence = {
        "sql_aggregates": sql_aggs,
        "x_encoding": enc.get("x"), "y_encoding": enc.get("y"), "primary_kpi": primary_kpi,
    }

    # (a) axis-aware conflict
    axis_funcs: Dict[str, Optional[str]] = {}
    for axis in ("x", "y"):
        expr = str(enc.get(axis) or "")
        func = _norm_agg_of(expr)
        axis_funcs[axis] = func
        if not func:
            continue
        base = extract_base_field(expr)
        if not base:
            continue
        src_funcs = by_base.get(base.lower())
        if src_funcs and func not in src_funcs:
            failed.append("kpi_sql_aggregation_conflict")
            evidence.setdefault("conflicts", []).append(
                {"axis": axis, "encoded": func, "base_field": base, "sql_funcs": sorted(src_funcs)}
            )

    # (b) mixed-aggregate ambiguity (both axes aggregate, different functions)
    if axis_funcs["x"] and axis_funcs["y"] and axis_funcs["x"] != axis_funcs["y"]:
        failed.append("mixed_aggregate_ambiguous_kpi")
        evidence["mixed_aggregate"] = {"x_func": axis_funcs["x"], "y_func": axis_funcs["y"]}

    # (c) query intent vs primary KPI aggregate
    kpi_func = _norm_agg_of(primary_kpi)
    if kpi_func:
        intent = detect_query_intent(nl_queries, allow_count_number_of=True)
        if intent and intent != kpi_func:
            failed.append("query_aggregation_conflict")
            evidence["query_intent"] = {"query": intent, "kpi": kpi_func}

    return {"failed_rules": failed, "warnings": warnings, "evidence": evidence}


# --------------------------------------------------------------------------- #
# v5: required time-grain / grouping preservation
# --------------------------------------------------------------------------- #
def _recorded_time_grain_fields(prov: Dict[str, Any]) -> set:
    tg = (prov.get("constraints") or {}).get("time_grain") or {}
    field = tg.get("field")
    return {str(field).lower()} if field else set()


def _name_is_temporal(name: str, cfg: Dict[str, Any]) -> bool:
    """Boundary-safe temporal name match (fallback when the DB can't confirm dtype).

    Matches a hint only as a whole ``_``-delimited token (so ``installation_date``
    and ``order_date`` hit on ``date``, ``time_of_day`` on ``time``, and bare
    ``year`` on ``year``), but ``birthday`` does not match ``day``.
    """
    hints = (cfg.get("time_grain") or {}).get("name_hints") or [
        "date", "time", "year", "month", "day", "week", "quarter", "hour",
    ]
    low = str(name).split(".")[-1].strip().lower()
    pattern = r"(^|_)(" + "|".join(re.escape(h) for h in hints) + r")(_|$)"
    return bool(re.search(pattern, low))


def check_required_constraints(
    record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Verify SQL/VQL-derived time-grain and grouping semantics are preserved.

    ``missing_required_time_grain`` fires when the source has an explicit time
    function (``strftime``/``date``/``EXTRACT``/``YEAR(...)``) or the chart's
    x-dimension is a database-confirmed ``datetime`` column that the query groups
    on, yet the transformed record records no matching ``constraints.time_grain``.
    ``missing_required_grouping`` fires when a SQL ``GROUP BY`` column is not
    represented anywhere in the transformed record (not x, not group_field, not
    an aggregate base, not a recorded time-grain field).
    """
    prov = _prov(record)
    m = _mapping0(record)
    enc = m.get("encoding") or {}
    axis_typing = prov.get("axis_typing") or {}
    sql = _sql_of(prov)
    vql = (prov.get("vis_query") or {}).get("VQL", "") or ""
    db_id = prov.get("db_id", "")
    chart_type = m.get("chart_type", "")
    temporal_charts = set((cfg.get("time_grain") or {}).get("temporal_charts") or ["line", "bar"])

    failed: List[str] = []
    evidence: Dict[str, Any] = {}
    recorded_tg = _recorded_time_grain_fields(prov)

    # Explicit source time-grain signals must each be preserved.
    signals = extract_time_grain_signals(sql, vql)
    evidence["time_grain_signals"] = signals
    for sig in signals:
        if sig["field"].lower() not in recorded_tg:
            failed.append("missing_required_time_grain")
            evidence.setdefault("missing_time_grain", []).append(sig)

    # x-dimension over a DB-confirmed datetime column, grouped/ordered in SQL,
    # on a temporal chart, with no recorded grain -> the grain was dropped.
    xt = axis_typing.get("x") or {}
    x_name = xt.get("name") or enc.get("x") or ""
    if chart_type in temporal_charts and x_name and not _norm_agg_of(str(x_name)):
        group_fields = {g.lower() for g in extract_group_by_fields(sql)}
        x_low = str(x_name).split(".")[-1].strip().lower()
        if x_low in group_fields and x_low not in recorded_tg:
            profile = profiler.profile_field(db_id, str(x_name), sql_context=sql)
            db_dt = profile.get("normalized_dtype") == "datetime" or xt.get("dtype") == "datetime"
            # Fall back to a boundary-safe temporal name hint when the database
            # can't positively type the column as datetime. This is applied only
            # to the chart's grouped x-*dimension* (never a y measure), and the
            # token regex avoids compound measure names (e.g. ``day_count`` only
            # matches the ``day`` token, ``number_of_days``/``birthday`` do not).
            name_dt = not db_dt and _name_is_temporal(str(x_name), cfg)
            if db_dt or name_dt:
                failed.append("missing_required_time_grain")
                evidence.setdefault("missing_time_grain", []).append({
                    "field": x_name, "grain": "DATE" if db_dt else "NAME_HINT",
                    "source": "grouped_datetime_dimension" if db_dt else "grouped_temporal_name",
                })

    # Grouping: every SQL GROUP BY column must be represented somewhere.
    represented = set()
    for axis in ("x", "y"):
        t = axis_typing.get(axis) or {}
        nm = t.get("name")
        if nm:
            represented.add(str(nm).split(".")[-1].strip().lower())
            base = extract_base_field(str(nm))
            if base:
                represented.add(base.split(".")[-1].strip().lower())
    gf = enc.get("group_field")
    if gf:
        represented.add(str(gf).split(".")[-1].strip().lower())
    represented |= recorded_tg
    missing_group = [g for g in extract_group_by_fields(sql)
                     if g.split(".")[-1].strip().lower() not in represented]
    if missing_group:
        failed.append("missing_required_grouping")
        evidence["missing_grouping"] = missing_group

    return {"failed_rules": failed, "evidence": evidence}


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


# Fallback defaults, used only when a config omits these keys (e.g. older
# fixtures); the real quality-rules YAML states them all explicitly.
_PIE_DEFAULT_ADDITIVE_AGGS = ("COUNT", "SUM")
_PIE_DEFAULT_REASON_CODE = "pie_non_additive_kpi"


def chart_pie(record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prov, axis_typing, db_id, sql, (xf, xp, xid), (yf, yp, yid) = _base_chart_evidence(record, profiler, cfg)
    pie_cfg = cfg.get("chart", {}).get("pie", {}) or {}
    failed: List[str] = []
    warnings: List[str] = []
    max_categories = int(pie_cfg.get("max_categories", 8))
    allow_negative = bool(pie_cfg.get("allow_negative_values", False))
    allow_identifier = bool(pie_cfg.get("allow_identifier_category", False))
    additive_aggs = frozenset(a.upper() for a in pie_cfg.get("additive_aggregates", _PIE_DEFAULT_ADDITIVE_AGGS))
    reason_code = pie_cfg.get("non_additive_reason_code", _PIE_DEFAULT_REASON_CODE)

    if _chart_shape_evidence_insufficient(xp):
        failed.append("insufficient_category_evidence")
    elif xp.get("distinct_count") is not None and xp["distinct_count"] > max_categories:
        failed.append("high_cardinality_pie")
    if xid["is_identifier"] and not allow_identifier:
        failed.append("identifier_pie_category")
    if not allow_negative and yp is not None and yp.get("stats_available") and yp.get("n_negative"):
        # The base column carries negative values; the aggregate itself could
        # still be non-negative, but a pie slice built on a base field that admits
        # negative values has no defensible part-to-whole interpretation, so this
        # is treated as a hard Tier-A blocker rather than a soft warning.
        failed.append("negative_measure_values")
    # Part-to-whole composition requires an additive measure: slices must sum to
    # a meaningful whole. AVG/MIN/MAX over a category is not additive -- the
    # slices don't represent parts of one total -- so it's source-faithful but
    # design-invalid (Tier B), never a Tier-A pie. Allowed/prohibited functions
    # and the reason code are configuration, not hardcoded production logic.
    y_agg = (axis_typing.get("y") or {}).get("aggregate")
    if y_agg and y_agg.upper() not in additive_aggs:
        failed.append(reason_code)
    evidence = {"x_profile": xp, "y_profile": yp, "x_identifier": xid, "max_categories": max_categories,
                "y_aggregate": y_agg, "additive_aggregates": sorted(additive_aggs)}
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
    constraint_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    weights = cfg["scoring"]["weights"]
    tier_a_min_score = int(cfg["scoring"]["tier_a_min_score"])
    constraint_result = constraint_result or {"failed_rules": [], "evidence": {}}

    semantic_constraint_failed = [c for c in fidelity_failed if c in _CONSTRAINT_CHECK_NAMES]
    general_fidelity_failed = [c for c in fidelity_failed if c not in _CONSTRAINT_CHECK_NAMES]
    constraint_failed = list(semantic_constraint_failed) + list(constraint_result["failed_rules"])

    fidelity_score = 0 if general_fidelity_failed else weights["source_fidelity"]
    constraint_score = 0 if constraint_failed else weights["constraint_completeness"]
    kpi_score = 0 if not kpi_result["suitable"] else max(
        0, weights["kpi_validity"] - 2 * len(kpi_result.get("warnings", []))
    )
    chart_score = 0 if not chart_result["passed"] else max(
        0, weights["chart_suitability"] - 2 * len(chart_result.get("warnings", []))
    )

    # Evidence-graduated database support: fraction of the profiled axis/group
    # fields backed by real database statistics (vs a name heuristic / missing).
    # A record whose fields are all DB-confirmed scores the full weight; one that
    # leans on name heuristics scores proportionally less -- so clean records are
    # not uniformly 100.
    def _profile_state(p: Optional[Dict[str, Any]]) -> Optional[bool]:
        if p is None:
            return None  # no physical field to profile (e.g. COUNT(*)); not counted
        if p.get("resolution") in ("ambiguous_table", "field_not_found"):
            return False
        return bool(p.get("stats_available"))

    ev = chart_result.get("evidence") or {}
    states = [s for s in (_profile_state(ev.get("x_profile")),
                          _profile_state(ev.get("y_profile")),
                          _profile_state(ev.get("group_profile"))) if s is not None]
    if not states:
        db_fraction = 1.0
    else:
        db_fraction = sum(1 for s in states if s) / len(states)
    db_score = round(weights["db_profile_support"] * db_fraction)
    db_support_ok = db_fraction >= 1.0

    score = fidelity_score + constraint_score + kpi_score + chart_score + db_score

    failed_rules = list(general_fidelity_failed) + list(constraint_failed) + \
        list(kpi_result["failed_rules"]) + list(chart_result["failed_rules"])
    warnings = list(kpi_result.get("warnings", [])) + list(chart_result.get("warnings", []))

    # A partial (but non-zero) DB profile is a soft signal, not a hard blocker:
    # v3's builder already guarantees a defensible heuristic dtype. Only a fully
    # unsupported profile (fraction 0 with fields present) is a mandatory failure.
    if states and db_fraction == 0.0:
        failed_rules.append("insufficient_db_profile_support")

    mandatory_failure = bool(failed_rules)
    # Tier C is reserved for a newly discovered *severe* contradiction: a
    # meaningless identifier aggregate co-occurring with an outright aggregation
    # conflict (SQL/query disagreement), i.e. the record is not merely uncertain
    # but internally contradictory.
    kpi_failed = set(kpi_result["failed_rules"])
    severe_combo = bool(
        {"meaningless_identifier_aggregation"} & kpi_failed
        and {"kpi_sql_aggregation_conflict", "query_aggregation_conflict", "broad_intent_mismatch"} & kpi_failed
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
        "db_profile_fraction": round(db_fraction, 3),
        "db_support_ok": db_support_ok,
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
        constraint_result = check_required_constraints(rec, profiler, cfg)
        fidelity_failed = fidelity_failed_map.get(iid, [])
        quality = score_and_tier(rec, kpi_result, chart_result, fidelity_failed, cfg,
                                 constraint_result=constraint_result)
        quality["kpi_suitability"] = kpi_result
        quality["chart_suitability"] = chart_result
        quality["constraint_suitability"] = constraint_result
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
