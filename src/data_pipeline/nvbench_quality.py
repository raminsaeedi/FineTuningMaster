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
    classify_group_by_terms,
    detect_query_intent,
    extract_base_field,
    extract_group_by_fields,
    extract_limit,
    extract_nl_aggregate_conditions,
    extract_nl_required_dimensions,
    extract_nl_time_grains,
    extract_order_by,
    extract_select_aggregates,
    extract_time_grain_signals,
)
from src.data_pipeline.nvbench_identifier import detect_identifier
from src.data_pipeline.nvbench_pilot import _lineage, _mapping0, _prov, _record, semantic_checks
from src.data_pipeline.nvbench_profile import DbProfiler
from src.data_pipeline.nvbench_source import parse_aggregate
from src.utils.io import read_yaml

QUALITY_RULE_VERSION = "nvbench_quality_v6"

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
    "invalid_identifier_aggregation",
    "wrong_kpi",
    "identifier_as_measure",
    "identifier_as_continuous_kpi",
    "aggregate_dtype_conflict",
    "field_table_ambiguous",
    "pie_non_additive_kpi",
    "scatter_identifier_axis",
    "invalid_scatter_axes",
    "source_conflict",
    "missing_required_dimension",
    "missing_aggregate_condition",
    "time_grain_source_conflict",
    "missing_grouping",
    "goal_mismatch",
    "constraint_scope_error",
    "invalid_group_by_expression",
    "insufficient_scatter_observations",
    "chart_inappropriate",
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


_TOP_LEVEL_CLAUSES = (
    ("group_by", r"\bGROUP\s+BY\b"),
    ("having", r"\bHAVING\b"),
    ("order_by", r"\bORDER\s+BY\b"),
    ("limit", r"\bLIMIT\b"),
)


def _top_level_clause_positions(sql: str) -> Dict[str, int]:
    """Locate SQL clauses outside parentheses and quoted strings.

    The nvBench SQL is SQLite-oriented but may contain nested SELECTs. Regexes
    over the whole string confuse an inner preselection with an outer aggregate
    constraint, so scope-sensitive validation uses this small deterministic
    scanner instead.
    """
    text = str(sql or "")
    positions: Dict[str, int] = {}
    depth = 0
    quote: Optional[str] = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == quote:
                if quote != "]" and index + 1 < len(text) and text[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if text.startswith("--", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end < 0 else end + 2
            continue
        if char in ("'", '"', "`"):
            quote = char
            index += 1
            continue
        if char == "[":
            quote = "]"
            index += 1
            continue
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth == 0:
            for name, pattern in _TOP_LEVEL_CLAUSES:
                if name not in positions and re.match(pattern, text[index:], re.IGNORECASE):
                    positions[name] = index
        index += 1
    return positions


def _parenthesized_select_scopes(sql: str) -> List[str]:
    """Return complete parenthesized SELECT bodies, including CTE bodies."""
    text = str(sql or "")
    stack: List[int] = []
    scopes: List[str] = []
    quote: Optional[str] = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == quote:
                if quote != "]" and index + 1 < len(text) and text[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if text.startswith("--", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end < 0 else end + 2
            continue
        if char in ("'", '"', "`"):
            quote = char
        elif char == "[":
            quote = "]"
        elif char == "(":
            stack.append(index)
        elif char == ")" and stack:
            start = stack.pop()
            body = text[start + 1:index].strip()
            if re.match(r"^SELECT\b", body, re.IGNORECASE):
                scopes.append(body)
        index += 1
    return scopes


def _normalized_sort_key(expression: str) -> str:
    base = extract_base_field(expression) if parse_aggregate(expression) else expression
    return re.sub(r"[`\"\[\]]", "", str(base or "")).split(".")[-1].strip().lower()


def _verified_preaggregation_scope(sql: str, sort_expression: str, limit: int) -> Dict[str, Any]:
    """Verify that ORDER BY/LIMIT occur together inside a feeding SELECT.

    A CTE/subquery marker alone is not evidence of population preselection.
    The exact parsed sort key and limit must coexist in one nested SELECT, and
    neither may also be an outer query constraint.
    """
    outer_clauses = _top_level_clause_positions(sql)
    evidence: Dict[str, Any] = {
        "verified": False,
        "sort_key": _normalized_sort_key(sort_expression),
        "limit": int(limit),
        "outer_order_by": "order_by" in outer_clauses,
        "outer_limit": "limit" in outer_clauses,
        "nested_select_count": 0,
        "matching_scope_count": 0,
    }
    if evidence["outer_order_by"] or evidence["outer_limit"]:
        evidence["reason"] = "sort_or_limit_applies_after_outer_aggregation"
        return evidence

    scopes = _parenthesized_select_scopes(sql)
    evidence["nested_select_count"] = len(scopes)
    matches = 0
    for scope in scopes:
        inner_sort = extract_order_by(scope)
        inner_limit = extract_limit(scope)
        if (
            inner_sort.get("status") == "ok"
            and inner_limit.get("status") == "ok"
            and _normalized_sort_key(str(inner_sort.get("field") or "")) == evidence["sort_key"]
            and int(inner_limit.get("value")) == evidence["limit"]
        ):
            matches += 1
    evidence["matching_scope_count"] = matches
    evidence["verified"] = matches == 1
    evidence["reason"] = (
        "one_nested_select_contains_matching_sort_and_limit"
        if matches == 1
        else "no_unique_nested_select_contains_matching_sort_and_limit"
    )
    return evidence


def _sqlite_vql_bin_expression(field: str, grain: Optional[str]) -> Optional[str]:
    """Translate one validated VQL BIN field/grain to SQLite grouping syntax."""
    raw = str(field or "").strip()
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?", raw):
        return None
    qfield = ".".join(f'"{part}"' for part in raw.split("."))
    normalized_grain = str(grain or "").upper()
    expressions = {
        "YEAR": f"strftime('%Y', {qfield})",
        "MONTH": f"strftime('%Y-%m', {qfield})",
        "DAY": f"date({qfield})",
        "WEEK": f"strftime('%Y-%W', {qfield})",
        "WEEKDAY": f"strftime('%w', {qfield})",
        "QUARTER": (
            f"printf('%04d-Q%d', CAST(strftime('%Y', {qfield}) AS INTEGER), "
            f"((CAST(strftime('%m', {qfield}) AS INTEGER) - 1) / 3) + 1)"
        ),
        "HOUR": f"strftime('%Y-%m-%d %H', {qfield})",
    }
    return expressions.get(normalized_grain, qfield)


def _materialize_vql_grouping_sql(sql: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Build a validation-only SQL result for a source-authorized VQL BIN.

    This does not mutate the record or claim the generated SQL is source SQL;
    it only materializes the already-recorded VQL grouping so chart-shape tests
    can inspect the result that the visual grouping describes.
    """
    visual = (constraints or {}).get("visual_grouping") or {}
    if visual.get("origin") != "vql_bin":
        return {"status": "not_vql_bin", "sql": None}
    clean_sql = str(sql or "").strip().rstrip(";").strip()
    clauses = _top_level_clause_positions(clean_sql)
    if "group_by" in clauses:
        return {"status": "sql_grouping_already_present", "sql": None}
    time_grain = (constraints or {}).get("time_grain") or {}
    fields = list(visual.get("fields") or [])
    field = str(time_grain.get("field") or (fields[0] if fields else ""))
    expression = _sqlite_vql_bin_expression(field, time_grain.get("grain"))
    if not clean_sql or not expression:
        return {"status": "unmaterializable", "sql": None, "field": field}
    insertion_points = [
        clauses[name] for name in ("having", "order_by", "limit") if name in clauses
    ]
    insertion = min(insertion_points) if insertion_points else len(clean_sql)
    materialized = (
        clean_sql[:insertion].rstrip()
        + f" GROUP BY {expression} "
        + clean_sql[insertion:].lstrip()
    ).strip()
    return {
        "status": "materialized_for_validation",
        "sql": materialized,
        "field": field,
        "grain": time_grain.get("grain"),
        "group_expression": expression,
    }


# --------------------------------------------------------------------------- #
# Phase 4 -- KPI suitability
# --------------------------------------------------------------------------- #
_DIRECT_COUNT_PATTERNS = (
    r"\bhow\s+many\b",
    r"\btotal\s+number\b",
    r"\bcount(?:s|ed|ing)?(?:\s+of)?\b",
    r"\bfrequenc(?:y|ies)\b",
)

_NUMBER_OF_PATTERNS = (
    r"\bnumber\s+of\b",
    r"\bnumber\s+[a-z][\w-]*s\b",
)


def _matches_any_nl_pattern(nl_queries: List[str], patterns: Tuple[str, ...]) -> bool:
    return any(
        re.search(pattern, str(query or ""), re.IGNORECASE)
        for query in nl_queries
        for pattern in patterns
    )


def _explicit_count_intent(
    nl_queries: List[str],
    *,
    base_profile: Optional[Dict[str, Any]] = None,
    identifier: Optional[Dict[str, Any]] = None,
    count_star: bool = False,
) -> Dict[str, Any]:
    """Return conservative evidence for an explicitly requested entity count.

    Part-to-whole words such as ``percentage``, ``share``, ``ratio`` and
    ``proportion`` do not, by themselves, request COUNT.  Bare ``number of`` is
    also ambiguous for numeric measure attributes (for example a stored
    ``number_of_platforms``); it is accepted only for COUNT(*), an
    identifier-like field, or a non-numeric entity/category field.  This keeps
    the documented COUNT(id) exception while failing closed on COUNT(measure).
    """
    direct = _matches_any_nl_pattern(nl_queries, _DIRECT_COUNT_PATTERNS)
    number_of = _matches_any_nl_pattern(nl_queries, _NUMBER_OF_PATTERNS)
    identifier_like = bool((identifier or {}).get("is_identifier"))
    normalized_dtype = (base_profile or {}).get("normalized_dtype")
    number_of_entity_supported = bool(
        count_star
        or identifier_like
        or (base_profile is not None and normalized_dtype not in (None, "number"))
    )
    explicit = direct or (number_of and number_of_entity_supported)
    return {
        "explicit": explicit,
        "direct_count_wording": direct,
        "number_of_wording": number_of,
        "number_of_entity_supported": number_of_entity_supported,
        "count_star": count_star,
        "base_normalized_dtype": normalized_dtype,
        "identifier_like_base": identifier_like,
    }


def _unambiguous_count_intent(nl_queries: List[str]) -> bool:
    """Count wording safe for cross-aggregate conflict detection.

    Bare ``number of`` can denote a numeric source attribute (for example
    ``number_of_platforms``), so it is acceptable for an encoded COUNT but is
    not strong enough by itself to reject a source SUM/AVG/MIN/MAX.
    """
    return _matches_any_nl_pattern(nl_queries, _DIRECT_COUNT_PATTERNS)


def _aggregate_intent_for_quality(nl_queries: List[str]) -> Optional[str]:
    """Narrow aggregate intent used for mandatory cross-function checks."""
    intent = detect_query_intent(nl_queries, allow_count_number_of=False)
    if intent is None and _unambiguous_count_intent(nl_queries):
        return "COUNT"
    return intent


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
        base = extract_base_field(primary_kpi)
        evidence["policy"] = "count-requires-explicit-entity-count-intent"
        profile: Optional[Dict[str, Any]] = None
        id_flag: Optional[Dict[str, Any]] = None
        if base:
            profile = profiler.profile_field(db_id, base, sql_context=sql)
            id_flag = detect_identifier(profile, base, cfg)
            evidence["base_field_profile"] = profile
            evidence["identifier"] = id_flag
            if profile.get("resolution") == "ambiguous_table":
                failed.append("field_table_ambiguous")
        count_intent = _explicit_count_intent(
            nl_queries,
            base_profile=profile,
            identifier=id_flag,
            count_star=base is None,
        )
        evidence["count_intent"] = count_intent
        evidence["explicit_count_intent"] = count_intent["explicit"]
        if not count_intent["explicit"]:
            failed.extend(["wrong_kpi", "goal_mismatch"])

    elif outer in ("SUM", "AVG", "MIN", "MAX"):
        base = extract_base_field(primary_kpi)
        if base:
            profile = profiler.profile_field(db_id, base, sql_context=sql)
            evidence["base_field_profile"] = profile
            id_flag = detect_identifier(profile, base, cfg)
            evidence["identifier"] = id_flag
            if profile.get("resolution") == "ambiguous_table":
                failed.append("field_table_ambiguous")
            elif id_flag["is_identifier"] and id_flag["confidence"] == "strong":
                failed.extend([
                    "meaningless_identifier_aggregation",
                    "identifier_as_measure",
                    "invalid_identifier_aggregation",
                    "wrong_kpi",
                ])
            elif id_flag["is_identifier"] and id_flag["confidence"] == "ambiguous":
                failed.append("possible_identifier_aggregation")
            if profile.get("stats_available") and profile.get("normalized_dtype") not in (None, "number"):
                failed.append("aggregate_dtype_conflict")
        intent = _aggregate_intent_for_quality(nl_queries)
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

    failed = list(dict.fromkeys(failed))
    warnings = list(dict.fromkeys(warnings))
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
        intent = _aggregate_intent_for_quality(nl_queries)
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

    # Classify GROUP BY terms before checking preservation. Aggregate calls are
    # invalid keys in the supported SQLite source dialect and cannot supply an
    # analytical observation unit.
    group_terms = classify_group_by_terms(sql)
    invalid_group_terms = [
        term for term in group_terms if term["kind"] == "invalid_aggregate_expression"
    ]
    valid_sql_group_fields = [
        str(term["field"]) for term in group_terms if term["kind"] == "field" and term.get("field")
    ]
    evidence["group_by_terms"] = group_terms
    if invalid_group_terms:
        failed.append("invalid_group_by_expression")
        evidence["invalid_group_by_expressions"] = [term["expression"] for term in invalid_group_terms]

    # Grouping: every valid SQL GROUP BY column must be represented somewhere.
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
    recorded_visual_grouping = (prov.get("constraints") or {}).get("visual_grouping") or {}
    for field in recorded_visual_grouping.get("fields") or []:
        represented.add(str(field).split(".")[-1].strip().lower())
    missing_group = [g for g in valid_sql_group_fields
                     if g.split(".")[-1].strip().lower() not in represented]
    if missing_group:
        failed.append("missing_required_grouping")
        evidence["missing_grouping"] = missing_group

    # Aggregate tasks that explicitly request a per-dimension result need a
    # valid SQL field group or a matching VQL BIN. A selected raw axis alone is
    # not grouping evidence, and an unrelated BIN cannot rescue the task.
    #
    # When SQL already has a valid GROUP BY field, free-text-to-field conflicts
    # are adjudicated by check_source_consistency(), which first requires the
    # requested phrase to map to the query's real schema. This separation is
    # intentional: nvBench wording frequently appends non-dimensions after
    # ``each`` (for example "each received" or "each Visualize by bar chart").
    # Treating those noisy phrases as authoritative here would reject valid
    # source groupings. Missing SQL grouping and visual-only grouping remain
    # fail-closed because their structural evidence is unambiguous.
    has_aggregate = bool(extract_select_aggregates(sql)) or any(
        (axis_typing.get(axis) or {}).get("aggregate") for axis in ("x", "y")
    )
    visual_grouping = (prov.get("constraints") or {}).get("visual_grouping") or {}
    vql_group_fields = (
        [str(field) for field in (visual_grouping.get("fields") or [])]
        if visual_grouping.get("origin") == "vql_bin"
        else []
    )
    valid_group_fields = valid_sql_group_fields + vql_group_fields
    recorded_time_grain = (prov.get("constraints") or {}).get("time_grain") or {}
    if recorded_time_grain.get("grain"):
        valid_group_fields.append(str(recorded_time_grain["grain"]))
    requested_dimensions = extract_nl_required_dimensions(prov.get("nl_query", "") or "")
    evidence["requested_grouping_dimensions"] = requested_dimensions
    evidence["valid_grouping_fields"] = valid_group_fields

    def grouping_matches(dimension: str, field: str) -> bool:
        requested = _normalized_field_tokens(dimension)
        supplied = _normalized_field_tokens(field)
        aliases = {"gender": "sex", "sex": "gender", "appellation": "appelation",
                   "appelation": "appellation", "apt": "apartment",
                   "apartment": "apt",
                   "fac": "faculty", "act": "activity"}
        requested |= {aliases[token] for token in requested if token in aliases}
        supplied |= {aliases[token] for token in supplied if token in aliases}
        requested_core = requested - {
            "id", "key", "code", "member", "entity", "record", "item", "row"
        }
        supplied_core = supplied - {"id", "key", "code"}
        if requested_core and supplied_core and (
            requested_core <= supplied_core or supplied_core <= requested_core
        ):
            return True

        def token_matches(left: str, right: str) -> bool:
            return left == right or (
                len(left) >= 5 and len(right) >= 5 and left[:5] == right[:5]
            )

        if requested_core and supplied_core and (
            all(any(token_matches(left, right) or left in right for right in supplied_core)
                for left in requested_core)
            or all(any(token_matches(left, right) or right in left for left in requested_core)
                   for right in supplied_core)
        ):
            return True

        profile = profiler.profile_field(db_id, field, sql_context=sql)
        table_tokens = _normalized_field_tokens(profile.get("table") or "")
        table_tokens |= {aliases[token] for token in table_tokens if token in aliases}
        identifier = detect_identifier(profile, field, cfg)
        representative = (
            identifier.get("is_identifier") and identifier.get("confidence") == "strong"
        ) or (
            profile.get("stats_available")
            and profile.get("unique_ratio") is not None
            and profile.get("distinct_count") is not None
            and profile["unique_ratio"] >= 0.9
            and profile["distinct_count"] >= 2
        )
        return bool(requested_core & table_tokens) and representative

    missing_requested_grouping = [
        item for item in requested_dimensions
        if not any(grouping_matches(item["dimension"], field) for field in valid_group_fields)
    ]
    structurally_unverified_grouping = not valid_sql_group_fields
    if has_aggregate and structurally_unverified_grouping and missing_requested_grouping:
        failed.extend(["missing_grouping", "goal_mismatch"])
        evidence["missing_requested_grouping"] = missing_requested_grouping

    # ORDER BY/LIMIT after aggregation can rank groups only by a grouping key or
    # aggregate result. A row-level field outside that scope cannot implement a
    # top-N population request (e.g. top players before weekday aggregation).
    constraints = prov.get("constraints") or {}
    sort = constraints.get("sort") or {}
    limit = constraints.get("limit")
    if has_aggregate and limit is not None and sort.get("status") == "ok" and sort.get("field"):
        sort_expression = str(sort["field"])
        sort_key = _normalized_sort_key(sort_expression)
        aggregate_items = extract_select_aggregates(sql)
        aggregate_bases = {
            str(item.get("base_field") or "").split(".")[-1].strip().lower()
            for item in aggregate_items
            if item.get("base_field")
        }
        aggregate_aliases = {
            str(item.get("alias") or "").split(".")[-1].strip().lower()
            for item in aggregate_items
            if item.get("alias")
        }
        grouping_keys = {
            str(field).split(".")[-1].strip().lower() for field in valid_group_fields
        }
        sort_is_aggregate = bool(parse_aggregate(sort_expression))
        preaggregation_scope = _verified_preaggregation_scope(
            sql, sort_expression, int(limit)
        )
        sort_scope_valid = (
            sort_is_aggregate
            or sort_key in grouping_keys
            or sort_key in aggregate_aliases
            or preaggregation_scope["verified"]
        )
        if not sort_scope_valid:
            failed.extend(["constraint_scope_error", "goal_mismatch"])
            evidence["constraint_scope_error"] = {
                "sort": sort,
                "limit": limit,
                "valid_grouping_fields": valid_group_fields,
                "aggregate_base_fields": sorted(aggregate_bases),
                "aggregate_aliases": sorted(aggregate_aliases),
                "preaggregation_scope": preaggregation_scope,
            }

    return {"failed_rules": list(dict.fromkeys(failed)), "evidence": evidence}


# --------------------------------------------------------------------------- #
# v5: query / SQL / VQL source-consistency checks
# --------------------------------------------------------------------------- #
def _normalized_field_tokens(value: str) -> set:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value or ""))
    normalized = re.sub(r"[^a-z0-9]+", "_", camel_split.lower()).strip("_")
    tokens = set()
    for token in normalized.split("_"):
        if not token or token in {"in", "of", "the", "a", "an"}:
            continue
        if token.endswith("ies") and token != "series":
            token = token[:-3] + "y"
        elif token.endswith("s") and not token.endswith(("ss", "series")):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _dimension_tokens_match(requested: set, source: set) -> bool:
    """Conservative token/stem match for schema-backed dimension phrases."""
    aliases = {"gender": "sex", "sex": "gender", "appellation": "appelation",
               "appelation": "appellation"}
    expanded_requested = requested | {aliases[t] for t in requested if t in aliases}
    expanded_source = source | {aliases[t] for t in source if t in aliases}
    if expanded_requested & expanded_source:
        return True
    return any(
        len(left) >= 5 and len(right) >= 5 and left[:5] == right[:5]
        for left in expanded_requested for right in expanded_source
    )


def check_source_consistency(
    record: Dict[str, Any], profiler: Optional[DbProfiler] = None,
) -> Dict[str, Any]:
    """Detect deterministic query/SQL/VQL conflicts without inventing clauses.

    Natural-language evidence is deliberately narrow: explicit ``for each`` or
    entity-series dimensions, exact count thresholds, and explicit bin/interval
    grains. A failure records the conflict; it never rewrites SQL or VQL.
    """
    prov = _prov(record)
    mapping = _mapping0(record)
    encoding = mapping.get("encoding") or {}
    axis_typing = prov.get("axis_typing") or {}
    grouping = prov.get("grouping") or {}
    constraints = prov.get("constraints") or {}
    sql = _sql_of(prov)
    nl_query = prov.get("nl_query", "") or ""
    db_id = prov.get("db_id", "") or ""

    failed: List[str] = []
    evidence: Dict[str, Any] = {}

    if not str(nl_query).strip():
        failed.extend(["missing_nl_goal", "goal_mismatch"])
        evidence["goal_text_status"] = "missing"
    else:
        evidence["goal_text_status"] = "present"

    source_dimensions: List[str] = []
    for axis in ("x", "y"):
        typed = axis_typing.get(axis) or {}
        if not typed.get("aggregate") and typed.get("name"):
            source_dimensions.append(str(typed["name"]))
    for value in (
        encoding.get("group_field"),
        *((grouping.get("normalized_fields") or [])),
        *((grouping.get("sql_group_by_fields") or [])),
    ):
        if value:
            source_dimensions.append(str(value))
    time_grain = constraints.get("time_grain") or {}
    if time_grain.get("field"):
        source_dimensions.append(str(time_grain["field"]))
    if time_grain.get("grain"):
        source_dimensions.append(str(time_grain["grain"]))

    query_dimensions = extract_nl_required_dimensions(nl_query)
    evidence["query_required_dimensions"] = query_dimensions
    evidence["source_dimensions"] = source_dimensions
    source_token_sets = [_normalized_field_tokens(value) for value in source_dimensions]
    schema_fields = set(
        profiler.query_schema_fields(db_id, sql_context=sql)
    ) if profiler is not None else set()
    schema_tables = set(
        profiler.query_schema_tables(db_id, sql_context=sql)
    ) if profiler is not None else set()
    schema_table_token_sets = [_normalized_field_tokens(table) for table in schema_tables]
    missing_dimensions: List[Dict[str, str]] = []
    for requested in query_dimensions:
        dimension = requested["dimension"]
        requested_tokens = _normalized_field_tokens(dimension)
        if dimension.endswith("_series"):
            entity = dimension[: -len("_series")]
            available = any({entity, "series"} <= tokens for tokens in source_token_sets)
        else:
            # Only enforce a free-text entity/dimension term when it maps to a
            # real schema field. Otherwise synonyms such as gender->Sex,
            # city->Official_Name, or apartment->apt_number are not safe enough
            # to support a deterministic rejection.
            schema_supported = profiler is None or any(
                _dimension_tokens_match(requested_tokens, _normalized_field_tokens(field))
                for field in schema_fields
            )
            if not schema_supported:
                continue
            available = any(
                requested_tokens <= tokens
                or _dimension_tokens_match(requested_tokens, tokens)
                for tokens in source_token_sets
            )
            # A generic selected ``Name`` column can represent an entity named
            # by its source table (e.g. technician name). This does not rescue a
            # requested series, which requires an actual series field above.
            generic_descriptors = {
                "name", "first", "last", "official", "number", "title", "label", "description"
            }
            if not available and any(tokens & generic_descriptors for tokens in source_token_sets):
                available = any(
                    _dimension_tokens_match(requested_tokens, table_tokens)
                    for table_tokens in schema_table_token_sets
                )
        if not available:
            missing_dimensions.append(requested)
    if missing_dimensions:
        failed.extend(["missing_required_dimension", "source_conflict"])
        evidence["missing_required_dimensions"] = missing_dimensions

    query_conditions = extract_nl_aggregate_conditions(nl_query)
    source_conditions = constraints.get("having") or []
    source_filters = constraints.get("filters") or []
    evidence["query_aggregate_conditions"] = query_conditions
    evidence["source_having_conditions"] = source_conditions
    missing_conditions = []
    for requested in query_conditions:
        represented = any(
            str(condition.get("aggregate", "")).upper() == requested["aggregate"]
            and str(condition.get("operator")) == requested["operator"]
            and str(condition.get("value")) == requested["value"]
            for condition in source_conditions
        )
        if not represented:
            represented = any(
                requested["subject"] in str(source_filter.get("field") or "").lower()
                and str(source_filter.get("operator")) == requested["operator"]
                and str(source_filter.get("value")) == requested["value"]
                for source_filter in source_filters
            )
        if not represented:
            missing_conditions.append(requested)
    if missing_conditions:
        failed.extend(["missing_aggregate_condition", "source_conflict"])
        evidence["missing_aggregate_conditions"] = missing_conditions

    query_grains = extract_nl_time_grains(nl_query)
    source_grain = str(time_grain.get("grain") or "").upper() or None
    evidence["query_time_grains"] = query_grains
    evidence["source_time_grain"] = source_grain
    if query_grains and source_grain and source_grain not in query_grains:
        failed.extend(["time_grain_source_conflict", "source_conflict"])
    elif query_grains and not source_grain:
        failed.extend(["missing_required_time_grain", "source_conflict"])

    return {"failed_rules": list(dict.fromkeys(failed)), "evidence": evidence}


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


def _is_count_axis(axis_typed: Dict[str, Any]) -> bool:
    """True when the plotted value is a count, not the identifier base itself."""
    return str(axis_typed.get("aggregate") or "").upper() == "COUNT"


def chart_bar(record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prov, axis_typing, db_id, sql, (xf, xp, xid), (yf, yp, yid) = _base_chart_evidence(record, profiler, cfg)
    failed: List[str] = []
    warnings: List[str] = []
    yt = axis_typing.get("y") or {}
    if (yt.get("role") == "measure" and not _is_count_axis(yt)
            and yid["is_identifier"] and yid["confidence"] == "strong"):
        failed.append("identifier_as_measure")
    elif (yt.get("role") == "measure" and not _is_count_axis(yt)
          and yid["is_identifier"] and yid["confidence"] == "ambiguous"):
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
    if (yt.get("role") == "measure" and not _is_count_axis(yt)
            and yid["is_identifier"] and yid["confidence"] == "strong"):
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
    # Percentages over independent dates are separate observations, not slices
    # of one additive whole. Preserve the Pie source label but demote it.
    x_typed = axis_typing.get("x") or {}
    y_name = str((axis_typing.get("y") or {}).get("name") or "")
    temporal_x = x_typed.get("dtype") == "datetime" or _name_is_temporal(str(x_typed.get("name") or ""), cfg)
    percentage_measure = bool(re.search(r"(^|_)(percent|percentage|rate|share)($|_)", y_name, re.IGNORECASE))
    if not y_agg and temporal_x and percentage_measure:
        failed.append("pie_not_part_to_whole")
    evidence = {"x_profile": xp, "y_profile": yp, "x_identifier": xid, "max_categories": max_categories,
                "y_aggregate": y_agg, "additive_aggregates": sorted(additive_aggs)}
    return {"passed": not failed, "failed_rules": failed, "warnings": warnings, "evidence": evidence}


def chart_scatter(record: Dict[str, Any], profiler: DbProfiler, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prov, axis_typing, db_id, sql, (xf, xp, xid), (yf, yp, yid) = _base_chart_evidence(record, profiler, cfg)
    failed: List[str] = []
    warnings: List[str] = []
    min_distinct = int(cfg["chart"]["scatter"]["min_distinct_values"])
    scatter_cfg = cfg.get("chart", {}).get("scatter", {}) or {}
    identifier_reason = scatter_cfg.get("identifier_axis_reason_code", "scatter_identifier_axis")
    invalid_reason = scatter_cfg.get("invalid_axes_reason_code", "invalid_scatter_axes")
    invalid = False
    identifier_axis = False
    for axis_name, t, profile, idf in (("x", axis_typing.get("x") or {}, xp, xid),
                                        ("y", axis_typing.get("y") or {}, yp, yid)):
        if t.get("dtype") != "number" or t.get("role") != "measure":
            invalid = True
        if idf["is_identifier"] and not _is_count_axis(t):
            identifier_axis = True
            invalid = True
            if t.get("role") == "measure":
                failed.append("identifier_as_measure")
        # Aggregate output variation must be measured from the actual result
        # below. Its base column (or COUNT(*)) is not the plotted distribution.
        if not t.get("aggregate"):
            if _chart_shape_evidence_insufficient(profile):
                invalid = True
                failed.append(f"insufficient_axis_evidence:{axis_name}")
            elif profile.get("distinct_count") is not None and profile["distinct_count"] < min_distinct:
                invalid = True
                failed.append(f"low_variation_axis:{axis_name}")
    if identifier_axis:
        failed.append(identifier_reason)

    # A Scatter needs an observation unit. Aggregate axes without a valid raw
    # SQL grouping field or VQL BIN collapse to one point (or invalid SQL), so
    # base-table column variation cannot certify the chart.
    group_terms = classify_group_by_terms(sql)
    valid_observation_fields = [
        str(term["field"]) for term in group_terms if term["kind"] == "field" and term.get("field")
    ]
    visual_grouping = (prov.get("constraints") or {}).get("visual_grouping") or {}
    if visual_grouping.get("origin") == "vql_bin":
        valid_observation_fields.extend(str(field) for field in (visual_grouping.get("fields") or []))
    aggregate_axes = [
        axis for axis in ("x", "y") if (axis_typing.get(axis) or {}).get("aggregate")
    ]
    observation_profiles = [
        profiler.profile_field(db_id, field, sql_context=sql) for field in valid_observation_fields
    ]
    result_max_rows = int(scatter_cfg.get("result_profile_max_rows", 1000))
    source_result_profile = profiler.profile_query_result(db_id, sql, max_rows=result_max_rows)
    result_profile = source_result_profile
    result_profile_origin = "source_sql"
    vql_materialization: Dict[str, Any] = {"status": "not_attempted", "sql": None}
    if aggregate_axes and visual_grouping.get("origin") == "vql_bin":
        vql_materialization = _materialize_vql_grouping_sql(
            sql, prov.get("constraints") or {}
        )
        if vql_materialization.get("sql"):
            materialized_profile = profiler.profile_query_result(
                db_id, str(vql_materialization["sql"]), max_rows=result_max_rows
            )
            vql_materialization["result_profile"] = materialized_profile
            result_profile = materialized_profile
            result_profile_origin = "vql_bin_validation_materialization"
    result_columns = result_profile.get("columns") or []
    result_axes_valid = (
        result_profile.get("execution_status") == "ok"
        and len(result_columns) >= 2
        and result_profile.get("paired_numeric_row_count", 0) >= 2
        and result_profile.get("paired_numeric_distinct_x_count", 0) >= 2
        and result_profile.get("paired_numeric_distinct_y_count", 0) >= 2
        and result_profile.get("paired_numeric_distinct_pair_count", 0) >= 2
    )
    if not result_axes_valid:
        failed.extend([
            "insufficient_scatter_observations",
            invalid_reason,
            "chart_inappropriate",
        ])
        invalid = True
    if invalid:
        failed.append(invalid_reason)
    evidence = {"x_profile": xp, "y_profile": yp, "x_identifier": xid, "y_identifier": yid,
                "min_distinct_values": min_distinct,
                "group_by_terms": group_terms,
                "valid_observation_fields": valid_observation_fields,
                "observation_profiles": observation_profiles,
                "aggregate_axes": aggregate_axes,
                "query_result_profile_origin": result_profile_origin,
                "source_query_result_profile": source_result_profile,
                "vql_bin_validation_materialization": vql_materialization,
                "query_result_profile": result_profile}
    failed = list(dict.fromkeys(failed))
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
    if (yt.get("role") == "measure" and not _is_count_axis(yt)
            and yid["is_identifier"] and yid["confidence"] == "strong"):
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
    consistency_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    weights = cfg["scoring"]["weights"]
    tier_a_min_score = int(cfg["scoring"]["tier_a_min_score"])
    constraint_result = constraint_result or {"failed_rules": [], "evidence": {}}
    consistency_result = consistency_result or {"failed_rules": [], "evidence": {}}

    semantic_constraint_failed = [c for c in fidelity_failed if c in _CONSTRAINT_CHECK_NAMES]
    general_fidelity_failed = [c for c in fidelity_failed if c not in _CONSTRAINT_CHECK_NAMES]
    general_fidelity_failed.extend(consistency_result.get("failed_rules", []))
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

    # One scientific defect may be detected independently by multiple
    # components (for example identifier_as_measure in both KPI and chart
    # checks). Preserve the evidence within each component, but expose each
    # record-level reason code once so summaries count affected records rather
    # than double-counting detector paths.
    failed_rules = list(dict.fromkeys(failed_rules))
    warnings = list(dict.fromkeys(warnings))

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

        # Rule precedence: constraint extraction and source consistency are
        # evaluated before KPI/identifier/chart suitability.
        constraint_result = check_required_constraints(rec, profiler, cfg)
        consistency_result = check_source_consistency(rec, profiler)
        kpi_result = kpi_suitability(rec, profiler, cfg)
        checker = CHART_CHECKERS.get(chart_type)
        chart_result = checker(rec, profiler, cfg) if checker else {
            "passed": False, "failed_rules": [f"unknown_chart_type:{chart_type}"], "warnings": [], "evidence": {},
        }
        fidelity_failed = fidelity_failed_map.get(iid, [])
        quality = score_and_tier(rec, kpi_result, chart_result, fidelity_failed, cfg,
                                 constraint_result=constraint_result,
                                 consistency_result=consistency_result)
        quality["kpi_suitability"] = kpi_result
        quality["chart_suitability"] = chart_result
        quality["constraint_suitability"] = constraint_result
        quality["source_consistency"] = consistency_result
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
