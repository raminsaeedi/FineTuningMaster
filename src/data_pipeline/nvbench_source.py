"""Pure, reusable logic for mapping nvBench source records into GoldItems.

This module is deliberately free of I/O side effects beyond opening the cached
SQLite databases read-only for column metadata. It is imported by
``NvBenchBuilder`` and exercised directly by the tests.

Key guarantees:
- Chart labels are mapped through a versioned YAML config; unsupported labels are
  rejected (never silently coerced).
- Stable, group-aware IDs: all queries of one visualization key share a split.
- Full field-level lineage and source provenance are preserved on every item.
- ``task_type`` is rule-derived (versioned) and never presented as a source label.
"""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.schemas import (
    ChartType,
    DashboardBrief,
    DesignOutput,
    GoldItem,
    KPIChartMapping,
    Rationale,
    TaskType,
)
from src.data_pipeline.nvbench_extract import (
    check_aggregate_intent_conflict,
    check_chart_query_conflict,
    classify_group_by_terms,
    extract_base_field,
    extract_group_by_fields,
    extract_having_conditions,
    extract_limit,
    extract_nested,
    extract_order_by,
    extract_select_projection_fields,
    extract_time_grain,
    extract_where_filters,
    resolve_nested,
)
from src.data_pipeline.splits import assign_split
from src.utils.io import read_yaml

_AGG_RE = re.compile(r"^\s*(\w+)\s*\(", re.IGNORECASE)
_DATETIME_HINT_RE = re.compile(r"date|time|year|month|day|week|quarter", re.IGNORECASE)


class RejectedRecord(Exception):
    """Raised when a source record cannot be mapped without an unsupported assumption."""

    def __init__(self, reason: str, detail: str = "", evidence: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail
        self.evidence = evidence or {}


# --------------------------------------------------------------------------- #
# mapping config
# --------------------------------------------------------------------------- #
def load_mapping(path: str | Path) -> Dict[str, Any]:
    """Load and lightly validate the versioned nvBench mapping YAML."""
    cfg = read_yaml(path)
    if "chart_map" not in cfg or "task_rules" not in cfg:
        raise ValueError(f"invalid nvBench mapping config: {path}")
    return cfg


def map_chart(label: str, mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Map an nvBench chart label to a project chart type.

    Returns ``{"chart_type": ChartType, "grouped": bool}``. Raises
    ``RejectedRecord`` for unsupported labels (no silent coercion).
    """
    chart_map = mapping["chart_map"]
    entry = chart_map.get(label)
    if entry is None:
        raise RejectedRecord("unsupported_chart", f"chart label not in mapping: {label!r}")
    return {"chart_type": ChartType(entry["chart_type"]), "grouped": bool(entry["grouped"])}


def infer_task(base_chart: ChartType, grouped: bool, mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-derive a task_type with version, confidence, evidence and status."""
    rules = mapping["task_rules"]
    rule = rules["by_chart"].get(base_chart.value, rules["default"])
    evidence = rule["evidence"]
    if grouped and rules.get("grouped_evidence"):
        evidence = f"{evidence}; {rules['grouped_evidence']}"
    return {
        "task_type": TaskType(rule["task_type"]),
        "rule_version": rules["version"],
        "confidence": float(rule["confidence"]),
        "evidence": evidence,
        "derivation_status": "rule-derived",
    }


# --------------------------------------------------------------------------- #
# stable ids + split
# --------------------------------------------------------------------------- #
def base_key(key: str) -> str:
    """Strip nvBench sort-ordering suffixes to the base visualization key.

    nvBench encodes ordering variants as ``<base>@x_name@ASC`` etc. These share
    the same chart, database and (near-duplicate) NL queries, so they belong to
    one visualization for grouping/splitting purposes.
    """
    return key.split("@", 1)[0]


def source_group_id(key: str) -> str:
    return f"nvbench:{base_key(key)}"


def source_record_id(key: str, query_index: int) -> str:
    return f"nvbench:{key}:query:{query_index}"


def group_split(key: str) -> str:
    """Group-aware split: every query of one visualization shares a bucket.

    Splits are derived from the *base-visualization group* id, not the per-query
    id, so all NL queries for a visualization (including its sort variants) share
    a split and can never straddle train/val. Augmentation is train/val only:
    the test bucket is remapped to train so augmentation never lands in test.
    """
    split = assign_split(source_group_id(key))
    return "train" if split == "test" else split


# --------------------------------------------------------------------------- #
# encoding + column metadata
# --------------------------------------------------------------------------- #
def parse_aggregate(expr: str) -> Optional[str]:
    """Extract an aggregate function name from a measure expression, if any."""
    if not expr:
        return None
    m = _AGG_RE.match(expr)
    return m.group(1).upper() if m else None


def _strip_alias(name: str) -> str:
    return name.split(".")[-1].strip() if name else name


def normalize_sqlite_type(sqlite_type: str) -> str:
    """Normalize a declared SQLite column type to {datetime, number, categorical}."""
    t = (sqlite_type or "").lower()
    if "date" in t or "time" in t:
        return "datetime"
    if "int" in t or any(k in t for k in ("real", "floa", "doub", "num", "dec")):
        return "number"
    return "categorical"


def is_finite_number(value: Any) -> int:
    """SQLite UDF: return 1 only for complete, finite numeric scalars."""
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(math.isfinite(float(value)))
    text = str(value).strip()
    if not text or not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return 0
    try:
        return int(math.isfinite(float(text)))
    except (TypeError, ValueError, OverflowError):
        return 0


class DbMetadataResolver:
    """Look up real column data types from the cached SQLite databases.

    Missing cache / missing database is handled gracefully: lookups return
    ``None`` and callers fall back to a name heuristic (recorded in lineage).
    """

    def __init__(self, cache_root: Optional[str | Path]) -> None:
        self.cache_root = Path(cache_root) if cache_root else None
        self._cache: Dict[str, Dict[str, str]] = {}

    @property
    def available(self) -> bool:
        return self.cache_root is not None and self.cache_root.exists()

    def _db_candidates(self, db_id: str) -> List[Path]:
        """Existing SQLite paths for a db, best first.

        nvBench ships an empty top-level ``<db_id>.sqlite`` stub alongside the
        real database at ``<db_id>/<db_id>.sqlite``; nested paths are tried first
        and any candidate with no tables is skipped by ``columns``.
        """
        if not self.available:
            return []
        ordered = [
            self.cache_root / db_id / f"{db_id}.sqlite",
            self.cache_root / "database" / db_id / f"{db_id}.sqlite",
            self.cache_root / f"{db_id}.sqlite",
            self.cache_root / "database" / f"{db_id}.sqlite",
        ]
        return [c for c in ordered if c.exists() and c.stat().st_size > 0]

    @staticmethod
    def _read_columns(path: Path) -> Dict[str, str]:
        result: Dict[str, str] = {}
        try:
            con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            con.create_function("nvbench_is_finite_number", 1, is_finite_number)
        except (sqlite3.Error, ValueError, OSError):
            return result
        try:
            tables = [
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            for t in tables:
                for row in con.execute(f'PRAGMA table_info("{t}")').fetchall():
                    col_name = str(row[1]).lower()
                    dtype = normalize_sqlite_type(str(row[2]))
                    if dtype == "categorical":
                        qtable = '"' + str(t).replace('"', '""') + '"'
                        qcol = '"' + str(row[1]).replace('"', '""') + '"'
                        non_null, numeric = con.execute(
                            f"SELECT COUNT({qcol}), "
                            f"COALESCE(SUM(nvbench_is_finite_number({qcol})), 0) FROM {qtable}"
                        ).fetchone()
                        if non_null and numeric == non_null:
                            dtype = "number"
                    result.setdefault(col_name, dtype)
        except sqlite3.Error:
            return {}
        finally:
            con.close()
        return result

    def columns(self, db_id: str) -> Dict[str, str]:
        """Return ``{column_name_lower: normalized_dtype}`` for a database."""
        if db_id in self._cache:
            return self._cache[db_id]
        result: Dict[str, str] = {}
        for path in self._db_candidates(db_id):
            cols = self._read_columns(path)
            if cols:  # first candidate that actually has tables wins
                result = cols
                break
        self._cache[db_id] = result
        return result

    def dtype_of(self, db_id: str, name: str) -> Optional[str]:
        return self.columns(db_id).get(_strip_alias(name).lower())


def _heuristic_dtype(name: str) -> str:
    return "datetime" if _DATETIME_HINT_RE.search(name or "") else "categorical"


def _axis_role(axis: str, chart_type: ChartType, dtype: str) -> str:
    """Role from chart semantics + dtype (not axis position alone).

    Scatter: both axes are measures when numeric, else dimensions. Other charts:
    x is the dimension, y is a measure only when numeric.
    """
    if chart_type == ChartType.SCATTER:
        return "measure" if dtype == "number" else "dimension"
    if axis == "x":
        return "dimension"
    return "measure" if dtype == "number" else "dimension"


def type_axis(
    axis: str, name: str, chart_type: ChartType, db_id: str, resolver: DbMetadataResolver
) -> Dict[str, Any]:
    """Deterministically type one axis expression.

    Aggregate expressions (COUNT/SUM/AVG/MIN/MAX(...)) are always
    ``number``/``measure`` with the aggregate preserved. Non-aggregate fields are
    resolved from SQLite metadata; a name heuristic is used only when metadata is
    genuinely unavailable. The y-axis is never forced to ``number`` without
    source evidence.
    """
    name = (name or "").strip()
    agg = parse_aggregate(name)
    if agg:
        return {"axis": axis, "name": name, "aggregate": agg, "dtype": "number",
                "role": "measure", "dtype_source": "aggregate-expression"}
    dtype = resolver.dtype_of(db_id, name)
    if dtype is not None:
        dtype_source = "source-provided(db)"
    else:
        dtype = _heuristic_dtype(name)
        dtype_source = "heuristic"
    return {"axis": axis, "name": name, "aggregate": None, "dtype": dtype,
            "role": _axis_role(axis, chart_type, dtype), "dtype_source": dtype_source}


def recover_group_field(
    record: Dict[str, Any], x_name: str, db_id: str, resolver: DbMetadataResolver
) -> Dict[str, Any]:
    """Recover a grouping series field from the source SQL ``GROUP BY``.

    Recovers only when exactly one non-x group-by column remains AND (when the
    database is available) that column is a real column. Otherwise returns an
    ``ambiguous``/``unresolved`` status and never invents a field name.
    """
    sql = ((record.get("vis_query") or {}).get("data_part") or {}).get("sql_part", "")
    x_low = _strip_alias(x_name).lower()
    seen: set = set()
    candidates: List[str] = []
    for col in extract_group_by_fields(sql):
        low = col.lower()
        if low == x_low or low in seen:
            continue
        seen.add(low)
        candidates.append(col)

    if len(candidates) != 1:
        status = "ambiguous" if len(candidates) > 1 else "unresolved"
        return {"series_field": None, "dtype": None, "dtype_source": None, "recovery_status": status}

    field = candidates[0]
    dtype = resolver.dtype_of(db_id, field)
    if dtype is None and resolver.available:
        # Named in SQL but not a real column (alias/expression) -> do not invent.
        return {"series_field": None, "dtype": None, "dtype_source": None, "recovery_status": "unresolved"}
    if dtype is None:
        dtype, dtype_source = _heuristic_dtype(field), "heuristic"
    else:
        dtype_source = "source-provided(db)"
    return {"series_field": field, "dtype": dtype, "dtype_source": dtype_source,
            "recovery_status": "recovered"}


def select_kpi(x_typed: Dict[str, Any], y_typed: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, documented KPI policy.

    - Exactly one aggregate axis -> that aggregate is the KPI.
    - Both axes aggregate -> y is primary, both preserved.
    - No aggregate but numeric, non-identifier-like measures -> the numeric
      measure axis (y preferred).
    - Identifier-shaped and categorical axes are never added to ``brief.kpis``.
      When no valid KPI remains, the source y expression stays in the mapping
      solely for source fidelity and quality rules demote the record.
    """
    axes = {"x": x_typed, "y": y_typed}
    agg_axes = [a for a in ("x", "y") if axes[a]["aggregate"]]
    obvious_identifier = re.compile(
        r"(?:^|_)(?:id|identifier|key|code)$|^(?:id|identifier|key|code)(?:_|$)",
        re.IGNORECASE,
    )
    measure_axes = [
        a for a in ("x", "y")
        if axes[a]["role"] == "measure"
        and axes[a].get("dtype") == "number"
        and not obvious_identifier.search(_strip_alias(str(axes[a].get("name") or "")))
    ]

    if len(agg_axes) == 1:
        primary, policy = agg_axes[0], "single-aggregate"
        evidence = f"exactly one aggregate axis ({primary}); its expression is the KPI"
    elif len(agg_axes) == 2:
        primary, policy = "y", "dual-aggregate-y-primary"
        evidence = "both axes are aggregate expressions; y chosen as primary, both preserved"
    elif measure_axes:
        primary = "y" if "y" in measure_axes else measure_axes[0]
        policy = "numeric-measure-y-primary"
        evidence = "no aggregate; primary numeric measure axis chosen (y preferred)"
    else:
        primary, policy = "y", "no-valid-kpi"
        evidence = "no aggregate or eligible numeric measure; source y is retained only in the encoding"

    eligible_axes = agg_axes if agg_axes else measure_axes
    ordered = ([axes[primary]["name"]] if primary in eligible_axes else []) + [
        axes[a]["name"] for a in ("x", "y") if a != primary and a in eligible_axes
    ]
    seen: set = set()
    kpis: List[str] = []
    for k in ordered:
        if k not in seen:
            seen.add(k)
            kpis.append(k)
    return {"primary_kpi": axes[primary]["name"], "primary_axis": primary, "kpis": kpis,
            "policy": policy, "aggregate_axes": agg_axes, "measure_axes": measure_axes,
            "evidence": evidence, "derivation_status": "rule-derived"}


def _evidence(key: str, query_index: int, nl_query: str, label: str, sql: str, vql: str) -> Dict[str, Any]:
    return {
        "source_group_id": source_group_id(key),
        "source_record_id": source_record_id(key, query_index),
        "original_query": nl_query,
        "source_chart": label,
        "sql": sql,
        "vql": vql,
    }


class _RawColumns:
    """Ordered, deduplicated raw-column collector for ``brief.columns``.

    First registration of a field name wins (callers register in x/y/group/
    filter/sort/time-bin priority order), so a field referenced by more than one
    construct keeps its most semantically specific role.
    """

    def __init__(self, db_id: str, resolver: DbMetadataResolver) -> None:
        self.db_id = db_id
        self.resolver = resolver
        self._by_name: Dict[str, Dict[str, str]] = {}
        self.lineage: Dict[str, str] = {}

    def add(self, name: Optional[str], role: str) -> None:
        if not name:
            return
        low = name.lower()
        if low in self._by_name:
            return
        dtype = self.resolver.dtype_of(self.db_id, name)
        source = "source-provided(db)" if dtype is not None else "heuristic"
        if dtype is None:
            dtype = _heuristic_dtype(name)
        self._by_name[low] = {"name": name, "dtype": dtype, "role": role}
        self.lineage[name] = source

    def has(self, name: Optional[str]) -> bool:
        return bool(name) and name.lower() in self._by_name

    def as_list(self) -> List[Dict[str, str]]:
        return list(self._by_name.values())


# --------------------------------------------------------------------------- #
# record -> GoldItem
# --------------------------------------------------------------------------- #
def build_gold_item(
    key: str,
    record: Dict[str, Any],
    query_index: int,
    nl_query: str,
    mapping: Dict[str, Any],
    resolver: DbMetadataResolver,
) -> GoldItem:
    """Map one (visualization key, NL query) pair into a schema-valid GoldItem.

    ``brief.columns`` contains only physical raw source fields (never an
    aggregate expression); derived expressions live in ``kpis``/
    ``kpi_chart_mapping.kpi``/``encoding``. Every source constraint (filter, sort,
    time grain, grouping) that is present but cannot be represented reliably
    causes a ``RejectedRecord`` rather than a silent omission.
    """
    label = record.get("chart", "")
    vis_obj = record.get("vis_obj", {}) or {}
    vis_query = record.get("vis_query", {}) or {}
    data_part = vis_query.get("data_part", {}) or {}
    sql = data_part.get("sql_part", "") or ""
    vql = vis_query.get("VQL", "") or ""
    binning = data_part.get("binning", "") or ""
    db_id = record.get("db_id", "")
    x_name = str(vis_obj.get("x_name", "")).strip()
    y_name = str(vis_obj.get("y_name", "")).strip()
    classify = list(vis_obj.get("classify", []) or [])
    ev = lambda: _evidence(key, query_index, nl_query, label, sql, vql)  # noqa: E731

    # Untouched source encoding, captured before any nested-aggregate collapse.
    raw_encoding = {"x_name": x_name, "y_name": y_name, "classify": classify}

    # 1) Chart requested by the query vs. the source chart.
    conflict = check_chart_query_conflict(label, [nl_query])
    if conflict:
        raise RejectedRecord("chart_query_conflict", conflict, evidence=ev())

    # 2) Chart-label mapping (unsupported labels rejected, never coerced).
    try:
        chart = map_chart(label, mapping)
    except RejectedRecord as exc:
        raise RejectedRecord(exc.reason, exc.detail, evidence=ev()) from exc
    chart_type: ChartType = chart["chart_type"]
    grouped: bool = chart["grouped"]

    # 3) Nested-aggregate resolution (normalize or reject) on each axis.
    nested_normalized: Dict[str, str] = {}
    for axis_name, expr in (("x", x_name), ("y", y_name)):
        nested = extract_nested(expr)
        if not nested:
            continue
        res = resolve_nested(nested, [nl_query])
        if res["action"] == "reject":
            raise RejectedRecord(res["reason"], res["detail"], evidence=ev())
        nested_normalized[axis_name] = res["collapsed"]
    if "x" in nested_normalized:
        x_name = nested_normalized["x"]
    if "y" in nested_normalized:
        y_name = nested_normalized["y"]

    # 4) Query-aggregate-intent vs. encoded-aggregate conflict (non-nested axes;
    #    nested axes were already checked inside resolve_nested).
    for axis_name, expr in (("x", x_name), ("y", y_name)):
        if axis_name in nested_normalized:
            continue
        conflict = check_aggregate_intent_conflict(expr, [nl_query], resolver, db_id)
        if conflict:
            raise RejectedRecord("aggregate_intent_conflict", conflict, evidence=ev())

    # 5) Expression-aware, independent typing of both axes.
    x_typed = type_axis("x", x_name, chart_type, db_id, resolver)
    y_typed = type_axis("y", y_name, chart_type, db_id, resolver)

    # 6) Scatter charts require two numeric axes for positive training data.
    if chart_type == ChartType.SCATTER:
        cat_axes = [t for t in (x_typed, y_typed) if t["dtype"] == "categorical"]
        if cat_axes:
            fields = ", ".join(f"{t['axis']}={t['name']}" for t in cat_axes)
            raise RejectedRecord(
                "categorical_scatter_axis",
                f"scatter axis is categorical per database metadata: {fields}",
                evidence=ev(),
            )

    raw_cols = _RawColumns(db_id, resolver)
    for typed in (x_typed, y_typed):
        if typed["aggregate"]:
            base = extract_base_field(typed["name"])
            if base:  # COUNT(*) has no physical column; never invented
                raw_cols.add(base, "measure")
        else:
            raw_cols.add(typed["name"], typed["role"])
    # Preserve every physical top-level projection, including source fields
    # that are not chosen as the chart's x/y axes. Never infer NL-only fields.
    for projected_field in extract_select_projection_fields(sql):
        raw_cols.add(projected_field, "source_projection")

    # 7) Grouping series-field recovery — mandatory (reject) for every grouped
    #    chart, including Stacked Bar, which always carries classify values.
    grouping: Dict[str, Any] = {"is_grouped": grouped, "classify": classify,
                                "series_field": None, "recovery_status": "not_grouped"}
    group_field = None
    if grouped:
        rec_g = recover_group_field(record, x_name, db_id, resolver)
        grouping.update(rec_g)
        if not rec_g["series_field"]:
            raise RejectedRecord(
                "missing_group_field",
                f"grouped chart '{label}' has classify values but no unambiguous "
                f"GROUP BY series field (status={rec_g['recovery_status']})",
                evidence=ev(),
            )
        group_field = rec_g["series_field"]
        raw_cols.add(group_field, "series")

    # 8) Filters: represent only unambiguous AND-only/single WHERE clauses.
    filters_info = extract_where_filters(sql)
    if filters_info["status"] == "unrepresentable":
        raise RejectedRecord("unpreserved_filter", filters_info["detail"], evidence=ev())
    for f in filters_info["filters"]:
        raw_cols.add(f["field"], "filter")

    # 9) Sort: represent only a single, unambiguous ORDER BY key.
    sort_info = extract_order_by(vql or sql)
    if sort_info["status"] == "unrepresentable":
        raise RejectedRecord("unpreserved_sort", sort_info["detail"], evidence=ev())
    if sort_info["status"] == "ok":
        sort_expr = sort_info["field"]
        # Same "no fake column" rule as x/y: COUNT(*) (and any aggregate with no
        # base field) never becomes a raw column, it just isn't re-added — it was
        # already added by the x/y aggregate handling above under its own name.
        sort_base = extract_base_field(sort_expr) if parse_aggregate(sort_expr) else sort_expr
        raw_cols.add(sort_base, "sort")

    # 10) Row limits and aggregate conditions are independent constraints.
    limit_info = extract_limit(sql or vql)
    if limit_info["status"] == "unrepresentable":
        raise RejectedRecord("unpreserved_limit", limit_info["detail"], evidence=ev())
    having_info = extract_having_conditions(sql or vql)
    if having_info["status"] == "unrepresentable":
        raise RejectedRecord("unpreserved_having", having_info["detail"], evidence=ev())
    for condition in having_info["conditions"]:
        raw_cols.add(condition.get("field"), "aggregate_filter")

    # 11) VQL time grain / visual binning. Some nvBench variants preserve BIN
    # only in VQL, so inspect VQL when data_part.binning is empty.
    time_grain = extract_time_grain(binning or vql)
    if binning.strip() and time_grain is None:
        raise RejectedRecord("unpreserved_time_grain", f"unparseable BIN clause: {binning!r}", evidence=ev())
    if time_grain:
        raw_cols.add(time_grain["field"], "time_bin")

    sql_group_fields = extract_group_by_fields(sql)
    # Every valid physical SQL grouping field is part of the analytical
    # observation unit and must be represented in raw columns, even when it is
    # not projected as x/y or used as a multi-series group_field. Invalid
    # aggregate/computed grouping expressions remain source evidence only.
    for term in classify_group_by_terms(sql):
        if term["kind"] == "field" and term.get("field"):
            raw_cols.add(term["field"], "group")
    normalized_group_fields: List[str] = []
    if time_grain:
        normalized_group_fields.append(time_grain["field"])
    for field in sql_group_fields:
        if field.lower() not in {name.lower() for name in normalized_group_fields}:
            normalized_group_fields.append(field)
    grouping.update({
        "sql_group_by_fields": sql_group_fields,
        "normalized_fields": normalized_group_fields,
        "grouping_origin": "vql_bin" if time_grain else ("sql_group_by" if sql_group_fields else "none"),
        "series_grouping_origin": "sql_group_by" if group_field else None,
        "grouping_status": "implicit_visual_grouping" if time_grain else (
            "explicit_sql_grouping" if sql_group_fields else "none"
        ),
    })

    if not raw_cols.as_list():
        raise RejectedRecord("missing_source_field", "no raw source field could be recovered", evidence=ev())

    kpi_sel = select_kpi(x_typed, y_typed)
    primary_kpi = kpi_sel["primary_kpi"]

    encoding: Dict[str, Any] = {
        "x": x_name,
        "y": y_name,
        "x_aggregate": x_typed["aggregate"],
        "y_aggregate": y_typed["aggregate"],
        "aggregate": (x_typed if kpi_sel["primary_axis"] == "x" else y_typed)["aggregate"],
        "grouped": grouped,
        "classify": classify,
        "group_field": group_field,
        "filters": filters_info["filters"],
        "sort": sort_info if sort_info["status"] == "ok" else None,
        "limit": limit_info["value"] if limit_info["status"] == "ok" else None,
        "having": having_info["conditions"],
        "time_grain": time_grain,
        "visual_grouping": {
            "fields": normalized_group_fields,
            "origin": grouping["grouping_origin"],
            "status": grouping["grouping_status"],
        },
        "source_x": x_name,
        "source_y": y_name,
    }

    task = infer_task(chart_type, grouped, mapping)

    provenance: Dict[str, Any] = {
        "source": "nvbench",
        "source_group_id": source_group_id(key),
        "source_record_id": source_record_id(key, query_index),
        "visualization_key": key,
        "base_visualization_key": base_key(key),
        "query_index": query_index,
        "original_chart_label": label,
        "db_id": db_id,
        "vis_query": vis_query,
        "vis_obj": {
            "chart": vis_obj.get("chart"),
            "x_name": raw_encoding["x_name"],
            "y_name": raw_encoding["y_name"],
            "classify": classify,
            "describe": vis_obj.get("describe", ""),
        },
        "nl_query": nl_query,
        # Verbatim source encoding, unchanged by nested-aggregate normalization or typing.
        "raw_encoding": raw_encoding,
        "nested_aggregate_normalization": nested_normalized or None,
        "axis_typing": {"x": x_typed, "y": y_typed},
        "kpi_selection": kpi_sel,
        "grouping": grouping,
        "constraints": {
            "filters": filters_info["filters"],
            "sort": sort_info if sort_info["status"] == "ok" else None,
            "limit": limit_info["value"] if limit_info["status"] == "ok" else None,
            "limit_origin": limit_info["syntax"] if limit_info["status"] == "ok" else None,
            "having": having_info["conditions"],
            "time_grain": time_grain,
            "visual_grouping": {
                "fields": normalized_group_fields,
                "origin": grouping["grouping_origin"],
                "status": grouping["grouping_status"],
            },
            "source_order": sort_info if sort_info["status"] == "ok" else None,
        },
        "build_warnings": [],
    }

    lineage: Dict[str, Any] = {
        "chart_type": "source-provided",
        "encoding": "source-provided",
        "goal": "source-provided",
        "kpi": "source-provided",  # KPI is a verbatim source axis expression
        "task_type": "rule-derived",
        "kpi_selection": "rule-derived",  # WHICH axis is primary is a rule
        "layout": "template-derived",
        "styling": "template-derived",
        "interactions": "template-derived",
        "rationales": "template-derived",
        # Two independent lineage buckets (Task 1): raw physical columns vs.
        # derived aggregate expressions never conflate their dtype origin.
        "raw_columns": dict(raw_cols.lineage),
        "derived_expressions": {
            t["name"]: "aggregate-expression" for t in (x_typed, y_typed) if t["aggregate"]
        },
    }
    if nested_normalized:
        lineage["nested_aggregate"] = "rule-normalized-nested-aggregate"

    brief = DashboardBrief(
        item_id=source_record_id(key, query_index),
        users="Data analyst exploring a relational database",
        goals=[nl_query],
        kpis=kpi_sel["kpis"],
        columns=raw_cols.as_list(),
        constraints=None,
        extra={
            "source": "nvbench",
            "usage_tier": "train_aug",
            "provenance": provenance,
            "lineage": lineage,
            "task_inference": task,
        },
    )

    recommendation = DesignOutput(
        context_summary={"db_id": db_id, "source": "nvbench", "n_kpis": len(kpi_sel["kpis"])},
        kpi_chart_mapping=[
            KPIChartMapping(
                kpi=primary_kpi,
                task_type=task["task_type"],
                chart_type=chart_type,
                alternatives=[],
                encoding=encoding,
            )
        ],
        layout={"type": "single", "blocks": [{"kpi": primary_kpi, "chart": chart_type.value}]},
        styling={"theme": "minimal"},
        # Template-derived default UI affordance (not source data, not LLM); keeps
        # the DesignOutput complete for the project's schema-completeness contract.
        interactions=[{"type": "tooltip", "fields": [x_name, y_name]}],
        rationales=[
            Rationale(
                claim=f"{chart_type.value} chart selected from the nvBench source label '{label}'.",
                principle="source-provided visualization label",
            )
        ],
    )

    return GoldItem(
        item_id=source_record_id(key, query_index),
        brief=brief,
        recommendation=recommendation,
        split=group_split(key),
    )


# --------------------------------------------------------------------------- #
# accessors + deterministic selection
# --------------------------------------------------------------------------- #
def item_group_id(item: GoldItem) -> str:
    return item.brief.extra["provenance"]["source_group_id"]


def item_chart(item: GoldItem) -> str:
    return item.recommendation.kpi_chart_mapping[0].chart_type.value


def select_one_per_group(items: List[GoldItem], seed: int = 42) -> List[GoldItem]:
    """Keep exactly one query per visualization group, deterministically.

    Within each group the query with the lowest ``(hash(seed,id))`` is kept, so
    the choice is stable for a given seed and independent of input order.
    """
    import hashlib

    best: Dict[str, Tuple[str, GoldItem]] = {}
    for it in items:
        gid = item_group_id(it)
        h = hashlib.md5(f"{seed}:{it.item_id}".encode("utf-8")).hexdigest()
        if gid not in best or h < best[gid][0]:
            best[gid] = (h, it)
    kept = [v[1] for v in best.values()]
    kept.sort(key=lambda it: it.item_id)
    return kept


def apply_limit(
    items: List[GoldItem],
    limit: Optional[int],
    stratify_by_chart: bool = False,
    seed: int = 42,
) -> List[GoldItem]:
    """Deterministically cap ``items`` to ``limit``.

    With ``stratify_by_chart`` the cap is distributed round-robin across chart
    types so no single chart dominates a small sample.
    """
    ordered = sorted(items, key=lambda it: it.item_id)
    if limit is None or limit >= len(ordered):
        return ordered
    if not stratify_by_chart:
        return ordered[:limit]

    buckets: Dict[str, List[GoldItem]] = {}
    for it in ordered:
        buckets.setdefault(item_chart(it), []).append(it)
    result: List[GoldItem] = []
    chart_keys = sorted(buckets)
    idx = 0
    while len(result) < limit and any(buckets.values()):
        chart = chart_keys[idx % len(chart_keys)]
        if buckets[chart]:
            result.append(buckets[chart].pop(0))
        idx += 1
        if idx > len(ordered) * 2:  # safety
            break
    result.sort(key=lambda it: it.item_id)
    return result[:limit]
