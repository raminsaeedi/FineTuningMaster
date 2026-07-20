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
from src.data_pipeline.splits import assign_split
from src.utils.io import read_yaml

_AGG_RE = re.compile(r"^\s*(\w+)\s*\(", re.IGNORECASE)
_DATETIME_HINT_RE = re.compile(r"date|time|year|month|day|week|quarter", re.IGNORECASE)


class RejectedRecord(Exception):
    """Raised when a source record cannot be mapped (e.g. unsupported chart)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


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
            con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        except sqlite3.Error:
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
                    result.setdefault(col_name, normalize_sqlite_type(str(row[2])))
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


def _dimension_dtype(
    db_id: str, name: str, resolver: DbMetadataResolver
) -> Tuple[str, str]:
    """Resolve an x/dimension column dtype. Returns ``(dtype, source)``."""
    dtype = resolver.dtype_of(db_id, name)
    if dtype is not None:
        return dtype, "source-provided(db)"
    heuristic = "datetime" if _DATETIME_HINT_RE.search(name or "") else "categorical"
    return heuristic, "heuristic"


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

    Raises ``RejectedRecord`` if the chart label is unsupported.
    """
    label = record.get("chart", "")
    chart = map_chart(label, mapping)  # may raise RejectedRecord
    chart_type: ChartType = chart["chart_type"]
    grouped: bool = chart["grouped"]

    vis_obj = record.get("vis_obj", {}) or {}
    vis_query = record.get("vis_query", {}) or {}
    db_id = record.get("db_id", "")
    x_name = str(vis_obj.get("x_name", "")).strip()
    y_name = str(vis_obj.get("y_name", "")).strip()
    classify = list(vis_obj.get("classify", []) or [])
    aggregate = parse_aggregate(y_name)

    x_dtype, x_dtype_source = _dimension_dtype(db_id, x_name, resolver)

    columns: List[Dict[str, str]] = [
        {"name": x_name, "dtype": x_dtype, "role": "dimension"},
        {"name": y_name, "dtype": "number", "role": "measure"},
    ]

    encoding: Dict[str, Any] = {
        "x": x_name,
        "y": y_name,
        "aggregate": aggregate,
        "grouped": grouped,
        # Preserve raw grouping/classify information verbatim.
        "classify": classify,
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
            "x_name": x_name,
            "y_name": y_name,
            "classify": classify,
            "describe": vis_obj.get("describe", ""),
        },
        "nl_query": nl_query,
        "raw_encoding": {"x_name": x_name, "y_name": y_name, "aggregate": aggregate, "classify": classify},
        "grouping": {"is_grouped": grouped, "classify": classify},
    }

    lineage: Dict[str, str] = {
        "chart_type": "source-provided",
        "encoding": "source-provided",
        "goal": "source-provided",
        "kpi": "source-provided",
        "task_type": "rule-derived",
        "layout": "template-derived",
        "styling": "template-derived",
        "interactions": "template-derived",
        "rationales": "template-derived",
        "column_dtype": x_dtype_source,
    }

    brief = DashboardBrief(
        item_id=source_record_id(key, query_index),
        users="Data analyst exploring a relational database",
        goals=[nl_query],
        kpis=[y_name],
        columns=columns,
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
        context_summary={"db_id": db_id, "source": "nvbench", "n_kpis": 1},
        kpi_chart_mapping=[
            KPIChartMapping(
                kpi=y_name,
                task_type=task["task_type"],
                chart_type=chart_type,
                alternatives=[],
                encoding=encoding,
            )
        ],
        layout={"type": "single", "blocks": [{"kpi": y_name, "chart": chart_type.value}]},
        styling={"theme": "minimal"},
        interactions=[],
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
