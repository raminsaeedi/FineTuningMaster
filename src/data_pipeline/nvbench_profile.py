"""Deterministic database field profiler for the nvBench quality layer.

Profiles one physical column at a time from the cached, read-only SQLite
databases: declared/normalized dtype, primary-key and unique-index status,
row/null/distinct counts, numeric range and sign counts, and a few deterministic
example values. Used by ``nvbench_identifier.detect_identifier`` and
``nvbench_quality`` to tell a real numeric measure apart from an identifier
that merely happens to be declared ``INTEGER``.

Table-ambiguity policy: when a field name matches columns in more than one
table of the same database, the profiler does **not** guess (e.g. by picking
the first table in schema order). It only resolves the table when the caller
supplies the record's own source SQL (``sql_context``) and that SQL either
qualifies the field with a table alias that maps to exactly one of the
candidate tables, or only joins one of the candidate tables at all. Otherwise
the profile is returned with ``resolution="ambiguous_table"`` and
``stats_available=False`` -- callers must treat this as a hard signal, never a
silently-passed guess (see ``nvbench_quality.py``).
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data_pipeline.nvbench_source import DbMetadataResolver, normalize_sqlite_type

PROFILE_RULE_VERSION = "nvbench_profile_v1"

_FROM_JOIN_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+[`\"\[]?(\w+)[`\"\]]?(?:\s+(?:AS\s+)?[`\"\[]?(\w+)[`\"\]]?)?",
    re.IGNORECASE,
)


def _table_alias_map(sql: str) -> Dict[str, str]:
    """``{alias_lower: table_name}`` from FROM/JOIN clauses; identity for bare tables."""
    alias_map: Dict[str, str] = {}
    for m in _FROM_JOIN_RE.finditer(sql or ""):
        table, alias = m.group(1), m.group(2)
        if not table:
            continue
        alias_map[table.lower()] = table
        if alias:
            alias_map[alias.lower()] = table
    return alias_map


def _joined_tables(sql: str) -> List[str]:
    return sorted({m.group(1) for m in _FROM_JOIN_RE.finditer(sql or "") if m.group(1)})


def _qualified_ref_present(sql: str, alias: str, field: str) -> bool:
    pat = re.compile(rf"\b{re.escape(alias)}\s*\.\s*{re.escape(field)}\b", re.IGNORECASE)
    return bool(pat.search(sql or ""))


def _resolve_table(
    field_name: str, candidate_tables: List[str], sql_context: str
) -> Tuple[Optional[str], str, List[str]]:
    """Deterministically resolve which table a field belongs to.

    Returns ``(table_or_None, resolution, notes)``. Never guesses: only
    resolves via an explicit qualified reference or an unambiguous join
    context; otherwise returns ``(None, "ambiguous_table", notes)``.
    """
    if len(candidate_tables) == 1:
        return candidate_tables[0], "unique_table_match", []
    if len(candidate_tables) == 0:
        return None, "field_not_found", [f"field {field_name!r} not found in any table"]

    notes = [f"field {field_name!r} present in multiple tables: {candidate_tables}"]
    if sql_context:
        alias_map = _table_alias_map(sql_context)
        matches = {
            table
            for alias, table in alias_map.items()
            if table in candidate_tables and _qualified_ref_present(sql_context, alias, field_name)
        }
        if len(matches) == 1:
            return next(iter(matches)), "resolved_via_sql_alias", notes

        joined = set(_joined_tables(sql_context))
        joined_candidates = [t for t in candidate_tables if t in joined]
        if len(joined_candidates) == 1:
            return joined_candidates[0], "resolved_via_join_context", notes

    return None, "ambiguous_table", notes


class DbProfiler:
    """Deterministic, cached field profiler over the nvBench SQLite cache."""

    def __init__(self, resolver: DbMetadataResolver, cache_path: Optional[str | Path] = None) -> None:
        self.resolver = resolver
        self.cache_path = Path(cache_path) if cache_path else None
        self._schema_cache: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        self._unique_index_cache: Dict[str, Dict[Tuple[str, str], bool]] = {}
        self._stats_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._loaded_from_disk = False
        if self.cache_path and self.cache_path.exists():
            self._load_disk_cache()

    # -- disk cache (stats only; schema/resolution are cheap and always live) -- #
    def _load_disk_cache(self) -> None:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if payload.get("rule_version") != PROFILE_RULE_VERSION:
            return  # stale cache from a different rule version; ignore, rebuild
        for key_str, stats in (payload.get("stats") or {}).items():
            db_id, table, col = key_str.split("\x1f")
            self._stats_cache[(db_id, table, col)] = stats
        self._loaded_from_disk = True

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        payload = {
            "rule_version": PROFILE_RULE_VERSION,
            "stats": {
                f"{db_id}\x1f{table}\x1f{col}": stats
                for (db_id, table, col), stats in self._stats_cache.items()
            },
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- schema (cheap: PRAGMA table_info / index_list, no full-table scans) -- #
    def _schema(self, db_id: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """``{table: {col_lower: {"declared_type":.., "is_pk":bool}}}``."""
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]
        schema: Dict[str, Dict[str, Dict[str, Any]]] = {}
        unique_idx: Dict[Tuple[str, str], bool] = {}
        for path in self.resolver._db_candidates(db_id):  # reuse path resolution
            con = self._connect(path)
            if con is None:
                continue
            try:
                tables = [
                    r[0] for r in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                ]
                any_table = False
                for table in tables:
                    cols: Dict[str, Dict[str, Any]] = {}
                    for row in con.execute(f'PRAGMA table_info("{table}")').fetchall():
                        col_name, decl_type, pk_flag = str(row[1]), str(row[2]), int(row[5] or 0)
                        cols[col_name.lower()] = {
                            "name": col_name, "declared_type": decl_type, "is_pk": pk_flag > 0,
                        }
                    if not cols:
                        continue
                    any_table = True
                    schema[table] = cols
                    for idx_row in con.execute(f'PRAGMA index_list("{table}")').fetchall():
                        idx_name, idx_unique = str(idx_row[1]), int(idx_row[2])
                        if not idx_unique:
                            continue
                        idx_cols = con.execute(f'PRAGMA index_info("{idx_name}")').fetchall()
                        if len(idx_cols) == 1:
                            unique_idx[(table, str(idx_cols[0][2]).lower())] = True
                if any_table:
                    break  # first candidate path with real tables wins (same rule as DbMetadataResolver)
            except sqlite3.Error:
                continue
            finally:
                con.close()
        self._schema_cache[db_id] = schema
        self._unique_index_cache[db_id] = unique_idx
        return schema

    @staticmethod
    def _connect(path: Path) -> Optional[sqlite3.Connection]:
        try:
            return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        except (sqlite3.Error, ValueError, OSError):
            return None

    def _db_path(self, db_id: str) -> Optional[Path]:
        candidates = self.resolver._db_candidates(db_id)
        return candidates[0] if candidates else None

    # -- expensive stats (full-table aggregate queries; cached) -- #
    def _compute_stats(self, db_id: str, table: str, col_name: str, declared_type: str) -> Dict[str, Any]:
        normalized = normalize_sqlite_type(declared_type)
        path = self._db_path(db_id)
        con = self._connect(path) if path else None
        if con is None:
            return {"stats_available": False, "notes": ["database unavailable for stats query"]}
        qtable, qcol = f'"{table}"', f'"{col_name}"'
        try:
            row_count = con.execute(f"SELECT COUNT(*) FROM {qtable}").fetchone()[0]
            non_null_count = con.execute(f"SELECT COUNT({qcol}) FROM {qtable}").fetchone()[0]
            distinct_count = con.execute(f"SELECT COUNT(DISTINCT {qcol}) FROM {qtable}").fetchone()[0]
            null_ratio = (1 - non_null_count / row_count) if row_count else None
            unique_ratio = (distinct_count / row_count) if row_count else None
            min_value = max_value = None
            n_positive = n_zero = n_negative = None
            if normalized == "number":
                min_value, max_value = con.execute(f"SELECT MIN({qcol}), MAX({qcol}) FROM {qtable}").fetchone()
                n_positive, n_zero, n_negative = con.execute(
                    f"SELECT SUM(CASE WHEN {qcol} > 0 THEN 1 ELSE 0 END), "
                    f"SUM(CASE WHEN {qcol} = 0 THEN 1 ELSE 0 END), "
                    f"SUM(CASE WHEN {qcol} < 0 THEN 1 ELSE 0 END) FROM {qtable}"
                ).fetchone()
            example_values: List[Any] = []
            try:
                rows = con.execute(
                    f"SELECT DISTINCT {qcol} FROM {qtable} WHERE {qcol} IS NOT NULL "
                    f"ORDER BY {qcol} LIMIT 3"
                ).fetchall()
                example_values = [r[0] for r in rows]
            except sqlite3.Error:
                example_values = []
            return {
                "stats_available": True,
                "row_count": row_count,
                "non_null_count": non_null_count,
                "null_ratio": null_ratio,
                "distinct_count": distinct_count,
                "unique_ratio": unique_ratio,
                "min_value": min_value,
                "max_value": max_value,
                "n_positive": n_positive,
                "n_zero": n_zero,
                "n_negative": n_negative,
                "example_values": example_values,
                "notes": [],
            }
        except sqlite3.Error as exc:
            return {"stats_available": False, "notes": [f"stats query failed: {exc}"]}
        finally:
            con.close()

    # -- public API -- #
    def profile_field(self, db_id: str, field_name: str, *, sql_context: str = "") -> Dict[str, Any]:
        """Return a ``FieldProfile`` dict for one raw field (see module docstring)."""
        raw_field = (field_name or "").strip()
        qualifier_alias: Optional[str] = None
        if "." in raw_field:
            qualifier_alias, raw_field = raw_field.rsplit(".", 1)

        schema = self._schema(db_id)
        candidate_tables = [t for t, cols in schema.items() if raw_field.lower() in cols]

        table: Optional[str]
        resolution: str
        notes: List[str]
        if qualifier_alias and sql_context:
            alias_map = _table_alias_map(sql_context)
            aliased_table = alias_map.get(qualifier_alias.lower())
            if aliased_table and aliased_table in candidate_tables:
                table, resolution, notes = aliased_table, "qualified_in_field_name", []
            else:
                table, resolution, notes = _resolve_table(raw_field, candidate_tables, sql_context)
        else:
            table, resolution, notes = _resolve_table(raw_field, candidate_tables, sql_context)

        if table is None:
            return {
                "db_id": db_id, "table": None, "field_name": raw_field,
                "declared_dtype": None, "normalized_dtype": None,
                "is_primary_key": False, "is_unique_index": False,
                "stats_available": False, "resolution": resolution,
                "rule_version": PROFILE_RULE_VERSION, "notes": notes,
            }

        col_meta = schema[table][raw_field.lower()]
        is_unique_index = self._unique_index_cache.get(db_id, {}).get((table, raw_field.lower()), False)

        cache_key = (db_id, table, raw_field.lower())
        if cache_key not in self._stats_cache:
            self._stats_cache[cache_key] = self._compute_stats(
                db_id, table, col_meta["name"], col_meta["declared_type"]
            )
        stats = self._stats_cache[cache_key]

        profile = {
            "db_id": db_id,
            "table": table,
            "field_name": col_meta["name"],
            "declared_dtype": col_meta["declared_type"],
            "normalized_dtype": normalize_sqlite_type(col_meta["declared_type"]),
            "is_primary_key": bool(col_meta["is_pk"]),
            "is_unique_index": bool(is_unique_index),
            "resolution": resolution,
            "rule_version": PROFILE_RULE_VERSION,
            "notes": notes + list(stats.get("notes", [])),
        }
        profile.update({k: v for k, v in stats.items() if k != "notes"})
        profile.setdefault("stats_available", False)
        return profile
