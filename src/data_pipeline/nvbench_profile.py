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

from src.data_pipeline.nvbench_source import (
    DbMetadataResolver,
    is_finite_number,
    normalize_sqlite_type,
)

PROFILE_RULE_VERSION = "nvbench_profile_v3"

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
        self._query_result_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
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
        """Return live schema evidence, including key roles, for each column."""
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
                            "name": col_name,
                            "declared_type": decl_type,
                            "is_pk": pk_flag > 0,
                            "is_fk": False,
                        }
                    if not cols:
                        continue
                    any_table = True
                    schema[table] = cols
                    for fk_row in con.execute(f'PRAGMA foreign_key_list("{table}")').fetchall():
                        # SQLite foreign_key_list: source column is index 3.
                        source_column = str(fk_row[3]).lower()
                        if source_column in cols:
                            cols[source_column]["is_fk"] = True
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
            connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            connection.create_function("nvbench_is_finite_number", 1, is_finite_number)
            return connection
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
            non_null_count, numeric_value_count = con.execute(
                f"SELECT COUNT({qcol}), COALESCE(SUM(nvbench_is_finite_number({qcol})), 0) "
                f"FROM {qtable}"
            ).fetchone()
            distinct_count = con.execute(f"SELECT COUNT(DISTINCT {qcol}) FROM {qtable}").fetchone()[0]
            null_ratio = (1 - non_null_count / row_count) if row_count else None
            unique_ratio = (distinct_count / row_count) if row_count else None
            numeric_value_ratio = numeric_value_count / non_null_count if non_null_count else None
            effective_normalized = (
                "number" if normalized == "number" or numeric_value_ratio == 1.0 else normalized
            )
            min_value = max_value = None
            n_positive = n_zero = n_negative = None
            if effective_normalized == "number":
                min_value, max_value = con.execute(
                    f"SELECT MIN(CAST({qcol} AS REAL)), MAX(CAST({qcol} AS REAL)) FROM {qtable}"
                ).fetchone()
                n_positive, n_zero, n_negative = con.execute(
                    f"SELECT SUM(CASE WHEN CAST({qcol} AS REAL) > 0 THEN 1 ELSE 0 END), "
                    f"SUM(CASE WHEN CAST({qcol} AS REAL) = 0 THEN 1 ELSE 0 END), "
                    f"SUM(CASE WHEN CAST({qcol} AS REAL) < 0 THEN 1 ELSE 0 END) FROM {qtable}"
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
                "numeric_value_count": numeric_value_count,
                "numeric_value_ratio": numeric_value_ratio,
                "normalized_dtype": effective_normalized,
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
    def query_schema_tables(self, db_id: str, *, sql_context: str = "") -> List[str]:
        """Return canonical schema table names referenced by the source SQL."""
        schema = self._schema(db_id)
        table_names = {table.lower(): table for table in schema}
        return sorted({
            table_names[table.lower()]
            for table in _joined_tables(sql_context)
            if table.lower() in table_names
        }, key=str.lower)

    def query_schema_fields(self, db_id: str, *, sql_context: str = "") -> List[str]:
        """Return fields from tables referenced by ``sql_context``.

        This is intentionally stricter than ``resolver.columns(db_id)``, which
        merges columns from every table in a database.  Source-consistency
        checks must not treat a same-named field in an unrelated table as
        evidence that the query requested a missing dimension.
        """
        schema = self._schema(db_id)
        if not schema:
            return []

        referenced = set(self.query_schema_tables(db_id, sql_context=sql_context))
        if not referenced:
            return []
        return sorted({
            column["name"]
            for table in referenced
            for column in schema[table].values()
        }, key=str.lower)

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
                "is_primary_key": False, "is_foreign_key": False, "is_unique_index": False,
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
            "is_foreign_key": bool(col_meta.get("is_fk")),
            "is_unique_index": bool(is_unique_index),
            "resolution": resolution,
            "rule_version": PROFILE_RULE_VERSION,
            "notes": notes + list(stats.get("notes", [])),
        }
        profile.update({k: v for k, v in stats.items() if k != "notes"})
        profile.setdefault("stats_available", False)
        return profile

    def profile_query_result(
        self, db_id: str, sql: str, *, max_rows: int = 1000
    ) -> Dict[str, Any]:
        """Profile the actual read-only SQL result, capped deterministically.

        Scatter validity depends on rows surviving the query's joins, filters,
        grouping, HAVING, ordering, and LIMIT—not on global base-table
        cardinality. This method executes the supplied source query as a
        read-only subquery, fetches at most ``max_rows`` rows, and records only
        aggregate result statistics. Raw values are never persisted.
        """
        clean_sql = str(sql or "").strip().rstrip(";").strip()
        cap = max(2, int(max_rows))
        cache_key = (db_id, clean_sql, cap)
        if cache_key in self._query_result_cache:
            return dict(self._query_result_cache[cache_key])

        result: Dict[str, Any]
        path = self._db_path(db_id)
        con = self._connect(path) if path else None
        if con is None or not clean_sql:
            result = {
                "execution_status": "unavailable",
                "execution_error": "database or SQL unavailable",
                "observed_row_count": 0,
                "row_count_capped": False,
                "at_least_two_rows": False,
                "paired_numeric_row_count": 0,
                "paired_numeric_distinct_x_count": 0,
                "paired_numeric_distinct_y_count": 0,
                "paired_numeric_distinct_pair_count": 0,
                "columns": [],
            }
            if con is not None:
                con.close()
        else:
            progress_calls = 0

            def progress_guard() -> int:
                nonlocal progress_calls
                progress_calls += 1
                # About 50 million SQLite virtual-machine instructions. This
                # prevents pathological source queries from blocking a rebuild.
                return 1 if progress_calls > 5000 else 0

            try:
                con.execute("PRAGMA query_only = ON")
                con.set_progress_handler(progress_guard, 10_000)
                wrapped = f'SELECT * FROM ({clean_sql}) AS "_nvbench_result" LIMIT {cap}'
                cursor = con.execute(wrapped)
                rows = cursor.fetchall()
                column_count = len(cursor.description or [])
                columns: List[Dict[str, Any]] = []
                for index in range(column_count):
                    values = [row[index] for row in rows if row[index] is not None]
                    numeric_values = [value for value in values if is_finite_number(value)]
                    distinct_values = {
                        (type(value).__name__, repr(value)) for value in values
                    }
                    columns.append({
                        "index": index,
                        "non_null_count": len(values),
                        "numeric_value_count": len(numeric_values),
                        "numeric_value_ratio": (
                            len(numeric_values) / len(values) if values else None
                        ),
                        "distinct_count": len(distinct_values),
                    })
                paired_numeric_rows = [
                    (row[0], row[1])
                    for row in rows
                    if len(row) >= 2
                    and is_finite_number(row[0])
                    and is_finite_number(row[1])
                ]
                paired_x = {
                    (type(x).__name__, repr(x)) for x, _y in paired_numeric_rows
                }
                paired_y = {
                    (type(y).__name__, repr(y)) for _x, y in paired_numeric_rows
                }
                paired_xy = {
                    ((type(x).__name__, repr(x)), (type(y).__name__, repr(y)))
                    for x, y in paired_numeric_rows
                }
                result = {
                    "execution_status": "ok",
                    "execution_error": None,
                    "observed_row_count": len(rows),
                    "row_count_capped": len(rows) == cap,
                    "at_least_two_rows": len(rows) >= 2,
                    "paired_numeric_row_count": len(paired_numeric_rows),
                    "paired_numeric_distinct_x_count": len(paired_x),
                    "paired_numeric_distinct_y_count": len(paired_y),
                    "paired_numeric_distinct_pair_count": len(paired_xy),
                    "columns": columns,
                }
            except sqlite3.Error as exc:
                result = {
                    "execution_status": "error",
                    "execution_error": str(exc),
                    "observed_row_count": 0,
                    "row_count_capped": False,
                    "at_least_two_rows": False,
                    "paired_numeric_row_count": 0,
                    "paired_numeric_distinct_x_count": 0,
                    "paired_numeric_distinct_y_count": 0,
                    "paired_numeric_distinct_pair_count": 0,
                    "columns": [],
                }
            finally:
                con.set_progress_handler(None, 0)
                con.close()

        self._query_result_cache[cache_key] = dict(result)
        return dict(result)
