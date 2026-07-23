"""Tests for the deterministic DB field profiler (src/data_pipeline/nvbench_profile.py).

Uses tiny, hand-built SQLite databases in tmp_path -- never the real nvBench cache.
"""

import sqlite3

from src.data_pipeline.nvbench_profile import DbProfiler
from src.data_pipeline.nvbench_source import DbMetadataResolver


def _make_db(tmp_path, db_id: str, ddl: str, rows_sql: list) -> None:
    db_dir = tmp_path / db_id
    db_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_dir / f"{db_id}.sqlite")
    try:
        con.executescript(ddl)
        for stmt in rows_sql:
            con.execute(stmt)
        con.commit()
    finally:
        con.close()


def _profiler(tmp_path) -> DbProfiler:
    resolver = DbMetadataResolver(tmp_path)
    return DbProfiler(resolver)


def _employee_db(tmp_path, db_id="db1"):
    _make_db(
        tmp_path, db_id,
        "CREATE TABLE employee (Employee_ID INTEGER PRIMARY KEY, salary REAL, dept TEXT);",
        [
            "INSERT INTO employee VALUES (1, 50000, 'eng')",
            "INSERT INTO employee VALUES (2, 60000, 'eng')",
            "INSERT INTO employee VALUES (3, 55000, 'sales')",
            "INSERT INTO employee VALUES (4, 70000, 'sales')",
        ],
    )


def test_primary_key_detected(tmp_path):
    _employee_db(tmp_path)
    p = _profiler(tmp_path).profile_field("db1", "Employee_ID")
    assert p["is_primary_key"] is True
    assert p["stats_available"] is True
    assert p["distinct_count"] == 4


def test_unique_index_detected(tmp_path):
    _make_db(
        tmp_path, "db2",
        'CREATE TABLE t (id INTEGER PRIMARY KEY, code TEXT); '
        'CREATE UNIQUE INDEX idx_code ON t(code);',
        ["INSERT INTO t VALUES (1, 'A')", "INSERT INTO t VALUES (2, 'B')"],
    )
    p = _profiler(tmp_path).profile_field("db2", "code")
    assert p["is_unique_index"] is True
    assert p["is_primary_key"] is False


def test_distinct_null_unique_ratio(tmp_path):
    _make_db(
        tmp_path, "db3",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, cat TEXT);",
        ["INSERT INTO t VALUES (1, 'a')", "INSERT INTO t VALUES (2, 'a')",
         "INSERT INTO t VALUES (3, NULL)", "INSERT INTO t VALUES (4, 'b')"],
    )
    p = _profiler(tmp_path).profile_field("db3", "cat")
    assert p["row_count"] == 4
    assert p["non_null_count"] == 3
    assert p["distinct_count"] == 2  # 'a', 'b'
    assert p["null_ratio"] == 0.25
    assert p["unique_ratio"] == 0.5  # distinct/row_count


def test_numeric_measure_not_flagged_as_stats_missing(tmp_path):
    _employee_db(tmp_path)
    p = _profiler(tmp_path).profile_field("db1", "salary")
    assert p["stats_available"] is True
    assert p["normalized_dtype"] == "number"
    assert p["min_value"] == 50000
    assert p["max_value"] == 70000
    assert p["n_negative"] == 0


def test_deterministic_repeat_profiling(tmp_path):
    _employee_db(tmp_path)
    profiler = _profiler(tmp_path)
    p1 = profiler.profile_field("db1", "salary")
    p2 = profiler.profile_field("db1", "salary")
    assert p1 == p2


def test_missing_db_stats_unavailable(tmp_path):
    p = _profiler(tmp_path).profile_field("missing_db", "anything")
    assert p["stats_available"] is False
    assert p["resolution"] == "field_not_found"


def test_missing_field_in_existing_db(tmp_path):
    _employee_db(tmp_path)
    p = _profiler(tmp_path).profile_field("db1", "nonexistent_column")
    assert p["stats_available"] is False
    assert p["resolution"] == "field_not_found"


def test_ambiguous_table_without_sql_context(tmp_path):
    _make_db(
        tmp_path, "db4",
        "CREATE TABLE a (Name TEXT, id INTEGER PRIMARY KEY); "
        "CREATE TABLE b (Name TEXT, id2 INTEGER PRIMARY KEY);",
        ["INSERT INTO a VALUES ('x', 1)", "INSERT INTO b VALUES ('y', 1)"],
    )
    p = _profiler(tmp_path).profile_field("db4", "Name")
    assert p["resolution"] == "ambiguous_table"
    assert p["stats_available"] is False
    assert p["table"] is None


def test_ambiguous_table_resolved_via_sql_alias(tmp_path):
    _make_db(
        tmp_path, "db4",
        "CREATE TABLE a (Name TEXT, id INTEGER PRIMARY KEY); "
        "CREATE TABLE b (Name TEXT, id2 INTEGER PRIMARY KEY);",
        ["INSERT INTO a VALUES ('x', 1)", "INSERT INTO b VALUES ('y', 1)"],
    )
    sql = "SELECT T1.Name FROM a AS T1 JOIN b AS T2 ON T1.id = T2.id2 WHERE T1.Name = 'x'"
    p = _profiler(tmp_path).profile_field("db4", "Name", sql_context=sql)
    assert p["resolution"] == "resolved_via_sql_alias"
    assert p["table"] == "a"
    assert p["stats_available"] is True


def test_ambiguous_table_resolved_via_join_context(tmp_path):
    _make_db(
        tmp_path, "db4",
        "CREATE TABLE a (Name TEXT, id INTEGER PRIMARY KEY); "
        "CREATE TABLE b (Name TEXT, id2 INTEGER PRIMARY KEY);",
        ["INSERT INTO a VALUES ('x', 1)", "INSERT INTO b VALUES ('y', 1)"],
    )
    sql = "SELECT Name FROM a WHERE Name = 'x'"
    p = _profiler(tmp_path).profile_field("db4", "Name", sql_context=sql)
    assert p["resolution"] == "resolved_via_join_context"
    assert p["table"] == "a"


def test_qualified_field_name_resolves_directly(tmp_path):
    _make_db(
        tmp_path, "db4",
        "CREATE TABLE a (Name TEXT, id INTEGER PRIMARY KEY); "
        "CREATE TABLE b (Name TEXT, id2 INTEGER PRIMARY KEY);",
        ["INSERT INTO a VALUES ('x', 1)", "INSERT INTO b VALUES ('y', 1)"],
    )
    sql = "SELECT T1.Name FROM a AS T1 JOIN b AS T2 ON T1.id = T2.id2"
    p = _profiler(tmp_path).profile_field("db4", "T1.Name", sql_context=sql)
    assert p["resolution"] == "qualified_in_field_name"
    assert p["table"] == "a"


def test_save_and_load_disk_cache(tmp_path):
    _employee_db(tmp_path)
    cache_path = tmp_path / "field_profiles.json"
    profiler = DbProfiler(DbMetadataResolver(tmp_path), cache_path=cache_path)
    p1 = profiler.profile_field("db1", "salary")
    profiler.save_cache()
    assert cache_path.exists()

    profiler2 = DbProfiler(DbMetadataResolver(tmp_path), cache_path=cache_path)
    p2 = profiler2.profile_field("db1", "salary")
    assert p1["distinct_count"] == p2["distinct_count"]
