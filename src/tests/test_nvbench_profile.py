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


def test_all_numeric_text_column_is_profiled_as_number(tmp_path):
    _make_db(
        tmp_path, "numeric_text",
        "CREATE TABLE dogs (name TEXT, age VARCHAR(20));",
        ["INSERT INTO dogs VALUES ('A', '2')", "INSERT INTO dogs VALUES ('B', '3.5')"],
    )
    p = _profiler(tmp_path).profile_field(
        "numeric_text", "age", sql_context="SELECT name, age FROM dogs"
    )
    assert p["declared_dtype"] == "VARCHAR(20)"
    assert p["normalized_dtype"] == "number"
    assert p["numeric_value_ratio"] == 1.0
    assert p["min_value"] == 2.0


def test_mixed_text_column_remains_categorical(tmp_path):
    _make_db(
        tmp_path, "mixed_text",
        "CREATE TABLE t (value TEXT);",
        ["INSERT INTO t VALUES ('2')", "INSERT INTO t VALUES ('unknown')"],
    )
    p = _profiler(tmp_path).profile_field(
        "mixed_text", "value", sql_context="SELECT value FROM t"
    )
    assert p["normalized_dtype"] == "categorical"
    assert p["numeric_value_ratio"] == 0.5


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


def test_query_schema_fields_excludes_unreferenced_tables(tmp_path):
    _make_db(
        tmp_path, "db5",
        "CREATE TABLE student (Sex TEXT, score REAL); "
        "CREATE TABLE audit (gender TEXT, event TEXT);",
        ["INSERT INTO student VALUES ('F', 90)", "INSERT INTO audit VALUES ('F', 'created')"],
    )
    fields = _profiler(tmp_path).query_schema_fields(
        "db5", sql_context="SELECT Sex, AVG(score) FROM student GROUP BY Sex"
    )
    assert fields == ["score", "Sex"]
    assert "gender" not in fields
    assert _profiler(tmp_path).query_schema_tables(
        "db5", sql_context="SELECT Sex FROM student"
    ) == ["student"]


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


def test_query_result_profile_uses_filtered_grouped_result_rows(tmp_path):
    _employee_db(tmp_path)
    profile = _profiler(tmp_path).profile_query_result(
        "db1",
        "SELECT dept, SUM(salary) FROM employee GROUP BY dept ORDER BY dept",
    )
    assert profile["execution_status"] == "ok"
    assert profile["observed_row_count"] == 2
    assert profile["at_least_two_rows"] is True
    assert profile["columns"][1]["numeric_value_count"] == 2
    assert profile["columns"][1]["distinct_count"] == 2


def test_query_result_profile_detects_one_surviving_observation_and_sql_error(tmp_path):
    _employee_db(tmp_path)
    profiler = _profiler(tmp_path)
    one = profiler.profile_query_result(
        "db1", "SELECT dept, SUM(salary) FROM employee WHERE dept = 'eng' GROUP BY dept"
    )
    assert one["execution_status"] == "ok"
    assert one["observed_row_count"] == 1
    assert one["at_least_two_rows"] is False
    invalid = profiler.profile_query_result(
        "db1", "SELECT MAX(salary), COUNT(*) FROM employee GROUP BY MAX(salary)"
    )
    assert invalid["execution_status"] == "error"
    assert invalid["at_least_two_rows"] is False


def test_query_result_profile_records_complete_numeric_pairs(tmp_path):
    _employee_db(tmp_path)
    profiler = _profiler(tmp_path)
    disjoint = profiler.profile_query_result(
        "db1",
        "SELECT 1 AS x, NULL AS y UNION ALL SELECT 2, NULL "
        "UNION ALL SELECT NULL, 10 UNION ALL SELECT NULL, 20",
    )
    assert disjoint["columns"][0]["numeric_value_count"] == 2
    assert disjoint["columns"][1]["numeric_value_count"] == 2
    assert disjoint["paired_numeric_row_count"] == 0
    assert disjoint["paired_numeric_distinct_pair_count"] == 0

    complete = profiler.profile_query_result(
        "db1", "SELECT 1 AS x, 10 AS y UNION ALL SELECT 2, 20"
    )
    assert complete["paired_numeric_row_count"] == 2
    assert complete["paired_numeric_distinct_x_count"] == 2
    assert complete["paired_numeric_distinct_y_count"] == 2
    assert complete["paired_numeric_distinct_pair_count"] == 2
