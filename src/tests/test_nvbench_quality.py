"""Tests for the KPI/chart suitability rule engine (src/data_pipeline/nvbench_quality.py).

Two fixture styles are used:
- ``_build_one``: builds a record through the real ``NvBenchBuilder`` against a
  tiny real SQLite database, so DB-evidence-dependent checks (identifier
  detection, cardinality thresholds, dtype conflicts) are exercised against
  genuine profiler output.
- ``_minimal_record``: a hand-built record dict for pure structural edge cases
  (e.g. a missing group field) that don't need real DB evidence.
"""

import json
import sqlite3
from pathlib import Path

from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder
from src.data_pipeline.nvbench_pilot import _record
from src.data_pipeline.nvbench_profile import DbProfiler
from src.data_pipeline.nvbench_quality import (
    build_quality_pool,
    chart_bar,
    chart_line,
    chart_pie,
    chart_scatter,
    chart_stacked_bar,
    check_kpi_sql_aggregation,
    check_required_constraints,
    kpi_suitability,
    score_and_tier,
)
from src.data_pipeline.nvbench_source import DbMetadataResolver

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAPPING_PATH = _REPO_ROOT / "src" / "config" / "data" / "nvbench_mapping.yaml"

_CFG = {
    "identifier": {
        "strong_unique_ratio": 0.98, "strong_min_distinct": 20, "ambiguous_unique_ratio": 0.5,
        "name_patterns": ["(^|_)id$", "^id(_|$)", "identifier", "(^|_)key$", "(^|_)code$"],
    },
    "chart": {
        "pie": {"max_categories": 8, "additive_aggregates": ["COUNT", "SUM"],
                "prohibited_aggregates": ["AVG", "MIN", "MAX"], "non_additive_reason_code": "pie_non_additive_kpi",
                "allow_negative_values": False, "allow_identifier_category": False},
        "scatter": {"min_distinct_values": 10},
        "stacked_bar": {"max_group_cardinality": 8},
    },
    "scoring": {
        "weights": {"source_fidelity": 30, "kpi_validity": 20, "chart_suitability": 25,
                    "constraint_completeness": 15, "db_profile_support": 10},
        "tier_a_min_score": 90,
    },
    "sampling": {"seed": 42, "target_per_chart": 20, "db_cap": 10, "near_dup_threshold": 0.8},
}


def _entry(chart, nl, db_id, x, y, sql, classify=None):
    return {
        "chart": chart, "db_id": db_id, "hardness": "Easy",
        "vis_query": {"vis_part": f"Visualize {chart.upper()}", "data_part": {"sql_part": sql, "binning": ""}, "VQL": sql},
        "vis_obj": {"chart": chart.lower(), "x_name": x, "y_name": y, "x_data": [[1]], "y_data": [[1]],
                    "classify": classify or [], "describe": ""},
        "nl_queries": nl,
    }


def _build_one(tmp_path, entries, db_setup_sql, db_id="db1", expect_rejections=0):
    nv_path = tmp_path / f"NVBench_{db_id}.json"
    nv_path.write_text(json.dumps(entries), encoding="utf-8")
    cache_root = tmp_path / f"databases_{db_id}"
    db_dir = cache_root / db_id
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / f"{db_id}.sqlite")
    con.executescript(db_setup_sql)
    con.commit()
    con.close()

    builder = NvBenchBuilder(str(nv_path), cache_root=str(cache_root), mapping_path=str(_MAPPING_PATH))
    result = builder.build()
    assert len(result.rejections) == expect_rejections, result.rejections
    assert len(result.accepted) == 1
    record = _record(result.accepted[0])
    resolver = DbMetadataResolver(str(cache_root))
    profiler = DbProfiler(resolver)
    return record, resolver, profiler


_EMPLOYEE_DDL = """
CREATE TABLE employee (Employee_ID INTEGER PRIMARY KEY, salary REAL, dept TEXT, region TEXT, delta REAL);
INSERT INTO employee VALUES (1, 50000, 'eng', 'east', 10);
INSERT INTO employee VALUES (2, 60000, 'eng', 'east', -5);
INSERT INTO employee VALUES (3, 55000, 'sales', 'west', 3);
INSERT INTO employee VALUES (4, 70000, 'sales', 'west', -1);
INSERT INTO employee VALUES (5, 48000, 'hr', 'east', 2);
"""


# --------------------------------------------------------------------------- #
# KPI suitability
# --------------------------------------------------------------------------- #
def test_kpi_valid_count(tmp_path):
    entries = {"1": _entry("Bar", ["how many employees per dept"], "e1", "dept", "count(*)",
                           "SELECT dept , count(*) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="e1")
    result = kpi_suitability(record, profiler, _CFG)
    assert result["suitable"] is True


def test_kpi_valid_sum(tmp_path):
    entries = {"1": _entry("Bar", ["total salary by dept"], "e2", "dept", "SUM(salary)",
                           "SELECT dept , SUM(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="e2")
    result = kpi_suitability(record, profiler, _CFG)
    assert result["suitable"] is True


def test_kpi_valid_average(tmp_path):
    entries = {"1": _entry("Bar", ["average salary by dept"], "e3", "dept", "AVG(salary)",
                           "SELECT dept , AVG(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="e3")
    result = kpi_suitability(record, profiler, _CFG)
    assert result["suitable"] is True


def test_kpi_valid_min_max(tmp_path):
    entries = {"1": _entry("Bar", ["minimal salary by dept"], "e4", "dept", "MIN(salary)",
                           "SELECT dept , MIN(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="e4")
    result = kpi_suitability(record, profiler, _CFG)
    assert result["suitable"] is True


def test_kpi_identifier_aggregation_rejected(tmp_path):
    entries = {"1": _entry("Bar", ["dept chart"], "e5", "dept", "SUM(Employee_ID)",
                           "SELECT dept , SUM(Employee_ID) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="e5")
    result = kpi_suitability(record, profiler, _CFG)
    assert result["suitable"] is False
    assert "meaningless_identifier_aggregation" in result["failed_rules"]


def test_kpi_aggregate_dtype_conflict(tmp_path):
    entries = {"1": _entry("Bar", ["region chart"], "e6", "region", "SUM(dept)",
                           "SELECT region , SUM(dept) FROM employee GROUP BY region")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="e6")
    result = kpi_suitability(record, profiler, _CFG)
    assert result["suitable"] is False
    assert "aggregate_dtype_conflict" in result["failed_rules"]


def test_kpi_bare_identifier_as_continuous_kpi_rejected(tmp_path):
    entries = {"1": _entry("Bar", ["dept chart"], "e7", "dept", "Employee_ID",
                           "SELECT dept , Employee_ID FROM employee GROUP BY dept , region")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="e7")
    result = kpi_suitability(record, profiler, _CFG)
    assert result["suitable"] is False
    assert "identifier_as_continuous_kpi" in result["failed_rules"]


def test_kpi_query_aggregate_mismatch_is_a_warning_signal(tmp_path):
    # "average" in the query but encoded SUM -> broad_intent_mismatch (Tier-B signal,
    # never silently escalated here -- see test_score_and_tier_query_mismatch_is_tier_b).
    entries = {"1": _entry("Bar", ["average salary total by dept"], "e8", "dept", "SUM(salary)",
                           "SELECT dept , SUM(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="e8")
    result = kpi_suitability(record, profiler, _CFG)
    assert "broad_intent_mismatch" in result["failed_rules"]


# --------------------------------------------------------------------------- #
# chart suitability -- bar / line
# --------------------------------------------------------------------------- #
def test_chart_bar_valid(tmp_path):
    entries = {"1": _entry("Bar", ["total salary by dept"], "b1", "dept", "SUM(salary)",
                           "SELECT dept , SUM(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="b1")
    result = chart_bar(record, profiler, _CFG)
    assert result["passed"] is True


def test_chart_bar_identifier_measure_rejected(tmp_path):
    entries = {"1": _entry("Bar", ["dept chart"], "b2", "dept", "Employee_ID",
                           "SELECT dept , Employee_ID FROM employee GROUP BY dept , region")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="b2")
    result = chart_bar(record, profiler, _CFG)
    assert result["passed"] is False
    assert "identifier_as_measure" in result["failed_rules"]


_SALES_DDL = """
CREATE TABLE sales (event_date DATE, amount REAL, dept TEXT);
INSERT INTO sales VALUES ('2020-01-01', 100, 'eng');
INSERT INTO sales VALUES ('2020-02-01', 150, 'eng');
INSERT INTO sales VALUES ('2020-03-01', 120, 'sales');
"""


def test_chart_line_valid_time_series(tmp_path):
    entries = {"1": _entry("Line", ["total amount over time"], "l1", "event_date", "SUM(amount)",
                           "SELECT event_date , SUM(amount) FROM sales GROUP BY event_date")}
    record, _r, profiler = _build_one(tmp_path, entries, _SALES_DDL, db_id="l1")
    result = chart_line(record, profiler, _CFG)
    assert result["passed"] is True


def test_chart_line_unordered_category_rejected(tmp_path):
    entries = {"1": _entry("Line", ["amount by dept"], "l2", "dept", "SUM(amount)",
                           "SELECT dept , SUM(amount) FROM sales GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _SALES_DDL, db_id="l2")
    result = chart_line(record, profiler, _CFG)
    assert result["passed"] is False
    assert "unordered_line_dimension" in result["failed_rules"]


# --------------------------------------------------------------------------- #
# chart suitability -- pie
# --------------------------------------------------------------------------- #
def test_chart_pie_valid_low_cardinality(tmp_path):
    entries = {"1": _entry("Pie", ["salary share by dept"], "p1", "dept", "SUM(salary)",
                           "SELECT dept , SUM(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="p1")
    result = chart_pie(record, profiler, _CFG)
    assert result["passed"] is True


_WIDE_DDL = """
CREATE TABLE t (city TEXT, amount REAL);
INSERT INTO t VALUES ('c1', 10); INSERT INTO t VALUES ('c2', 10); INSERT INTO t VALUES ('c3', 10);
INSERT INTO t VALUES ('c4', 10); INSERT INTO t VALUES ('c5', 10); INSERT INTO t VALUES ('c6', 10);
INSERT INTO t VALUES ('c7', 10); INSERT INTO t VALUES ('c8', 10); INSERT INTO t VALUES ('c9', 10);
"""


def test_chart_pie_high_cardinality_rejected(tmp_path):
    entries = {"1": _entry("Pie", ["amount share by city"], "p2", "city", "SUM(amount)",
                           "SELECT city , SUM(amount) FROM t GROUP BY city")}
    record, _r, profiler = _build_one(tmp_path, entries, _WIDE_DDL, db_id="p2")
    result = chart_pie(record, profiler, _CFG)
    assert result["passed"] is False
    assert "high_cardinality_pie" in result["failed_rules"]


def test_chart_pie_negative_values_rejected(tmp_path):
    entries = {"1": _entry("Pie", ["delta share by dept"], "p3", "dept", "SUM(delta)",
                           "SELECT dept , SUM(delta) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="p3")
    result = chart_pie(record, profiler, _CFG)
    assert result["passed"] is False
    assert "negative_measure_values" in result["failed_rules"]


# --------------------------------------------------------------------------- #
# chart suitability -- pie non-additive KPI (COUNT/SUM only; AVG/MIN/MAX reject)
# --------------------------------------------------------------------------- #
def test_chart_pie_count_passes(tmp_path):
    entries = {"1": _entry("Pie", ["employee count by dept"], "p4", "dept", "count(*)",
                           "SELECT dept , count(*) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="p4")
    result = chart_pie(record, profiler, _CFG)
    assert result["passed"] is True
    assert "pie_non_additive_kpi" not in result["failed_rules"]


def test_chart_pie_sum_passes(tmp_path):
    entries = {"1": _entry("Pie", ["total salary share by dept"], "p5", "dept", "SUM(salary)",
                           "SELECT dept , SUM(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="p5")
    result = chart_pie(record, profiler, _CFG)
    assert result["passed"] is True
    assert "pie_non_additive_kpi" not in result["failed_rules"]


def test_chart_pie_avg_rejected(tmp_path):
    entries = {"1": _entry("Pie", ["average salary share by dept"], "p6", "dept", "AVG(salary)",
                           "SELECT dept , AVG(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="p6")
    result = chart_pie(record, profiler, _CFG)
    assert result["passed"] is False
    assert "pie_non_additive_kpi" in result["failed_rules"]


def test_chart_pie_min_rejected(tmp_path):
    entries = {"1": _entry("Pie", ["minimal salary share by dept"], "p7", "dept", "MIN(salary)",
                           "SELECT dept , MIN(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="p7")
    result = chart_pie(record, profiler, _CFG)
    assert result["passed"] is False
    assert "pie_non_additive_kpi" in result["failed_rules"]


def test_chart_pie_max_rejected(tmp_path):
    entries = {"1": _entry("Pie", ["maximal salary share by dept"], "p8", "dept", "MAX(salary)",
                           "SELECT dept , MAX(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="p8")
    result = chart_pie(record, profiler, _CFG)
    assert result["passed"] is False
    assert "pie_non_additive_kpi" in result["failed_rules"]


def test_chart_pie_identifier_category_rejected(tmp_path):
    entries = {"1": _entry("Pie", ["salary share by employee id"], "p9", "Employee_ID", "SUM(salary)",
                           "SELECT Employee_ID , SUM(salary) FROM employee GROUP BY Employee_ID")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="p9")
    result = chart_pie(record, profiler, _CFG)
    assert result["passed"] is False
    assert "identifier_pie_category" in result["failed_rules"]


def test_chart_pie_reason_code_is_config_driven(tmp_path):
    # The reason code and allowed-aggregate list come from cfg, not a hardcoded
    # constant: overriding them in cfg changes chart_pie's output accordingly.
    entries = {"1": _entry("Pie", ["average salary share by dept"], "p10", "dept", "AVG(salary)",
                           "SELECT dept , AVG(salary) FROM employee GROUP BY dept")}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="p10")
    custom_cfg = json.loads(json.dumps(_CFG))  # deep copy
    custom_cfg["chart"]["pie"]["non_additive_reason_code"] = "custom_pie_reason"
    custom_cfg["chart"]["pie"]["additive_aggregates"] = ["COUNT", "SUM", "AVG"]  # now AVG allowed
    result = chart_pie(record, profiler, custom_cfg)
    assert "pie_non_additive_kpi" not in result["failed_rules"]  # AVG now allowed under custom cfg
    assert result["passed"] is True


def test_pie_non_additive_kpi_overrides_high_score():
    kpi_result = {"suitable": True, "failed_rules": [], "warnings": [], "evidence": {}}
    chart_result = {"passed": False, "failed_rules": ["pie_non_additive_kpi"], "warnings": [], "evidence": {}}
    quality = score_and_tier({}, kpi_result, chart_result, [], _CFG)
    assert quality["tier"] != "A"
    assert "pie_non_additive_kpi" in quality["failed_rules"]


# --------------------------------------------------------------------------- #
# chart suitability -- scatter
# --------------------------------------------------------------------------- #
_SCATTER_DDL = """
CREATE TABLE s (Employee_ID INTEGER PRIMARY KEY, salary REAL, age INTEGER, flag INTEGER);
INSERT INTO s VALUES (1, 50000, 25, 0); INSERT INTO s VALUES (2, 52000, 30, 1);
INSERT INTO s VALUES (3, 54000, 35, 0); INSERT INTO s VALUES (4, 56000, 40, 1);
INSERT INTO s VALUES (5, 58000, 45, 0); INSERT INTO s VALUES (6, 60000, 50, 1);
INSERT INTO s VALUES (7, 62000, 55, 0); INSERT INTO s VALUES (8, 64000, 60, 1);
INSERT INTO s VALUES (9, 66000, 65, 0); INSERT INTO s VALUES (10, 68000, 70, 1);
INSERT INTO s VALUES (11, 70000, 75, 0);
"""


def test_chart_scatter_valid(tmp_path):
    entries = {"1": _entry("Scatter", ["correlation between salary and age"], "sc1", "salary", "age",
                           "SELECT salary , age FROM s")}
    record, _r, profiler = _build_one(tmp_path, entries, _SCATTER_DDL, db_id="sc1")
    result = chart_scatter(record, profiler, _CFG)
    assert result["passed"] is True


def test_chart_scatter_low_variation_rejected(tmp_path):
    entries = {"1": _entry("Scatter", ["correlation between salary and flag"], "sc2", "salary", "flag",
                           "SELECT salary , flag FROM s")}
    record, _r, profiler = _build_one(tmp_path, entries, _SCATTER_DDL, db_id="sc2")
    result = chart_scatter(record, profiler, _CFG)
    assert result["passed"] is False
    assert any(f.startswith("low_variation_axis") for f in result["failed_rules"])


def test_chart_scatter_identifier_axis_rejected(tmp_path):
    entries = {"1": _entry("Scatter", ["correlation between id and salary"], "sc3", "Employee_ID", "salary",
                           "SELECT Employee_ID , salary FROM s")}
    record, _r, profiler = _build_one(tmp_path, entries, _SCATTER_DDL, db_id="sc3")
    result = chart_scatter(record, profiler, _CFG)
    assert result["passed"] is False
    assert any(f.startswith("identifier_scatter_axis") for f in result["failed_rules"])


# --------------------------------------------------------------------------- #
# chart suitability -- stacked bar
# --------------------------------------------------------------------------- #
def test_chart_stacked_bar_valid(tmp_path):
    entries = {"1": _entry("Stacked Bar", ["total salary by dept grouped by region"], "sb1", "dept", "SUM(salary)",
                           "SELECT dept , SUM(salary) FROM employee GROUP BY dept , region",
                           classify=["east", "west"])}
    record, _r, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="sb1")
    result = chart_stacked_bar(record, profiler, _CFG)
    assert result["passed"] is True


_WIDE_GROUP_DDL = """
CREATE TABLE g (dept TEXT, region TEXT, employee_name TEXT, salary REAL);
INSERT INTO g VALUES ('eng','e1','n1',10); INSERT INTO g VALUES ('eng','e2','n2',10);
INSERT INTO g VALUES ('eng','e3','n3',10); INSERT INTO g VALUES ('eng','e4','n4',10);
INSERT INTO g VALUES ('eng','e5','n5',10); INSERT INTO g VALUES ('eng','e6','n6',10);
INSERT INTO g VALUES ('eng','e7','n7',10); INSERT INTO g VALUES ('eng','e8','n8',10);
INSERT INTO g VALUES ('eng','e9','n9',10);
"""


def test_chart_stacked_bar_high_cardinality_group_rejected(tmp_path):
    entries = {"1": _entry("Stacked Bar", ["total salary by dept grouped by name"], "sb2", "dept", "SUM(salary)",
                           "SELECT dept , SUM(salary) FROM g GROUP BY dept , employee_name",
                           classify=["n1", "n2"])}
    record, _r, profiler = _build_one(tmp_path, entries, _WIDE_GROUP_DDL, db_id="sb2")
    result = chart_stacked_bar(record, profiler, _CFG)
    assert result["passed"] is False
    assert "high_cardinality_group" in result["failed_rules"]


def _minimal_record(*, chart_type, db_id="m1", x, y, group_field=None):
    kpi = x["name"] if x.get("role") == "measure" else y["name"]
    return {
        "item_id": "test:1", "split": "train",
        "brief": {
            "item_id": "test:1", "users": "u", "goals": ["query"], "kpis": [kpi],
            "columns": [], "constraints": None,
            "extra": {"provenance": {
                "db_id": db_id, "nl_query": "query", "original_chart_label": chart_type,
                "vis_query": {"data_part": {"sql_part": "SELECT 1"}, "VQL": "SELECT 1"},
                "axis_typing": {"x": x, "y": y},
                "kpi_selection": {"primary_kpi": kpi},
                "constraints": {},
            }},
        },
        "recommendation": {"kpi_chart_mapping": [{
            "kpi": kpi, "chart_type": chart_type,
            "encoding": {"x": x["name"], "y": y["name"], "group_field": group_field},
        }]},
    }


def test_chart_stacked_bar_missing_group_field_rejected(tmp_path):
    x = {"axis": "x", "name": "dept", "aggregate": None, "dtype": "categorical", "role": "dimension"}
    y = {"axis": "y", "name": "salary", "aggregate": "SUM", "dtype": "number", "role": "measure"}
    record = _minimal_record(chart_type="stacked_bar", x=x, y=y, group_field=None)
    profiler = DbProfiler(DbMetadataResolver(None))
    result = chart_stacked_bar(record, profiler, _CFG)
    assert result["passed"] is False
    assert "missing_group_field" in result["failed_rules"]


# --------------------------------------------------------------------------- #
# scoring + tiering
# --------------------------------------------------------------------------- #
def test_score_and_tier_all_pass_is_tier_a(tmp_path):
    entries = {"1": _entry("Bar", ["total salary by dept"], "t1", "dept", "SUM(salary)",
                           "SELECT dept , SUM(salary) FROM employee GROUP BY dept")}
    record, resolver, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="t1")
    kpi_result = kpi_suitability(record, profiler, _CFG)
    chart_result = chart_bar(record, profiler, _CFG)
    quality = score_and_tier(record, kpi_result, chart_result, [], _CFG)
    assert quality["tier"] == "A"
    assert quality["quality_score"] >= 90
    assert quality["failed_rules"] == []


def test_score_and_tier_query_mismatch_is_tier_b_not_c(tmp_path):
    entries = {"1": _entry("Bar", ["average salary total by dept"], "t2", "dept", "SUM(salary)",
                           "SELECT dept , SUM(salary) FROM employee GROUP BY dept")}
    record, resolver, profiler = _build_one(tmp_path, entries, _EMPLOYEE_DDL, db_id="t2")
    kpi_result = kpi_suitability(record, profiler, _CFG)
    chart_result = chart_bar(record, profiler, _CFG)
    quality = score_and_tier(record, kpi_result, chart_result, [], _CFG)
    assert quality["tier"] == "B"


def test_score_and_tier_mandatory_failure_overrides_high_score():
    kpi_result = {"suitable": False, "failed_rules": ["meaningless_identifier_aggregation"], "warnings": [], "evidence": {}}
    chart_result = {"passed": True, "failed_rules": [], "warnings": [], "evidence": {}}
    quality = score_and_tier({}, kpi_result, chart_result, [], _CFG)
    assert quality["tier"] != "A"
    assert quality["failed_rules"]


def test_score_and_tier_severe_combo_is_tier_c():
    kpi_result = {
        "suitable": False,
        "failed_rules": ["meaningless_identifier_aggregation", "broad_intent_mismatch"],
        "warnings": [], "evidence": {},
    }
    chart_result = {"passed": True, "failed_rules": [], "warnings": [], "evidence": {}}
    quality = score_and_tier({}, kpi_result, chart_result, [], _CFG)
    assert quality["tier"] == "C"


def test_score_and_tier_deterministic():
    kpi_result = {"suitable": True, "failed_rules": [], "warnings": [], "evidence": {}}
    chart_result = {"passed": True, "failed_rules": [], "warnings": [], "evidence": {}}
    q1 = score_and_tier({}, kpi_result, chart_result, [], _CFG)
    q2 = score_and_tier({}, kpi_result, chart_result, [], _CFG)
    assert q1 == q2


# --------------------------------------------------------------------------- #
# build_quality_pool (small integration)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# v5: KPI / SQL aggregation agreement
# --------------------------------------------------------------------------- #
_PRICE_DDL = """
CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_price REAL, product_type TEXT);
INSERT INTO products VALUES (1, 10.0, 'A'); INSERT INTO products VALUES (2, 20.0, 'A');
INSERT INTO products VALUES (3, 30.0, 'B'); INSERT INTO products VALUES (4, 40.0, 'B');
INSERT INTO products VALUES (5, 50.0, 'C'); INSERT INTO products VALUES (6, 60.0, 'C');
"""


def test_kpi_sql_agg_sum_vs_avg_conflict(tmp_path):
    # encoded AVG(product_price) but SQL SUM(product_price) for the same field.
    entries = {"1": _entry("Bar", ["price by type"], "a1", "product_type", "AVG(product_price)",
                           "SELECT product_type , SUM(product_price) FROM products GROUP BY product_type")}
    record, _r, profiler = _build_one(tmp_path, entries, _PRICE_DDL, db_id="a1")
    result = check_kpi_sql_aggregation(record)
    assert "kpi_sql_aggregation_conflict" in result["failed_rules"]


def test_kpi_sql_agg_matching_avg_passes(tmp_path):
    entries = {"1": _entry("Bar", ["avg price by type"], "a2", "product_type", "AVG(product_price)",
                           "SELECT product_type , AVG(product_price) FROM products GROUP BY product_type")}
    record, _r, profiler = _build_one(tmp_path, entries, _PRICE_DDL, db_id="a2")
    result = check_kpi_sql_aggregation(record)
    assert "kpi_sql_aggregation_conflict" not in result["failed_rules"]
    assert "mixed_aggregate_ambiguous_kpi" not in result["failed_rules"]


def test_kpi_sql_agg_table_alias_no_false_conflict(tmp_path):
    # SQL uses a table alias on the aggregate arg; must still match the encoding.
    entries = {"1": _entry("Bar", ["avg price"], "a3", "product_type", "AVG(product_price)",
                           "SELECT product_type , AVG(T1.product_price) FROM products AS T1 GROUP BY product_type")}
    record, _r, profiler = _build_one(tmp_path, entries, _PRICE_DDL, db_id="a3")
    result = check_kpi_sql_aggregation(record)
    assert result["failed_rules"] == []


def test_kpi_mixed_aggregate_scatter_flagged(tmp_path):
    # x=max, y=min over the same field: a single KPI can't represent both.
    entries = {"1": _entry("Scatter", ["max vs min price by type"], "a4", "max(product_price)", "min(product_price)",
                           "SELECT max(product_price) , min(product_price) FROM products GROUP BY product_type")}
    record, _r, profiler = _build_one(tmp_path, entries, _PRICE_DDL, db_id="a4")
    result = check_kpi_sql_aggregation(record)
    assert "mixed_aggregate_ambiguous_kpi" in result["failed_rules"]


def test_kpi_query_average_vs_sum_conflict(tmp_path):
    # query says "average" (AVG intent) but the encoded/KPI aggregate is SUM.
    # The builder's own narrow rule (a) only fires on "total"->SUM, so it accepts
    # this record; the broader v5 query-intent rule catches the AVG-vs-SUM gap.
    entries = {"1": _entry("Bar", ["average price by type"], "a5", "product_type", "SUM(product_price)",
                           "SELECT product_type , SUM(product_price) FROM products GROUP BY product_type")}
    record, _r, profiler = _build_one(tmp_path, entries, _PRICE_DDL, db_id="a5")
    result = check_kpi_sql_aggregation(record)
    assert "query_aggregation_conflict" in result["failed_rules"]


def test_kpi_conflict_prevents_tier_a(tmp_path):
    entries = {"1": _entry("Scatter", ["max vs min price"], "a6", "max(product_price)", "min(product_price)",
                           "SELECT max(product_price) , min(product_price) FROM products GROUP BY product_type")}
    record, resolver, profiler = _build_one(tmp_path, entries, _PRICE_DDL, db_id="a6")
    kpi_result = kpi_suitability(record, profiler, _CFG)
    chart_result = chart_scatter(record, profiler, _CFG)
    constraint_result = check_required_constraints(record, profiler, _CFG)
    quality = score_and_tier(record, kpi_result, chart_result, [], _CFG, constraint_result=constraint_result)
    assert quality["tier"] != "A"
    assert quality["quality_score"] < 100


# --------------------------------------------------------------------------- #
# v5: required time-grain / grouping preservation
# --------------------------------------------------------------------------- #
_DATED_DDL = """
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, order_date TEXT, amount REAL, year INTEGER);
INSERT INTO orders VALUES (1, '2020-01-01', 10, 2020);
INSERT INTO orders VALUES (2, '2020-02-01', 20, 2020);
INSERT INTO orders VALUES (3, '2021-01-01', 30, 2021);
"""


def test_missing_time_grain_on_grouped_date_line(tmp_path):
    # line grouped by a date-named column, no recorded time_grain -> mandatory fail.
    entries = {"1": _entry("Line", ["count by order date"], "t1", "order_date", "count(*)",
                           "SELECT order_date , count(*) FROM orders GROUP BY order_date")}
    record, _r, profiler = _build_one(tmp_path, entries, _DATED_DDL, db_id="t1")
    result = check_required_constraints(record, profiler, _CFG)
    assert "missing_required_time_grain" in result["failed_rules"]


def test_missing_time_grain_on_integer_year(tmp_path):
    entries = {"1": _entry("Bar", ["count by year"], "t2", "year", "count(*)",
                           "SELECT year , count(*) FROM orders GROUP BY year")}
    record, _r, profiler = _build_one(tmp_path, entries, _DATED_DDL, db_id="t2")
    result = check_required_constraints(record, profiler, _CFG)
    assert "missing_required_time_grain" in result["failed_rules"]


def test_time_grain_present_passes(tmp_path):
    # BIN clause makes the builder record a time_grain -> no missing-grain failure.
    entry = _entry("Line", ["count by month"], "t3", "order_date", "count(*)",
                   "SELECT order_date , count(*) FROM orders GROUP BY order_date")
    entry["vis_query"]["data_part"]["binning"] = "BIN order_date BY MONTH"
    record, _r, profiler = _build_one(tmp_path, {"1": entry}, _DATED_DDL, db_id="t3")
    result = check_required_constraints(record, profiler, _CFG)
    assert "missing_required_time_grain" not in result["failed_rules"]


def test_non_temporal_grouping_no_false_time_grain(tmp_path):
    entries = {"1": _entry("Bar", ["amount by type"], "t4", "product_type", "SUM(product_price)",
                           "SELECT product_type , SUM(product_price) FROM products GROUP BY product_type")}
    record, _r, profiler = _build_one(tmp_path, entries, _PRICE_DDL, db_id="t4")
    result = check_required_constraints(record, profiler, _CFG)
    assert "missing_required_time_grain" not in result["failed_rules"]


def test_missing_time_grain_prevents_tier_a(tmp_path):
    entries = {"1": _entry("Line", ["count by order date"], "t5", "order_date", "count(*)",
                           "SELECT order_date , count(*) FROM orders GROUP BY order_date")}
    record, resolver, profiler = _build_one(tmp_path, entries, _DATED_DDL, db_id="t5")
    kpi_result = kpi_suitability(record, profiler, _CFG)
    chart_result = chart_line(record, profiler, _CFG)
    constraint_result = check_required_constraints(record, profiler, _CFG)
    quality = score_and_tier(record, kpi_result, chart_result, [], _CFG, constraint_result=constraint_result)
    assert quality["tier"] != "A"


def test_scores_not_unconditionally_100(tmp_path):
    # A clean count-by-category bar over a heuristic-typed field yields a valid
    # Tier-A record whose db_profile_support is graduated -- proving scores are
    # computed from evidence, not pinned at 100.
    entries = {"1": _entry("Bar", ["count per type"], "s1", "product_type", "count(*)",
                           "SELECT product_type , count(*) FROM products GROUP BY product_type")}
    record, resolver, profiler = _build_one(tmp_path, entries, _PRICE_DDL, db_id="s1")
    kpi_result = kpi_suitability(record, profiler, _CFG)
    chart_result = chart_bar(record, profiler, _CFG)
    constraint_result = check_required_constraints(record, profiler, _CFG)
    quality = score_and_tier(record, kpi_result, chart_result, [], _CFG, constraint_result=constraint_result)
    # component scores must add up to the total, and the total is evidence-derived
    assert sum(quality["component_scores"].values()) == quality["quality_score"]
    assert 0 <= quality["quality_score"] <= 100


def test_build_quality_pool_buckets_by_tier(tmp_path):
    good = _entry("Bar", ["total salary by dept"], "q1", "dept", "SUM(salary)",
                  "SELECT dept , SUM(salary) FROM employee GROUP BY dept")
    bad = _entry("Bar", ["dept chart"], "q1", "dept", "SUM(Employee_ID)",
                 "SELECT dept , SUM(Employee_ID) FROM employee GROUP BY dept")
    entries = {"1": good, "2": bad}
    nv_path = tmp_path / "NVBench.json"
    nv_path.write_text(json.dumps(entries), encoding="utf-8")
    cache_root = tmp_path / "databases"
    db_dir = cache_root / "q1"
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / "q1.sqlite")
    con.executescript(_EMPLOYEE_DDL)
    con.commit()
    con.close()

    builder = NvBenchBuilder(str(nv_path), cache_root=str(cache_root), mapping_path=str(_MAPPING_PATH))
    result = builder.build()
    assert len(result.accepted) == 2

    from src.data_pipeline.nvbench_source import load_mapping
    mapping = load_mapping(str(_MAPPING_PATH))
    resolver = DbMetadataResolver(str(cache_root))
    profiler = DbProfiler(resolver)
    pool = build_quality_pool(result.accepted, mapping, resolver, profiler, _CFG)

    assert pool["summary"]["total_candidates"] == 2
    assert len(pool["tier_a"]) == 1
    assert len(pool["tier_b"]) + len(pool["tier_c"]) == 1
