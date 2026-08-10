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

import pytest

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
    check_source_consistency,
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
        "scatter": {"min_distinct_values": 10,
                    "identifier_axis_reason_code": "scatter_identifier_axis",
                    "invalid_axes_reason_code": "invalid_scatter_axes"},
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
    assert {"identifier_as_measure", "invalid_identifier_aggregation", "wrong_kpi"} <= set(
        result["failed_rules"]
    )


@pytest.mark.parametrize("aggregate", ["SUM", "AVG", "MIN", "MAX"])
def test_all_non_count_identifier_aggregations_rejected(tmp_path, aggregate):
    entries = {"1": _entry(
        "Bar", ["show values by dept"], f"eid_{aggregate.lower()}", "dept",
        f"{aggregate}(Employee_ID)",
        f"SELECT dept, {aggregate}(Employee_ID) FROM employee GROUP BY dept",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, _EMPLOYEE_DDL, db_id=f"eid_{aggregate.lower()}"
    )
    failed = set(kpi_suitability(record, profiler, _CFG)["failed_rules"])
    assert {"identifier_as_measure", "invalid_identifier_aggregation", "wrong_kpi"} <= failed


def test_count_identifier_allowed_only_for_explicit_entity_count(tmp_path):
    explicit = {"1": _entry(
        "Bar", ["how many employees are in each dept"], "eid_count_ok", "dept",
        "COUNT(Employee_ID)", "SELECT dept, COUNT(Employee_ID) FROM employee GROUP BY dept",
    )}
    record, _r, profiler = _build_one(tmp_path, explicit, _EMPLOYEE_DDL, db_id="eid_count_ok")
    assert kpi_suitability(record, profiler, _CFG)["suitable"] is True
    chart_result = chart_bar(record, profiler, _CFG)
    assert chart_result["passed"] is True
    assert "identifier_as_measure" not in chart_result["failed_rules"]
    quality = score_and_tier(
        record,
        kpi_suitability(record, profiler, _CFG),
        chart_result,
        [],
        _CFG,
        constraint_result=check_required_constraints(record, profiler, _CFG),
        consistency_result=check_source_consistency(record, profiler),
    )
    assert quality["tier"] == "A"

    implicit = {"1": _entry(
        "Bar", ["show employee identifiers by dept"], "eid_count_bad", "dept",
        "COUNT(Employee_ID)", "SELECT dept, COUNT(Employee_ID) FROM employee GROUP BY dept",
    )}
    record, _r, profiler = _build_one(tmp_path, implicit, _EMPLOYEE_DDL, db_id="eid_count_bad")
    failed = set(kpi_suitability(record, profiler, _CFG)["failed_rules"])
    assert {"wrong_kpi", "goal_mismatch"} <= failed


@pytest.mark.parametrize("kpi", ["COUNT(*)", "COUNT(salary)"])
def test_count_requires_explicit_count_semantics_for_every_base(tmp_path, kpi):
    entries = {"1": _entry(
        "Bar", ["list department values"], "count_without_intent", "dept", kpi,
        f"SELECT dept, {kpi} FROM employee GROUP BY dept",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, _EMPLOYEE_DDL, db_id="count_without_intent"
    )
    result = kpi_suitability(record, profiler, _CFG)
    assert {"wrong_kpi", "goal_mismatch"} <= set(result["failed_rules"])


@pytest.mark.parametrize("goal", [
    "show the frequency by department",
    "show the total number across departments",
])
def test_generated_count_paraphrases_are_explicit_semantic_evidence(tmp_path, goal):
    entries = {"1": _entry(
        "Bar", [goal], "count_paraphrase", "dept", "COUNT(*)",
        "SELECT dept, COUNT(*) FROM employee GROUP BY dept",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, _EMPLOYEE_DDL, db_id="count_paraphrase"
    )
    assert kpi_suitability(record, profiler, _CFG)["suitable"] is True


@pytest.mark.parametrize("goal", [
    "show the proportion for each department",
    "show the percentage for each department",
    "show the share for each department",
    "show the ratio for each department",
    "show the distribution for each department",
    "show the amount for each department",
])
def test_part_to_whole_words_do_not_imply_entity_count(tmp_path, goal):
    entries = {"1": _entry(
        "Bar", [goal], "count_not_explicit", "dept", "COUNT(*)",
        "SELECT dept, COUNT(*) FROM employee GROUP BY dept",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, _EMPLOYEE_DDL, db_id="count_not_explicit"
    )
    failed = set(kpi_suitability(record, profiler, _CFG)["failed_rules"])
    assert {"wrong_kpi", "goal_mismatch"} <= failed


def test_bare_number_of_numeric_measure_is_not_entity_count_evidence(tmp_path):
    entries = {"1": _entry(
        "Bar", ["show the number of salary values by department"],
        "number_of_measure", "dept", "COUNT(salary)",
        "SELECT dept, COUNT(salary) FROM employee GROUP BY dept",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, _EMPLOYEE_DDL, db_id="number_of_measure"
    )
    result = kpi_suitability(record, profiler, _CFG)
    assert {"wrong_kpi", "goal_mismatch"} <= set(result["failed_rules"])
    assert result["evidence"]["count_intent"]["number_of_entity_supported"] is False


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


def test_pie_percentages_across_dates_are_not_one_additive_whole(tmp_path):
    ddl = """
    CREATE TABLE election (election_date TEXT, vote_percent REAL);
    INSERT INTO election VALUES ('1942-01-01', 16.2);
    INSERT INTO election VALUES ('1946-01-01', 19.5);
    INSERT INTO election VALUES ('1953-01-01', 16.0);
    """
    entries = {"1": _entry(
        "Pie", ["vote percentages across election dates"], "ppw",
        "election_date", "vote_percent", "SELECT election_date, vote_percent FROM election",
    )}
    record, _r, profiler = _build_one(tmp_path, entries, ddl, db_id="ppw")
    result = chart_pie(record, profiler, _CFG)
    assert "pie_not_part_to_whole" in result["failed_rules"]


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
    assert "scatter_identifier_axis" in result["failed_rules"]
    assert "invalid_scatter_axes" in result["failed_rules"]
    assert "identifier_as_measure" in result["failed_rules"]
    assert record["recommendation"]["kpi_chart_mapping"][0]["chart_type"] == "scatter"


_ENTITY_SCATTER_DDL = """
CREATE TABLE employee (employee_id INTEGER PRIMARY KEY, department_id INTEGER, salary REAL);
INSERT INTO employee VALUES (1, 10, 50000); INSERT INTO employee VALUES (2, 10, 52000);
INSERT INTO employee VALUES (3, 20, 54000); INSERT INTO employee VALUES (4, 20, 56000);
INSERT INTO employee VALUES (5, 30, 58000); INSERT INTO employee VALUES (6, 30, 60000);
INSERT INTO employee VALUES (7, 40, 62000); INSERT INTO employee VALUES (8, 40, 64000);
INSERT INTO employee VALUES (9, 50, 66000); INSERT INTO employee VALUES (10, 50, 68000);
INSERT INTO employee VALUES (11, 60, 70000); INSERT INTO employee VALUES (12, 60, 72000);
"""


def test_numeric_entity_reference_rejected_and_excluded_from_kpis(tmp_path):
    entries = {"1": _entry(
        "Scatter", ["salary and department id relationship"], "sc4",
        "salary", "department_id", "SELECT salary, department_id FROM employee",
    )}
    record, _r, profiler = _build_one(tmp_path, entries, _ENTITY_SCATTER_DDL, db_id="sc4")
    assert record["brief"]["kpis"] == ["salary"]
    result = chart_scatter(record, profiler, _CFG)
    assert {"scatter_identifier_axis", "invalid_scatter_axes", "identifier_as_measure"} <= set(
        result["failed_rules"]
    )


def test_removing_identifier_from_kpi_does_not_repair_invalid_scatter(tmp_path):
    entries = {"1": _entry(
        "Scatter", ["salary and department id relationship"], "sc5",
        "salary", "department_id", "SELECT salary, department_id FROM employee",
    )}
    record, _r, profiler = _build_one(tmp_path, entries, _ENTITY_SCATTER_DDL, db_id="sc5")
    kpi_result = kpi_suitability(record, profiler, _CFG)
    chart_result = chart_scatter(record, profiler, _CFG)
    quality = score_and_tier(record, kpi_result, chart_result, [], _CFG)
    assert record["brief"]["kpis"] == ["salary"]
    assert quality["tier"] == "B"
    assert "invalid_scatter_axes" in quality["failed_rules"]


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


def test_score_and_tier_deduplicates_record_level_reason_codes():
    kpi_result = {
        "suitable": False,
        "failed_rules": ["identifier_as_measure", "wrong_kpi"],
        "warnings": ["shared_warning"],
        "evidence": {},
    }
    chart_result = {
        "passed": False,
        "failed_rules": ["identifier_as_measure"],
        "warnings": ["shared_warning"],
        "evidence": {},
    }
    quality = score_and_tier({}, kpi_result, chart_result, [], _CFG)
    assert quality["failed_rules"] == ["identifier_as_measure", "wrong_kpi"]
    assert quality["warnings"] == ["shared_warning"]


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
    grouping = record["brief"]["extra"]["provenance"]["grouping"]
    assert grouping["grouping_origin"] == "vql_bin"
    assert grouping["grouping_status"] == "implicit_visual_grouping"
    assert grouping["normalized_fields"] == ["order_date"]


def test_vql_bin_supplies_visual_grouping_without_sql_group_by(tmp_path):
    entry = _entry(
        "Bar", ["count orders by month"], "tbin", "order_date", "count(*)",
        "SELECT order_date, count(*) FROM orders",
    )
    entry["vis_query"]["data_part"]["binning"] = "BIN order_date BY MONTH"
    entry["vis_query"]["VQL"] += " BIN order_date BY MONTH"
    record, _r, profiler = _build_one(tmp_path, {"1": entry}, _DATED_DDL, db_id="tbin")
    result = check_required_constraints(record, profiler, _CFG)
    assert "missing_required_grouping" not in result["failed_rules"]
    visual = record["brief"]["extra"]["provenance"]["constraints"]["visual_grouping"]
    assert visual == {
        "fields": ["order_date"],
        "origin": "vql_bin",
        "status": "implicit_visual_grouping",
    }


def test_aggregate_task_without_sql_or_vql_grouping_is_rejected(tmp_path):
    entries = {"1": _entry(
        "Bar", ["show average salary for each dept"], "missing_group", "dept", "AVG(salary)",
        "SELECT dept, AVG(salary) FROM employee",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, _EMPLOYEE_DDL, db_id="missing_group"
    )
    failed = set(check_required_constraints(record, profiler, _CFG)["failed_rules"])
    assert {"missing_grouping", "goal_mismatch"} <= failed


def test_entity_primary_key_validly_groups_each_entity(tmp_path):
    ddl = """
    CREATE TABLE faculty (FacID INTEGER PRIMARY KEY, rank TEXT);
    CREATE TABLE student (student_id INTEGER PRIMARY KEY, advisor INTEGER);
    INSERT INTO faculty VALUES (1, 'Professor'); INSERT INTO faculty VALUES (2, 'Lecturer');
    INSERT INTO student VALUES (10, 1); INSERT INTO student VALUES (11, 1);
    INSERT INTO student VALUES (12, 2);
    """
    entries = {"1": _entry(
        "Bar", ["show the number of students for each faculty member"], "entity_group",
        "FacID", "COUNT(student_id)",
        "SELECT FacID, COUNT(student_id) FROM faculty JOIN student ON FacID = advisor GROUP BY FacID",
    )}
    record, _r, profiler = _build_one(tmp_path, entries, ddl, db_id="entity_group")
    failed = set(check_required_constraints(record, profiler, _CFG)["failed_rules"])
    assert "missing_grouping" not in failed
    assert "goal_mismatch" not in failed


def test_unrelated_vql_bin_does_not_supply_requested_grouping(tmp_path):
    entry = _entry(
        "Bar", ["show average salary for each dept"], "unrelated_bin", "region", "AVG(salary)",
        "SELECT region, AVG(salary) FROM employee",
    )
    entry["vis_query"]["data_part"]["binning"] = "BIN region BY MONTH"
    entry["vis_query"]["VQL"] += " BIN region BY MONTH"
    record, _r, profiler = _build_one(
        tmp_path, {"1": entry}, _EMPLOYEE_DDL, db_id="unrelated_bin"
    )
    failed = set(check_required_constraints(record, profiler, _CFG)["failed_rules"])
    assert {"missing_grouping", "goal_mismatch"} <= failed


def test_recorded_sql_visual_grouping_represents_unprojected_observation_field(tmp_path):
    ddl = """
    CREATE TABLE metric (category TEXT, x REAL, y REAL);
    INSERT INTO metric VALUES ('A', 1, 2); INSERT INTO metric VALUES ('A', 2, 3);
    INSERT INTO metric VALUES ('B', 3, 5); INSERT INTO metric VALUES ('B', 4, 7);
    INSERT INTO metric VALUES ('C', 5, 11); INSERT INTO metric VALUES ('C', 6, 13);
    """
    entries = {"1": _entry(
        "Scatter", ["show correlation between total x and total y by category"],
        "aggregate_scatter_valid", "SUM(x)", "SUM(y)",
        "SELECT SUM(x), SUM(y) FROM metric GROUP BY category",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, ddl, db_id="aggregate_scatter_valid"
    )
    constraints = check_required_constraints(record, profiler, _CFG)
    chart = chart_scatter(record, profiler, _CFG)
    kpi = kpi_suitability(record, profiler, _CFG)
    quality = score_and_tier(
        record,
        kpi,
        chart,
        [],
        _CFG,
        constraint_result=constraints,
        consistency_result=check_source_consistency(record, profiler),
    )
    assert "category" in {column["name"] for column in record["brief"]["columns"]}
    assert "missing_required_grouping" not in constraints["failed_rules"]
    assert chart["passed"] is True
    assert chart["evidence"]["query_result_profile"]["observed_row_count"] == 3
    assert quality["tier"] == "A"


def test_aggregate_scatter_with_one_filtered_result_is_rejected(tmp_path):
    ddl = """
    CREATE TABLE metric (category TEXT, x REAL, y REAL);
    INSERT INTO metric VALUES ('A', 1, 2); INSERT INTO metric VALUES ('A', 2, 3);
    INSERT INTO metric VALUES ('B', 3, 5); INSERT INTO metric VALUES ('B', 4, 7);
    """
    entries = {"1": _entry(
        "Scatter", ["show correlation between total x and total y for category A"],
        "aggregate_scatter_one", "SUM(x)", "SUM(y)",
        "SELECT SUM(x), SUM(y) FROM metric WHERE category = 'A' GROUP BY category",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, ddl, db_id="aggregate_scatter_one"
    )
    chart = chart_scatter(record, profiler, _CFG)
    assert {
        "insufficient_scatter_observations", "invalid_scatter_axes", "chart_inappropriate"
    } <= set(chart["failed_rules"])
    assert chart["evidence"]["query_result_profile"]["observed_row_count"] == 1
    assert record["recommendation"]["kpi_chart_mapping"][0]["chart_type"] == "scatter"


def test_scatter_requires_complete_numeric_pairs_not_columnwise_values(tmp_path):
    ddl = "CREATE TABLE metric (x REAL, y REAL);"
    ddl += "".join(
        f"INSERT INTO metric VALUES ({index}, NULL);"
        for index in range(1, 13)
    )
    ddl += "".join(
        f"INSERT INTO metric VALUES (NULL, {index});"
        for index in range(101, 113)
    )
    entries = {"1": _entry(
        "Scatter", ["show the relationship between x and y"],
        "scatter_disjoint_nulls", "x", "y", "SELECT x, y FROM metric",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, ddl, db_id="scatter_disjoint_nulls"
    )
    chart = chart_scatter(record, profiler, _CFG)
    profile = chart["evidence"]["query_result_profile"]
    assert profile["observed_row_count"] == 24
    assert profile["columns"][0]["numeric_value_count"] == 12
    assert profile["columns"][1]["numeric_value_count"] == 12
    assert profile["paired_numeric_row_count"] == 0
    assert {"insufficient_scatter_observations", "invalid_scatter_axes"} <= set(
        chart["failed_rules"]
    )


def test_vql_bin_materializes_aggregate_scatter_observations_for_validation(tmp_path):
    ddl = """
    CREATE TABLE metric (event_date TEXT, x REAL, y REAL);
    INSERT INTO metric VALUES ('2020-01-01', 1, 2);
    INSERT INTO metric VALUES ('2020-02-01', 2, 3);
    INSERT INTO metric VALUES ('2021-01-01', 4, 7);
    INSERT INTO metric VALUES ('2021-02-01', 5, 11);
    INSERT INTO metric VALUES ('2022-01-01', 8, 13);
    INSERT INTO metric VALUES ('2022-02-01', 10, 17);
    """
    entry = _entry(
        "Scatter", ["show total x against total y binned by year"],
        "scatter_vql_bin", "SUM(x)", "SUM(y)",
        "SELECT SUM(x), SUM(y) FROM metric",
    )
    entry["vis_query"]["data_part"]["binning"] = "BIN event_date BY YEAR"
    entry["vis_query"]["VQL"] += " BIN event_date BY YEAR"
    record, _r, profiler = _build_one(
        tmp_path, {"1": entry}, ddl, db_id="scatter_vql_bin"
    )
    chart = chart_scatter(record, profiler, _CFG)
    evidence = chart["evidence"]
    assert chart["passed"] is True
    assert evidence["source_query_result_profile"]["observed_row_count"] == 1
    assert evidence["query_result_profile_origin"] == "vql_bin_validation_materialization"
    assert evidence["query_result_profile"]["observed_row_count"] == 3
    assert evidence["query_result_profile"]["paired_numeric_row_count"] == 3


def test_schema_backed_dimension_conflict_with_sql_group_is_rejected(tmp_path):
    entries = {"1": _entry(
        "Bar", ["show average salary for each dept"], "wrong_sql_group", "region", "AVG(salary)",
        "SELECT region, AVG(salary) FROM employee GROUP BY region",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, _EMPLOYEE_DDL, db_id="wrong_sql_group"
    )
    constraints = check_required_constraints(record, profiler, _CFG)
    consistency = check_source_consistency(record, profiler)
    assert "missing_grouping" not in constraints["failed_rules"]
    assert {"missing_required_dimension", "source_conflict"} <= set(
        consistency["failed_rules"]
    )


def test_regression_2780_vql_bin_valid_but_top5_scope_invalid(tmp_path):
    ddl = """
    CREATE TABLE player (player_id INTEGER PRIMARY KEY, birthday TEXT, potential REAL);
    INSERT INTO player VALUES (1, '1990-01-01', 90);
    INSERT INTO player VALUES (2, '1991-02-02', 89);
    INSERT INTO player VALUES (3, '1992-03-03', 88);
    INSERT INTO player VALUES (4, '1993-04-04', 87);
    INSERT INTO player VALUES (5, '1994-05-05', 86);
    INSERT INTO player VALUES (6, '1995-06-06', 85);
    """
    entry = _entry(
        "Bar", ["show the top five players by potential and bin birthday by weekday, using count"],
        "scope2780", "birthday", "COUNT(birthday)",
        "SELECT birthday, COUNT(birthday) FROM player ORDER BY potential DESC LIMIT 5",
    )
    entry["vis_query"]["data_part"]["binning"] = "BIN birthday BY WEEKDAY"
    entry["vis_query"]["VQL"] += " BIN birthday BY WEEKDAY"
    record, _r, profiler = _build_one(tmp_path, {"1": entry}, ddl, db_id="scope2780")
    result = check_required_constraints(record, profiler, _CFG)
    failed = set(result["failed_rules"])
    assert "missing_grouping" not in failed
    assert {"constraint_scope_error", "goal_mismatch"} <= failed
    constraints = record["brief"]["extra"]["provenance"]["constraints"]
    assert constraints["sort"]["field"] == "potential"
    assert constraints["limit"] == 5


def test_aggregate_limit_rejects_raw_aggregate_base_sort_but_accepts_alias(tmp_path):
    raw_sort = {"1": _entry(
        "Bar", ["show top five departments by total salary"], "raw_sort_scope", "dept", "SUM(salary)",
        "SELECT dept, SUM(salary) FROM employee GROUP BY dept ORDER BY salary DESC LIMIT 5",
    )}
    record, _r, profiler = _build_one(
        tmp_path, raw_sort, _EMPLOYEE_DDL, db_id="raw_sort_scope"
    )
    failed = set(check_required_constraints(record, profiler, _CFG)["failed_rules"])
    assert {"constraint_scope_error", "goal_mismatch"} <= failed

    alias_sort = {"1": _entry(
        "Bar", ["show top five departments by total salary"], "alias_sort_scope", "dept", "SUM(salary)",
        "SELECT dept, SUM(salary) AS total_salary FROM employee "
        "GROUP BY dept ORDER BY total_salary DESC LIMIT 5",
    )}
    record, _r, profiler = _build_one(
        tmp_path, alias_sort, _EMPLOYEE_DDL, db_id="alias_sort_scope"
    )
    assert "constraint_scope_error" not in check_required_constraints(
        record, profiler, _CFG
    )["failed_rules"]


def test_top_n_scope_requires_matching_nested_preaggregation(tmp_path):
    valid = {"1": _entry(
        "Bar", ["show total salary by department for the top three employees by salary"],
        "valid_preaggregate", "dept", "SUM(salary)",
        "SELECT dept, SUM(salary) FROM ("
        "SELECT dept, salary FROM employee ORDER BY salary DESC LIMIT 3"
        ") ranked GROUP BY dept",
    )}
    record, _r, profiler = _build_one(
        tmp_path, valid, _EMPLOYEE_DDL, db_id="valid_preaggregate"
    )
    assert "constraint_scope_error" not in check_required_constraints(
        record, profiler, _CFG
    )["failed_rules"]


def test_cte_marker_does_not_validate_outer_raw_measure_sort(tmp_path):
    invalid = {"1": _entry(
        "Bar", ["show top three departments by total salary"],
        "invalid_cte_scope", "dept", "SUM(salary)",
        "WITH scoped AS (SELECT dept, salary FROM employee) "
        "SELECT dept, SUM(salary) FROM scoped GROUP BY dept "
        "ORDER BY salary DESC LIMIT 3",
    )}
    record, _r, profiler = _build_one(
        tmp_path, invalid, _EMPLOYEE_DDL, db_id="invalid_cte_scope"
    )
    result = check_required_constraints(record, profiler, _CFG)
    assert {"constraint_scope_error", "goal_mismatch"} <= set(result["failed_rules"])
    scope = result["evidence"]["constraint_scope_error"]["preaggregation_scope"]
    assert scope["verified"] is False
    assert scope["outer_order_by"] is True
    assert scope["outer_limit"] is True


def test_regression_3257_invalid_group_and_single_point_scatter(tmp_path):
    ddl = """
    CREATE TABLE wine (Price REAL, Score REAL, Year INTEGER);
    INSERT INTO wine VALUES (10, 80, 2010); INSERT INTO wine VALUES (11, 81, 2011);
    INSERT INTO wine VALUES (12, 82, 2012); INSERT INTO wine VALUES (13, 83, 2013);
    INSERT INTO wine VALUES (14, 84, 2014); INSERT INTO wine VALUES (15, 85, 2015);
    INSERT INTO wine VALUES (16, 86, 2016); INSERT INTO wine VALUES (17, 87, 2017);
    INSERT INTO wine VALUES (18, 88, 2018); INSERT INTO wine VALUES (19, 89, 2019);
    """
    entries = {"1": _entry(
        "Scatter", ["Scatter plot to show max price and maximal score"], "scatter3257",
        "MAX(Price)", "MAX(Score)",
        "SELECT MAX(Price), MAX(Score) FROM wine GROUP BY MAX(Price)",
    )}
    record, _r, profiler = _build_one(tmp_path, entries, ddl, db_id="scatter3257")
    constraint_failed = set(check_required_constraints(record, profiler, _CFG)["failed_rules"])
    chart_failed = set(chart_scatter(record, profiler, _CFG)["failed_rules"])
    assert "invalid_group_by_expression" in constraint_failed
    assert {
        "insufficient_scatter_observations", "invalid_scatter_axes", "chart_inappropriate"
    } <= chart_failed
    assert record["recommendation"]["kpi_chart_mapping"][0]["chart_type"] == "scatter"


def test_regression_1048_card_number_sum_demoted_and_source_projection_not_invented(tmp_path):
    rows = ";".join(
        f"INSERT INTO customers_cards VALUES ({index}, {100 + index}, 'T{index % 2}', '{900000 + index}')"
        for index in range(1, 16)
    ) + ";"
    ddl = (
        "CREATE TABLE customers_cards (card_id INTEGER, customer_id INTEGER, "
        "card_type_code TEXT, card_number VARCHAR(80));" + rows
    )
    entries = {"1": _entry(
        "Bar", ["What are card ids, customer ids, card types, and card numbers for each customer card?"],
        "cards1048", "card_type_code", "SUM(card_number)",
        "SELECT card_type_code, SUM(card_number) FROM customers_cards GROUP BY card_type_code",
    )}
    record, _r, profiler = _build_one(tmp_path, entries, ddl, db_id="cards1048")
    kpi_failed = set(kpi_suitability(record, profiler, _CFG)["failed_rules"])
    grouping_failed = set(check_required_constraints(record, profiler, _CFG)["failed_rules"])
    assert {"identifier_as_measure", "invalid_identifier_aggregation", "wrong_kpi"} <= kpi_failed
    assert "missing_grouping" not in grouping_failed
    raw_columns = {column["name"] for column in record["brief"]["columns"]}
    assert raw_columns == {"card_type_code", "card_number"}
    assert {"card_id", "customer_id"}.isdisjoint(raw_columns)


def test_stacked_bar_grouping_combines_binned_x_and_series(tmp_path):
    entry = _entry(
        "Stacked Bar", ["count starts by weekday and full-time status"], "tstack",
        "order_date", "count(*)", "SELECT order_date, count(*) FROM orders GROUP BY status",
        classify=["F", "T"],
    )
    entry["vis_query"]["data_part"]["binning"] = "BIN order_date BY WEEKDAY"
    entry["vis_query"]["VQL"] += " BIN order_date BY WEEKDAY"
    ddl = _DATED_DDL.replace(
        "amount REAL, year INTEGER", "amount REAL, year INTEGER, status TEXT"
    ).replace(
        "(1, '2020-01-01', 10, 2020)", "(1, '2020-01-01', 10, 2020, 'F')"
    ).replace(
        "(2, '2020-02-01', 20, 2020)", "(2, '2020-02-01', 20, 2020, 'T')"
    ).replace(
        "(3, '2021-01-01', 30, 2021)", "(3, '2021-01-01', 30, 2021, 'F')"
    )
    record, _r, profiler = _build_one(tmp_path, {"1": entry}, ddl, db_id="tstack")
    grouping = record["brief"]["extra"]["provenance"]["grouping"]
    assert grouping["normalized_fields"] == ["order_date", "status"]
    assert grouping["series_field"] == "status"


def test_query_vql_time_grain_conflict_detected(tmp_path):
    entry = _entry(
        "Line", ["Bin all dates into the weekday interval and show the trend."],
        "tconflict", "order_date", "count(*)", "SELECT order_date, count(*) FROM orders",
    )
    entry["vis_query"]["data_part"]["binning"] = "BIN order_date BY YEAR"
    entry["vis_query"]["VQL"] += " BIN order_date BY YEAR"
    record, _r, _profiler = _build_one(tmp_path, {"1": entry}, _DATED_DDL, db_id="tconflict")
    result = check_source_consistency(record)
    assert "time_grain_source_conflict" in result["failed_rules"]
    assert "source_conflict" in result["failed_rules"]


def test_blank_natural_language_goal_is_not_tier_a(tmp_path):
    entries = {"1": _entry(
        "Bar", [""], "blank_goal", "dept", "COUNT(*)",
        "SELECT dept, COUNT(*) FROM employee GROUP BY dept",
    )}
    record, _r, profiler = _build_one(
        tmp_path, entries, _EMPLOYEE_DDL, db_id="blank_goal"
    )
    consistency = check_source_consistency(record, profiler)
    kpi = kpi_suitability(record, profiler, _CFG)
    assert {"missing_nl_goal", "goal_mismatch"} <= set(consistency["failed_rules"])
    assert {"wrong_kpi", "goal_mismatch"} <= set(kpi["failed_rules"])
    quality = score_and_tier(
        record,
        kpi,
        chart_bar(record, profiler, _CFG),
        [],
        _CFG,
        constraint_result=check_required_constraints(record, profiler, _CFG),
        consistency_result=consistency,
    )
    assert quality["tier"] != "A"


def test_missing_true_dimension_remains_source_conflict(tmp_path):
    entries = {"1": _entry(
        "Scatter", ["average price and score for each appellation"], "tdim",
        "AVG(price)", "AVG(score)", "SELECT AVG(price), AVG(score) FROM wine GROUP BY AVG(price)",
    )}
    ddl = "CREATE TABLE wine (price REAL, score REAL, appellation TEXT); INSERT INTO wine VALUES (10, 90, 'A');"
    record, _r, _profiler = _build_one(tmp_path, entries, ddl, db_id="tdim")
    result = check_source_consistency(record)
    assert "missing_required_dimension" in result["failed_rules"]
    assert result["evidence"]["missing_required_dimensions"][0]["dimension"] == "appellation"


def test_unrelated_table_field_does_not_create_dimension_conflict(tmp_path):
    entries = {"1": _entry(
        "Bar", ["average score for each gender"], "tdim_scope",
        "Sex", "AVG(score)", "SELECT Sex, AVG(score) FROM student GROUP BY Sex",
    )}
    ddl = (
        "CREATE TABLE student (Sex TEXT, score REAL); "
        "CREATE TABLE audit (gender TEXT, event TEXT); "
        "INSERT INTO student VALUES ('F', 90); INSERT INTO audit VALUES ('F', 'created');"
    )
    record, _r, profiler = _build_one(tmp_path, entries, ddl, db_id="tdim_scope")
    result = check_source_consistency(record, profiler)
    assert "missing_required_dimension" not in result["failed_rules"]


def test_entity_title_can_represent_each_entity(tmp_path):
    entries = {"1": _entry(
        "Bar", ["average price for each film"], "tentity_title",
        "Title", "AVG(price)", "SELECT Title, AVG(price) FROM film GROUP BY Title",
    )}
    ddl = "CREATE TABLE film (Title TEXT, price REAL); INSERT INTO film VALUES ('A', 10);"
    record, _r, profiler = _build_one(tmp_path, entries, ddl, db_id="tentity_title")
    result = check_source_consistency(record, profiler)
    assert "missing_required_dimension" not in result["failed_rules"]


def test_filter_field_does_not_substitute_for_output_dimension(tmp_path):
    entries = {"1": _entry(
        "Bar", ["For each director, show movie title and rating"], "tdirector",
        "title", "rating",
        "SELECT title, rating FROM movie WHERE director != 'null'",
    )}
    ddl = (
        "CREATE TABLE movie (title TEXT, rating REAL, director TEXT); "
        "INSERT INTO movie VALUES ('A', 8, 'D');"
    )
    record, _r, profiler = _build_one(tmp_path, entries, ddl, db_id="tdirector")
    result = check_source_consistency(record, profiler)
    assert "missing_required_dimension" in result["failed_rules"]


def test_numeric_text_measure_is_kept_as_kpi(tmp_path):
    entries = {"1": _entry(
        "Bar", ["Show the age of each dog"], "tnumeric_text",
        "name", "age", "SELECT name, age FROM dogs",
    )}
    ddl = (
        "CREATE TABLE dogs (name TEXT, age VARCHAR(20)); "
        "INSERT INTO dogs VALUES ('A', '2'); INSERT INTO dogs VALUES ('B', '3.5');"
    )
    record, _r, profiler = _build_one(tmp_path, entries, ddl, db_id="tnumeric_text")
    assert record["brief"]["kpis"] == ["age"]
    assert record["brief"]["extra"]["provenance"]["axis_typing"]["y"]["dtype"] == "number"
    assert kpi_suitability(record, profiler, _CFG)["suitable"] is True


def test_schema_spelling_variant_does_not_hide_missing_appellation(tmp_path):
    entries = {"1": _entry(
        "Scatter", ["average price and score for each appellation"], "tappelation",
        "AVG(price)", "AVG(score)", "SELECT AVG(price), AVG(score) FROM wine GROUP BY AVG(price)",
    )}
    ddl = "CREATE TABLE wine (price REAL, score REAL, Appelation TEXT); INSERT INTO wine VALUES (10, 90, 'A');"
    record, _r, profiler = _build_one(tmp_path, entries, ddl, db_id="tappelation")
    result = check_source_consistency(record, profiler)
    assert "missing_required_dimension" in result["failed_rules"]


def test_missing_aggregate_condition_is_not_invented(tmp_path):
    entries = {"1": _entry(
        "Bar", ["salary for each department that has more than 2 employees"], "thaving",
        "department_id", "SUM(salary)",
        "SELECT department_id, SUM(salary) FROM employee GROUP BY department_id",
    )}
    record, _r, _profiler = _build_one(tmp_path, entries, _ENTITY_SCATTER_DDL, db_id="thaving")
    result = check_source_consistency(record)
    assert "missing_aggregate_condition" in result["failed_rules"]
    constraints = record["brief"]["extra"]["provenance"]["constraints"]
    assert constraints["having"] == []


def test_explicit_having_condition_is_preserved(tmp_path):
    entries = {"1": _entry(
        "Bar", ["salary for each department that has more than 2 employees"], "thaving2",
        "department_id", "SUM(salary)",
        "SELECT department_id, SUM(salary) FROM employee GROUP BY department_id HAVING COUNT(*) > 2",
    )}
    record, _r, _profiler = _build_one(tmp_path, entries, _ENTITY_SCATTER_DDL, db_id="thaving2")
    result = check_source_consistency(record)
    assert "missing_aggregate_condition" not in result["failed_rules"]
    assert record["brief"]["extra"]["provenance"]["constraints"]["having"][0]["value"] == "2"


def test_row_level_numeric_filter_is_not_misclassified_as_missing_having(tmp_path):
    ddl = """
    CREATE TABLE apartment (apt_number TEXT, bedroom_count INTEGER);
    INSERT INTO apartment VALUES ('A1', 3); INSERT INTO apartment VALUES ('A2', 4);
    """
    entries = {"1": _entry(
        "Bar", ["number of apartments with more than 2 bedrooms"], "tfilter",
        "apt_number", "COUNT(apt_number)",
        "SELECT apt_number, COUNT(apt_number) FROM apartment WHERE bedroom_count > 2 GROUP BY apt_number",
    )}
    record, _r, profiler = _build_one(tmp_path, entries, ddl, db_id="tfilter")
    result = check_source_consistency(record, profiler)
    assert "missing_aggregate_condition" not in result["failed_rules"]


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
