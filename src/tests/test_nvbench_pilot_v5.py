"""Tests for Pilot v5 strict selection + corpus-level before/after."""

import json
import sqlite3
from pathlib import Path

from src.core.schemas import ChartType, DashboardBrief, DesignOutput, GoldItem, KPIChartMapping, TaskType
from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder
from src.data_pipeline.nvbench_pilot import _record
from src.data_pipeline.nvbench_pilot_v4 import NORMALIZED_CHART_TYPES
from src.data_pipeline.nvbench_pilot_v5 import before_after_v4_v5, select_pilot_v4
from src.data_pipeline.nvbench_profile import DbProfiler
from src.data_pipeline.nvbench_source import DbMetadataResolver, load_mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAPPING_PATH = _REPO_ROOT / "src" / "config" / "data" / "nvbench_mapping.yaml"

_CFG = {
    "identifier": {"strong_unique_ratio": 0.98, "strong_min_distinct": 20, "ambiguous_unique_ratio": 0.5,
                   "name_patterns": ["(^|_)id$", "^id(_|$)", "identifier", "(^|_)key$", "(^|_)code$"]},
    "chart": {"pie": {"max_categories": 8}, "scatter": {"min_distinct_values": 10},
              "stacked_bar": {"max_group_cardinality": 8}},
    "scoring": {"weights": {"source_fidelity": 30, "kpi_validity": 20, "chart_suitability": 25,
                            "constraint_completeness": 15, "db_profile_support": 10}, "tier_a_min_score": 90},
    "sampling": {"seed": 42, "target_per_chart": 20, "db_cap": 10, "near_dup_threshold": 0.8},
    "time_grain": {"supported": ["YEAR", "MONTH", "DAY", "WEEKDAY", "WEEK", "QUARTER", "HOUR", "DATE"],
                   "name_hints": ["date", "time", "year", "month", "day", "week", "quarter", "hour"],
                   "temporal_charts": ["line", "bar"]},
}


_WORDBANK = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
             "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()


def _distinct_goal(chart, i):
    # High-entropy goal so distinct items never collide as char-3gram near-dups,
    # varied by BOTH chart and index so cross-chart pairs are distinct too.
    offset = sum(ord(c) for c in chart)
    words = " ".join(_WORDBANK[(offset + i * 7 + j * 5) % len(_WORDBANK)] for j in range(8))
    return f"{chart} {words} scenario {i} cohort {offset + i}"


def _gold_item(item_id, chart_type, group_id, goal, db_id):
    kpi = f"k_{item_id}"
    brief = DashboardBrief(item_id=item_id, users=f"user {item_id}", goals=[goal], kpis=[kpi],
                           columns=[{"name": kpi, "dtype": "number", "role": "measure"}],
                           extra={"provenance": {"source_group_id": group_id, "db_id": db_id}})
    rec = DesignOutput(context_summary={},
                       kpi_chart_mapping=[KPIChartMapping(kpi=kpi, task_type=TaskType.COMPARISON,
                                                          chart_type=ChartType(chart_type), encoding={})],
                       layout={}, styling={}, interactions=[], rationales=[])
    return GoldItem(item_id=item_id, brief=brief, recommendation=rec, split="train")


def _pool(per_chart_map):
    items = []
    for chart, n in per_chart_map.items():
        for i in range(n):
            items.append(_gold_item(f"{chart}:{i}", chart, f"grp:{chart}:{i}",
                                     _distinct_goal(chart, i), db_id=f"db_{chart}_{i}"))
    return items


def test_v5_strict_exactly_20_per_chart_gives_100():
    items = _pool({c: 20 for c in NORMALIZED_CHART_TYPES})
    selected, report = select_pilot_v4(items, seed=42, target_per_chart=20, db_cap=10)
    assert selected is not None
    assert report["status"] == "ok"
    assert len(selected) == 100


def test_v5_strict_15_scatter_is_insufficient_no_partial():
    supply = {c: 20 for c in NORMALIZED_CHART_TYPES}
    supply["scatter"] = 15
    items = _pool(supply)
    selected, report = select_pilot_v4(items, seed=42, target_per_chart=20, db_cap=10)  # strict: no allow_partial
    assert selected is None
    assert report["status"] == "insufficient_tier_a_candidates"
    assert "scatter" in report["short_charts"]
    assert report["available_per_chart"]["scatter"] == 15


def test_v5_strict_deterministic():
    items = _pool({c: 25 for c in NORMALIZED_CHART_TYPES})
    s1, _ = select_pilot_v4(items, seed=42, target_per_chart=20, db_cap=10)
    s2, _ = select_pilot_v4(items, seed=42, target_per_chart=20, db_cap=10)
    assert [it.item_id for it in s1] == [it.item_id for it in s2]


# --------------------------------------------------------------------------- #
# before_after_v4_v5 corpus-level
# --------------------------------------------------------------------------- #
_EMP_DDL = """
CREATE TABLE employee (Employee_ID INTEGER PRIMARY KEY, salary REAL, dept TEXT);
INSERT INTO employee VALUES (1, 50000, 'eng'); INSERT INTO employee VALUES (2, 60000, 'eng');
INSERT INTO employee VALUES (3, 55000, 'sales');
"""


def _entry(chart, nl, db_id, x, y, sql):
    return {"chart": chart, "db_id": db_id, "hardness": "Easy",
            "vis_query": {"vis_part": f"Visualize {chart.upper()}", "data_part": {"sql_part": sql, "binning": ""}, "VQL": sql},
            "vis_obj": {"chart": chart.lower(), "x_name": x, "y_name": y, "x_data": [[1]], "y_data": [[1]],
                        "classify": [], "describe": ""},
            "nl_queries": nl}


def test_before_after_v4_v5_is_corpus_level(tmp_path):
    good = _entry("Bar", ["total salary by dept"], "b1", "dept", "SUM(salary)",
                  "SELECT dept , SUM(salary) FROM employee GROUP BY dept")
    nv = tmp_path / "NVBench.json"
    nv.write_text(json.dumps({"1": good}), encoding="utf-8")
    cache = tmp_path / "databases"; (cache / "b1").mkdir(parents=True)
    con = sqlite3.connect(cache / "b1" / "b1.sqlite"); con.executescript(_EMP_DDL); con.commit(); con.close()
    result = NvBenchBuilder(str(nv), cache_root=str(cache), mapping_path=str(_MAPPING_PATH)).build()
    records = [_record(it) for it in result.accepted]
    mapping = load_mapping(str(_MAPPING_PATH))
    resolver = DbMetadataResolver(str(cache)); profiler = DbProfiler(resolver)
    ba = before_after_v4_v5(records, records, mapping, resolver, profiler, _CFG)
    assert ba["v4"]["n"] == 1 and ba["v5"]["n"] == 1
    assert "item_ids" not in json.dumps(ba)
    assert "kpi_conflict_count" in ba["v5"]
