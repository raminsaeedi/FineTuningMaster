"""Tests for Pilot v4 selection + before/after comparison (nvbench_pilot_v4.py)."""

import json
import sqlite3
from pathlib import Path

from src.core.schemas import ChartType, DashboardBrief, DesignOutput, GoldItem, KPIChartMapping, TaskType
from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder
from src.data_pipeline.nvbench_pilot import _record
from src.data_pipeline.nvbench_pilot_v4 import NORMALIZED_CHART_TYPES, before_after_v3_v4, select_pilot_v4
from src.data_pipeline.nvbench_profile import DbProfiler
from src.data_pipeline.nvbench_source import DbMetadataResolver, load_mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAPPING_PATH = _REPO_ROOT / "src" / "config" / "data" / "nvbench_mapping.yaml"

_CFG = {
    "identifier": {
        "strong_unique_ratio": 0.98, "strong_min_distinct": 20, "ambiguous_unique_ratio": 0.5,
        "name_patterns": ["(^|_)id$", "^id(_|$)", "identifier", "(^|_)key$", "(^|_)code$"],
    },
    "chart": {
        "pie": {"max_categories": 8}, "scatter": {"min_distinct_values": 10},
        "stacked_bar": {"max_group_cardinality": 8},
    },
    "scoring": {
        "weights": {"source_fidelity": 30, "kpi_validity": 20, "chart_suitability": 25,
                    "constraint_completeness": 15, "db_profile_support": 10},
        "tier_a_min_score": 90,
    },
    "sampling": {"seed": 42, "target_per_chart": 20, "db_cap": 10, "near_dup_threshold": 0.8},
}


def _gold_item(item_id, chart_type, group_id, goal, db_id="db1"):
    # kpi/column names and user text embed item_id so distinct items never
    # collide as near-duplicates by accident (char-3gram Jaccard over very
    # short, templated text is otherwise easy to trip above the 0.8 threshold).
    kpi_name = f"k_{item_id}"
    brief = DashboardBrief(
        item_id=item_id, users=f"user for {item_id}", goals=[goal], kpis=[kpi_name],
        columns=[{"name": kpi_name, "dtype": "number", "role": "measure"}],
        extra={"provenance": {"source_group_id": group_id, "db_id": db_id}},
    )
    rec = DesignOutput(
        context_summary={},
        kpi_chart_mapping=[KPIChartMapping(kpi="k", task_type=TaskType.COMPARISON,
                                            chart_type=ChartType(chart_type), encoding={})],
        layout={}, styling={}, interactions=[], rationales=[],
    )
    return GoldItem(item_id=item_id, brief=brief, recommendation=rec, split="train")


def _items(per_chart, charts=NORMALIZED_CHART_TYPES):
    # Each item gets its own database: db_cap defaults are large enough that this
    # never interferes with these tests; the dedicated cap-fallback test below
    # constructs its own shared-database scenario explicitly.
    items = []
    for chart in charts:
        for i in range(per_chart):
            items.append(_gold_item(f"{chart}:{i}", chart, f"grp:{chart}:{i}",
                                     f"goal for {chart} number {i} unique text {i}", db_id=f"db_{chart}_{i}"))
    return items


# --------------------------------------------------------------------------- #
# select_pilot_v4
# --------------------------------------------------------------------------- #
def test_select_pilot_v4_happy_path():
    items = _items(per_chart=3)
    selected, report = select_pilot_v4(items, seed=42, target_per_chart=3, db_cap=10)
    assert selected is not None
    assert report["status"] == "ok"
    assert len(selected) == 3 * len(NORMALIZED_CHART_TYPES)
    counts = {}
    for it in selected:
        chart = it.recommendation.kpi_chart_mapping[0].chart_type.value
        counts[chart] = counts.get(chart, 0) + 1
    assert all(counts[c] == 3 for c in NORMALIZED_CHART_TYPES)
    group_ids = {it.brief.extra["provenance"]["source_group_id"] for it in selected}
    assert len(group_ids) == len(selected)


def test_select_pilot_v4_deterministic_seed():
    items = _items(per_chart=3)
    sel1, _ = select_pilot_v4(items, seed=42, target_per_chart=3, db_cap=10)
    sel2, _ = select_pilot_v4(items, seed=42, target_per_chart=3, db_cap=10)
    assert [it.item_id for it in sel1] == [it.item_id for it in sel2]


def test_select_pilot_v4_insufficient_tier_a_supply():
    items = _items(per_chart=3)
    # Cut the bar bucket down to 2 (below target 3).
    items = [it for it in items if not (it.item_id.startswith("bar:") and it.item_id == "bar:2")]
    selected, report = select_pilot_v4(items, seed=42, target_per_chart=3, db_cap=10)
    assert selected is None
    assert report["status"] == "insufficient_tier_a_candidates"
    assert "bar" in report["short_charts"]
    assert report["available_per_chart"]["bar"] == 2


def test_select_pilot_v4_no_silent_db_cap_fallback():
    # All bar items share one database; db_cap smaller than target forces v3's
    # sampler to fall back over-cap. v4 must refuse rather than accept it silently.
    items = []
    for i in range(3):
        items.append(_gold_item(f"bar:{i}", "bar", f"grp:bar:{i}", f"goal bar {i} unique", db_id="shared_db"))
    for chart in NORMALIZED_CHART_TYPES:
        if chart == "bar":
            continue
        for i in range(3):
            items.append(_gold_item(f"{chart}:{i}", chart, f"grp:{chart}:{i}", f"goal {chart} {i} unique",
                                     db_id=f"db_{chart}_{i}"))
    selected, report = select_pilot_v4(items, seed=42, target_per_chart=3, db_cap=1)
    assert selected is None
    assert report["status"] == "insufficient_tier_a_candidates_within_db_cap"
    assert "bar" in report["short_charts"]


def test_select_pilot_v4_allow_partial_accepts_documented_shortfall():
    items = _items(per_chart=3)
    # Cut bar down to 2 (below target 3): a genuine post-dedup-style shortfall.
    items = [it for it in items if it.item_id != "bar:2"]
    selected, report = select_pilot_v4(items, seed=42, target_per_chart=3, db_cap=10, allow_partial=True)
    assert selected is not None
    assert report["status"] == "ok_partial_documented_shortfall"
    assert report["shortfall"] == {"bar": 2}
    assert len(selected) == 2 + 3 * (len(NORMALIZED_CHART_TYPES) - 1)


def test_select_pilot_v4_allow_partial_still_refuses_db_cap_fallback():
    # allow_partial must not weaken the separate, always-refused db-cap-fallback rule.
    items = []
    for i in range(3):
        items.append(_gold_item(f"bar:{i}", "bar", f"grp:bar:{i}", f"goal bar {i} unique", db_id="shared_db"))
    for chart in NORMALIZED_CHART_TYPES:
        if chart == "bar":
            continue
        for i in range(3):
            items.append(_gold_item(f"{chart}:{i}", chart, f"grp:{chart}:{i}", f"goal {chart} {i} unique",
                                     db_id=f"db_{chart}_{i}"))
    selected, report = select_pilot_v4(items, seed=42, target_per_chart=3, db_cap=1, allow_partial=True)
    assert selected is None
    assert report["status"] == "insufficient_tier_a_candidates_within_db_cap"


# --------------------------------------------------------------------------- #
# before_after_v3_v4 (corpus-level, never by item id)
# --------------------------------------------------------------------------- #
_EMPLOYEE_DDL = """
CREATE TABLE employee (Employee_ID INTEGER PRIMARY KEY, salary REAL, dept TEXT);
INSERT INTO employee VALUES (1, 50000, 'eng');
INSERT INTO employee VALUES (2, 60000, 'eng');
INSERT INTO employee VALUES (3, 55000, 'sales');
"""


def _entry(chart, nl, db_id, x, y, sql):
    return {
        "chart": chart, "db_id": db_id, "hardness": "Easy",
        "vis_query": {"vis_part": f"Visualize {chart.upper()}", "data_part": {"sql_part": sql, "binning": ""}, "VQL": sql},
        "vis_obj": {"chart": chart.lower(), "x_name": x, "y_name": y, "x_data": [[1]], "y_data": [[1]],
                    "classify": [], "describe": ""},
        "nl_queries": nl,
    }


def test_before_after_v3_v4_is_corpus_level(tmp_path):
    good = _entry("Bar", ["total salary by dept"], "ba1", "dept", "SUM(salary)",
                  "SELECT dept , SUM(salary) FROM employee GROUP BY dept")
    bad = _entry("Bar", ["dept chart"], "ba1", "dept", "SUM(Employee_ID)",
                 "SELECT dept , SUM(Employee_ID) FROM employee GROUP BY dept")
    nv_path = tmp_path / "NVBench.json"
    nv_path.write_text(json.dumps({"1": good, "2": bad}), encoding="utf-8")
    cache_root = tmp_path / "databases"
    db_dir = cache_root / "ba1"
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / "ba1.sqlite")
    con.executescript(_EMPLOYEE_DDL)
    con.commit()
    con.close()

    builder = NvBenchBuilder(str(nv_path), cache_root=str(cache_root), mapping_path=str(_MAPPING_PATH))
    result = builder.build()
    records = [_record(it) for it in result.accepted]

    mapping = load_mapping(str(_MAPPING_PATH))
    resolver = DbMetadataResolver(str(cache_root))
    profiler = DbProfiler(resolver)

    comparison = before_after_v3_v4(records, records, mapping, resolver, profiler, _CFG)
    assert comparison["v3"]["n"] == 2
    assert comparison["v4"]["n"] == 2
    assert comparison["v3"]["identifier_as_measure_count"] == 1
    # corpus-level only: no per-item id lists anywhere in the comparison
    assert "item_ids" not in json.dumps(comparison)
