"""Focused tests for the hybrid dataset build (offline, no API call)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_pipeline.hybrid_dataset import (
    SOURCE_NVBENCH,
    SOURCE_SYNTHETIC,
    cross_source_duplicates,
    distribution_rows,
    has_provenance,
    holdout_collisions,
    normalized_goal,
    required_field_problems,
    schema_problems,
    split_overlap,
    summarize,
    tag_source,
)

HYBRID_DIR = Path("data/staging/dashboard_v3/hybrid_dataset_v1")


def _nvbench(item_id="nvbench:1:query:0", split="train", goal="Bar chart of product counts.",
             group="grp:1", chart="bar") -> dict:
    return {
        "item_id": item_id, "split": split,
        "brief": {
            "item_id": item_id, "users": "analyst", "goals": [goal], "kpis": ["COUNT(x)"],
            "columns": [{"name": "x", "dtype": "categorical"}],
            "extra": {"source": "nvbench",
                      "provenance": {"db_id": "db", "source_group_id": group},
                      "lineage": {"chart_type": "source-provided", "task_type": "rule-derived"}},
        },
        "recommendation": {
            "context_summary": {"n_kpis": 1},
            "kpi_chart_mapping": [{"kpi": "COUNT(x)", "task_type": "comparison",
                                   "chart_type": chart, "encoding": {"x": "x", "y": "COUNT(x)"}}],
            "layout": {"type": "single"}, "styling": {"theme": "minimal"},
            "interactions": [{"type": "tooltip", "fields": ["x"]}],
            "rationales": [{"claim": "c", "principle": "p"}],
        },
    }


def _synthetic(item_id="v2_abc", split="train", goal="optimize product mix", n_kpis=2) -> dict:
    mapping = [{"kpi": f"KPI {i}", "task_type": "comparison", "chart_type": "bar",
                "encoding": {"x": "segment", "y": f"kpi_{i}"}} for i in range(n_kpis)]
    return {
        "item_id": item_id, "split": split,
        "brief": {
            "item_id": item_id, "users": "Category Managers", "goals": [goal],
            "kpis": [f"KPI {i}" for i in range(n_kpis)],
            "columns": [{"name": "segment", "dtype": "categorical"}],
            "extra": {"source_id": "tmpl_e-commerce", "generator_version": "v2.0-sample",
                      "domain": "E-Commerce"},
        },
        "recommendation": {
            "context_summary": {"domain": "E-Commerce"}, "kpi_chart_mapping": mapping,
            "layout": {"type": "grid"}, "styling": {"theme": "light"},
            "interactions": [{"type": "filter", "fields": ["segment"]}],
            "rationales": [{"claim": "c", "principle": "p"}],
        },
    }


# --------------------------------------------------------------- provenance


def test_source_tagging_marks_both_origins_and_lineage():
    nv = tag_source(_nvbench(), SOURCE_NVBENCH)
    syn = tag_source(_synthetic(), SOURCE_SYNTHETIC)
    assert nv["brief"]["extra"]["dataset_source"] == SOURCE_NVBENCH
    assert syn["brief"]["extra"]["dataset_source"] == SOURCE_SYNTHETIC
    assert has_provenance(nv) and has_provenance(syn)
    assert "llm_generated" in nv["brief"]["extra"]["lineage_classes"]
    assert syn["brief"]["extra"]["lineage_classes"]["llm_generated"] == []


def test_untagged_record_has_no_provenance():
    assert has_provenance(_nvbench()) is False


# ---------------------------------------------------------------- dedup


def test_exact_item_id_collision_drops_the_synthetic_record():
    nv = [_nvbench(item_id="shared")]
    kept, dropped = cross_source_duplicates(nv, [_synthetic(item_id="shared")])
    assert kept == []
    assert dropped[0]["reason"] == "duplicate_item_id"


def test_identical_goal_drops_the_synthetic_record():
    goal = "Bar chart of product counts."
    kept, dropped = cross_source_duplicates([_nvbench(goal=goal)], [_synthetic(goal=goal)])
    assert kept == []
    assert dropped[0]["reason"] == "duplicate_normalized_goal"
    assert "nvBench" in dropped[0]["policy"]


def test_complementary_synthetic_record_with_same_chart_class_is_kept():
    nv = [_nvbench(goal="Bar chart of product counts.", chart="bar")]
    syn = [_synthetic(goal="optimize product mix and reduce cart abandonment")]
    kept, dropped = cross_source_duplicates(nv, syn)
    assert len(kept) == 1 and dropped == []


def test_near_duplicate_goal_is_dropped_even_with_different_kpi_wording():
    nv = [_nvbench(goal="Show revenue by region for the sales team")]
    syn = [_synthetic(goal="Show revenue by region for the sales teams")]
    kept, dropped = cross_source_duplicates(nv, syn)
    assert kept == []
    assert dropped[0]["reason"] == "near_duplicate_goal"


def test_near_duplicate_full_brief_is_dropped():
    nv = [_nvbench(goal="Compare count of x by segment")]
    syn = [dict(_synthetic(goal="Compare count of x by segment"),
                brief={**_synthetic()["brief"], "goals": ["Compare count of x by segment"],
                       "kpis": ["COUNT(x)"], "columns": [{"name": "x", "dtype": "categorical"}]})]
    kept, dropped = cross_source_duplicates(nv, syn)
    assert kept == []
    assert dropped[0]["reason"] in ("near_duplicate_brief", "near_duplicate_goal",
                                    "duplicate_normalized_goal")


def test_normalized_goal_ignores_punctuation_and_case():
    assert normalized_goal(_nvbench(goal="Revenue by Region!")) == \
           normalized_goal(_nvbench(goal="revenue  by region"))


# ------------------------------------------------------------ split safety


def test_split_overlap_detects_shared_ids_and_source_groups():
    train = [tag_source(_nvbench(item_id="a", group="g1"), SOURCE_NVBENCH)]
    val = [tag_source(_nvbench(item_id="a", split="val", group="g1"), SOURCE_NVBENCH)]
    overlap = split_overlap(train, val)
    assert overlap["duplicate_item_ids"] == ["a"]
    assert overlap["shared_source_groups"] == ["g1"]


def test_disjoint_splits_report_no_overlap():
    train = [tag_source(_nvbench(item_id="a", group="g1"), SOURCE_NVBENCH)]
    val = [tag_source(_nvbench(item_id="b", split="val", group="g2"), SOURCE_NVBENCH)]
    assert split_overlap(train, val) == {"duplicate_item_ids": [], "shared_source_groups": []}


def test_holdout_collision_detects_shared_item_and_group():
    records = [tag_source(_nvbench(item_id="a", group="g1"), SOURCE_NVBENCH)]
    holdout = [_nvbench(item_id="a", split="test", group="g1")]
    collisions = holdout_collisions(records, holdout)
    assert collisions["shared_item_ids"] == ["a"]
    assert collisions["shared_source_groups"] == ["g1"]


# ------------------------------------------------------------- validation


def test_schema_and_required_field_checks_catch_broken_records():
    broken = _nvbench()
    broken["recommendation"]["rationales"] = []
    broken["brief"]["kpis"] = []
    problems = required_field_problems([broken])
    assert any("empty rationales" in p for p in problems)
    assert any("empty kpis" in p for p in problems)
    assert schema_problems([{"item_id": "x"}])  # missing brief/recommendation


def test_distribution_rows_cover_the_required_dimensions():
    rows = distribution_rows([tag_source(_nvbench(), SOURCE_NVBENCH),
                              tag_source(_synthetic(), SOURCE_SYNTHETIC)], "train")
    dimensions = {r["dimension"] for r in rows}
    assert dimensions == {"source", "chart_type", "task_type", "domain", "kpi_cardinality"}
    kpi_rows = {r["value"]: r["records"] for r in rows if r["dimension"] == "kpi_cardinality"}
    assert kpi_rows == {"single_kpi": 1, "multi_kpi": 1}


def test_summary_counts_sources_and_kpi_cardinality():
    summary = summarize([tag_source(_nvbench(), SOURCE_NVBENCH),
                         tag_source(_synthetic(), SOURCE_SYNTHETIC)])
    assert summary["records"] == 2
    assert summary["by_source"] == {SOURCE_NVBENCH: 1, SOURCE_SYNTHETIC: 1}
    assert summary["multi_kpi_records"] == 1


# ------------------------------------------------- the built artifact itself


@pytest.mark.skipif(not (HYBRID_DIR / "train.jsonl").exists(), reason="hybrid dataset not built")
def test_built_hybrid_dataset_passes_every_documented_check():
    report = json.loads((HYBRID_DIR / "validation_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS_HYBRID_DATASET_READY_FOR_FREEZE"
    assert report["failures"] == []
    assert all(report["checks"].values())


@pytest.mark.skipif(not (HYBRID_DIR / "train.jsonl").exists(), reason="hybrid dataset not built")
def test_built_hybrid_dataset_has_no_holdout_leakage():
    leakage = json.loads((HYBRID_DIR / "leakage_report.json").read_text(encoding="utf-8"))
    for key in ("shared_item_ids", "shared_normalized_goals", "shared_source_groups",
                "near_duplicates"):
        assert leakage["against_nvbench_test"][key] == []
    assert leakage["against_human_eval_items"]["shared_item_ids"] == []
    assert leakage["cross_split"]["shared_source_groups"] == []


@pytest.mark.skipif(not (HYBRID_DIR / "train.jsonl").exists(), reason="hybrid dataset not built")
def test_built_hybrid_records_carry_split_and_provenance():
    for split in ("train", "val"):
        records = [json.loads(l) for l in (HYBRID_DIR / f"{split}.jsonl").read_text(
            encoding="utf-8").splitlines()]
        assert records
        assert {r["split"] for r in records} == {split}
        assert all(has_provenance(r) for r in records)
