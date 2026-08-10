"""Focused tests for version-2 selection, splitting, and human-R1 sampling."""

from __future__ import annotations

import json

from src.data_pipeline.nvbench_large_v2 import (
    repair_selected_v2,
    select_r1_sample,
    snapshot_tree,
    split_train_val_test_v2,
)


def _record(
    index: int,
    chart: str,
    *,
    group: str | None = None,
    source_record: str | None = None,
    tier: str = "A",
    score: int = 100,
    failed: list[str] | None = None,
    features: set[str] | None = None,
) -> dict:
    features = features or set()
    item_id = f"{chart}:{index}"
    group_id = group or f"group:{chart}:{index}"
    constraints = {
        "filters": [{"field": "region", "operator": "=", "value": str(index)}]
        if "filters" in features else [],
        "sort": {"field": "value", "direction": "desc", "status": "ok"}
        if "sort" in features else None,
        "limit": 5 if "limit" in features else None,
        "having": [],
        "time_grain": {"field": "event_date", "grain": "MONTH"}
        if "time_grain" in features else None,
        "visual_grouping": {
            "fields": ["event_date"], "origin": "vql_bin", "status": "implicit_visual_grouping"
        } if "vql_bin" in features else {
            "fields": ["category"] if "grouping" in features else [],
            "origin": "sql_group_by" if "grouping" in features else "none",
            "status": "explicit_sql_grouping" if "grouping" in features else "none",
        },
    }
    grouping_fields = ["category"] if "grouping" in features else []
    if "vql_bin" in features:
        grouping_fields = ["event_date"]
    kpi_evidence = {
        "evidence": {
            "policy": "count-requires-explicit-entity-count-intent",
            "explicit_count_intent": True,
            "identifier": {"is_identifier": True, "confidence": "strong"},
        }
    } if "identifier_count" in features else {"evidence": {}}
    goal = f"{chart} scientifically distinct goal tokens {index} database {index % 7}"
    provenance = {
        "source_group_id": group_id,
        "source_record_id": source_record or item_id,
        "db_id": f"db:{index % 7}",
        "nl_query": goal,
        "original_chart_label": chart,
        "vis_query": {"data_part": {"sql_part": "SELECT category, value FROM t"}, "VQL": "SELECT category, value FROM t"},
        "constraints": constraints,
        "grouping": {"sql_group_by_fields": grouping_fields, "normalized_fields": grouping_fields},
        "axis_typing": {
            "x": {
                "name": "SUM(value_x)" if "aggregate_scatter" in features else "category",
                "aggregate": "SUM" if "aggregate_scatter" in features else None,
            },
            "y": {
                "name": (
                    "SUM(value_y)" if "aggregate_scatter" in features
                    else "COUNT(value)" if "aggregate_limit" in features
                    else "value"
                ),
                "aggregate": (
                    "SUM" if "aggregate_scatter" in features
                    else "COUNT" if "aggregate_limit" in features
                    else None
                ),
            },
        },
        "kpi_selection": {"primary_kpi": "value"},
    }
    brief = {
        "users": "researcher",
        "goals": [goal],
        "kpis": ["value"],
        "columns": [{"name": "value", "dtype": "number", "role": "measure"}],
        "constraints": None,
        "extra": {"provenance": provenance},
    }
    return {
        "item_id": item_id,
        "source_record_id": source_record or item_id,
        "source_group_id": group_id,
        "db_id": f"db:{index % 7}",
        "chart_type": chart,
        "quality_tier": tier,
        "quality_score": score,
        "failed_rules": failed or [],
        "warnings": [],
        "evidence": {"kpi_suitability": kpi_evidence},
        "record": {
            "item_id": item_id,
            "split": "train",
            "brief": brief,
            "recommendation": {"kpi_chart_mapping": [{
                "kpi": "value", "chart_type": chart,
                "encoding": {"x": "category", "y": "value", "group_field": None},
            }]},
        },
    }


def _no_eval() -> list[dict]:
    return [{"name": "none", "records": [], "kind": "top", "present": False}]


def test_v2_replacement_uses_only_tier_a_and_deduplicates_source_records():
    previous = [f"bar:{index}" for index in range(5)]
    pool = [_record(index, "bar") for index in range(4)]
    pool.append(_record(4, "bar", tier="B", failed=["wrong_kpi"]))
    pool.append(_record(5, "bar"))
    pool.append(_record(6, "bar", source_record="bar:5"))
    selected, report = repair_selected_v2(
        pool, _no_eval(), previous,
        previous_chart_distribution={"bar": 5},
        preferred_target=5, minimum_acceptable=5, db_cap=100, near_dup_threshold=1.0,
    )
    assert selected is not None and len(selected) == 5
    assert all(record["quality_tier"] == "A" and not record["failed_rules"] for record in selected)
    assert "bar:4" in report["removed_previous_ids"]
    assert len({record["source_record_id"] for record in selected}) == len(selected)
    assert report["source_record_deduplication"]["dropped_duplicate_item_ids"]


def test_v2_selection_excludes_empty_normalized_goals():
    pool = [_record(index, "bar") for index in range(5)]
    pool[4]["record"]["brief"]["goals"] = ["   "]
    selected, report = repair_selected_v2(
        pool,
        _no_eval(),
        [],
        preferred_target=4,
        minimum_acceptable=4,
        db_cap=100,
        near_dup_threshold=1.0,
    )
    assert selected is not None and len(selected) == 4
    assert pool[4]["item_id"] not in {record["item_id"] for record in selected}
    assert report["excluded_empty_normalized_goal_ids"] == [pool[4]["item_id"]]


def test_v2_split_is_deterministic_approximately_70_15_15_and_group_safe():
    records = [_record(index, ["bar", "line", "pie", "scatter", "stacked_bar"][index % 5]) for index in range(120)]
    records.extend([
        _record(200, "bar", group="paired"),
        _record(201, "line", group="paired"),
    ])
    first = split_train_val_test_v2(records, seed=42)
    second = split_train_val_test_v2(records, seed=42)
    assert [[record["item_id"] for record in split] for split in first[:3]] == [
        [record["item_id"] for record in split] for split in second[:3]
    ]
    train, val, test, report = first
    assert len(train) + len(val) + len(test) == len(records)
    assert not report["cross_split_group_overlap"]
    assert 65 <= report["actual_percentages"]["train"] <= 75
    assert 10 <= report["actual_percentages"]["val"] <= 20
    assert 10 <= report["actual_percentages"]["test"] <= 20


def test_r1_sample_is_deterministic_blank_input_selection_with_one_scatter_and_coverage():
    charts = ["bar", "line", "pie", "scatter", "stacked_bar"]
    records = []
    feature_sets = [
        {"filters"}, {"sort", "limit", "aggregate_limit"}, {"limit"},
        {"grouping"}, {"vql_bin", "time_grain"},
        {"identifier_count"}, set(), set(), {"aggregate_scatter"}, set(),
    ]
    for index in range(60):
        chart = charts[index % len(charts)]
        features = feature_sets[index % len(feature_sets)]
        score = 90 if index == 0 else (100 if index == 59 else 95)
        records.append(_record(index, chart, score=score, features=features))
    records[10]["source_group_id"] = "two-record-group"
    records[10]["record"]["brief"]["extra"]["provenance"]["source_group_id"] = "two-record-group"
    records[11]["source_group_id"] = "two-record-group"
    records[11]["record"]["brief"]["extra"]["provenance"]["source_group_id"] = "two-record-group"

    sample1, coverage1 = select_r1_sample(records, seed=42, size=30)
    sample2, coverage2 = select_r1_sample(records, seed=42, size=30)
    assert [record["item_id"] for record in sample1] == [record["item_id"] for record in sample2]
    assert coverage1 == coverage2
    assert len(sample1) == 30
    assert coverage1["scatter_count"] == 1
    assert coverage1["database_count"] >= 3
    assert not coverage1["missing_available_tags"]
    assert set(coverage1["chart_counts"]) == set(charts)
    assert "feature:valid_identifier_count" in coverage1["covered_tags"]
    assert "feature:valid_constraint_scope" in coverage1["covered_tags"]
    assert "feature:valid_aggregate_scatter" in coverage1["covered_tags"]


def test_snapshot_tree_detects_any_historical_file_change(tmp_path):
    root = tmp_path / "v1"
    root.mkdir()
    file_path = root / "manifest.json"
    file_path.write_text(json.dumps({"status": "historical"}), encoding="utf-8")
    before = snapshot_tree(root)
    file_path.write_text(json.dumps({"status": "changed"}), encoding="utf-8")
    after = snapshot_tree(root)
    assert before != after
