"""Tests for Phase 2 (nvbench_large_v1.py): 2,000-record selection, split, spot-check.

All fixtures are synthetic enriched dicts matching Phase 1's
``tier_a_candidates.jsonl`` shape -- no real DB, no real Phase-1 output needed.
"""

import collections
import json
from pathlib import Path

from experiments.scripts.run_nvbench_large_v1 import _human_eval_item, _write_independent_reference
from src.data_pipeline.nvbench_pilot import duplicate_checks
from src.data_pipeline.nvbench_large_v1 import (
    NON_SCATTER_CHARTS,
    NORMALIZED_CHART_TYPES,
    compute_availability,
    select_human_eval_sample,
    select_large_v1,
    select_spotcheck_sample,
    semantic_signature,
    signature_diff,
    split_train_val,
    split_train_val_test,
    validate_phase1_input,
)

_WORDBANK = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
             "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()


def _goal(chart, item_id):
    # Hash-derived word indices (not a small-period linear formula) so goal
    # text stays distinct across pools of any size -- a periodic formula here
    # would make items i and i+27 near-identical and trip the near-dup filter
    # for reasons that have nothing to do with the code under test.
    import hashlib
    digest = hashlib.md5(f"{chart}:{item_id}".encode("utf-8")).digest()
    words = " ".join(_WORDBANK[digest[j] % len(_WORDBANK)] for j in range(8))
    return f"{chart} {words} scenario {item_id}"


def _rec(item_id, chart, group_id, db_id, *, score=100, failed=None, filters=None,
        sort=None, grouping=False, time_grain=None, y_agg=None, goal=None, x_field=None, y_field=None, kpi=None,
        group_field=None):
    goal = goal if goal is not None else _goal(chart, item_id)
    kpi = kpi if kpi is not None else f"k_{item_id}"
    prov = {
        "source_group_id": group_id, "db_id": db_id, "nl_query": goal,
        "original_chart_label": chart.capitalize().replace("_", " "),
        "vis_query": {"data_part": {"sql_part": "SELECT 1"}, "VQL": "SELECT 1"},
        "constraints": {"filters": filters or [], "sort": sort, "time_grain": time_grain},
        "grouping": {"is_grouped": grouping},
        "axis_typing": {"x": {"name": x_field}, "y": {"name": y_field, "aggregate": y_agg}},
        "kpi_selection": {"primary_kpi": kpi},
    }
    brief = {
        "item_id": item_id, "users": f"user {item_id}", "goals": [goal], "kpis": [kpi],
        "columns": [{"name": kpi, "dtype": "number", "role": "measure"}],
        "constraints": None, "extra": {"provenance": prov},
    }
    return {
        "item_id": item_id, "split": None, "source_record_id": item_id, "source_group_id": group_id,
        "db_id": db_id, "chart_type": chart, "quality_tier": "A", "quality_score": score,
        "failed_rules": failed or [], "warnings": [], "rule_version": "test", "evidence": {},
        "record": {"item_id": item_id, "split": "train", "brief": brief,
                  "recommendation": {"kpi_chart_mapping": [
                      {"kpi": kpi, "chart_type": chart, "encoding": {"group_field": group_field}}]}},
    }


def _pool(per_chart, db_per_item=True):
    items = []
    for chart, n in per_chart.items():
        for i in range(n):
            db = f"db_{chart}_{i}" if db_per_item else f"db_{chart}"
            items.append(_rec(f"{chart}:{i}", chart, f"grp:{chart}:{i}", db))
    return items


def _no_eval():
    return [{"name": "x", "records": [], "kind": "top", "present": False}]


# --------------------------------------------------------------------------- #
# validate_phase1_input
# --------------------------------------------------------------------------- #
def test_validate_phase1_input_all_pass():
    tier_a = _pool({"bar": 3})
    manifest = {"passed": True}
    hashes = {"tier_a_candidates": "h1", "quality_pool_summary": "h2"}
    recomputed = dict(hashes)
    checks = validate_phase1_input(manifest, hashes, recomputed, tier_a)
    assert all(c["passed"] for c in checks)


def test_validate_phase1_input_low_score_fails():
    tier_a = _pool({"bar": 2})
    tier_a[0]["quality_score"] = 50
    manifest = {"passed": True}
    hashes = {"tier_a_candidates": "h1", "quality_pool_summary": "h2"}
    checks = validate_phase1_input(manifest, hashes, dict(hashes), tier_a)
    by_name = {c["check"]: c for c in checks}
    assert not by_name["phase1_all_tier_a_score_at_least_90"]["passed"]


def test_validate_phase1_input_pie_avg_fails():
    tier_a = [_rec("pie:0", "pie", "grp:pie:0", "db1", y_agg="AVG")]
    manifest = {"passed": True}
    hashes = {"tier_a_candidates": "h1", "quality_pool_summary": "h2"}
    checks = validate_phase1_input(manifest, hashes, dict(hashes), tier_a)
    by_name = {c["check"]: c for c in checks}
    assert not by_name["phase1_no_tier_a_pie_avg_min_max"]["passed"]


def test_validate_phase1_input_hash_mismatch_fails():
    tier_a = _pool({"bar": 2})
    manifest = {"passed": True}
    hashes = {"tier_a_candidates": "h1", "quality_pool_summary": "h2"}
    recomputed = {"tier_a_candidates": "DIFFERENT", "quality_pool_summary": "h2"}
    checks = validate_phase1_input(manifest, hashes, recomputed, tier_a)
    by_name = {c["check"]: c for c in checks}
    assert not by_name["phase1_hashes_valid"]["passed"]


# --------------------------------------------------------------------------- #
# compute_availability
# --------------------------------------------------------------------------- #
def test_compute_availability_reports_group_and_dedup_drops():
    items = _pool({"bar": 4})
    # Make two records share a group (group-uniqueness drop) and two share an
    # exact goal (goal-dedup drop, after one-per-group).
    items[1]["source_group_id"] = items[0]["source_group_id"]
    avail = compute_availability(items, _no_eval())
    bar = avail["bar"]
    assert bar["tier_a_records"] == 4
    assert bar["unique_source_groups"] == 3  # items[0]/[1] share a group
    assert bar["dropped_by_group_uniqueness"] == 1


def test_compute_availability_counts_eval_leakage():
    items = _pool({"bar": 3})
    eval_sources = [{"name": "x", "kind": "top", "present": True,
                    "records": [items[0]["record"]]}]
    avail = compute_availability(items, eval_sources)
    assert avail["bar"]["excluded_by_eval_leakage"] == 1


# --------------------------------------------------------------------------- #
# select_large_v1
# --------------------------------------------------------------------------- #
def test_select_large_v1_scatter_takes_all_available():
    items = _pool({"scatter": 5, "bar": 100, "line": 100, "pie": 100, "stacked_bar": 100})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=50, db_cap=100)
    assert selected is not None
    assert report["achieved_distribution"]["scatter"] == 5  # all 5 scatter admitted, no cap


def test_select_large_v1_exact_total_and_dynamic_allocation():
    items = _pool({"scatter": 10, "bar": 100, "line": 100, "pie": 100, "stacked_bar": 100})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=100, db_cap=100)
    assert selected is not None
    assert report["status"] == "ok"
    assert len(selected) == 100
    counts = collections.Counter(r["chart_type"] for r in selected)
    assert counts["scatter"] == 10
    remainder = 100 - 10
    assert counts["bar"] == round(remainder * 0.40)
    assert counts["line"] == round(remainder * 0.20)


def test_select_large_v1_bar_capped_at_50_percent():
    items = _pool({"scatter": 0, "bar": 1000, "line": 1000, "pie": 1000, "stacked_bar": 1000})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=200, db_cap=1000)
    counts = collections.Counter(r["chart_type"] for r in selected)
    assert counts["bar"] <= 100  # 50% of 200


def test_select_large_v1_redistributes_shortfall():
    # Line chart has almost no supply; others must absorb the shortage to
    # still reach the total, without ever touching Tier B.
    items = _pool({"scatter": 0, "bar": 200, "line": 2, "pie": 200, "stacked_bar": 200})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=100, db_cap=200)
    assert selected is not None
    assert report["status"] == "ok"
    assert len(selected) == 100
    counts = collections.Counter(r["chart_type"] for r in selected)
    assert counts["line"] == 2  # took everything line had
    assert sum(counts.values()) == 100


def test_select_large_v1_insufficient_supply_fails_clearly():
    items = _pool({"scatter": 1, "bar": 5, "line": 5, "pie": 5, "stacked_bar": 5})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=100, db_cap=100)
    assert selected is None
    assert report["status"] == "insufficient_unique_tier_a_candidates"
    assert report["achieved_total"] < 100
    assert report["deficit"] == 100 - report["achieved_total"]


def test_select_large_v1_no_tier_b_fallback():
    items = _pool({"bar": 50})
    items[0]["quality_tier"] = "B"  # contaminated input
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=10, db_cap=100)
    assert selected is not None
    assert all(r["quality_tier"] == "A" for r in selected)
    assert items[0]["item_id"] not in {r["item_id"] for r in selected}


def test_select_large_v1_deterministic():
    items = _pool({"scatter": 10, "bar": 100, "line": 100, "pie": 100, "stacked_bar": 100})
    s1, _ = select_large_v1(items, _no_eval(), seed=42, total=100, db_cap=100)
    s2, _ = select_large_v1(items, _no_eval(), seed=42, total=100, db_cap=100)
    assert [r["item_id"] for r in s1] == [r["item_id"] for r in s2]


def test_select_large_v1_db_cap_forces_redistribution_not_overcap():
    # All bar supply concentrated in 2 databases; db_cap should force spillover
    # to be absorbed elsewhere rather than exceeding the cap.
    items = []
    for i in range(60):
        items.append(_rec(f"bar:{i}", "bar", f"grp:bar:{i}", f"shared_db_{i % 2}"))
    items += _pool({"line": 60, "pie": 60, "stacked_bar": 60})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=100, db_cap=10)
    assert selected is not None
    db_counts = collections.Counter(r["db_id"] for r in selected)
    assert all(c <= 10 for c in db_counts.values())


# --------------------------------------------------------------------------- #
# split_train_val
# --------------------------------------------------------------------------- #
def test_split_train_val_ratio_and_no_overlap():
    items = _pool({"bar": 100, "line": 100})
    train, val, report = split_train_val(items, seed=42, val_fraction=0.10)
    assert len(train) + len(val) == 200
    assert 15 <= len(val) <= 25  # ~10% per chart bucket (100*0.1=10 each -> 20 total)
    assert not report["cross_split_group_overlap"]


def test_split_train_val_preserves_chart_ratio():
    items = _pool({"bar": 100, "scatter": 20})
    train, val, report = split_train_val(items, seed=42, val_fraction=0.10)
    assert report["per_chart"]["bar"]["val"] == 10
    assert report["per_chart"]["scatter"]["val"] == 2


def test_split_train_val_deterministic():
    items = _pool({"bar": 50})
    t1, v1, _ = split_train_val(items, seed=42)
    t2, v2, _ = split_train_val(items, seed=42)
    assert [r["item_id"] for r in t1] == [r["item_id"] for r in t2]
    assert [r["item_id"] for r in v1] == [r["item_id"] for r in v2]


# --------------------------------------------------------------------------- #
# select_spotcheck_sample
# --------------------------------------------------------------------------- #
def test_spotcheck_sample_size_and_all_scatter_included():
    items = _pool({"scatter": 8, "bar": 50, "line": 50, "pie": 50, "stacked_bar": 50})
    sample = select_spotcheck_sample(items, seed=42, size=30)
    assert len(sample) == 30
    scatter_ids = {r["item_id"] for r in items if r["chart_type"] == "scatter"}
    sample_ids = {r["item_id"] for r in sample}
    assert scatter_ids <= sample_ids  # all 8 scatter (<=10) must be included


def test_spotcheck_sample_covers_multiple_chart_types():
    items = _pool({"scatter": 15, "bar": 50, "line": 50, "pie": 50, "stacked_bar": 50})
    sample = select_spotcheck_sample(items, seed=42, size=30)
    charts_present = {r["chart_type"] for r in sample}
    assert len(charts_present) >= 3


def test_spotcheck_sample_covers_constraint_presence():
    items = _pool({"bar": 20, "line": 20, "pie": 20, "stacked_bar": 20})
    items.append(_rec("bar:filtered", "bar", "grp:bar:filtered", "dbf", filters=[{"field": "x", "operator": "=", "value": "1"}]))
    items.append(_rec("line:sorted", "line", "grp:line:sorted", "dbs", sort={"field": "y", "direction": "asc"}))
    items.append(_rec("pie:grouped", "pie", "grp:pie:grouped", "dbg", grouping=True))
    items.append(_rec("stacked_bar:timegrain", "stacked_bar", "grp:sb:tg", "dbt", time_grain={"field": "d", "grain": "YEAR"}))
    sample = select_spotcheck_sample(items, seed=42, size=30)
    sample_ids = {r["item_id"] for r in sample}
    assert "bar:filtered" in sample_ids
    assert "line:sorted" in sample_ids
    assert "pie:grouped" in sample_ids
    assert "stacked_bar:timegrain" in sample_ids


def test_spotcheck_sample_deterministic():
    items = _pool({"scatter": 15, "bar": 50, "line": 50, "pie": 50, "stacked_bar": 50})
    s1 = select_spotcheck_sample(items, seed=42, size=30)
    s2 = select_spotcheck_sample(items, seed=42, size=30)
    assert [r["item_id"] for r in s1] == [r["item_id"] for r in s2]


# --------------------------------------------------------------------------- #
# Phase 2B: controlled max-2-per-group policy
# --------------------------------------------------------------------------- #
def test_semantic_signature_ignores_wording_but_not_structure():
    a = _rec("a", "bar", "g", "db", kpi="rev", sort=None, goal="alpha bravo charlie")
    b = _rec("b", "bar", "g", "db", kpi="rev", sort=None, goal="totally different wording here")
    assert semantic_signature(a) == semantic_signature(b)  # same structure, different wording
    c = _rec("c", "bar", "g", "db", kpi="rev", sort={"field": "x", "direction": "desc"}, goal="alpha bravo charlie")
    assert semantic_signature(a) != semantic_signature(c)
    assert "sort" in signature_diff(semantic_signature(a), semantic_signature(c))


def test_max_per_group_2_accepts_analytically_distinct_pair():
    # Bar's ENTIRE supply is this one group, so both candidates are always
    # tried during admission regardless of hash order; other charts have ample
    # supply to absorb the redistributed remainder.
    primary = _rec("g1:a", "bar", "grp1", "db1", kpi="rev", sort=None,
                  goal="alpha bravo charlie delta echo foxtrot report one")
    secondary = _rec("g1:b", "bar", "grp1", "db1", kpi="rev", sort={"field": "x", "direction": "desc"},
                     goal="zulu yankee xray whiskey victor uniform tango different scenario entirely")
    items = [primary, secondary] + _pool({"line": 40, "pie": 40, "stacked_bar": 40})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=10, db_cap=100, max_per_group=2)
    assert selected is not None
    sel_ids = {r["item_id"] for r in selected}
    assert {"g1:a", "g1:b"} <= sel_ids
    assert report["groups_with_two_records"] >= 1
    pair = next(e for e in report["multi_record_groups"] if e["source_group_id"] == "grp1")
    assert "sort" in pair["differing_components"]


def test_max_per_group_2_rejects_identical_signature_paraphrase():
    primary = _rec("g2:a", "bar", "grp2", "db2", kpi="rev", sort=None,
                  goal="show total revenue by region for the year in a bar chart")
    secondary = _rec("g2:b", "bar", "grp2", "db2", kpi="rev", sort=None,
                     goal="show total revenue by region for the year in a bar chart!")
    items = [primary, secondary] + _pool({"line": 40, "pie": 40, "stacked_bar": 40})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=10, db_cap=100, max_per_group=2)
    assert selected is not None
    sel_ids = {r["item_id"] for r in selected}
    # identical signature + near-identical wording -> paraphrase: exactly one
    # of the pair survives (whichever hash order picks as the group's
    # representative), never both.
    assert len(sel_ids & {"g2:a", "g2:b"}) == 1
    assert not any(e["source_group_id"] == "grp2" for e in report["multi_record_groups"])


def test_max_per_group_2_rejects_similar_wording_even_with_different_signature():
    primary = _rec("g3:a", "bar", "grp3", "db3", kpi="rev", sort=None,
                  goal="show total revenue by region for the year in a bar chart")
    secondary = _rec("g3:b", "bar", "grp3", "db3", kpi="rev", sort={"field": "x", "direction": "asc"},
                     goal="show total revenue by region for the year in a bar chart!")
    items = [primary, secondary] + _pool({"line": 40, "pie": 40, "stacked_bar": 40})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=10, db_cap=100, max_per_group=2)
    assert selected is not None
    sel_ids = {r["item_id"] for r in selected}
    # wording gate alone is not enough when similarity exceeds the threshold,
    # even though the signature (sort direction) differs -- exactly one survives.
    assert len(sel_ids & {"g3:a", "g3:b"}) == 1


def test_max_two_per_group_never_admits_three():
    primary = _rec("g4:a", "bar", "grp4", "db4", kpi="rev", sort=None, goal="alpha bravo charlie report a")
    second = _rec("g4:b", "bar", "grp4", "db4", kpi="rev2", sort=None, goal="delta echo foxtrot report b")
    third = _rec("g4:c", "bar", "grp4", "db4", kpi="rev3", sort=None, goal="golf hotel india report c")
    items = [primary, second, third] + _pool({"line": 40, "pie": 40, "stacked_bar": 40})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=10, db_cap=100, max_per_group=2)
    assert selected is not None
    sel_ids = {r["item_id"] for r in selected}
    from_group4 = sel_ids & {"g4:a", "g4:b", "g4:c"}
    assert len(from_group4) == 2  # never more than 2, even with 2 valid distinct candidates available


def test_split_train_val_keeps_two_record_groups_together():
    items = []
    for i in range(40):
        items.append(_rec(f"bar:{i}:a", "bar", f"grp:{i}", f"db:{i}", kpi=f"rev{i}",
                          goal=_goal("bar", f"{i}:a")))
        items.append(_rec(f"bar:{i}:b", "bar", f"grp:{i}", f"db:{i}", kpi=f"rev{i}b",
                          goal=_goal("bar", f"{i}:b")))
    train, val, report = split_train_val(items, seed=42, val_fraction=0.10)
    train_groups = {r["source_group_id"] for r in train}
    val_groups = {r["source_group_id"] for r in val}
    assert not (train_groups & val_groups)
    # every group's two records land in the same split
    by_group = collections.defaultdict(set)
    split_of = {r["item_id"]: "train" for r in train}
    split_of.update({r["item_id"]: "val" for r in val})
    for it in items:
        by_group[it["source_group_id"]].add(split_of[it["item_id"]])
    assert all(len(splits) == 1 for splits in by_group.values())


def test_select_large_v1_max_per_group_2_deterministic():
    items = _pool({"scatter": 10, "bar": 100, "line": 100, "pie": 100, "stacked_bar": 100})
    s1, _ = select_large_v1(items, _no_eval(), seed=42, total=100, db_cap=100, max_per_group=2)
    s2, _ = select_large_v1(items, _no_eval(), seed=42, total=100, db_cap=100, max_per_group=2)
    assert [r["item_id"] for r in s1] == [r["item_id"] for r in s2]


def test_select_large_v1_max_per_group_2_no_tier_b_fallback():
    items = _pool({"bar": 50})
    items[0]["quality_tier"] = "B"
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=10, db_cap=100, max_per_group=2)
    assert selected is not None
    assert all(r["quality_tier"] == "A" for r in selected)


def test_select_large_v1_max_per_group_2_reaches_more_than_one_per_group():
    # Every group has exactly one genuinely distinct, sufficiently-differently
    # worded secondary candidate available -- max_per_group=2 should roughly
    # double the achievable total for a supply-constrained chart.
    items = []
    for i in range(30):
        items.append(_rec(f"line:{i}:a", "line", f"grpL:{i}", f"dbL:{i}", kpi=f"kL{i}a",
                          goal=_goal("line", f"L{i}a")))
        items.append(_rec(f"line:{i}:b", "line", f"grpL:{i}", f"dbL:{i}", kpi=f"kL{i}b", sort={"field": "x", "direction": "desc"},
                          goal=_goal("line", f"L{i}b")))
    items += _pool({"bar": 200, "pie": 200, "stacked_bar": 200, "scatter": 0})
    _sel1, report1 = select_large_v1(items, _no_eval(), seed=42, total=200, db_cap=100, max_per_group=1)
    sel2, report2 = select_large_v1(items, _no_eval(), seed=42, total=200, db_cap=100, max_per_group=2)
    assert report2["max_achievable_per_chart"]["line"] > report1["max_achievable_per_chart"]["line"]


def test_select_large_v1_insufficient_even_with_two_per_group():
    items = _pool({"scatter": 1, "bar": 5, "line": 5, "pie": 5, "stacked_bar": 5})
    selected, report = select_large_v1(items, _no_eval(), seed=42, total=100, db_cap=100, max_per_group=2)
    assert selected is None
    assert report["status"] == "insufficient_unique_tier_a_candidates"


# --------------------------------------------------------------------------- #
# Phase 2C: 70/15/15 group-aware train/val/test split
# --------------------------------------------------------------------------- #
def test_split_train_val_test_ratios_and_group_safety():
    items = _pool({"bar": 100, "line": 100})
    train, val, test, report = split_train_val_test(items, seed=42, val_fraction=0.15, test_fraction=0.15)
    assert len(train) + len(val) + len(test) == 200
    assert not report["cross_split_group_overlap"]
    # ~70/15/15 per chart (single-record groups here, so group ratio == row ratio)
    assert report["per_chart"]["bar"]["test_groups"] == 15
    assert report["per_chart"]["bar"]["val_groups"] == 15


def test_split_train_val_test_keeps_two_record_groups_together():
    items = []
    for i in range(60):
        items.append(_rec(f"bar:{i}:a", "bar", f"grp:{i}", f"db:{i}", kpi=f"rev{i}", goal=_goal("bar", f"{i}:a")))
        items.append(_rec(f"bar:{i}:b", "bar", f"grp:{i}", f"db:{i}", kpi=f"rev{i}b", goal=_goal("bar", f"{i}:b")))
    train, val, test, report = split_train_val_test(items, seed=42)
    split_of = {}
    for bucket, name in ((train, "train"), (val, "val"), (test, "test")):
        for r in bucket:
            split_of[r["item_id"]] = name
    by_group = collections.defaultdict(set)
    for it in items:
        by_group[it["source_group_id"]].add(split_of[it["item_id"]])
    assert all(len(s) == 1 for s in by_group.values())
    assert not report["cross_split_group_overlap"]


def test_split_train_val_test_deterministic():
    items = _pool({"bar": 100, "pie": 100})
    t1, v1, te1, _ = split_train_val_test(items, seed=42)
    t2, v2, te2, _ = split_train_val_test(items, seed=42)
    assert [r["item_id"] for r in t1] == [r["item_id"] for r in t2]
    assert [r["item_id"] for r in te1] == [r["item_id"] for r in te2]


def test_split_train_val_test_totals_match_input():
    items = _pool({"bar": 37, "line": 23, "pie": 19, "stacked_bar": 11, "scatter": 7})
    train, val, test, report = split_train_val_test(items, seed=42)
    assert len(train) + len(val) + len(test) == 37 + 23 + 19 + 11 + 7
    assert report["train_count"] == len(train)
    assert report["val_count"] == len(val)
    assert report["test_count"] == len(test)


# --------------------------------------------------------------------------- #
# Phase 2C: 40-item human-eval subset (test split only)
# --------------------------------------------------------------------------- #
def test_human_eval_sample_size_and_scatter_coverage():
    items = _pool({"scatter": 8, "bar": 60, "line": 60, "pie": 60, "stacked_bar": 60})
    sample = select_human_eval_sample(items, seed=42, size=40)
    assert len(sample) == 40
    scatter_ids = {r["item_id"] for r in items if r["chart_type"] == "scatter"}
    sample_ids = {r["item_id"] for r in sample}
    assert scatter_ids <= sample_ids


def test_human_eval_sample_covers_one_and_two_record_groups():
    items = [
        _rec("pair:a", "bar", "grpP", "dbP", kpi="k1", goal=_goal("bar", "pairA")),
        _rec("pair:b", "bar", "grpP", "dbP", kpi="k2", goal=_goal("bar", "pairB")),
    ]
    items += _pool({"bar": 60, "line": 60, "pie": 60, "stacked_bar": 60})
    sample = select_human_eval_sample(items, seed=42, size=40)
    sample_ids = {r["item_id"] for r in sample}
    assert {"pair:a", "pair:b"} <= sample_ids  # both records of the 2-record group present


def test_human_eval_sample_deterministic():
    items = _pool({"scatter": 15, "bar": 60, "line": 60, "pie": 60, "stacked_bar": 60})
    s1 = select_human_eval_sample(items, seed=42, size=40)
    s2 = select_human_eval_sample(items, seed=42, size=40)
    assert [r["item_id"] for r in s1] == [r["item_id"] for r in s2]


def test_select_large_v1_accepts_documented_maximum_above_minimum():
    items = _pool({"scatter": 1, "bar": 5, "line": 5, "pie": 5, "stacked_bar": 5})
    selected, report = select_large_v1(
        items, _no_eval(), seed=42, total=100, minimum_acceptable=20,
        db_cap=100, max_per_group=2,
    )
    assert selected is not None
    assert len(selected) == 21
    assert report["status"] == "maximum_valid_corpus_accepted"
    assert report["preferred_target"] == 100
    assert report["minimum_acceptable"] == 20


def test_split_train_val_test_keeps_mixed_chart_group_together():
    items = _pool({"bar": 40, "line": 40, "pie": 40})
    items += [
        _rec("mixed:bar", "bar", "mixed-group", "mixed-db"),
        _rec("mixed:line", "line", "mixed-group", "mixed-db"),
    ]
    train, val, test, report = split_train_val_test(items, seed=42)
    split_of = {
        rec["item_id"]: split_name
        for split_name, records in (("train", train), ("val", val), ("test", test))
        for rec in records
    }
    assert split_of["mixed:bar"] == split_of["mixed:line"]
    assert not report["cross_split_group_overlap"]


def test_human_eval_payload_contains_no_output_or_rating():
    payload = _human_eval_item(_rec("human:1", "bar", "human-group", "human-db"))
    assert set(payload) == {"item_id", "input_brief", "source_evidence", "provenance", "review"}
    assert all(value == "" for value in payload["review"].values())
    assert "recommendation" not in payload


def test_external_l1_reference_records_verified_repository_path(tmp_path):
    root = Path(__file__).resolve().parents[2]
    l1_path = root / "data" / "eval" / "l1_chart_effectiveness_v1.csv"
    assert l1_path.is_file()
    output = tmp_path / "independent_evaluation_reference.json"
    reference = _write_independent_reference(output, l1_path)
    assert reference["literature_based_human_effectiveness_gold"]["path"] == \
        "data/eval/l1_chart_effectiveness_v1.csv"
    assert reference["held_out_nvbench_test"]["fully_external"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == reference

def test_select_large_v1_deduplicates_goals_across_chart_types():
    shared_goal = "show the same analytical request"
    items = [
        _rec("bar:shared", "bar", "g:bar:shared", "db:bar:shared", goal=shared_goal),
        _rec("bar:unique", "bar", "g:bar:unique", "db:bar:unique"),
        _rec("line:unique", "line", "g:line:unique", "db:line:unique"),
        _rec("pie:shared", "pie", "g:pie:shared", "db:pie:shared", goal=shared_goal),
        _rec("pie:unique:1", "pie", "g:pie:unique:1", "db:pie:unique:1"),
        _rec("pie:unique:2", "pie", "g:pie:unique:2", "db:pie:unique:2"),
        _rec("stacked:unique", "stacked_bar", "g:stacked:unique", "db:stacked:unique"),
    ]
    selected, report = select_large_v1(
        items, _no_eval(), seed=42, total=5, db_cap=100, max_per_group=2,
    )
    assert selected is not None
    assert report["achieved_total"] == 5
    assert sum(_norm_goal_for_test(rec) == shared_goal for rec in selected) == 1


def _norm_goal_for_test(rec):
    return " ".join(rec["record"]["brief"]["goals"][0].strip().lower().split())

def test_duplicate_check_ignores_absent_goal_text_but_not_real_duplicates():
    blank_a = _rec("blank:a", "bar", "blank:g1", "blank:db1", goal="")["record"]
    blank_b = _rec("blank:b", "pie", "blank:g2", "blank:db2", goal="")["record"]
    checks, _ = duplicate_checks([blank_a, blank_b], strict=True)
    normalized_check = next(check for check in checks if check["check"] == "no_normalized_duplicate_goals")
    assert normalized_check["passed"]

    blank_b["brief"]["goals"] = ["same substantive goal"]
    blank_a["brief"]["goals"] = ["same substantive goal"]
    checks, _ = duplicate_checks([blank_a, blank_b], strict=True)
    normalized_check = next(check for check in checks if check["check"] == "no_normalized_duplicate_goals")
    assert not normalized_check["passed"]