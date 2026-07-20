"""Tests for the versioned staging nvBench pilot build + validation logic.

Records are produced by the real builder from a tiny synthetic NVBench.json so
their shape always matches the builder; mutations exercise each failure mode.
Never hashes real datasets.
"""

import collections
import json
from copy import deepcopy
from pathlib import Path

from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder
from src.data_pipeline.nvbench_pilot import (
    build_pilot_records,
    distribution_rows,
    duplicate_checks,
    leakage_checks,
    select_pilot_v3,
    semantic_checks,
    structural_checks,
)
from src.data_pipeline.nvbench_source import load_mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAPPING_PATH = _REPO_ROOT / "src" / "config" / "data" / "nvbench_mapping.yaml"


def _entry(chart, nl, db_id="db1", x="cat", y="count(*)", sql="SELECT", classify=None):
    return {"chart": chart, "db_id": db_id, "hardness": "Easy",
            "vis_query": {"vis_part": "V", "data_part": {"sql_part": sql, "binning": ""}, "VQL": sql},
            "vis_obj": {"chart": chart.lower(), "x_name": x, "y_name": y,
                        "x_data": [[1]], "y_data": [[1]], "classify": classify or [], "describe": ""},
            "nl_queries": nl}


def _write_json(tmp_path, entries):
    path = tmp_path / "NVBench.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _make_all(tmp_path, entries, one_per_group=True, limit=None):
    path = tmp_path / "NVBench.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return build_pilot_records(str(path), None, str(_MAPPING_PATH),
                               limit=limit, seed=42, one_per_group=one_per_group,
                               stratify=False)


def _make_records(tmp_path, entries, one_per_group=True, limit=None):
    recs, _rej, _meta = _make_all(tmp_path, entries, one_per_group=one_per_group, limit=limit)
    return recs


def _good(tmp_path, n=5):
    charts = ["Bar", "Pie", "Line", "Bar", "Pie"]
    entries = {str(i): _entry(charts[i % len(charts)], [f"query number {i}"]) for i in range(n)}
    return _make_records(tmp_path, entries)


def _passed(checks, name):
    return next(c for c in checks if c["check"] == name)["passed"]


# --------------------------------------------------------------------------- #
# structural
# --------------------------------------------------------------------------- #
def test_structural_all_pass(tmp_path):
    recs = _good(tmp_path, 5)
    checks = structural_checks(recs, expected=5)
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]


def test_structural_duplicate_item_id_fails(tmp_path):
    recs = _good(tmp_path, 3)
    recs[1]["item_id"] = recs[0]["item_id"]
    assert not _passed(structural_checks(recs, expected=3), "unique_item_ids")


def test_structural_cross_split_group_fails(tmp_path):
    recs = _good(tmp_path, 2)
    recs[1]["brief"]["extra"]["provenance"]["source_group_id"] = \
        recs[0]["brief"]["extra"]["provenance"]["source_group_id"]
    recs[1]["split"] = "val" if recs[0]["split"] == "train" else "train"
    assert not _passed(structural_checks(recs, expected=2), "no_cross_split_group_leakage")


def test_structural_group_count(tmp_path):
    recs = _good(tmp_path, 4)
    assert not _passed(structural_checks(recs, expected=100), "unique_source_group_count")


# --------------------------------------------------------------------------- #
# semantic
# --------------------------------------------------------------------------- #
def test_semantic_all_pass(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    checks, warnings = semantic_checks(_good(tmp_path, 5), mapping)
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]
    assert warnings == []


def test_semantic_chart_mismatch_fails(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    recs = _good(tmp_path, 2)
    recs[0]["recommendation"]["kpi_chart_mapping"][0]["chart_type"] = "sankey"
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "chart_type_matches_source")


def test_semantic_kpi_not_in_brief_fails(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    recs = _good(tmp_path, 2)
    recs[0]["recommendation"]["kpi_chart_mapping"][0]["kpi"] = "not_a_kpi"
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "kpi_present_in_brief")


def test_semantic_llm_field_fails(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    recs = _good(tmp_path, 2)
    recs[0]["brief"]["extra"]["lineage"]["goal"] = "LLM-generated"
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "no_llm_generated_fields")


def test_semantic_aggregate_dtype_fail(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    recs = _good(tmp_path, 2)
    # corrupt: aggregate axis relabeled categorical
    recs[0]["brief"]["extra"]["provenance"]["axis_typing"]["y"]["dtype"] = "categorical"
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "aggregate_not_categorical")


def test_unrecoverable_grouping_is_rejected_not_warned(tmp_path):
    # v3: an unrecoverable group field is a mandatory build-time rejection, not a
    # warning — the record never reaches the accepted pool at all.
    mapping = load_mapping(_MAPPING_PATH)
    entries = {"1": _entry("Grouping Line", ["grouped q"])}
    entries["1"]["vis_obj"]["classify"] = ["A", "B"]
    recs, rejections, _meta = _make_all(tmp_path, entries)
    assert recs == []
    assert len(rejections) == 1
    assert rejections[0]["reason"] == "missing_group_field"


# --------------------------------------------------------------------------- #
# duplicates
# --------------------------------------------------------------------------- #
def test_duplicate_fingerprint_detected(tmp_path):
    recs = _good(tmp_path, 2)
    dup = deepcopy(recs[0])
    dup["item_id"] = "nvbench:99:query:0"
    dup["brief"]["extra"]["provenance"]["source_record_id"] = "nvbench:99:query:0"
    dup["brief"]["extra"]["provenance"]["source_group_id"] = "nvbench:99"
    recs.append(dup)
    checks, findings = duplicate_checks(recs)
    assert not _passed(checks, "no_exact_duplicate_briefs")
    assert any(f["type"] == "duplicate_brief_fingerprint" for f in findings)


def test_near_duplicate_pairs_unique(tmp_path):
    # two queries of one visualization -> near-duplicate briefs; report once.
    entries = {"1": _entry("Bar", ["show total count of items by category",
                                    "display the total count of items per category"])}
    recs = _make_records(tmp_path, entries, one_per_group=False)
    assert len(recs) == 2
    checks, findings = duplicate_checks(recs)
    pairs = [f for f in findings if f["type"] == "near_duplicate_pair"]
    # each unordered pair stored once (A,B) not also (B,A)
    keys = {tuple(sorted((f["left_id"], f["right_id"]))) for f in pairs}
    assert len(pairs) == len(keys)


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
def test_leakage_fingerprint_overlap_mandatory(tmp_path):
    recs = _good(tmp_path, 3)
    eval_rec = {"item_id": "eval_x", "brief": deepcopy(recs[0]["brief"])}
    sources = [{"name": "internal_test", "records": [eval_rec], "kind": "nested", "present": True}]
    checks, findings = leakage_checks(recs, sources)
    assert not _passed(checks, "no_exact_fingerprint_overlap")
    assert any(f["type"] == "exact_fingerprint_overlap" for f in findings)


def test_leakage_clean_when_disjoint(tmp_path):
    recs = _good(tmp_path, 3)
    eval_rec = {"item_id": "eval_y", "users": "x", "goals": ["totally different goal text"],
                "kpis": ["Z"], "columns": [{"name": "q", "dtype": "number"}]}
    sources = [{"name": "real_briefs_v1", "records": [eval_rec], "kind": "top", "present": True},
               {"name": "missing", "records": [], "kind": "nested", "present": False}]
    checks, findings = leakage_checks(recs, sources)
    assert _passed(checks, "no_exact_fingerprint_overlap")
    assert any(f["type"] == "eval_source_skipped" for f in findings)


# --------------------------------------------------------------------------- #
# distributions + determinism
# --------------------------------------------------------------------------- #
def test_distribution_rows_dimensions(tmp_path):
    rows = distribution_rows(_good(tmp_path, 5))
    dims = {d for d, _, _ in rows}
    assert {"source_chart_label", "normalized_chart_type", "inferred_task_type",
            "split", "database", "field_lineage", "column_dtype_origin",
            "grouping_recovery", "grouped"} <= dims
    fl_values = {v.split("=")[0] for d, v, _ in rows if d == "field_lineage"}
    assert {"chart_type", "encoding", "goal", "kpi", "task_type", "layout",
            "styling", "interactions", "rationales"} == fl_values


def test_build_pilot_records_deterministic(tmp_path):
    entries = {str(i): _entry("Bar", [f"q{i}a", f"q{i}b"]) for i in range(1, 11)}
    path = tmp_path / "NVBench.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    a, ar, _ = build_pilot_records(str(path), None, str(_MAPPING_PATH),
                                   limit=5, seed=42, one_per_group=True, stratify=True)
    b, br, _ = build_pilot_records(str(path), None, str(_MAPPING_PATH),
                                   limit=5, seed=42, one_per_group=True, stratify=True)
    assert a == b and ar == br
    assert len(a) == 5
    gids = {r["brief"]["extra"]["provenance"]["source_group_id"] for r in a}
    assert len(gids) == 5


# --------------------------------------------------------------------------- #
# Task 7: strengthened mandatory semantic checks
# --------------------------------------------------------------------------- #
def test_no_aggregate_in_columns_pass_and_fail(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    recs = _good(tmp_path, 2)
    checks, _ = semantic_checks(recs, mapping)
    assert _passed(checks, "no_aggregate_in_columns")
    # corrupt: inject an aggregate expression into brief.columns
    recs[0]["brief"]["columns"].append({"name": "SUM(x)", "dtype": "number", "role": "measure"})
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "no_aggregate_in_columns")


def test_aggregate_base_field_missing_from_columns_fails(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    entries = {"1": _entry("Bar", ["q"], y="sum(price)")}
    recs = _make_records(tmp_path, entries)
    checks, _ = semantic_checks(recs, mapping)
    assert _passed(checks, "aggregate_base_field_in_columns")
    # corrupt: drop the "price" raw column the y-aggregate depends on
    recs[0]["brief"]["columns"] = [c for c in recs[0]["brief"]["columns"] if c["name"] != "price"]
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "aggregate_base_field_in_columns")


def test_constraint_field_missing_from_columns_fails(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    recs = _good(tmp_path, 2)
    checks, _ = semantic_checks(recs, mapping)
    assert _passed(checks, "constraint_fields_in_columns")
    # inject a filter referencing a field that is not a raw column
    recs[0]["brief"]["extra"]["provenance"]["constraints"] = {
        "filters": [{"field": "not_a_real_column", "operator": "=", "value": "x"}],
        "sort": None, "time_grain": None, "source_order": None,
    }
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "constraint_fields_in_columns")


def test_stacked_bar_requires_group_field(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    entries = {"1": _entry("Stacked Bar", ["q"], sql="SELECT cat , count(*) FROM t GROUP BY cat , grp",
                           classify=["A", "B"])}
    recs = _make_records(tmp_path, entries)
    assert len(recs) == 1  # group field ("grp") recovered -> accepted
    checks, _ = semantic_checks(recs, mapping)
    assert _passed(checks, "stacked_bar_has_group_field")
    # corrupt: blank out the group_field on an otherwise-valid stacked bar record
    recs[0]["recommendation"]["kpi_chart_mapping"][0]["encoding"]["group_field"] = None
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "stacked_bar_has_group_field")


def test_scatter_requires_two_numeric_axes(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    # both axes aggregate -> both number/measure without any DB cache -> accepted.
    entries = {"1": _entry("Scatter", ["q"], x="count(*)", y="sum(price)")}
    recs = _make_records(tmp_path, entries)
    assert len(recs) == 1
    checks, _ = semantic_checks(recs, mapping)
    assert _passed(checks, "scatter_two_numeric_axes")
    # corrupt: relabel x as categorical post-hoc
    recs[0]["brief"]["extra"]["provenance"]["axis_typing"]["x"]["dtype"] = "categorical"
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "scatter_two_numeric_axes")


def test_no_nested_aggregate_remains_fails_on_corruption(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    recs = _good(tmp_path, 2)
    checks, _ = semantic_checks(recs, mapping)
    assert _passed(checks, "no_nested_aggregate_remains")
    recs[0]["recommendation"]["kpi_chart_mapping"][0]["encoding"]["y"] = "SUM(sum(x))"
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "no_nested_aggregate_remains")


def test_query_chart_consistent_fails_on_corrupted_query(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    recs = _good(tmp_path, 2)  # includes a Bar record
    bar = next(r for r in recs if r["brief"]["extra"]["provenance"]["original_chart_label"] == "Bar")
    checks, _ = semantic_checks(recs, mapping)
    assert _passed(checks, "query_chart_consistent")
    bar["brief"]["extra"]["provenance"]["nl_query"] = "Show this as a pie chart please."
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "query_chart_consistent")


def test_query_aggregate_agrees_fails_on_corrupted_query(tmp_path):
    mapping = load_mapping(_MAPPING_PATH)
    entries = {"1": _entry("Bar", ["q"], y="avg(price)")}
    recs = _make_records(tmp_path, entries)
    checks, _ = semantic_checks(recs, mapping)
    assert _passed(checks, "query_aggregate_agrees")
    # unambiguous SUM intent ("total price", no other aggregate keyword) vs
    # encoded AVG -> conflict, matching the real nvbench:1002 regression.
    recs[0]["brief"]["extra"]["provenance"]["nl_query"] = "Show the total price for each category."
    checks, _ = semantic_checks(recs, mapping)
    assert not _passed(checks, "query_aggregate_agrees")


# --------------------------------------------------------------------------- #
# strict duplicate mode (pilot v3: exact-goal and near-duplicates are mandatory)
# --------------------------------------------------------------------------- #
def test_duplicate_checks_strict_promotes_severity(tmp_path):
    entries = {"1": _entry("Bar", ["show total count of items by category",
                                    "display the total count of items per category"])}
    recs = _make_records(tmp_path, entries, one_per_group=False)
    lenient, _ = duplicate_checks(recs, strict=False)
    strict, _ = duplicate_checks(recs, strict=True)
    near_lenient = next(c for c in lenient if c["check"] == "near_duplicate_within_pilot")
    near_strict = next(c for c in strict if c["check"] == "near_duplicate_within_pilot")
    assert near_lenient["severity"] == "warning"
    assert near_strict["severity"] == "mandatory"


# --------------------------------------------------------------------------- #
# Task 6: select_pilot_v3 — chart balance, database cap, near-dup-aware sampling
# --------------------------------------------------------------------------- #
def _accepted(tmp_path, entries):
    path = _write_json(tmp_path, entries)
    return NvBenchBuilder(str(path), cache_root=None, mapping_path=_MAPPING_PATH).to_gold_items()


# Real nvBench-style topics: long enough, varied enough vocabulary that briefs
# built from them stay well under the 0.8 near-duplicate threshold (verified:
# max pairwise similarity ~0.79 across all 12), unlike short templated fragments
# that differ only by a trailing number (which collapse to ~0.9+ similarity).
_TOPICS = [
    ("region", "sales region", "SalesRegion"),
    ("product", "product catalog entry", "ProductCatalog"),
    ("employee", "staff member record", "StaffMember"),
    ("branch", "retail branch office", "BranchOffice"),
    ("supplier", "vendor supply contract", "SupplyContract"),
    ("customer", "client account profile", "AccountProfile"),
    ("warehouse", "storage warehouse facility", "WarehouseFacility"),
    ("department", "academic department unit", "DepartmentUnit"),
    ("flight", "airline flight schedule", "FlightSchedule"),
    ("patient", "hospital patient chart", "PatientChart"),
    ("vehicle", "fleet vehicle inspection", "VehicleInspection"),
    ("course", "university course catalog", "CourseCatalog"),
]


def _diverse_entry(i, chart, db_id):
    _key, phrase, colname = _TOPICS[i % len(_TOPICS)]
    goal = f"How many {phrase} entries were recorded last quarter across every division ({i})?"
    return _entry(chart, [goal], db_id=db_id, x=colname, y=f"count({colname}_id)")


def test_select_pilot_v3_chart_balance_and_db_cap(tmp_path):
    entries = {}
    for i in range(12):
        entries[f"bar{i}"] = _diverse_entry(i, "Bar", db_id=f"db{i % 4}")
    for i in range(12):
        entries[f"pie{i}"] = _diverse_entry(i, "Pie", db_id=f"db{i % 4}")
    items = _accepted(tmp_path, entries)
    selected, report = select_pilot_v3(items, seed=42, target_per_chart=4, db_cap=2)

    from src.data_pipeline.nvbench_source import item_chart
    chart_counts = collections.Counter(item_chart(it) for it in selected)
    assert chart_counts == {"bar": 4, "pie": 4}
    assert report["chart_counts"] == {"bar": 4, "pie": 4}
    assert all(n <= 2 for n in report["db_counts"].values())
    assert report["fallbacks"] == []  # 4 distinct dbs available -> cap satisfiable without fallback


def test_select_pilot_v3_db_cap_fallback_when_supply_concentrated(tmp_path):
    # All 6 Bar candidates come from ONE database; db_cap=2 can supply only 2 of
    # the 4 target -> the fallback pass must relax the cap and document it.
    entries = {f"bar{i}": _diverse_entry(i, "Bar", db_id="onlydb") for i in range(6)}
    items = _accepted(tmp_path, entries)
    selected, report = select_pilot_v3(items, seed=42, target_per_chart=4, db_cap=2)
    assert report["chart_counts"]["bar"] == 4
    assert report["db_counts"]["onlydb"] == 4  # over the cap, but documented
    assert len(report["fallbacks"]) == 2
    assert all(f["reason"] == "db_cap_relaxed" for f in report["fallbacks"])


def test_select_pilot_v3_excludes_near_duplicate_during_selection(tmp_path):
    # Two distinct groups whose briefs are near-duplicates (small edit distance,
    # verified similarity ~0.92); with a target that would otherwise admit both,
    # only one may be selected.
    entries = {
        "1": _entry("Bar", ["show the total count of items by category"], db_id="db1"),
        "2": _entry("Bar", ["show the total count of items in category"], db_id="db2"),
    }
    items = _accepted(tmp_path, entries)
    selected, report = select_pilot_v3(items, seed=42, target_per_chart=2, db_cap=10)
    assert report["chart_counts"]["bar"] == 1  # the near-duplicate was skipped, not admitted
    assert len(selected) == 1


def test_select_pilot_v3_drops_exact_goal_duplicate_clusters(tmp_path):
    # Two different source groups sharing the exact same normalized goal text.
    entries = {
        "1": _entry("Bar", ["identical goal text here"], db_id="db1"),
        "2": _entry("Pie", ["identical goal text here"], db_id="db2"),
    }
    items = _accepted(tmp_path, entries)
    selected, report = select_pilot_v3(items, seed=42, target_per_chart=2, db_cap=10)
    assert report["dropped_exact_goal_duplicates"] == 1
    assert len(selected) == 1


def test_select_pilot_v3_deterministic(tmp_path):
    entries = {f"bar{i}": _diverse_entry(i, "Bar", db_id=f"db{i % 3}") for i in range(10)}
    items = _accepted(tmp_path, entries)
    a, _ = select_pilot_v3(items, seed=42, target_per_chart=3, db_cap=2)
    b, _ = select_pilot_v3(items, seed=42, target_per_chart=3, db_cap=2)
    assert [it.item_id for it in a] == [it.item_id for it in b]
