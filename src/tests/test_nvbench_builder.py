"""Tests for the functional NvBenchBuilder and its pure mapping helpers."""

import json
import sqlite3
from pathlib import Path

import pytest

from src.core.schemas import ChartType, TaskType
from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder
from src.data_pipeline.nvbench_source import (
    DbMetadataResolver,
    RejectedRecord,
    build_gold_item,
    group_split,
    load_mapping,
    map_chart,
    parse_aggregate,
    select_one_per_group,
    source_group_id,
    source_record_id,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAPPING_PATH = _REPO_ROOT / "src" / "config" / "data" / "nvbench_mapping.yaml"


@pytest.fixture(scope="module")
def mapping():
    return load_mapping(_MAPPING_PATH)


def _record(chart="Bar", db_id="db1", x="cat", y="count(*)", classify=None, nl=None):
    # GROUP BY includes a second field ("series_field") so grouped charts (Stacked
    # Bar / Grouping Line / Grouping Scatter) can recover a group field; harmless
    # for non-grouped charts, which never call recover_group_field.
    sql = f"SELECT {x} , {y} FROM t GROUP BY {x} , series_field"
    return {
        "chart": chart,
        "db_id": db_id,
        "hardness": "Easy",
        "vis_query": {"vis_part": f"Visualize {chart.upper()}", "data_part": {"sql_part": sql, "binning": ""}, "VQL": sql},
        "vis_obj": {
            "chart": chart.lower(),
            "x_name": x,
            "y_name": y,
            "x_data": [[1, 2, 3]],
            "y_data": [[1, 2, 3]],
            "classify": classify or [],
            "describe": "GROUP BY x",
        },
        "nl_queries": nl or ["show me the data", "render a chart"],
    }


# --------------------------------------------------------------------------- #
# chart mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "label,expected,grouped",
    [
        ("Bar", ChartType.BAR, False),
        ("Pie", ChartType.PIE, False),
        ("Line", ChartType.LINE, False),
        ("Scatter", ChartType.SCATTER, False),
        ("Stacked Bar", ChartType.STACKED_BAR, True),
        ("Grouping Line", ChartType.LINE, True),
        ("Grouping Scatter", ChartType.SCATTER, True),
    ],
)
def test_all_supported_chart_labels(mapping, label, expected, grouped):
    result = map_chart(label, mapping)
    assert result["chart_type"] == expected
    assert result["grouped"] is grouped


def test_unsupported_chart_rejected(mapping):
    with pytest.raises(RejectedRecord) as exc:
        map_chart("Radar", mapping)
    assert exc.value.reason == "unsupported_chart"


def test_parse_aggregate():
    assert parse_aggregate("count(*)") == "COUNT"
    assert parse_aggregate("AVG(monthly_rental)") == "AVG"
    assert parse_aggregate("plain_column") is None


# --------------------------------------------------------------------------- #
# stable ids + group-aware split
# --------------------------------------------------------------------------- #
def test_stable_ids(mapping):
    resolver = DbMetadataResolver(None)
    item = build_gold_item("42", _record(), 1, "some query", mapping, resolver)
    assert item.item_id == "nvbench:42:query:1"
    assert source_group_id("42") == "nvbench:42"
    assert source_record_id("42", 1) == "nvbench:42:query:1"
    prov = item.brief.extra["provenance"]
    assert prov["source_group_id"] == "nvbench:42"
    assert prov["source_record_id"] == "nvbench:42:query:1"


def test_group_aware_split_keeps_queries_together(mapping):
    resolver = DbMetadataResolver(None)
    rec = _record(nl=["q0", "q1", "q2"])
    splits = {
        build_gold_item("7", rec, qi, nl, mapping, resolver).split
        for qi, nl in enumerate(rec["nl_queries"])
    }
    assert len(splits) == 1  # all queries of one key share a split
    assert splits.pop() in {"train", "val"}  # augmentation is never test


def test_sort_variants_share_one_group_and_split(mapping):
    resolver = DbMetadataResolver(None)
    # base + sort variants of one visualization must map to one group + split.
    keys = ["5", "5@x_name@ASC", "5@x_name@DESC", "5@y_name@ASC", "5@y_name@DESC"]
    items = [build_gold_item(k, _record(), 0, "q", mapping, resolver) for k in keys]
    groups = {it.brief.extra["provenance"]["source_group_id"] for it in items}
    splits = {it.split for it in items}
    assert groups == {"nvbench:5"}
    assert len(splits) == 1
    # full key is still preserved for traceability, record ids stay unique.
    assert {it.item_id for it in items} == {f"nvbench:{k}:query:0" for k in keys}
    assert items[1].brief.extra["provenance"]["visualization_key"] == "5@x_name@ASC"
    assert items[1].brief.extra["provenance"]["base_visualization_key"] == "5"


def test_augmentation_never_lands_in_test(mapping):
    resolver = DbMetadataResolver(None)
    for k in range(300):
        assert group_split(str(k)) in {"train", "val"}


# --------------------------------------------------------------------------- #
# provenance + field lineage + grouping preservation
# --------------------------------------------------------------------------- #
def test_source_provenance_preserved(mapping):
    resolver = DbMetadataResolver(None)
    rec = _record(chart="Pie", db_id="activity_1", x="Rank", y="count(*)")
    item = build_gold_item("3", rec, 0, "a pie of ranks", mapping, resolver)
    prov = item.brief.extra["provenance"]
    assert prov["original_chart_label"] == "Pie"
    assert prov["db_id"] == "activity_1"
    assert prov["nl_query"] == "a pie of ranks"
    assert prov["vis_query"] == rec["vis_query"]
    assert prov["vis_obj"]["x_name"] == "Rank"


def test_field_lineage(mapping):
    resolver = DbMetadataResolver(None)
    item = build_gold_item("3", _record(), 0, "q", mapping, resolver)
    lineage = item.brief.extra["lineage"]
    assert lineage["chart_type"] == "source-provided"
    assert lineage["encoding"] == "source-provided"
    assert lineage["task_type"] == "rule-derived"
    assert lineage["layout"] == "template-derived"
    # no field is LLM-generated in this task
    assert "LLM-generated" not in lineage.values()


def test_task_inference_is_rule_derived(mapping):
    resolver = DbMetadataResolver(None)
    item = build_gold_item("3", _record(chart="Line"), 0, "q", mapping, resolver)
    task = item.brief.extra["task_inference"]
    assert task["derivation_status"] == "rule-derived"
    assert task["rule_version"] == "v1"
    assert 0.0 <= task["confidence"] <= 1.0
    assert task["evidence"]
    assert item.recommendation.kpi_chart_mapping[0].task_type == TaskType.TREND


def test_grouping_classify_preserved(mapping):
    resolver = DbMetadataResolver(None)
    rec = _record(chart="Grouping Line", classify=["apartment", "house"])
    item = build_gold_item("241", rec, 0, "grouped line", mapping, resolver)
    enc = item.recommendation.kpi_chart_mapping[0].encoding
    assert enc["grouped"] is True
    assert enc["classify"] == ["apartment", "house"]
    assert item.brief.extra["provenance"]["grouping"]["classify"] == ["apartment", "house"]
    assert item.recommendation.kpi_chart_mapping[0].chart_type == ChartType.LINE


# --------------------------------------------------------------------------- #
# database metadata lookup + missing cache handling
# --------------------------------------------------------------------------- #
def test_db_metadata_lookup(tmp_path, mapping):
    cache_root = tmp_path / "databases"
    cache_root.mkdir()
    db = cache_root / "salesdb.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE orders (order_date DATE, region TEXT, amount INTEGER)")
    con.commit()
    con.close()

    resolver = DbMetadataResolver(cache_root)
    assert resolver.available is True
    assert resolver.dtype_of("salesdb", "order_date") == "datetime"
    assert resolver.dtype_of("salesdb", "region") == "categorical"
    assert resolver.dtype_of("salesdb", "amount") == "number"

    item = build_gold_item(
        "1", _record(db_id="salesdb", x="region", y="sum(amount)"), 0, "q", mapping, resolver
    )
    x_col = item.brief.columns[0]
    assert x_col["name"] == "region"
    assert x_col["dtype"] == "categorical"
    assert x_col["role"] == "dimension"
    # sum(amount) is a derived expression: brief.columns holds the raw base field
    # "amount", never the aggregate expression itself.
    y_col = item.brief.columns[1]
    assert y_col["name"] == "amount"
    assert y_col["dtype"] == "number" and y_col["role"] == "measure"
    lineage = item.brief.extra["lineage"]
    assert lineage["raw_columns"]["region"] == "source-provided(db)"
    assert lineage["raw_columns"]["amount"] == "source-provided(db)"
    assert lineage["derived_expressions"]["sum(amount)"] == "aggregate-expression"


def test_missing_cache_falls_back_to_heuristic(tmp_path, mapping):
    resolver = DbMetadataResolver(tmp_path / "does_not_exist")
    assert resolver.available is False
    assert resolver.dtype_of("x", "y") is None
    item = build_gold_item(
        "1", _record(db_id="x", x="order_year", y="count(*)"), 0, "q", mapping, resolver
    )
    assert item.brief.columns[0]["dtype"] == "datetime"  # name heuristic
    assert item.brief.extra["lineage"]["raw_columns"]["order_year"] == "heuristic"
    # count(*) has no physical base column -> not added, not invented.
    assert "*" not in item.brief.extra["lineage"]["raw_columns"]
    assert len(item.brief.columns) == 1


def test_resolver_none_cache():
    resolver = DbMetadataResolver(None)
    assert resolver.available is False
    assert resolver.dtype_of("x", "y") is None


# --------------------------------------------------------------------------- #
# axis typing / KPI policy / grouping recovery regressions
# --------------------------------------------------------------------------- #
def _db(tmp_path):
    cache_root = tmp_path / "databases"
    cache_root.mkdir()
    con = sqlite3.connect(cache_root / "shop.sqlite")
    con.execute(
        "CREATE TABLE t (product_price REAL, account_name TEXT, staff_gender TEXT, "
        "company_id INTEGER, other_details TEXT, order_date DATE)"
    )
    con.commit()
    con.close()
    return DbMetadataResolver(cache_root)


def _rec2(chart="Bar", x="x", y="y", classify=None, sql="SELECT 1", db_id="shop"):
    return {
        "chart": chart, "db_id": db_id, "hardness": "Easy",
        "vis_query": {"vis_part": "V", "data_part": {"sql_part": sql, "binning": ""}, "VQL": sql},
        "vis_obj": {"chart": chart.lower(), "x_name": x, "y_name": y,
                    "x_data": [[1]], "y_data": [[1]], "classify": classify or [], "describe": ""},
        "nl_queries": ["q"],
    }


def _axes(item):
    at = item.brief.extra["provenance"]["axis_typing"]
    return at["x"], at["y"]


def test_aggregate_axis_always_number_measure(tmp_path, mapping):
    resolver = _db(tmp_path)
    for expr in ("max(product_price)", "min(product_price)", "count(*)", "SUM(company_id)", "AVG(product_price)"):
        item = build_gold_item("1", _rec2(chart="Bar", x="account_name", y=expr), 0, "q", mapping, resolver)
        _x, y = _axes(item)
        assert y["dtype"] == "number", expr
        assert y["role"] == "measure", expr
        assert y["aggregate"] is not None, expr
        assert y["dtype_source"] == "aggregate-expression"
        # original expression preserved
        assert item.recommendation.kpi_chart_mapping[0].encoding["y"] == expr


def test_non_numeric_y_not_forced_numeric(tmp_path, mapping):
    resolver = _db(tmp_path)
    # account_name on the y-axis of a bar chart must NOT become numeric.
    item = build_gold_item("1", _rec2(chart="Bar", x="company_id", y="account_name"), 0, "q", mapping, resolver)
    _x, y = _axes(item)
    assert y["dtype"] == "categorical"
    assert y["role"] == "dimension"
    assert y["dtype_source"] == "source-provided(db)"


def test_scatter_numeric_axes_typed_independently(tmp_path, mapping):
    resolver = _db(tmp_path)
    item = build_gold_item("1", _rec2(chart="Scatter", x="company_id", y="product_price"), 0, "q", mapping, resolver)
    x, y = _axes(item)
    assert x["dtype"] == "number" and x["role"] == "measure"
    assert y["dtype"] == "number" and y["role"] == "measure"
    assert not item.brief.extra["provenance"]["build_warnings"]


def test_scatter_categorical_axes_rejected(tmp_path, mapping):
    # v3: a categorical scatter axis is rejected outright (not just warned) —
    # unsuitable as positive dashboard-design training data.
    resolver = _db(tmp_path)
    with pytest.raises(RejectedRecord) as exc:
        build_gold_item("1", _rec2(chart="Scatter", x="account_name", y="staff_gender"), 0, "q", mapping, resolver)
    assert exc.value.reason == "categorical_scatter_axis"
    assert "x=account_name" in exc.value.detail and "y=staff_gender" in exc.value.detail


def test_aggregate_on_x_becomes_kpi(tmp_path, mapping):
    resolver = _db(tmp_path)
    item = build_gold_item("1", _rec2(chart="Bar", x="count(*)", y="account_name"), 0, "q", mapping, resolver)
    sel = item.brief.extra["provenance"]["kpi_selection"]
    assert sel["primary_axis"] == "x"
    assert sel["primary_kpi"] == "count(*)"
    assert item.recommendation.kpi_chart_mapping[0].kpi == "count(*)"
    assert "count(*)" in item.brief.kpis


def test_aggregate_on_y_becomes_kpi(tmp_path, mapping):
    resolver = _db(tmp_path)
    item = build_gold_item("1", _rec2(chart="Bar", x="account_name", y="count(*)"), 0, "q", mapping, resolver)
    sel = item.brief.extra["provenance"]["kpi_selection"]
    assert sel["primary_axis"] == "y" and sel["primary_kpi"] == "count(*)"


def test_aggregate_on_both_axes_preserved(tmp_path, mapping):
    resolver = _db(tmp_path)
    item = build_gold_item("1", _rec2(chart="Scatter", x="count(*)", y="sum(product_price)"), 0, "q", mapping, resolver)
    sel = item.brief.extra["provenance"]["kpi_selection"]
    assert set(sel["aggregate_axes"]) == {"x", "y"}
    assert sel["primary_axis"] == "y"
    assert set(item.brief.kpis) == {"count(*)", "sum(product_price)"}


def test_grouping_recovery_success(tmp_path, mapping):
    resolver = _db(tmp_path)
    sql = ("SELECT order_date , AVG(product_price) FROM t "
           "GROUP BY other_details , order_date ORDER BY product_price DESC")
    item = build_gold_item("1", _rec2(chart="Grouping Line", x="order_date", y="avg(product_price)",
                                      classify=["a", "b"], sql=sql), 0, "q", mapping, resolver)
    g = item.brief.extra["provenance"]["grouping"]
    assert g["recovery_status"] == "recovered"
    assert g["series_field"] == "other_details"
    col_names = {c["name"] for c in item.brief.columns}
    assert "other_details" in col_names
    assert item.recommendation.kpi_chart_mapping[0].encoding["group_field"] == "other_details"
    assert g["classify"] == ["a", "b"]


def test_grouping_recovery_ambiguous_rejected(tmp_path, mapping):
    # v3: an unrecoverable group field is mandatory-reject, not a warning —
    # Stacked Bar / Grouping Line / Grouping Scatter always require group_field.
    resolver = _db(tmp_path)
    sql = ("SELECT order_date , AVG(product_price) FROM t "
           "GROUP BY other_details , staff_gender , order_date")  # two extra -> ambiguous
    with pytest.raises(RejectedRecord) as exc:
        build_gold_item("1", _rec2(chart="Grouping Line", x="order_date", y="avg(product_price)",
                                   classify=["a"], sql=sql), 0, "q", mapping, resolver)
    assert exc.value.reason == "missing_group_field"
    assert "ambiguous" in exc.value.detail


# --------------------------------------------------------------------------- #
# builder end-to-end + determinism + selection
# --------------------------------------------------------------------------- #
def _write_nvbench(tmp_path):
    data = {
        "1": _record(chart="Bar", nl=["q1a", "q1b"]),
        "2": _record(chart="Pie", nl=["q2a"]),
        "3": _record(chart="Radar", nl=["bad"]),  # unsupported -> rejected
        # Stacked Bar (not Scatter): grouped, but does not require numeric axes,
        # so the default categorical x="cat" fixture stays valid.
        "4": _record(chart="Stacked Bar", classify=["A", "B"], nl=["q4a", "q4b"]),
    }
    path = tmp_path / "NVBench.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_builder_accept_reject_counts(tmp_path):
    path = _write_nvbench(tmp_path)
    result = NvBenchBuilder(path, cache_root=None, mapping_path=_MAPPING_PATH).build()
    # accepted queries: 2 (key1) + 1 (key2) + 2 (key4) = 5; rejected: 1 (key3)
    assert result.stats["n_accepted"] == 5
    assert result.stats["n_rejected"] == 1
    assert result.rejections[0]["reason"] == "unsupported_chart"
    assert result.rejections[0]["original_chart_label"] == "Radar"
    assert result.stats["chart_distribution"]["bar"] == 2


def test_builder_deterministic(tmp_path):
    path = _write_nvbench(tmp_path)
    a = NvBenchBuilder(path, mapping_path=_MAPPING_PATH).to_gold_items()
    b = NvBenchBuilder(path, mapping_path=_MAPPING_PATH).to_gold_items()
    assert [it.item_id for it in a] == [it.item_id for it in b]
    assert [it.split for it in a] == [it.split for it in b]


def test_one_query_per_group(tmp_path):
    path = _write_nvbench(tmp_path)
    items = NvBenchBuilder(path, mapping_path=_MAPPING_PATH).to_gold_items()
    picked = select_one_per_group(items, seed=42)
    groups = {it.brief.extra["provenance"]["source_group_id"] for it in picked}
    # one per accepted group (keys 1, 2, 4)
    assert len(picked) == len(groups) == 3
    # stable for a fixed seed
    picked2 = select_one_per_group(items, seed=42)
    assert [it.item_id for it in picked] == [it.item_id for it in picked2]
