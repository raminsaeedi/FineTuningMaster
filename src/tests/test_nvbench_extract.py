"""Tests for deterministic nvBench source-clause extractors.

All fixtures are small literal strings mirroring corpus-verified patterns; no real
dataset or cache access.
"""

from src.data_pipeline.nvbench_extract import (
    check_aggregate_intent_conflict,
    check_chart_query_conflict,
    detect_query_intent,
    extract_base_field,
    extract_group_by_fields,
    extract_nested,
    extract_order_by,
    extract_select_aggregates,
    extract_time_grain,
    extract_time_grain_signals,
    extract_where_filters,
    resolve_nested,
)


# --------------------------------------------------------------------------- #
# extract_select_aggregates (v5)
# --------------------------------------------------------------------------- #
def test_select_aggregates_basic_order_and_base():
    aggs = extract_select_aggregates("SELECT max(product_price) , min(product_price) , product_type_code FROM products")
    assert [(a["func"], a["base_field"]) for a in aggs] == [("MAX", "product_price"), ("MIN", "product_price")]
    assert [a["position"] for a in aggs] == [0, 1]


def test_select_aggregates_strips_alias_and_quotes():
    aggs = extract_select_aggregates('SELECT SUM(T2.order_quantity) , avg("salary") FROM t')
    assert aggs[0] == {"func": "SUM", "base_field": "order_quantity", "position": 0}
    assert aggs[1]["func"] == "AVG" and aggs[1]["base_field"] == "salary"


def test_select_aggregates_count_star_has_no_base():
    aggs = extract_select_aggregates("SELECT year , count(*) FROM postseason GROUP BY year")
    assert aggs == [{"func": "COUNT", "base_field": None, "position": 1}]


def test_select_aggregates_as_alias_stripped():
    aggs = extract_select_aggregates("SELECT SUM(x) AS total FROM t")
    assert aggs == [{"func": "SUM", "base_field": "x", "position": 0}]


def test_select_aggregates_none_when_no_select():
    assert extract_select_aggregates("UPDATE t SET x=1") == []


# --------------------------------------------------------------------------- #
# extract_time_grain_signals (v5)
# --------------------------------------------------------------------------- #
def test_time_grain_signal_strftime_year():
    sigs = extract_time_grain_signals("SELECT strftime('%Y', order_date) , count(*) FROM orders")
    assert {"field": "order_date", "grain": "YEAR", "source": "strftime"} in sigs


def test_time_grain_signal_strftime_month_and_weekday():
    sigs = extract_time_grain_signals("SELECT strftime('%m', d), strftime('%w', d) FROM t")
    grains = {s["grain"] for s in sigs}
    assert grains == {"MONTH", "WEEKDAY"}


def test_time_grain_signal_named_function():
    sigs = extract_time_grain_signals("SELECT YEAR(created_at) FROM t")
    assert sigs == [{"field": "created_at", "grain": "YEAR", "source": "named_function"}]


def test_time_grain_signal_extract():
    sigs = extract_time_grain_signals("SELECT EXTRACT(MONTH FROM ts) FROM t")
    assert {"field": "ts", "grain": "MONTH", "source": "extract"} in sigs


def test_time_grain_signal_none_when_absent():
    assert extract_time_grain_signals("SELECT city , count(*) FROM t GROUP BY city") == []


# --------------------------------------------------------------------------- #
# base field / nested aggregate
# --------------------------------------------------------------------------- #
def test_extract_base_field_simple():
    assert extract_base_field("SUM(order_quantity)") == "order_quantity"
    assert extract_base_field("AVG(monthly_rental)") == "monthly_rental"
    assert extract_base_field("COUNT(product_name)") == "product_name"
    assert extract_base_field("count(T1.product_name)") == "product_name"


def test_extract_base_field_count_star_returns_none():
    assert extract_base_field("COUNT(*)") is None
    assert extract_base_field("count( * )") is None


def test_extract_base_field_non_aggregate_returns_none():
    assert extract_base_field("product_name") is None
    assert extract_base_field("") is None


def test_extract_nested_detects_and_ignores_plain():
    assert extract_nested("SUM(sum(T2.order_quantity))") == {"outer": "SUM", "inner": "SUM", "arg": "T2.order_quantity"}
    assert extract_nested("AVG(sum(T2.order_quantity))") == {"outer": "AVG", "inner": "SUM", "arg": "T2.order_quantity"}
    assert extract_nested("SUM(count(*))") == {"outer": "SUM", "inner": "COUNT", "arg": "*"}
    assert extract_nested("SUM(order_quantity)") is None
    assert extract_nested("product_name") is None


def test_resolve_nested_same_outer_inner_normalizes():
    nested = extract_nested("SUM(sum(T2.order_quantity))")
    result = resolve_nested(nested, ["Show the product name and total order quantity for each product in a bar chart."])
    assert result == {"action": "normalize", "collapsed": "SUM(T2.order_quantity)"}


def test_resolve_nested_intent_conflict_rejects():
    nested = extract_nested("AVG(sum(T2.order_quantity))")
    result = resolve_nested(nested, ["Show the product name and total order quantity for each product with a bar chart."])
    assert result["action"] == "reject"
    assert result["reason"] == "aggregate_intent_conflict"


def test_resolve_nested_count_star_arg_ambiguous():
    nested = extract_nested("SUM(count(*))")
    result = resolve_nested(nested, ["Return the total number of times the team won."])
    assert result["action"] == "reject"
    assert result["reason"] == "nested_aggregate_ambiguous"


# --------------------------------------------------------------------------- #
# query intent + narrow conflict rules
# --------------------------------------------------------------------------- #
def test_detect_query_intent_total_only():
    assert detect_query_intent(["Show the product name and total order quantity for each product."]) == "SUM"


def test_detect_query_intent_total_minimal_is_min_not_sum():
    # "total minimal" is a template artifact meaning MIN, not SUM.
    assert detect_query_intent(["Compute the total minimal stu gpa across dept code as a pie chart."]) == "MIN"


def test_detect_query_intent_count_number_of_gated():
    assert detect_query_intent(["Show total number of shop id from each start from."], allow_count_number_of=True) == "COUNT"
    # "total number of" must never be mistaken for a plain SUM signal even when
    # the COUNT family itself is gated off for the caller's purposes.
    assert detect_query_intent(["Show total number of shop id from each start from."], allow_count_number_of=False) is None


def test_aggregate_intent_conflict_rule_a_fires():
    detail = check_aggregate_intent_conflict(
        "AVG(sum(T2.order_quantity))",
        ["Show the product name and total order quantity for each product with a bar chart."],
        resolver=None, db_id="shop",
    )
    assert detail is not None and "conflicts with encoded function=AVG" in detail


def test_aggregate_intent_conflict_rule_a_does_not_fire_on_template_idiom():
    # Regression: "total minimal X" must NOT be flagged as a SUM conflict.
    detail = check_aggregate_intent_conflict(
        "min(stu_gpa)",
        ["Compute the total minimal stu gpa across dept code as a pie chart."],
        resolver=None, db_id="db1",
    )
    assert detail is None


def test_aggregate_intent_conflict_rule_b_fires_on_id_shaped_sum():
    detail = check_aggregate_intent_conflict(
        "SUM(Shop_ID)",
        ["Stacked bar chart of total number of shop id for with each Is_full_time in each start from."],
        resolver=None, db_id="employee_hire_evaluation",
    )
    assert detail is not None and "id-shaped field" in detail


def test_aggregate_intent_conflict_none_for_ordinary_count():
    assert check_aggregate_intent_conflict(
        "COUNT(*)", ["Show the number of debates in each venue."], resolver=None, db_id="debate"
    ) is None


def test_chart_query_conflict_fires():
    detail = check_chart_query_conflict("Pie", ["A bar chart showing the number of debates in each venue."])
    assert detail is not None and "Bar" in detail


def test_chart_query_conflict_none_when_consistent():
    assert check_chart_query_conflict("Pie", ["A pie chart showing the number of debates in each venue."]) is None


def test_chart_query_conflict_stacked_bar_allows_bare_bar_mention():
    assert check_chart_query_conflict("Stacked Bar", ["Draw a bar chart about the distribution of X."]) is None


# --------------------------------------------------------------------------- #
# WHERE filter extraction
# --------------------------------------------------------------------------- #
def test_where_simple_condition():
    r = extract_where_filters('SELECT * FROM t WHERE rank = "AsstProf" GROUP BY sex')
    assert r["status"] == "ok"
    assert r["filters"] == [{"field": "rank", "operator": "=", "value": "AsstProf"}]


def test_where_and_multiple_conditions():
    r = extract_where_filters("SELECT * FROM t WHERE age > 30 AND dept = 'CS' ORDER BY age")
    assert r["status"] == "ok"
    assert r["filters"] == [
        {"field": "age", "operator": ">", "value": "30"},
        {"field": "dept", "operator": "=", "value": "CS"},
    ]


def test_where_or_unrepresentable():
    r = extract_where_filters("SELECT * FROM t WHERE grade = 'C' OR grade = 'A'")
    assert r["status"] == "unrepresentable"


def test_where_subquery_unrepresentable():
    r = extract_where_filters(
        "SELECT * FROM t WHERE StuID NOT IN (SELECT StuID FROM allergy WHERE type = 'food')"
    )
    assert r["status"] == "unrepresentable"


def test_where_no_clause():
    assert extract_where_filters("SELECT * FROM t GROUP BY x") == {"filters": [], "status": "none"}


def test_group_by_fields():
    assert extract_group_by_fields("SELECT x FROM t GROUP BY Is_full_time , other_details ORDER BY x") == \
        ["Is_full_time", "other_details"]
    assert extract_group_by_fields("SELECT x FROM t") == []


# --------------------------------------------------------------------------- #
# ORDER BY extraction
# --------------------------------------------------------------------------- #
def test_order_by_ascending():
    r = extract_order_by("SELECT x FROM t ORDER BY x ASC")
    assert r == {"field": "x", "expression": "x", "direction": "asc", "status": "ok"}


def test_order_by_descending():
    r = extract_order_by("SELECT x FROM t ORDER BY SUM(Shop_ID) DESC")
    assert r["status"] == "ok" and r["direction"] == "desc" and r["field"] == "SUM(Shop_ID)"


def test_order_by_default_direction_when_omitted():
    r = extract_order_by("SELECT x FROM t ORDER BY x")
    assert r["status"] == "ok" and r["direction"] == "asc"


def test_order_by_multi_key_unrepresentable():
    r = extract_order_by("SELECT x FROM t ORDER BY x, y DESC")
    assert r["status"] == "unrepresentable"


def test_order_by_none():
    assert extract_order_by("SELECT x FROM t")["status"] == "none"


# --------------------------------------------------------------------------- #
# BIN / time grain extraction
# --------------------------------------------------------------------------- #
def test_extract_time_grain_year():
    assert extract_time_grain("BIN order_date BY YEAR") == {"field": "order_date", "grain": "YEAR"}


def test_extract_time_grain_weekday():
    assert extract_time_grain("BIN Start_from BY WEEKDAY") == {"field": "Start_from", "grain": "WEEKDAY"}


def test_extract_time_grain_none():
    assert extract_time_grain("") is None
    assert extract_time_grain("GROUP BY x") is None
