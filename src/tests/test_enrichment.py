"""Tests for the Phase-3 enrichment contract (offline, no API call)."""

from __future__ import annotations

import copy

import pytest

from src.data_pipeline.enrichment import (
    ENRICHABLE_FIELDS,
    LINEAGE_ENRICHED_VALUE,
    SYSTEM_PROMPT,
    immutable_diff,
    immutable_fingerprint,
    merge_enrichment,
    normalize_structure,
    parse_payload,
    select_records,
    selection_summary,
    source_columns,
    validate_payload,
)


def _record(item_id="nvbench:1@k:query:1", split="train", chart="bar", task="comparison",
            group="nvbench:1") -> dict:
    return {
        "item_id": item_id,
        "split": split,
        "brief": {
            "item_id": item_id,
            "users": "Data analyst exploring a relational database",
            "goals": ["Bar chart x axis product name y axis how many product name."],
            "kpis": ["COUNT(product_name)"],
            "columns": [{"name": "product_name", "dtype": "categorical", "role": "dimension"}],
            "constraints": None,
            "extra": {
                "source": "nvbench",
                "usage_tier": "train_aug",
                "provenance": {"db_id": "customers_and_products_contacts", "source_group_id": group,
                               "source_record_id": item_id},
                "task_inference": {"task_type": task, "confidence": 0.55},
                "lineage": {"chart_type": "source-provided", "kpi": "source-provided",
                            "layout": "template-derived", "styling": "template-derived",
                            "interactions": "template-derived", "rationales": "template-derived"},
            },
        },
        "recommendation": {
            "context_summary": {"db_id": "customers_and_products_contacts", "source": "nvbench", "n_kpis": 1},
            "kpi_chart_mapping": [{
                "kpi": "COUNT(product_name)",
                "task_type": task,
                "chart_type": chart,
                "alternatives": [],
                "encoding": {"x": "product_name", "y": "COUNT(product_name)", "aggregate": "COUNT",
                             "grouped": False, "filters": [], "limit": None, "time_grain": None},
            }],
            "layout": {"type": "single", "blocks": [{"kpi": "COUNT(product_name)", "chart": chart}]},
            "styling": {"theme": "minimal"},
            "interactions": [{"type": "tooltip", "fields": ["product_name"]}],
            "rationales": [{"claim": "bar chart from source label", "principle": "source label"}],
        },
    }


def _payload_obj(**overrides) -> dict:
    obj = {
        "users": "Procurement analyst reviewing product counts in the products table.",
        "context_summary": {"db_id": "customers_and_products_contacts", "source": "nvbench", "n_kpis": 1,
                            "data_scope": "product records grouped by product name"},
        "layout": {"type": "single", "blocks": [{"kpi": "COUNT(product_name)", "chart": "bar"}]},
        "styling": {"theme": "minimal", "emphasis": "ordered categories"},
        "interactions": [{"type": "tooltip", "fields": ["product_name", "COUNT(product_name)"]},
                         {"type": "sort", "fields": ["COUNT(product_name)"]}],
        "rationales": [
            {"claim": "A bar chart supports comparison of COUNT(product_name) across product_name.",
             "principle": "position encoding on a common scale (Cleveland & McGill)"},
        ],
    }
    obj.update(overrides)
    return obj


def _accepted(record, obj):
    payload, codes, details = parse_payload(obj)
    assert payload is not None, (codes, details)
    return payload


# --------------------------------------------------------------- schema


def test_valid_payload_passes_all_content_rules():
    record = _record()
    payload = _accepted(record, _payload_obj())
    codes, details = validate_payload(record, payload)
    assert codes == [], details


def test_extra_field_in_reply_is_rejected():
    payload, codes, _ = parse_payload({**_payload_obj(), "kpi_chart_mapping": []})
    assert payload is None
    assert codes == ["extra_fields_returned"]


def test_missing_required_field_is_schema_invalid():
    obj = _payload_obj()
    obj.pop("users")
    payload, codes, _ = parse_payload(obj)
    assert payload is None
    assert codes == ["schema_invalid"]


# ------------------------------------------------------------- content


def test_interaction_field_outside_source_columns_is_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(
        interactions=[{"type": "filter", "fields": ["customer_segment"]}]))
    codes, _ = validate_payload(record, payload)
    assert "interaction_field_not_in_source" in codes


def test_unsupported_interaction_verb_is_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(
        interactions=[{"type": "predict_future", "fields": ["product_name"]}]))
    codes, _ = validate_payload(record, payload)
    assert "unsupported_interaction_type" in codes


def test_comparative_mention_is_allowed_but_a_proposal_is_not():
    record = _record()
    comparison = _accepted(record, _payload_obj(
        rationales=[{"claim": "The selected chart beats a pie chart for comparison of "
                              "COUNT(product_name) across product_name.",
                     "principle": "position over angle"}]))
    assert "invented_chart_type" not in validate_payload(record, comparison)[0]

    proposal = _accepted(record, _payload_obj(
        rationales=[{"claim": "Switch to a pie chart for comparison of COUNT(product_name) "
                              "across product_name.",
                     "principle": "part-to-whole"}]))
    assert "invented_chart_type" in validate_payload(record, proposal)[0]


def test_plain_english_use_of_a_chart_word_is_not_a_violation():
    record = _record()
    payload = _accepted(record, _payload_obj(
        users="Analyst reading the products table and the line of business context.",
        styling={"theme": "minimal", "note": "keep the plot area uncluttered"}))
    codes, details = validate_payload(record, payload)
    assert "invented_chart_type" not in codes, details


def test_foreign_chart_word_as_a_field_value_is_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(styling={"theme": "minimal", "fallback_view": "table"}))
    codes, _ = validate_payload(record, payload)
    assert "invented_chart_type" in codes


def test_layout_block_with_foreign_kpi_is_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(
        layout={"type": "single", "blocks": [{"kpi": "SUM(revenue)", "chart": "bar"}]}))
    codes, _ = validate_payload(record, payload)
    assert "invented_kpi" in codes


def test_invented_number_is_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(
        styling={"theme": "minimal", "target": "keep counts above 250 units"}))
    codes, _ = validate_payload(record, payload)
    assert "invented_numeric_value" in codes


def test_context_summary_conflict_is_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(
        context_summary={"db_id": "some_other_db", "source": "nvbench", "n_kpis": 1}))
    codes, _ = validate_payload(record, payload)
    assert "context_summary_conflicts_source" in codes


def test_rationale_missing_task_or_encoding_is_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(
        rationales=[{"claim": "A bar chart looks clean.", "principle": "aesthetics"}]))
    codes, _ = validate_payload(record, payload)
    assert "rationale_disagrees_with_task_chart_or_encoding" in codes


def test_empty_enrichment_field_is_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(interactions=[]))
    codes, _ = validate_payload(record, payload)
    assert "empty_enrichment_field" in codes


# ------------------------------- regressions from the first sample run
# Every case below is taken verbatim from data/staging/enrichment/sample_10.


def test_layout_returned_as_list_is_normalized_not_rejected():
    """nvbench:2516, 2477, 94, 2871 all returned layout as a list of blocks."""
    obj = _payload_obj(layout=[{"type": "kpi", "metric": "COUNT(product_name)"},
                               {"type": "chart", "chart_type": "bar", "position": "bottom"}])
    payload, codes, notes = parse_payload(obj)
    assert payload is not None, codes
    assert payload.layout["blocks"][0]["metric"] == "COUNT(product_name)"
    assert any("normalized layout" in n for n in notes)


def test_normalization_moves_values_without_inventing_any():
    obj = {"layout": [{"a": 1}], "interactions": {"type": "tooltip"}, "users": ["A", "B"]}
    normalized, notes = normalize_structure(obj)
    assert normalized["layout"] == {"blocks": [{"a": 1}]}
    assert normalized["interactions"] == [{"type": "tooltip"}]
    assert normalized["users"] == "A; B"
    assert len(notes) == 3


def test_extra_top_level_field_still_rejected_after_normalization():
    payload, codes, _ = parse_payload({**_payload_obj(), "kpi_chart_mapping": [{"kpi": "x"}]})
    assert payload is None
    assert codes == ["extra_fields_returned"]


def test_stacked_bar_record_may_call_itself_a_stacked_bar_chart():
    """nvbench:134, 1391: 'stacked bar chart' matched the foreign word 'bar'."""
    record = _record(chart="stacked_bar", task="composition")
    payload = _accepted(record, _payload_obj(
        layout={"type": "single", "blocks": [{"kpi": "COUNT(product_name)", "chart": "stacked_bar"}]},
        rationales=[{"claim": "The stacked bar chart shows the composition of COUNT(product_name) "
                              "across product_name.",
                     "principle": "part-to-whole with stacked segments"}]))
    codes, details = validate_payload(record, payload)
    assert codes == [], details


def test_negative_comparison_with_another_chart_is_accepted():
    record = _record()
    payload = _accepted(record, _payload_obj(
        rationales=[{"claim": "The selected chart supports comparison of COUNT(product_name) across "
                              "product_name; a pie chart would hide the ordering.",
                     "principle": "position beats angle (Cleveland & McGill)"}]))
    codes, details = validate_payload(record, payload)
    assert "invented_chart_type" not in codes, details


def test_actual_chart_replacement_is_still_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(
        rationales=[{"claim": "Use a pie chart instead to show COUNT(product_name) for comparison.",
                     "principle": "part-to-whole"}]))
    codes, _ = validate_payload(record, payload)
    assert "invented_chart_type" in codes


@pytest.mark.parametrize("chart_value", [
    "the selected chart",       # the wording the prompt asks for (run 2: all 10 rejected on this)
    "selected chart",
    "bar",                      # the chart type verbatim
    "bar chart",
    "chart",
])
def test_layout_block_may_point_at_the_given_chart_in_any_valid_form(chart_value):
    record = _record(chart="bar")
    payload = _accepted(record, _payload_obj(
        layout={"type": "single", "blocks": [{"kpi": "COUNT(product_name)", "chart": chart_value}]}))
    codes, details = validate_payload(record, payload)
    assert codes == [], details


@pytest.mark.parametrize("chart_value", ["stacked_bar", "stacked bar", "stacked bar chart"])
def test_normalized_aliases_of_the_given_chart_are_accepted(chart_value):
    record = _record(chart="stacked_bar", task="composition")
    payload = _accepted(record, _payload_obj(
        layout={"type": "single", "blocks": [{"kpi": "COUNT(product_name)", "chart": chart_value}]},
        rationales=[{"claim": "The selected chart shows the composition of COUNT(product_name) "
                              "across product_name.",
                     "principle": "part-to-whole with stacked segments"}]))
    codes, details = validate_payload(record, payload)
    assert codes == [], details


def test_source_column_named_like_a_chart_type_is_not_a_chart_proposal():
    """nvbench:3224 (val run): the wine DB column is literally named 'Area'."""
    record = _record(chart="bar")
    record["brief"]["columns"] = [{"name": "Area", "dtype": "categorical", "role": "dimension"}]
    record["brief"]["kpis"] = ["COUNT(Area)"]
    mapping = record["recommendation"]["kpi_chart_mapping"][0]
    mapping["kpi"] = "COUNT(Area)"
    mapping["encoding"] = {"x": "Area", "y": "COUNT(Area)"}
    payload = _accepted(record, _payload_obj(
        layout={"type": "single", "blocks": [{"kpi": "COUNT(Area)", "chart": "the selected chart"}]},
        interactions=[{"type": "tooltip", "fields": ["Area", "COUNT(Area)"]},
                      {"type": "filter", "fields": ["Area"]}],
        rationales=[{"claim": "The selected chart supports comparison of COUNT(Area) across Area.",
                     "principle": "position on a common scale"}]))
    codes, details = validate_payload(record, payload)
    assert codes == [], details


def test_layout_block_naming_a_different_chart_is_rejected():
    record = _record(chart="bar")
    payload = _accepted(record, _payload_obj(
        layout={"type": "single", "blocks": [{"kpi": "COUNT(product_name)", "chart": "pie"}]}))
    codes, _ = validate_payload(record, payload)
    assert "invented_chart_type" in codes


def test_negated_alternative_mention_is_accepted():
    record = _record()
    payload = _accepted(record, _payload_obj(
        rationales=[{"claim": "Do not replace the selected chart with a pie chart; it supports "
                              "comparison of COUNT(product_name) across product_name.",
                     "principle": "position on a common scale"}]))
    codes, details = validate_payload(record, payload)
    assert "invented_chart_type" not in codes, details


def test_explicit_addition_of_another_chart_is_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(
        rationales=[{"claim": "Add a scatter plot next to the selected chart for comparison of "
                              "COUNT(product_name) across product_name.",
                     "principle": "small multiples"}]))
    codes, _ = validate_payload(record, payload)
    assert "invented_chart_type" in codes


def test_immutable_chart_field_change_is_still_rejected():
    """The payload cannot write chart_type; a tampered merge must still be caught."""
    record = _record(chart="bar")
    payload = _accepted(record, _payload_obj())
    merged = merge_enrichment(record, payload)
    merged["recommendation"]["kpi_chart_mapping"][0]["chart_type"] = "pie"
    assert immutable_fingerprint(merged) != immutable_fingerprint(record)
    assert "kpi_chart_mapping" in immutable_diff(record, merged)


def test_citation_years_are_not_business_facts():
    """nvbench:1148, 156: '(Cleveland & McGill, 1984)', '(Few, 2004)'."""
    record = _record()
    payload = _accepted(record, _payload_obj(
        rationales=[{"claim": "The selected chart supports comparison of COUNT(product_name) across "
                              "product_name.",
                     "principle": "Position on a common scale ranks first (Cleveland & McGill, 1984); "
                                  "see also Few, 2004 and Tufte, 1983."}]))
    codes, details = validate_payload(record, payload)
    assert "invented_numeric_value" not in codes, details


def test_design_numbers_and_hex_colors_are_allowed():
    """nvbench:1391: grid width/height 1..3 and '#377eb8' were flagged as facts."""
    record = _record()
    payload = _accepted(record, _payload_obj(
        layout={"type": "grid", "columns": 2, "blocks": [{"kpi": "COUNT(product_name)", "chart": "bar",
                                                          "width": 3, "height": 2}]},
        styling={"theme": "minimal", "font_size": 14, "palette": {"Mass suicide": "#377eb8"},
                 "contrast_ratio": 4.5}))
    codes, details = validate_payload(record, payload)
    assert "invented_numeric_value" not in codes, details


def test_nested_grid_coordinates_are_design_not_facts():
    """nvbench:2871: position {"x": 0, "y": 0, "width": 2} was flagged as an invented fact."""
    record = _record()
    payload = _accepted(record, _payload_obj(
        layout={"type": "grid", "blocks": [
            {"kpi": "COUNT(product_name)", "chart": "bar",
             "position": {"x": 0, "y": 1, "width": 4, "height": 3}}]}))
    codes, details = validate_payload(record, payload)
    assert "invented_numeric_value" not in codes, details


def test_invented_kpi_target_is_still_rejected():
    record = _record()
    payload = _accepted(record, _payload_obj(
        rationales=[{"claim": "The selected chart tracks COUNT(product_name) against the target of 250 "
                              "units and the top 5 products for comparison across product_name.",
                     "principle": "goal-referenced comparison"}]))
    codes, _ = validate_payload(record, payload)
    assert "invented_numeric_value" in codes


def test_concise_payload_using_selected_chart_phrase_is_accepted():
    record = _record()
    payload = _accepted(record, _payload_obj(
        users="Product analyst comparing product counts.",
        context_summary={"db_id": "customers_and_products_contacts", "source": "nvbench", "n_kpis": 1},
        layout={"type": "single", "blocks": [{"kpi": "COUNT(product_name)", "chart": "bar"}]},
        styling={"theme": "minimal"},
        interactions=[{"type": "tooltip", "fields": ["product_name", "COUNT(product_name)"]}],
        rationales=[{"claim": "The selected chart supports comparison of COUNT(product_name) across "
                              "product_name.",
                     "principle": "position on a common scale"}]))
    codes, details = validate_payload(record, payload)
    assert codes == [], details


def test_prompt_states_the_exact_schema_and_chart_rule():
    for fragment in ("layout           object", "interactions     list of objects",
                     'Refer to the visualization only as "the selected chart"',
                     "Do not name, introduce, compare or recommend alternative chart types",
                     "copy the given chart_type verbatim", "Avoid explicit numeric values"):
        assert fragment in SYSTEM_PROMPT


# ---------------------------------------------------- immutable contract


def test_merge_writes_only_enrichable_fields_and_keeps_fingerprint():
    record = _record()
    before = immutable_fingerprint(record)
    payload = _accepted(record, _payload_obj())
    merged = merge_enrichment(record, payload, {"model": "test-model"})
    assert immutable_fingerprint(merged) == before
    assert immutable_diff(record, merged) == []
    assert merged["brief"]["users"] == payload.users
    assert merged["recommendation"]["styling"] == payload.styling
    assert merged["recommendation"]["kpi_chart_mapping"] == record["recommendation"]["kpi_chart_mapping"]
    assert merged["brief"]["extra"]["lineage"]["interactions"] == LINEAGE_ENRICHED_VALUE
    assert merged["brief"]["extra"]["lineage"]["chart_type"] == "source-provided"
    assert merged["brief"]["extra"]["enrichment"] == {"model": "test-model"}


def test_source_record_is_not_mutated_by_merge():
    record = _record()
    snapshot = copy.deepcopy(record)
    merge_enrichment(record, _accepted(record, _payload_obj()))
    assert record == snapshot


@pytest.mark.parametrize("mutate", [
    lambda r: r["recommendation"]["kpi_chart_mapping"][0].__setitem__("chart_type", "pie"),
    lambda r: r["recommendation"]["kpi_chart_mapping"][0]["encoding"].__setitem__("x", "other_col"),
    lambda r: r["brief"].__setitem__("kpis", ["SUM(x)"]),
    lambda r: r["brief"]["columns"][0].__setitem__("dtype", "number"),
    lambda r: r["brief"]["extra"]["provenance"].__setitem__("db_id", "other_db"),
    lambda r: r.__setitem__("split", "test"),
    lambda r: r.__setitem__("item_id", "nvbench:changed"),
    lambda r: r["brief"]["extra"]["lineage"].__setitem__("chart_type", "llm-enriched"),
])
def test_any_immutable_change_breaks_the_fingerprint(mutate):
    record = _record()
    before = immutable_fingerprint(record)
    tampered = copy.deepcopy(record)
    mutate(tampered)
    assert immutable_fingerprint(tampered) != before
    assert immutable_diff(record, tampered) != []


def test_quality_fields_are_part_of_the_fingerprint_when_present():
    record = _record()
    record["quality_score"] = 91
    record["quality_tier"] = "A"
    before = immutable_fingerprint(record)
    tampered = copy.deepcopy(record)
    tampered["quality_tier"] = "B"
    assert immutable_fingerprint(tampered) != before


def test_source_columns_include_aggregate_inner_fields():
    columns = source_columns(_record())
    assert "product_name" in columns
    assert "COUNT(product_name)" in columns


# -------------------------------------------------------------- selection


def _corpus():
    train, val = [], []
    charts = ["bar"] * 20 + ["pie"] * 6 + ["line"] * 4 + ["stacked_bar"] * 3 + ["scatter"] * 2
    for i, chart in enumerate(charts):
        train.append(_record(item_id=f"nvbench:t{i}:query:1", split="train", chart=chart,
                             group=f"nvbench:t{i % 20}"))
    for i, chart in enumerate(["bar"] * 5 + ["pie"] * 2 + ["line"] * 2):
        val.append(_record(item_id=f"nvbench:v{i}:query:1", split="val", chart=chart,
                           group=f"nvbench:v{i}"))
    return train, val


def test_selection_is_deterministic_and_group_disjoint():
    train, val = _corpus()
    first = [r["item_id"] for r in select_records(train, val, n=10, seed=42)]
    second = [r["item_id"] for r in select_records(train, val, n=10, seed=42)]
    assert first == second
    summary = selection_summary(select_records(train, val, n=10, seed=42))
    assert summary["n"] == 10
    assert summary["unique_source_groups"] == 10


def test_selection_covers_charts_and_both_splits():
    train, val = _corpus()
    summary = selection_summary(select_records(train, val, n=10, seed=42))
    assert set(summary["chart_type"]) == {"bar", "pie", "line", "stacked_bar", "scatter"}
    assert set(summary["split"]) == {"train", "val"}


def test_test_split_records_are_never_selected():
    train, val = _corpus()
    train.append(_record(item_id="nvbench:heldout:query:1", split="test", group="nvbench:heldout"))
    selected = select_records(train, val, n=30, seed=42)
    assert all(r["split"] != "test" for r in selected)
    assert "nvbench:heldout:query:1" not in {r["item_id"] for r in selected}


def test_exclusion_list_is_respected():
    train, val = _corpus()
    sample = select_records(train, val, n=10, seed=42)
    sample_ids = {r["item_id"] for r in sample}
    pilot = select_records(train, val, n=10, seed=42, exclude_item_ids=sample_ids)
    assert sample_ids.isdisjoint({r["item_id"] for r in pilot})


def test_enrichable_field_set_is_the_agreed_six():
    assert ENRICHABLE_FIELDS == ("users", "context_summary", "layout", "styling",
                                 "interactions", "rationales")
