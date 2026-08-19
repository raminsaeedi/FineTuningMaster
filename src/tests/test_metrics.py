"""Metric computation on small synthetic predictions (masterplan schema)."""

from pathlib import Path

import yaml

from src.core.schemas import DesignOutput, GenerationResult
from src.evaluation.metrics.macro_f1 import NONE_LABEL, MacroF1ChartType
from src.evaluation.metrics.latency import Latency
from src.evaluation.metrics.schema_compliance import SchemaCompliance
from src.evaluation.metrics.structured_exact_match import StructuredExactMatch
from src.evaluation.metrics.topk_accuracy import TopKAccuracy


def _result(item_id: str, mappings, raw: str = "{}") -> GenerationResult:
    parsed = DesignOutput(context_summary={"a": 1}, kpi_chart_mapping=mappings)
    return GenerationResult(
        item_id=item_id, method_name="m", model_name="x", raw_text=raw, parsed=parsed
    )


def _ref(item_id: str, charts):
    return {
        "item_id": item_id,
        "recommendation": {
            "kpi_chart_mapping": [{"kpi": "k", "task_type": "trend", "chart_type": c} for c in charts]
        },
    }


def test_top3_invalid_when_fewer_than_three_distinct_recs():
    # 'b' carries one alternative -> only 2 distinct recs, never 3 -> top-3 invalid.
    results = [
        _result("a", [{"kpi": "k", "task_type": "trend", "chart_type": "line"}]),
        _result("b", [{"kpi": "k", "task_type": "trend", "chart_type": "bar", "alternatives": ["line"]}]),
    ]
    refs = [_ref("a", ["line"]), _ref("b", ["line"])]
    out = TopKAccuracy().compute(results, refs)
    assert out["top_1_accuracy"] == 50.0      # only 'a' has matching primary
    assert out["n"] == 2
    assert out["n_with_alternatives"] == 1    # 'b' has one alternative
    assert out["n_with_3_recs"] == 0          # but no item has 3 distinct recs
    assert out["top_3_support_rate"] == 0.0
    assert out["top_3_valid"] is False
    assert out["top_3_accuracy"] is None
    assert out["top_3_accuracy_supported"] is None


def test_top3_supported_with_three_distinct_recs():
    # Both items emit primary + 2 distinct valid alternatives.
    m_a = [{"kpi": "k", "task_type": "comparison", "chart_type": "bar",
            "alternatives": ["grouped_bar", "table"]}]
    m_b = [{"kpi": "k", "task_type": "correlation", "chart_type": "scatter",
            "alternatives": ["heatmap", "line"]}]
    results = [_result("a", m_a), _result("b", m_b)]
    # 'a' gold 'table' is in [bar, grouped_bar, table]; 'b' gold 'scatter' is the primary.
    refs = [_ref("a", ["table"]), _ref("b", ["scatter"])]
    out = TopKAccuracy().compute(results, refs)
    assert out["n_with_3_recs"] == 2
    assert out["top_3_support_rate"] == 1.0
    assert out["top_3_valid"] is True
    assert out["top_3_accuracy"] == 100.0
    assert out["top_3_accuracy_supported"] == 100.0
    # top-1: 'a' bar != table (wrong); 'b' scatter == scatter (right) -> 50%.
    assert out["top_1_accuracy"] == 50.0


def test_top3_dedupes_and_filters_alternatives():
    # Alternatives include the primary, a duplicate, and an invalid token; only
    # distinct valid non-primary alternatives count toward the 3-rec requirement.
    m = [{"kpi": "k", "task_type": "comparison", "chart_type": "bar",
          "alternatives": ["bar", "table", "table", "not_a_chart"]}]
    results = [_result("a", m)]
    refs = [_ref("a", ["bar"])]
    out = TopKAccuracy().compute(results, refs)
    assert out["n_with_3_recs"] == 0          # recs = [bar, table] -> only 2 distinct valid
    assert out["top_3_valid"] is False


def test_top1_counts_parse_failures_as_wrong():
    # 'a' parses and is correct; 'b' is a parse failure (no parsed output).
    good = _result("a", [{"kpi": "k", "task_type": "trend", "chart_type": "line"}])
    bad = GenerationResult(item_id="b", method_name="m", model_name="x",
                           raw_text="not json", parsed=None)
    refs = [_ref("a", ["line"]), _ref("b", ["bar"])]
    out = TopKAccuracy().compute([good, bad], refs)
    assert out["n"] == 2                  # denominator includes the failure
    assert out["n_parse_failures"] == 1
    assert out["top_1_accuracy"] == 50.0  # 1 correct out of 2 items (failure = wrong)


def test_top1_counts_a_completely_missing_prediction_as_wrong():
    good = _result("a", [{"kpi": "k", "task_type": "trend", "chart_type": "line"}])
    refs = [_ref("a", ["line"]), _ref("b", ["bar"])]

    out = TopKAccuracy().compute([good], refs)

    assert out["n"] == 2
    assert out["n_missing_predictions"] == 1
    assert out["top_1_accuracy"] == 50.0


def test_top3_invalid_when_no_alternatives():
    # No item emits alternatives -> top-3 is degenerate and reported invalid.
    results = [
        _result("a", [{"kpi": "k", "task_type": "trend", "chart_type": "line"}]),
        _result("b", [{"kpi": "k", "task_type": "trend", "chart_type": "bar"}]),
    ]
    refs = [_ref("a", ["line"]), _ref("b", ["line"])]
    out = TopKAccuracy().compute(results, refs)
    assert out["n_with_alternatives"] == 0
    assert out["n_with_3_recs"] == 0
    assert out["top_3_valid"] is False
    assert out["top_3_accuracy"] is None


def test_schema_full_validity_vs_required_keys():
    # Required keys all present but chart_type enum is invalid -> not full-valid.
    bad_enum = ('{"context_summary": {"x": 1}, "kpi_chart_mapping": '
                '[{"kpi": "k", "task_type": "trend", "chart_type": "column chart"}], '
                '"layout": {"a": 1}, "styling": {"a": 1}, "interactions": ["zoom"], '
                '"rationales": [{"claim": "c"}]}')
    valid = ('{"context_summary": {"x": 1}, "kpi_chart_mapping": '
             '[{"kpi": "k", "task_type": "trend", "chart_type": "line"}], '
             '"layout": {"a": 1}, "styling": {"a": 1}, "interactions": ["zoom"], '
             '"rationales": [{"claim": "c"}]}')
    results = [_result("a", [], bad_enum), _result("b", [], valid)]
    out = SchemaCompliance().compute(results, None)
    assert out["required_keys_rate"] == 100.0       # both have all keys
    assert out["schema_validity_rate"] == 50.0      # only the valid-enum one passes


def test_completeness_ignores_empty_containers():
    # All required keys present, but every value is empty -> completeness 0.
    empty = ('{"context_summary": {}, "kpi_chart_mapping": [], "layout": {}, '
             '"styling": {}, "interactions": [], "rationales": []}')
    full = ('{"context_summary": {"x": 1}, "kpi_chart_mapping": '
            '[{"kpi": "k", "task_type": "trend", "chart_type": "line"}], '
            '"layout": {"a": 1}, "styling": {"a": 1}, "interactions": ["zoom"], '
            '"rationales": [{"claim": "c"}]}')
    results = [_result("a", [], empty), _result("b", [], full)]
    out = SchemaCompliance().compute(results, None)
    assert out["required_keys_rate"] == 100.0   # keys present in both
    # 'a' contributes 0 (all empty), 'b' contributes 1 -> mean 0.5.
    assert out["completeness_score"] == 0.5


def test_schema_rates_include_a_completely_missing_prediction():
    valid = ('{"context_summary": {"x": 1}, "kpi_chart_mapping": '
             '[{"kpi": "k", "task_type": "trend", "chart_type": "line"}], '
             '"layout": {"a": 1}, "styling": {"a": 1}, "interactions": ["zoom"], '
             '"rationales": [{"claim": "c"}]}')
    result = _result("a", [], valid)
    refs = [_ref("a", ["line"]), _ref("b", ["bar"])]

    out = SchemaCompliance().compute([result], refs)

    assert out["n"] == 2
    assert out["n_missing_predictions"] == 1
    assert out["schema_validity_rate"] == 50.0


def test_per_class_f1_and_confusion_matrix():
    # gold: a->line, b->line, c->bar ; pred: a->line, b->bar, c->(parse fail).
    good_a = _result("a", [{"kpi": "k", "task_type": "trend", "chart_type": "line"}])
    good_b = _result("b", [{"kpi": "k", "task_type": "trend", "chart_type": "bar"}])
    bad_c = GenerationResult(item_id="c", method_name="m", model_name="x",
                             raw_text="not json", parsed=None)
    refs = [_ref("a", ["line"]), _ref("b", ["line"]), _ref("c", ["bar"])]
    out = MacroF1ChartType().compute([good_a, good_b, bad_c], refs)
    assert out["n"] == 3

    pcf = out["per_class_f1"]
    # line: tp=1 (a), fp=0, fn=1 (b) -> precision 1.0, recall 0.5, f1 = 2/3.
    assert pcf["line"]["precision"] == 1.0
    assert pcf["line"]["recall"] == 0.5
    assert pcf["line"]["f1"] == round(2 * 1.0 * 0.5 / 1.5, 4)
    assert pcf["line"]["support"] == 2
    # bar: tp=0 (c predicted none, b predicted bar but gold line) -> f1 0, support 1.
    assert pcf["bar"]["f1"] == 0.0
    assert pcf["bar"]["support"] == 1

    cm = out["confusion_matrix"]
    labels, matrix = cm["labels"], cm["matrix"]
    li, bi, ni = labels.index("line"), labels.index("bar"), labels.index(NONE_LABEL)
    assert matrix[li][li] == 1   # a: line -> line
    assert matrix[li][bi] == 1   # b: line -> bar
    assert matrix[bi][ni] == 1   # c: bar  -> (none)


def test_macro_f1_includes_a_completely_missing_prediction():
    good = _result("a", [{"kpi": "k", "task_type": "trend", "chart_type": "line"}])
    refs = [_ref("a", ["line"]), _ref("b", ["bar"])]

    out = MacroF1ChartType().compute([good], refs)

    assert out["n"] == 2
    assert out["per_class_f1"][NONE_LABEL]["support"] == 0
    labels = out["confusion_matrix"]["labels"]
    matrix = out["confusion_matrix"]["matrix"]
    assert matrix[labels.index("bar")][labels.index(NONE_LABEL)] == 1


def test_latency_reports_measurement_coverage_without_inventing_missing_times():
    result = _result("a", [])
    result.latency_ms = 1200.0
    refs = [_ref("a", ["line"]), _ref("b", ["bar"])]

    out = Latency().compute([result], refs)

    assert out["avg_latency_ms"] == 1200.0
    assert out["n_measured"] == 1
    assert out["n_requested"] == 2
    assert out["coverage_rate"] == 50.0


def test_structured_exact_match_uses_full_reference_cohort_and_strict_fields():
    parsed = DesignOutput(
        context_summary={"db_id": "db_a"},
        kpi_chart_mapping=[{
            "kpi": "SUM(value)", "task_type": "comparison", "chart_type": "bar",
            "encoding": {"aggregate": "SUM"},
        }],
    )
    result = GenerationResult(
        item_id="a", method_name="m", model_name="x", raw_text="{}", parsed=parsed
    )
    refs = [
        {"item_id": "a", "recommendation": {
            "context_summary": {"db_id": "db_a"},
            "kpi_chart_mapping": [{
                "kpi": "SUM(value)", "task_type": "comparison", "chart_type": "bar",
                "encoding": {"aggregate": "SUM"},
            }],
        }},
        {"item_id": "b", "recommendation": {
            "context_summary": {"db_id": "db_b"},
            "kpi_chart_mapping": [{
                "kpi": "COUNT(value)", "task_type": "ranking", "chart_type": "bar",
                "encoding": {"aggregate": "COUNT"},
            }],
        }},
    ]

    out = StructuredExactMatch().compute([result], refs)

    assert out["n"] == 2 and out["prediction_coverage_rate"] == 50.0
    assert out["exact_task_classification"] == 50.0
    assert out["exact_kpi_selection"] == 50.0
    assert out["exact_mapping_count"] == 50.0
    assert out["exact_encoding"] == 50.0
    assert out["exact_aggregate"] == 50.0
    assert out["hidden_context_diagnostics"]["db_id"] == {
        "presence_rate": 50.0,
        "exact_match_rate": 50.0,
        "n_applicable": 2,
        "visible_in_prompt": False,
    }


def test_full_eval_profile_enables_structured_exact_match():
    profile = yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "eval" / "full.yaml").read_text()
    )

    assert "structured_exact_match" in profile["metrics"]
