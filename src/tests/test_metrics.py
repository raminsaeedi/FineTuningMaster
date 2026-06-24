"""Metric computation on small synthetic predictions (masterplan schema)."""

from src.core.schemas import DesignOutput, GenerationResult
from src.evaluation.metrics.schema_compliance import SchemaCompliance
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


def test_top_k_accuracy_primary_and_alternatives():
    results = [
        _result("a", [{"kpi": "k", "task_type": "trend", "chart_type": "line"}]),
        _result("b", [{"kpi": "k", "task_type": "trend", "chart_type": "bar", "alternatives": ["line"]}]),
    ]
    refs = [_ref("a", ["line"]), _ref("b", ["line"])]
    out = TopKAccuracy().compute(results, refs)
    assert out["top_1_accuracy"] == 50.0      # only 'a' has matching primary
    assert out["n"] == 2
    # Only 'b' carries alternatives (50% coverage) -> top-3 is valid at threshold.
    assert out["top_3_valid"] is True
    # 'a' primary hit + 'b' alternative hit -> both counted in global top-3.
    assert out["top_3_accuracy"] == 100.0
    # Supported subset = items with alternatives ('b' only), which hits.
    assert out["top_3_accuracy_supported"] == 100.0
    assert out["n_with_alternatives"] == 1


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


def test_top3_invalid_when_no_alternatives():
    # No item emits alternatives -> top-3 is degenerate and reported invalid.
    results = [
        _result("a", [{"kpi": "k", "task_type": "trend", "chart_type": "line"}]),
        _result("b", [{"kpi": "k", "task_type": "trend", "chart_type": "bar"}]),
    ]
    refs = [_ref("a", ["line"]), _ref("b", ["line"])]
    out = TopKAccuracy().compute(results, refs)
    assert out["n_with_alternatives"] == 0
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
