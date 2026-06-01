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
    assert out["top_1_accuracy"] == 50.0   # only 'a' has matching primary
    assert out["top_3_accuracy"] == 100.0  # 'b' lists 'line' as an alternative
    assert out["n"] == 2


def test_schema_compliance_full_and_partial():
    full = ('{"context_summary": {}, "kpi_chart_mapping": [], "layout": {}, '
            '"styling": {}, "interactions": [], "rationales": []}')
    partial = '{"context_summary": {}}'
    broken = "not json at all"
    results = [
        _result("a", [], full),
        _result("b", [], partial),
        _result("c", [], broken),
    ]
    out = SchemaCompliance().compute(results, None)
    assert out["json_parse_rate"] == round(100 * 2 / 3, 2)
    assert out["schema_validity_rate"] == round(100 * 1 / 3, 2)
    assert 0 < out["completeness_score"] < 1
