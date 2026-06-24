"""Robustness metric checks: paraphrase accuracy + missing-info behaviour."""

from src.core.schemas import DesignOutput, GenerationResult
from src.evaluation.metrics.robustness import compute_robustness


def _res(item_id: str, chart: str, raw: str = "{}", variant: str = "original") -> GenerationResult:
    parsed = DesignOutput(
        kpi_chart_mapping=[{"kpi": "k", "task_type": "trend", "chart_type": chart}]
    )
    return GenerationResult(item_id=item_id, method_name="m", model_name="x",
                            raw_text=raw, parsed=parsed, variant=variant)


def _ref(item_id: str, chart: str):
    return {"item_id": item_id,
            "recommendation": {"kpi_chart_mapping": [
                {"kpi": "k", "task_type": "trend", "chart_type": chart}]}}


def test_paraphrase_accuracy_combines_with_consistency():
    # Original correct on both; paraphrase flips 'b' to a wrong chart.
    original = [_res("a", "line"), _res("b", "line")]
    paraphrased = [_res("a", "line", variant="paraphrased"),
                   _res("b", "bar", variant="paraphrased")]
    refs = [_ref("a", "line"), _ref("b", "line")]
    out = compute_robustness(original, paraphrased, references=refs)
    # 'a' stable, 'b' changed -> 50% consistency.
    assert out["paraphrase_consistency"] == 50.0
    # Paraphrased accuracy: 'a' correct, 'b' wrong -> 50%.
    assert out["paraphrase_accuracy"] == 50.0
    # Original was 100%, paraphrased 50% -> delta -50.
    assert out["paraphrase_accuracy_delta"] == -50.0


def test_missing_info_clarification_detected():
    asks = GenerationResult(item_id="a", method_name="m", model_name="x",
                            raw_text="I need more information about the KPIs to recommend a chart.",
                            parsed=None, variant="missing_info")
    guesses = _res("b", "bar",
                   raw='{"context_summary": {}, "kpi_chart_mapping": [], "layout": {}, '
                       '"styling": {}, "interactions": [], "rationales": []}',
                   variant="missing_info")
    out = compute_robustness([], missing_info=[asks, guesses])
    assert out["missing_info_clarification_rate"] == 50.0   # only 'a' asks
    assert out["missing_info_schema_rate"] == 50.0          # only 'b' emits full schema
