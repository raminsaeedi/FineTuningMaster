"""Phase-1 reporting artifacts: per-item scoring + layered metrics.json."""

from src.core.schemas import DesignOutput, GenerationResult
from src.evaluation.reporting import (
    NOT_APPLICABLE,
    NOT_AVAILABLE,
    build_metrics_json,
    legacy_per_item,
    mark_backfill,
    score_per_item,
)

_FULL_VALID = (
    '{"context_summary": {"x": 1}, "kpi_chart_mapping": '
    '[{"kpi": "k", "task_type": "trend", "chart_type": "line"}], '
    '"layout": {"a": 1}, "styling": {"a": 1}, "interactions": ["zoom"], '
    '"rationales": [{"claim": "c"}]}'
)


def _result(item_id: str, chart: str, raw: str) -> GenerationResult:
    parsed = DesignOutput(kpi_chart_mapping=[{"kpi": "k", "task_type": "trend", "chart_type": chart}])
    return GenerationResult(item_id=item_id, method_name="m", model_name="x",
                            raw_text=raw, parsed=parsed)


def _fail(item_id: str) -> GenerationResult:
    return GenerationResult(item_id=item_id, method_name="m", model_name="x",
                            raw_text="not json", parsed=None, parse_error="no_json_found")


def _ref(item_id: str, chart: str) -> dict:
    return {"item_id": item_id,
            "recommendation": {"kpi_chart_mapping": [
                {"kpi": "k", "task_type": "trend", "chart_type": chart}]}}


def test_score_per_item_flags():
    results = [_result("a", "line", _FULL_VALID), _result("b", "bar", "{}"), _fail("c"),
               _result("d", "line", "{}")]
    references = [_ref("a", "line"), _ref("b", "line"), _ref("c", "bar")]  # 'd' has no gold
    rows = {r["item_id"]: r for r in score_per_item(results, references)}

    assert rows["a"]["synthetic_top1_correct"] == 1      # line == line
    assert rows["b"]["synthetic_top1_correct"] == 0      # bar != line
    assert rows["c"]["synthetic_top1_correct"] == 0      # parse failure -> wrong
    assert rows["d"]["synthetic_top1_correct"] is None   # no gold -> not scored

    assert rows["a"]["parsed"] is True and rows["a"]["schema_valid"] is True
    assert rows["c"]["parsed"] is False and rows["c"]["parse_error"] == "no_json_found"
    assert rows["a"]["predicted_primary_chart"] == "line"
    assert rows["c"]["predicted_primary_chart"] is None
    # Independent L1 fields are not applicable until that scorer exists.
    assert rows["a"]["l1_covered"] == NOT_APPLICABLE
    assert rows["a"]["l1_correct"] == NOT_APPLICABLE


def test_build_metrics_json_structure_and_ci():
    payload = {
        "experiment_id": "E0X", "method": "prompt_only", "model": "x", "seed": 42,
        "n_predictions": 3,
        "metrics": {
            "schema_compliance": {"json_parse_rate": 66.67, "schema_validity_rate": 33.33,
                                  "completeness_score": 0.5, "n": 3},
            "top_k_accuracy": {"top_1_accuracy": 33.33, "top_3_accuracy": None,
                               "top_3_valid": False, "top_3_support_rate": 0.0, "n": 3},
            "macro_f1": {"macro_f1": 0.4, "n": 3},
            "robustness": {"paraphrase_accuracy": 50.0, "paraphrase_consistency": 80.0,
                           "missing_info_clarification_rate": 40.0, "missing_info_schema_rate": 20.0},
            # no grounding key -> non-RAG run
        },
    }
    results = [_result("a", "line", _FULL_VALID), _result("b", "bar", "{}"), _fail("c")]
    references = [_ref("a", "line"), _ref("b", "line"), _ref("c", "bar")]
    rows = score_per_item(results, references)

    mj = build_metrics_json(payload, rows)

    assert mj["report_schema_version"] == "1"
    assert mj["eval_tier"] == "internal-synthetic"
    layers = mj["layers"]
    assert set(layers) == {
        "L2_format_robustness", "L1_chart_selection", "L1c_grounding", "L3_realism", "L4_human",
    }

    # L2: parse rate value preserved + a computed bootstrap CI.
    jpr = layers["L2_format_robustness"]["json_parse_rate"]
    assert jpr["value"] == 66.67 and jpr["n"] == 3
    assert "ci_low" in jpr["ci"] and "ci_high" in jpr["ci"]
    # Robustness CIs honestly marked pending (no per-variant vectors in Phase 1).
    assert layers["L2_format_robustness"]["paraphrase_accuracy"]["ci"]["status"] == "pending"

    # L1: synthetic top-1 is diagnostic (internal-circular) with a computed CI.
    st1 = layers["L1_chart_selection"]["synthetic_top1"]
    assert st1["tier"] == "internal-circular" and "ci_low" in st1["ci"]
    # macro-F1 has no item-mean CI -> not_available.
    assert layers["L1_chart_selection"]["macro_f1_synthetic"]["ci"]["status"] == "not_available"
    # Independent L1 human-effectiveness + L3/L4 are pending; grounding not applicable.
    assert layers["L1_chart_selection"]["l1_human_effectiveness"]["status"] == "pending"
    assert layers["L3_realism"]["status"] == "pending"
    assert layers["L4_human"]["status"] == "pending"
    assert layers["L1c_grounding"]["status"] == NOT_APPLICABLE


def test_score_per_item_degraded_without_references():
    # No references available (backfill degraded mode): gold/top1 become null,
    # but prediction-derived fields are still populated.
    rows = {r["item_id"]: r for r in score_per_item([_result("a", "line", _FULL_VALID), _fail("b")], [])}
    assert rows["a"]["gold_primary_chart"] is None
    assert rows["a"]["synthetic_top1_correct"] is None
    assert rows["a"]["predicted_primary_chart"] == "line"
    assert rows["a"]["parsed"] is True and rows["a"]["schema_valid"] is True
    assert rows["b"]["synthetic_top1_correct"] is None


def test_legacy_per_item_uses_stored_fields_only():
    rows = {r["item_id"]: r for r in legacy_per_item([_result("a", "line", _FULL_VALID), _fail("b")])}
    # Predicted chart comes from the stored parse; correctness/schema/completeness deferred.
    assert rows["a"]["predicted_primary_chart"] == "line"
    assert rows["a"]["parsed"] is True
    assert rows["a"]["schema_valid"] == NOT_AVAILABLE
    assert rows["a"]["completeness"] == NOT_AVAILABLE
    assert rows["a"]["gold_primary_chart"] == NOT_AVAILABLE
    assert rows["a"]["synthetic_top1_correct"] == NOT_AVAILABLE
    assert rows["b"]["parsed"] is False


def test_backfill_metrics_json_no_ci_and_annotation():
    payload = {"metrics": {"schema_compliance": {"json_parse_rate": 66.0, "n": 50}}}
    rows = legacy_per_item([_result("a", "line", "{}")])
    mj = build_metrics_json(payload, rows, compute_ci=False)
    mark_backfill(mj, metrics_auto_present=True, references_present=True)
    jpr = mj["layers"]["L2_format_robustness"]["json_parse_rate"]
    assert jpr["value"] == 66.0                    # legacy value carried unchanged
    assert jpr["ci"]["status"] == "not_available"  # no fresh CI paired to a legacy value
    bf = mj["backfill"]
    assert bf["mode"] == "legacy-carry-forward"
    assert bf["references_used"] is False and bf["pre_task7_metrics_auto"] is True
