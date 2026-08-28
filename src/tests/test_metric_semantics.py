"""Focused metric-semantics guards (complement test_metrics.py).

Locks the scientifically important thresholds/definitions so they cannot silently
regress: Top-3 support gating at TOP3_MIN_SUPPORT, strict enum schema validity, and
non-empty completeness.
"""

from src.core.schemas import DesignOutput, GenerationResult
from src.evaluation.metrics.schema_compliance import (
    SchemaCompliance,
    completeness_fraction,
    strict_response_valid,
)
from src.evaluation.metrics.topk_accuracy import TOP3_MIN_SUPPORT, TopKAccuracy


def _res(item_id, mappings):
    return GenerationResult(item_id=item_id, method_name="m", model_name="x",
                            raw_text="{}", parsed=DesignOutput(kpi_chart_mapping=mappings))


def _ref(item_id, chart):
    return {"item_id": item_id,
            "recommendation": {"kpi_chart_mapping": [{"kpi": "k", "task_type": "trend", "chart_type": chart}]}}


def _three_rec(item_id):
    return _res(item_id, [{"kpi": "k", "task_type": "comparison", "chart_type": "bar",
                           "alternatives": ["grouped_bar", "table"]}])


def _one_rec(item_id):
    return _res(item_id, [{"kpi": "k", "task_type": "trend", "chart_type": "line"}])


def test_top3_valid_exactly_at_support_threshold():
    # 4 of 5 items carry 3 distinct recs -> support 0.8 == TOP3_MIN_SUPPORT -> valid.
    results = [_three_rec(f"t{i}") for i in range(4)] + [_one_rec("o0")]
    refs = [_ref(f"t{i}", "bar") for i in range(4)] + [_ref("o0", "line")]
    out = TopKAccuracy().compute(results, refs)
    assert TOP3_MIN_SUPPORT == 0.8
    assert out["top_3_support_rate"] == 0.8
    assert out["top_3_valid"] is True
    assert out["top_3_accuracy"] is not None


def test_top3_invalid_just_below_threshold():
    # 3 of 5 -> support 0.6 < 0.8 -> reported invalid (None).
    results = [_three_rec(f"t{i}") for i in range(3)] + [_one_rec("o0"), _one_rec("o1")]
    refs = [_ref(f"t{i}", "bar") for i in range(3)] + [_ref("o0", "line"), _ref("o1", "line")]
    out = TopKAccuracy().compute(results, refs)
    assert out["top_3_support_rate"] == 0.6
    assert out["top_3_valid"] is False
    assert out["top_3_accuracy"] is None


def test_strict_response_valid_rejects_bad_enum_accepts_good():
    base = {"context_summary": {"x": 1}, "layout": {"a": 1}, "styling": {"a": 1},
            "interactions": ["zoom"],
            "rationales": [{"claim": "c", "principle": "p"}]}
    mapping = {"kpi": "k", "task_type": "trend", "chart_type": "line",
               "alternatives": [],
               "encoding": {"x": "date", "y": "value", "aggregate": None}}
    bad = {**base, "kpi_chart_mapping": [{**mapping, "chart_type": "not_a_chart"}]}
    good = {**base, "kpi_chart_mapping": [mapping]}
    assert strict_response_valid(bad) is False
    assert strict_response_valid(good) is True


def test_strict_response_valid_rejects_missing_or_empty_encoding():
    base = {
        "context_summary": {"x": 1},
        "layout": {"a": 1},
        "styling": {"a": 1},
        "interactions": [],
        "rationales": [],
    }
    missing = {**base, "kpi_chart_mapping": [
        {"kpi": "k", "task_type": "trend", "chart_type": "line", "alternatives": []}
    ]}
    empty = {**base, "kpi_chart_mapping": [
        {"kpi": "k", "task_type": "trend", "chart_type": "line",
         "alternatives": [], "encoding": {}}
    ]}

    assert strict_response_valid(missing) is False
    assert strict_response_valid(empty) is False


def test_completeness_fraction_counts_only_nonempty():
    partial = {"context_summary": {"x": 1}, "kpi_chart_mapping": [{"kpi": "k"}],
               "layout": {}, "styling": {}, "interactions": [], "rationales": []}
    # 2 of 6 required keys are present AND non-empty.
    assert completeness_fraction(partial) == round(2 / 6, 4) or abs(completeness_fraction(partial) - 2 / 6) < 1e-6


def test_schema_validity_rate_is_full_pydantic():
    # A parse-fail counts against schema_validity_rate (over all results).
    bad = GenerationResult(item_id="b", method_name="m", model_name="x", raw_text="not json", parsed=None)
    out = SchemaCompliance().compute([bad], None)
    assert out["schema_validity_rate"] == 0.0
    assert out["json_parse_rate"] == 0.0


def test_schema_compliance_reports_encoding_object_rate_separately():
    good = GenerationResult(
        item_id="good",
        method_name="m",
        model_name="x",
        raw_text=(
            '{"kpi_chart_mapping": [{"encoding": '
            '{"x": "date", "y": "revenue", "aggregate": "sum"}}]}'
        ),
        parsed=None,
    )
    bad = GenerationResult(
        item_id="bad",
        method_name="m",
        model_name="x",
        raw_text='{"kpi_chart_mapping": [{"encoding": "x=date, y=revenue"}]}',
        parsed=None,
    )

    out = SchemaCompliance().compute([good, bad], None)

    assert out["json_parse_rate"] == 100.0
    assert out["encoding_object_rate"] == 50.0
    assert out["n_encoding_object_valid"] == 1
