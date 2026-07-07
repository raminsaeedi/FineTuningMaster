"""Tests for the independent benchmark scorer (score_benchmark), with mock predictions.

Uses NO synthetic gold — scoring is against benchmark `acceptable_chart_types`.
"""

from src.core.schemas import DesignOutput, GenerationResult
from src.evaluation.l1_independent import score_benchmark


def _res(item_id, task_type, chart_type):
    parsed = DesignOutput(kpi_chart_mapping=[{"kpi": "k", "task_type": task_type, "chart_type": chart_type}])
    return GenerationResult(item_id=item_id, method_name="m", model_name="x", raw_text="{}", parsed=parsed)


def _parse_fail(item_id):
    return GenerationResult(item_id=item_id, method_name="m", model_name="x", raw_text="not json", parsed=None)


def _item(bid, task, charts, auto=True, domain="Retail",
          source_type="real_public", label_source="literature_L1"):
    return {"benchmark_id": bid, "suitable_for_auto_scoring": auto, "task_type": task,
            "domain": domain, "acceptable_chart_types": charts,
            "source_type": source_type, "label_source": label_source}


def test_correct_prediction_scores_right():
    items = [_item("bm1", "comparison", ["bar", "table"])]
    out = score_benchmark([_res("bm1", "comparison", "bar")], items)
    assert out["n_total"] == 1 and out["n_covered"] == 1
    assert out["coverage_rate"] == 1.0
    assert out["covered_accuracy"] == 1.0
    assert out["parse_failures"] == 0


def test_wrong_prediction_scores_wrong():
    items = [_item("bm1", "correlation", ["scatter", "heatmap"])]
    out = score_benchmark([_res("bm1", "correlation", "line")], items)
    assert out["n_covered"] == 1
    assert out["covered_accuracy"] == 0.0
    assert out["parse_failures"] == 0


def test_parse_failure_counts_wrong():
    items = [_item("bm1", "comparison", ["bar", "table"])]
    out = score_benchmark([_parse_fail("bm1")], items)
    assert out["n_covered"] == 1
    assert out["covered_accuracy"] == 0.0
    assert out["parse_failures"] == 1


def test_uncovered_excluded_from_accuracy_but_counted_in_coverage():
    # bm2 is not auto-scorable: even a "correct" chart must not count toward covered_accuracy,
    # but it must still be counted in n_total (coverage denominator).
    items = [_item("bm1", "comparison", ["bar"]),
             _item("bm2", "flow", ["sankey"], auto=False, source_type="realistic_manual",
                   label_source="manual_expert")]
    out = score_benchmark([_res("bm1", "comparison", "bar"), _res("bm2", "flow", "sankey")], items)
    assert out["n_total"] == 2
    assert out["n_covered"] == 1                 # only bm1 is covered
    assert out["n_uncovered"] == 1
    assert out["coverage_rate"] == 0.5           # coverage reflects the excluded item
    assert out["covered_accuracy"] == 1.0        # computed over covered items only
    assert "flow" not in out["per_task_type"]    # uncovered task not in covered breakdown


def test_multiple_acceptable_chart_types_accepted():
    # 'table' is one of several acceptable charts -> accepted.
    items = [_item("bm1", "comparison", ["bar", "grouped_bar", "table"])]
    out = score_benchmark([_res("bm1", "comparison", "table")], items)
    assert out["covered_accuracy"] == 1.0


def test_evidence_and_domain_breakdown_present():
    items = [_item("bm1", "comparison", ["bar"], domain="Retail",
                   source_type="real_public", label_source="literature_L1"),
             _item("bm2", "ranking", ["bar", "table"], domain="Finance",
                   source_type="realistic_manual", label_source="literature_L1")]
    out = score_benchmark([_res("bm1", "comparison", "bar"), _res("bm2", "ranking", "line")], items)
    assert out["evidence_strength"]["strong"]["covered"] == 1   # bm1
    assert out["evidence_strength"]["weak"]["covered"] == 1      # bm2 (realistic_manual)
    assert set(out["per_domain"]) == {"Retail", "Finance"}
