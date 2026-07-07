"""Unit tests for the scientific-validity tooling: L1 scorer, leakage similarity,
provenance derivation, and benchmark validation."""

from src.core.schemas import DesignOutput, GenerationResult
from src.data_pipeline.benchmark_validation import (
    label_lineage_check,
    source_leakage,
    validate_item,
)
from src.data_pipeline.leakage_similarity import brief_text, jaccard, char_ngrams, near_duplicate_pairs
from src.data_pipeline.provenance import derive_benchmark, derive_real_brief, derive_synthetic
from src.evaluation.l1_independent import score_l1


# ── L1 independent scorer ────────────────────────────────────────────────────
def _res(item_id, mappings):
    return GenerationResult(item_id=item_id, method_name="m", model_name="x",
                            raw_text="{}", parsed=DesignOutput(kpi_chart_mapping=mappings))


def _ref(item_id, entries):
    return {"item_id": item_id, "recommendation": {"kpi_chart_mapping": entries}}


def test_l1_coverage_and_accuracy():
    eff = {"comparison": {"bar", "table"}, "trend": {"line"}}
    refs = [_ref("a", [
        {"kpi": "k1", "task_type": "comparison", "chart_type": "bar"},   # covered
        {"kpi": "k2", "task_type": "flow", "chart_type": "sankey"},      # uncovered -> excluded
    ])]
    results = [_res("a", [{"kpi": "k1", "task_type": "comparison", "chart_type": "bar"}])]
    out = score_l1(results, refs, eff)
    assert out["n_gold_kpi"] == 2
    assert out["n_covered"] == 1 and out["n_uncovered"] == 1
    assert out["coverage_rate"] == 0.5
    assert out["covered_accuracy"] == 1.0          # bar in {bar, table}
    assert out["uncovered_task_types"] == {"flow": 1}


def test_l1_parse_failure_on_covered_is_wrong():
    eff = {"comparison": {"bar", "table"}}
    refs = [_ref("a", [{"kpi": "k1", "task_type": "comparison", "chart_type": "bar"}])]
    bad = GenerationResult(item_id="a", method_name="m", model_name="x", raw_text="x", parsed=None)
    out = score_l1([bad], refs, eff)
    assert out["n_covered"] == 1
    assert out["covered_accuracy"] == 0.0


# ── near-duplicate similarity ────────────────────────────────────────────────
def test_near_duplicate_detects_identical_and_ignores_different():
    b1 = {"users": "analyst", "goals": ["compare sales vs last year"], "kpis": ["revenue"],
          "columns": [{"name": "date"}]}
    b2 = dict(b1)
    b3 = {"users": "zzzzz", "goals": ["xxxxx yyyyy"], "kpis": ["qqqqq"], "columns": [{"name": "wwwww"}]}
    pairs = near_duplicate_pairs([("x", b1)], [("y", b2), ("z", b3)], threshold=0.8)
    assert len(pairs) == 1 and pairs[0]["right_id"] == "y"
    assert pairs[0]["similarity"] == 1.0


def test_jaccard_bounds():
    a = char_ngrams("hello world")
    assert jaccard(a, a) == 1.0
    assert jaccard(a, set()) == 0.0
    assert "compare" in brief_text({"goals": ["Compare X"], "users": "", "kpis": [], "columns": []})


# ── provenance derivation ────────────────────────────────────────────────────
def test_provenance_synthetic_is_diagnostic_only():
    p = derive_synthetic({"item_id": "v2_1", "brief": {"extra": {"generator_version": "v2.0"}}},
                         "train.jsonl", "v2")
    assert p["is_synthetic"] is True
    assert p["independent_eval_safe"] is False
    assert p["label_source"] == "synthetic_generator"
    assert p["split"] == "train" and p["intended_use"] == "train"


def test_provenance_benchmark_and_real_are_independent():
    b = derive_benchmark({"benchmark_id": "bm1", "label_source": "literature_L1",
                          "source_type": "real_public", "license_or_usage_note": "x"}, "benchmark_v1")
    assert b["is_synthetic"] is False and b["independent_eval_safe"] is True
    assert "Saket2019" in b["label_lineage_id"]
    r = derive_real_brief({"item_id": "rb1", "extra": {"provenance_id": "rb_001"}}, "real_briefs_v1")
    assert r["label_source"] == "none" and r["independent_eval_safe"] is True


# ── benchmark validation ─────────────────────────────────────────────────────
def _bench_item(**over):
    item = {
        "benchmark_id": "bm_v1_001", "domain": "Retail", "users": "u", "goals": ["g"],
        "kpis": ["k"], "columns": [{"name": "date", "dtype": "datetime"}],
        "constraints": None, "task_type": "comparison", "acceptable_chart_types": ["bar", "table"],
        "rationale": "r", "source_name": "s", "source_type": "real_public",
        "source_reference": "ref", "license_or_usage_note": "note",
        "label_source": "literature_L1", "label_confidence": "high",
        "suitable_for_auto_scoring": True, "suitable_for_human_eval": True, "notes": "",
    }
    item.update(over)
    return item


def test_benchmark_validate_item_good_and_bad():
    assert validate_item(_bench_item()) == []
    bad = validate_item(_bench_item(acceptable_chart_types=["not_a_chart"], task_type="nope"))
    assert any("task_type" in p for p in bad)
    assert any("acceptable_chart_types" in p for p in bad)


def test_benchmark_label_lineage_check():
    eff = {"comparison": {"bar", "table"}}
    gen = {"comparison": {"bar", "grouped_bar", "table"}}
    items = [_bench_item()]  # literature_L1, acceptable {bar,table} == eff[comparison]
    out = label_lineage_check(items, eff, gen)
    assert out["label_source_ok"] is True
    assert out["l1_mismatch"] == []              # matches the L1 table
    assert out["identical_to_generator_set"] == []  # {bar,table} != generator {bar,grouped_bar,table}


def test_benchmark_source_leakage_detects_overlap():
    item = _bench_item()
    # a training brief identical to the benchmark brief content
    train_brief = {"users": "u", "goals": ["g"], "kpis": ["k"],
                   "columns": [{"name": "date", "dtype": "datetime"}]}
    assert source_leakage([item], [train_brief]) == ["bm_v1_001"]
    assert source_leakage([item], []) == []
