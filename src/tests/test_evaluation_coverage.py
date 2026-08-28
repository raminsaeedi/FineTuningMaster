"""Public evaluation contract for incomplete prediction cohorts."""

from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from src.core.schemas import GenerationResult
from src.pipeline.runner import ExperimentRunner

_RECOMMENDATION = {
    "context_summary": {"scope": "test"},
    "kpi_chart_mapping": [{
        "kpi": "COUNT(value)", "task_type": "comparison", "chart_type": "bar",
        "alternatives": [],
        "encoding": {"x": "category", "y": "value", "aggregate": "COUNT"},
    }],
    "layout": {"type": "single"},
    "styling": {"theme": "minimal"},
    "interactions": ["tooltip"],
    "rationales": [{"claim": "comparison", "principle": "common scale"}],
}


def test_run_eval_writes_coverage_aware_metrics_and_rejects_incomplete_run(tmp_path):
    test_file = tmp_path / "test.jsonl"
    rows = [
        {"item_id": item_id, "brief": {"item_id": item_id},
         "recommendation": _RECOMMENDATION, "split": "test"}
        for item_id in ("a", "b")
    ]
    test_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = OmegaConf.create({
        "output_root": str(tmp_path / "outputs"),
        "experiment_id": "coverage",
        "run_layout": "legacy",
        "profile": "final",
        "seed": 42,
        "method": {"name": "prompt_only"},
        "model": {"name": "fake"},
        "data": {"test_file": str(test_file), "max_samples": None,
                 "paraphrased_file": None, "missing_info_file": None},
        "eval": {"metrics": ["schema_compliance", "top_k_accuracy", "macro_f1",
                              "latency", "structured_exact_match"]},
    })
    runner = ExperimentRunner(config, tmp_path)
    runner.exp_dir.mkdir(parents=True)
    prediction = GenerationResult(
        item_id="a", method_name="prompt_only", model_name="fake",
        raw_text=json.dumps(_RECOMMENDATION), latency_ms=1000.0,
    )
    (runner.exp_dir / "predictions.jsonl").write_text(
        prediction.model_dump_json() + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="1 of 2 expected predictions are missing"):
        runner.run_eval()

    payload = json.loads((runner.exp_dir / "metrics_auto.json").read_text())
    assert payload["coverage"] == {
        "n_requested": 2,
        "n_predictions": 1,
        "n_missing": 1,
        "prediction_coverage_rate": 50.0,
        "missing_item_ids": ["b"],
    }
    assert payload["metrics"]["top_k_accuracy"]["top_1_accuracy"] == 50.0
    assert payload["metrics"]["schema_compliance"]["schema_validity_rate"] == 50.0


def test_run_eval_rejects_incomplete_robustness_variant(tmp_path):
    test_file = tmp_path / "test.jsonl"
    rows = [
        {"item_id": item_id, "brief": {"item_id": item_id},
         "recommendation": _RECOMMENDATION, "split": "test"}
        for item_id in ("a", "b")
    ]
    test_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = OmegaConf.create({
        "output_root": str(tmp_path / "outputs"), "experiment_id": "variants",
        "run_layout": "legacy", "profile": "final", "seed": 42,
        "method": {"name": "prompt_only"}, "model": {"name": "fake"},
        "data": {"test_file": str(test_file), "max_samples": None,
                 "paraphrased_file": str(test_file), "missing_info_file": None},
        "eval": {"metrics": ["schema_compliance"]},
    })
    runner = ExperimentRunner(config, tmp_path)
    runner.exp_dir.mkdir(parents=True)
    predictions = [
        GenerationResult(item_id=item_id, method_name="prompt_only", model_name="fake",
                         raw_text=json.dumps(_RECOMMENDATION), latency_ms=1000.0)
        for item_id in ("a", "b")
    ]
    (runner.exp_dir / "predictions.jsonl").write_text(
        "".join(result.model_dump_json() + "\n" for result in predictions), encoding="utf-8"
    )
    (runner.exp_dir / "predictions_paraphrased.jsonl").write_text(
        predictions[0].model_dump_json() + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="paraphrased.*1 of 2"):
        runner.run_eval()

    payload = json.loads((runner.exp_dir / "metrics_auto.json").read_text())
    assert payload["variant_coverage"]["paraphrased"]["n_missing"] == 1
