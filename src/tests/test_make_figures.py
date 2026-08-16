"""The figure builder must work from the aggregated table alone.

It is the last step of the professor's run, so it has to be robust: no model,
no GPU, no dataset access, and a metric that was never recorded must be skipped
rather than crash the run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "make_figures", _ROOT / "experiments" / "scripts" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_SPEC)
sys.modules["make_figures"] = make_figures
_SPEC.loader.exec_module(make_figures)


def _table(path: Path, *, with_robustness: bool = True) -> Path:
    rows = []
    for model in ("qwen3_1_7b", "qwen3_8b"):
        for method in ("A", "B", "C", "D"):
            for seed in (42, 43, 44):
                row = {
                    "experiment_id": f"{model}_{method}_seed_{seed}",
                    "model_key": model,
                    "method_key": method,
                    "seed": seed,
                    "metrics.top_k_accuracy.top_1_accuracy": 0.4 + 0.01 * seed,
                    "metrics.macro_f1.macro_f1": 0.3,
                    "metrics.schema_compliance.schema_validity_rate": 0.9,
                    "metrics.schema_compliance.completeness_score": 0.8,
                    "metrics.schema_compliance.json_parse_rate": 1.0,
                    "metrics.latency.avg_latency_ms": 1200.0,
                }
                if with_robustness:
                    row["metrics.robustness.paraphrase_consistency"] = 0.7
                    row["metrics.robustness.paraphrase_accuracy"] = 0.5
                    row["metrics.robustness.missing_info_clarification_rate"] = 0.2
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_all_figures_are_written(tmp_path):
    results = tmp_path / "results"
    _table(results / "comparison_table.csv")
    make_figures.main(["--results-dir", str(results), "--format", "png"])

    figures = results / "figures"
    for name in ("F1_accuracy_by_method", "F2_schema_quality",
                 "F3_seed_variability", "F4_robustness", "F5_latency"):
        assert (figures / f"{name}.png").exists(), name
    assert (figures / "README.md").exists()
    assert (figures / "figure_data.csv").exists()


def test_plotted_values_are_the_seed_means(tmp_path):
    results = tmp_path / "results"
    _table(results / "comparison_table.csv")
    make_figures.main(["--results-dir", str(results), "--format", "png"])

    data = pd.read_csv(results / "figures" / "figure_data.csv")
    top1 = data[(data["metric"] == "Top-1 accuracy") & (data["method_key"] == "A")]
    # 0.4 + 0.01 * mean(42, 43, 44) = 0.83
    assert top1["mean"].round(3).unique().tolist() == [0.83]
    assert top1["count"].unique().tolist() == [3]


def test_missing_metrics_are_skipped_not_fatal(tmp_path):
    results = tmp_path / "results"
    _table(results / "comparison_table.csv", with_robustness=False)
    make_figures.main(["--results-dir", str(results), "--format", "png"])

    figures = results / "figures"
    assert not (figures / "F4_robustness.png").exists()
    assert (figures / "F1_accuracy_by_method.png").exists()
    assert "F4_robustness" in (figures / "README.md").read_text(encoding="utf-8")


def test_missing_aggregation_fails_with_an_actionable_message(tmp_path):
    with pytest.raises(SystemExit) as exc:
        make_figures.main(["--results-dir", str(tmp_path / "nothing"), "--format", "png"])
    assert "run_professor" in str(exc.value)
