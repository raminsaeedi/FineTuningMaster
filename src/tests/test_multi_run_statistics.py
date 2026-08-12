"""Focused tests for independent-run discovery and multi-seed reporting."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.evaluation.aggregator import build_multi_seed_summary, collect_rows


_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "eval_stats_script", _ROOT / "experiments" / "scripts" / "eval_stats.py"
)
eval_stats = importlib.util.module_from_spec(_SPEC)
sys.modules["eval_stats_script"] = eval_stats
_SPEC.loader.exec_module(eval_stats)


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(seed: int, *, robustness: bool = True) -> dict:
    payload = {
        "experiment_id": f"E01_{seed}",
        "method": "prompt_only",
        "model": BASE_MODEL,
        "seed": seed,
        "n_predictions": 2,
        "metrics": {
            "schema_compliance": {
                "json_parse_rate": 90.0 + seed / 1000,
                "schema_validity_rate": 80.0,
                "completeness_score": 0.7,
                "n": 2,
            },
            "top_k_accuracy": {
                "top_1_accuracy": 50.0,
                "n": 2,
                "n_parse_failures": 0,
            },
            "macro_f1": {"macro_f1": 0.5, "n": 2},
            "latency": {"avg_latency_ms": 10.0, "n": 2},
        },
    }
    if robustness:
        payload["metrics"]["robustness"] = {
            "paraphrase_accuracy": 60.0,
            "paraphrase_consistency": 70.0,
            "missing_info_clarification_rate": 40.0,
        }
    return payload


def _write_run(root: Path, experiment: str, method: str, seed: int, *,
               dataset: str = "dashboard_v3", test_hash: str = "test-hash",
               robustness: bool = True) -> Path:
    run = root / f"{experiment}_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(seed, robustness=robustness)
    metrics["experiment_id"] = run.name
    metrics["method"] = method
    _write_json(run / "metrics_auto.json", metrics)
    _write_json(run / "manifest.json", {
        "experiment_id": run.name,
        "experiment_name": experiment,
        "method": method,
        "model": BASE_MODEL,
        "model_hf_id": BASE_MODEL,
        "seed": seed,
        "dataset_version": dataset,
        "data_file_sha256": {"test": test_hash},
        "config_hash": f"config-{seed}",
        "report_schema_version": "1",
        "status": "completed",
    })
    (run / "config_snapshot.yaml").write_text(
        "model:\n  name: qwen2_5_0_5b\nmethod:\n  name: "
        f"{method}\nseed: {seed}\n",
        encoding="utf-8",
    )
    return run


def _gold_and_predictions(tmp_path: Path, *, seed: int = 43):
    test_file = tmp_path / "test.jsonl"
    records = [
        {
            "item_id": "item-a",
            "brief": {"users": "analysts", "goals": [], "kpis": []},
            "recommendation": {
                "kpi_chart_mapping": [
                    {"kpi": "x", "task_type": "trend", "chart_type": "line"}
                ]
            },
        },
        {
            "item_id": "item-b",
            "brief": {"users": "analysts", "goals": [], "kpis": []},
            "recommendation": {
                "kpi_chart_mapping": [
                    {"kpi": "x", "task_type": "comparison", "chart_type": "bar"}
                ]
            },
        },
    ]
    test_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    raw_by_id = {
        "item-a": '{"kpi_chart_mapping": [{"task_type": "trend", "chart_type": "line"}]}',
        "item-b": '{"kpi_chart_mapping": [{"task_type": "comparison", "chart_type": "bar"}]}',
    }
    return test_file, raw_by_id


def _write_stat_run(root: Path, name: str, method: str, seed: int, test_file: Path,
                    raw_by_id: dict[str, str], *, test_hash: str = "same-test-hash",
                    dataset: str = "dashboard_v3", model: str = BASE_MODEL,
                    adapter_path: str | None = None) -> tuple[object, Path]:
    run_dir = root / f"{name}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_id": item_id,
            "method_name": method,
            "model_name": model,
            "config_hash": f"hash-{method}",
            "raw_text": raw,
            "parsed": None,
            "parse_error": None,
            "seed": seed,
            "variant": "original",
        }
        for item_id, raw in raw_by_id.items()
    ]
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    metadata = {
        "base_model": model,
        "seed": seed,
        "dataset_version": dataset,
        "experiment_id": f"E03_qwen0_5b_ft_{seed}",
    }
    manifest = {
        "experiment_id": run_dir.name,
        "experiment_name": name,
        "method": method,
        "model": "qwen2_5_0_5b",
        "model_hf_id": model,
        "seed": seed,
        "dataset_version": dataset,
        "data_file_sha256": {"test": test_hash},
        "report_schema_version": "1",
        "config_hash": f"config-{method}-{seed}",
        "status": "completed",
        "adapter": {
            "adapter_path": adapter_path,
            "adapter_training_metadata": metadata if adapter_path else None,
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    cfg = OmegaConf.create({
        "experiment_name": name,
        "experiment_id": run_dir.name,
        "output_root": str(root),
        "seed": seed,
        "model": {"name": "qwen2_5_0_5b", "hf_id": model},
        "method": {
            "name": method,
            "type": "fine_tuned" if method in {"ft", "ft_rag"} else method,
            "adapter_source_experiment": "E03_qwen0_5b_ft" if method == "ft_rag" else None,
        },
        "data": {"test_file": str(test_file), "dataset_version": dataset},
        "eval": {"metrics": ["schema_compliance"]},
    })
    return cfg, run_dir


def test_multi_seed_summary_keeps_raw_seeds_and_groups_by_dataset(tmp_path):
    root = tmp_path / "outputs"
    for seed in (42, 43, 44):
        _write_run(root, "E01_qwen0_5b_prompt", "prompt_only", seed)
    _write_run(root, "E01_qwen0_5b_prompt", "prompt_only", 45, dataset="dashboard_v2")

    df = pd.DataFrame(collect_rows(root))
    summary = build_multi_seed_summary(df)

    current = summary[
        (summary["dataset_version"] == "dashboard_v3")
        & (summary["metric"] == "top_1_accuracy")
    ].iloc[0]
    assert current["n_seeds"] == 3
    assert current["seed_list"] == "42,43,44"
    assert current["seed_42"] == pytest.approx(50.0)
    assert current["seed_43"] == pytest.approx(50.0)
    assert current["seed_44"] == pytest.approx(50.0)
    assert current["mean"] == pytest.approx(50.0)
    assert pd.isna(current["ci_low"])
    assert pd.isna(current["ci_high"])


def test_aggregation_records_run_provenance_and_na_robustness(tmp_path):
    root = tmp_path / "outputs"
    run = _write_run(root, "E01_qwen0_5b_prompt", "prompt_only", 43, robustness=False)

    rows = collect_rows(root)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_path"] == str(run.resolve())
    assert row["dataset_version"] == "dashboard_v3"
    assert row["test_file_sha256"] == "test-hash"
    assert row["config_hash"] == "config-43"

    summary = build_multi_seed_summary(pd.DataFrame(rows))
    robustness = summary[summary["metric"] == "paraphrase_accuracy"].iloc[0]
    assert pd.isna(robustness["seed_43"])
    assert pd.isna(robustness["mean"])


def test_non_default_root_seed_and_model_overrides_are_forwarded(tmp_path, monkeypatch):
    calls = []
    root = tmp_path / "final"
    test_file, raw_by_id = _gold_and_predictions(tmp_path, seed=43)
    cfg, run_dir = _write_stat_run(
        root, "E01_qwen0_5b_prompt", "prompt_only", 43, test_file, raw_by_id
    )

    def fake_load_cfg(*, experiment, overrides):
        calls.append((experiment, list(overrides)))
        return cfg

    monkeypatch.setattr(eval_stats, "load_cfg", fake_load_cfg)
    bundle = eval_stats.load_run_bundle(
        "E01_qwen0_5b_prompt",
        ["output_root=" + str(root), "seed=43", "model=qwen2_5_0_5b"],
    )

    assert calls == [(
        "E01_qwen0_5b_prompt",
        ["output_root=" + str(root), "seed=43", "model=qwen2_5_0_5b"],
    )]
    assert bundle["seed"] == 43
    assert bundle["model"] == BASE_MODEL
    assert bundle["run_dir"] == run_dir


def test_missing_run_fails_clearly(tmp_path, monkeypatch):
    cfg = OmegaConf.create({
        "experiment_id": "E01_43",
        "experiment_name": "E01_qwen0_5b_prompt",
        "output_root": str(tmp_path),
        "seed": 43,
        "model": {"name": "qwen2_5_0_5b", "hf_id": BASE_MODEL},
        "method": {"name": "prompt_only"},
        "data": {"test_file": str(tmp_path / "test.jsonl")},
        "eval": {"metrics": ["schema_compliance"]},
    })
    monkeypatch.setattr(eval_stats, "load_cfg", lambda **kwargs: cfg)

    with pytest.raises(FileNotFoundError, match="Missing run manifest"):
        eval_stats.load_run_bundle("E01_qwen0_5b_prompt", ["seed=43"])


def test_compatible_four_method_comparison_passes(tmp_path, monkeypatch):
    root = tmp_path / "final"
    test_file, raw_by_id = _gold_and_predictions(tmp_path, seed=43)
    configs = {}
    specs = [
        ("E01_qwen0_5b_prompt", "prompt_only"),
        ("E02_qwen0_5b_rag", "rag"),
        ("E03_qwen0_5b_ft", "ft"),
        ("E04_qwen0_5b_ft_rag", "ft_rag"),
    ]
    c_adapter = str(root / "E03_qwen0_5b_ft_43" / "adapter")
    Path(c_adapter).mkdir(parents=True)
    for name, method in specs:
        cfg, _ = _write_stat_run(
            root, name, method, 43, test_file, raw_by_id,
            adapter_path=c_adapter if method in {"ft", "ft_rag"} else None,
        )
        configs[name] = cfg

    monkeypatch.setattr(
        eval_stats, "load_cfg", lambda *, experiment, overrides: configs[experiment]
    )
    bundles = eval_stats.load_run_bundles(
        [name for name, _ in specs], ["output_root=" + str(root), "seed=43"]
    )
    validated = eval_stats.validate_paired_runs(bundles)

    assert validated["n_items"] == 2
    assert validated["methods"] == ["prompt_only", "rag", "ft", "ft_rag"]


def test_mismatched_test_items_are_rejected(tmp_path):
    root = tmp_path / "final"
    test_file, raw_by_id = _gold_and_predictions(tmp_path, seed=43)
    cfg_a, _ = _write_stat_run(root, "E01_qwen0_5b_prompt", "prompt_only", 43,
                               test_file, raw_by_id)
    bad_raw = dict(raw_by_id)
    bad_raw.pop("item-b")
    cfg_b, _ = _write_stat_run(root, "E02_qwen0_5b_rag", "rag", 43,
                               test_file, bad_raw)

    with pytest.raises(eval_stats.IncompatiblePairedRuns, match="item IDs"):
        eval_stats.validate_paired_runs([
            eval_stats.bundle_from_config(cfg_a),
            eval_stats.bundle_from_config(cfg_b),
        ])


def test_mismatched_dataset_hashes_are_rejected(tmp_path):
    root = tmp_path / "final"
    test_file, raw_by_id = _gold_and_predictions(tmp_path, seed=43)
    cfg_a, _ = _write_stat_run(root, "E01_qwen0_5b_prompt", "prompt_only", 43,
                               test_file, raw_by_id, test_hash="hash-a")
    cfg_b, _ = _write_stat_run(root, "E02_qwen0_5b_rag", "rag", 43,
                               test_file, raw_by_id, test_hash="hash-b")

    with pytest.raises(eval_stats.IncompatiblePairedRuns, match="test split hash"):
        eval_stats.validate_paired_runs([
            eval_stats.bundle_from_config(cfg_a),
            eval_stats.bundle_from_config(cfg_b),
        ])
