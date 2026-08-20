"""Regression checks for the reproducible tiny dashboard_v4 derivative."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from src.utils.adapter import check_adapter_compatibility, resolve_adapter_path


_ROOT = Path(__file__).resolve().parents[2]
_BUILDER_PATH = _ROOT / "experiments" / "scripts" / "build_dashboard_v4_tiny.py"
_SPEC = importlib.util.spec_from_file_location("build_dashboard_v4_tiny", _BUILDER_PATH)
builder = importlib.util.module_from_spec(_SPEC)
sys.modules["build_dashboard_v4_tiny"] = builder
_SPEC.loader.exec_module(builder)

_RUNNER_PATH = _ROOT / "experiments" / "scripts" / "run_tiny_v4_kaggle.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("run_tiny_v4_kaggle", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules["run_tiny_v4_kaggle"] = runner
_RUNNER_SPEC.loader.exec_module(runner)


def test_tiny_dataset_integrity_and_parent_lineage():
    builder.verify(
        source=_ROOT / "data" / "frozen" / "dashboard_v4",
        out=_ROOT / "data" / "frozen" / "dashboard_v4_tiny",
    )


def test_sports_config_can_reuse_dashboard_tiny_adapter(tmp_path):
    cfg = {
        "output_root": "runs",
        "run_layout": "final",
        "seed": 42,
        "model_key": "qwen2_5_0_5b",
        "method_key": "C",
        "model": {"hf_id": "Qwen/Qwen2.5-0.5B-Instruct", "key": "qwen2_5_0_5b"},
        "training": {},
        "data": {
            "dataset_version": "sports_v4_tiny",
            "adapter_dataset_version": "dashboard_v4_tiny",
        },
        "method": {},
    }
    resolved = resolve_adapter_path(cfg, tmp_path)
    assert resolved == (
        tmp_path / "runs" / "dashboard_v4_tiny" / "qwen2_5_0_5b" / "C" / "seed_42" / "adapter"
    )
    metadata = {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_key": "qwen2_5_0_5b",
        "seed": 42,
        "dataset_version": "dashboard_v4_tiny",
    }
    assert check_adapter_compatibility(metadata, cfg) == []


def test_tiny_run_verifier_enforces_format_acceptance_gate(tmp_path):
    run_dir = tmp_path / "demo" / "model" / "B" / "seed_42"
    run_dir.mkdir(parents=True)
    for name in (
        "predictions.jsonl",
        "config_snapshot.yaml",
        "config_hash.txt",
        "cache_identity.json",
    ):
        (run_dir / name).write_text("\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "completed", "dataset_version": "demo"}),
        encoding="utf-8",
    )
    (run_dir / "metrics_auto.json").write_text(
        json.dumps({
            "coverage": {"n_requested": 10, "n_predictions": 10, "n_missing": 0},
            "metrics": {
                "schema_compliance": {
                    "json_parse_rate": 94.0,
                    "schema_validity_rate": 89.0,
                    "required_keys_rate": 100.0,
                    "completeness_score": 1.0,
                }
            },
        }),
        encoding="utf-8",
    )

    problems = runner._verify_run(
        tmp_path,
        dataset="demo",
        model="model",
        method="B",
        seed=42,
        expected_items=10,
    )

    assert any("json_parse_rate" in problem for problem in problems)
    assert any("schema_validity_rate" in problem for problem in problems)
