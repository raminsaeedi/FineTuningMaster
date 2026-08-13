"""Focused release checks for the final multi-model experiment contract."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.evaluation.aggregator import collect_rows
from src.models.hf_utils import chat_template_kwargs, from_pretrained_kwargs
from src.training.sft_trainer import build_lora_kwargs
from src.utils.artifacts import experiment_dir, write_run_metadata
from src.utils.config import load_cfg
from src.utils.config_hash import hash_config
from src.utils.adapter import check_adapter_compatibility, resolve_adapter_path

ROOT = Path(__file__).resolve().parents[2]
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "final_matrix_architecture",
    ROOT / "experiments" / "scripts" / "run_final_matrix.py",
)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules["final_matrix_architecture"] = runner
RUNNER_SPEC.loader.exec_module(runner)


MODEL_EXPECTATIONS = {
    "qwen2_5_0_5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3_1_7b": "Qwen/Qwen3-1.7B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def _cfg(tmp_path: Path, *, model: str = "qwen3_8b", method: str = "D", seed: int = 42):
    return load_cfg(
        experiment="E04_qwen0_5b_ft_rag" if method == "D" else "E03_qwen0_5b_ft",
        overrides=[
            f"model={model}",
            "profile=final",
            "run_layout=final",
            f"model_key={model}",
            f"method_key={method}",
            f"seed={seed}",
            f"output_root={str(tmp_path / 'outputs').replace(chr(92), '/')}",
            f"experiment_id={model}_{method}_seed_{seed}",
            "method.adapter_source_experiment=E03_qwen0_5b_ft",
            "method.adapter_source_method_key=C",
        ],
    )


@pytest.mark.parametrize("model_key,hf_id", MODEL_EXPECTATIONS.items())
def test_all_five_model_profiles_compose(model_key, hf_id):
    cfg = load_cfg(experiment="E01_qwen0_5b_prompt", overrides=[f"model={model_key}"])
    assert cfg.model.key == model_key
    assert cfg.model.hf_id == hf_id


def test_qwen3_disables_thinking_without_prompt_markers():
    cfg = load_cfg(experiment="E01_qwen0_5b_prompt", overrides=["model=qwen3_8b"])
    assert chat_template_kwargs(cfg.model) == {"enable_thinking": False}
    assert "/think" not in str(cfg.model.chat_template)
    assert "/no_think" not in str(cfg.model.chat_template)


def test_llama_uses_generic_profile_and_marks_gated_access():
    cfg = load_cfg(experiment="E01_qwen0_5b_prompt", overrides=["model=llama3_1_8b"])
    assert cfg.model.trust_remote_code is False
    assert cfg.model.requires_hf_token is True
    assert chat_template_kwargs(cfg.model) == {}


def test_hf_token_is_forwarded_only_to_loader_kwargs(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-secret")
    cfg = {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "revision": None}
    kwargs = from_pretrained_kwargs(cfg)
    assert kwargs["token"] == "hf-test-secret"


def test_hf_token_never_enters_run_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-secret")
    monkeypatch.setattr("src.utils.artifacts._pip_freeze", lambda: "torch==2.6.0\n")
    monkeypatch.setattr("src.utils.artifacts._hardware_provenance", lambda: {})
    monkeypatch.setattr("src.utils.artifacts._package_versions", lambda: {})
    monkeypatch.setattr("src.utils.artifacts.get_git_hash", lambda: "deadbeef")
    monkeypatch.setattr("src.utils.artifacts.is_git_dirty", lambda: False)
    cfg = {
        "output_root": str(tmp_path / "outputs"),
        "experiment_id": "qwen3_8b_A_seed_42",
        "experiment_name": "qwen3_8b_A",
        "profile": "final",
        "run_layout": "final",
        "model_key": "qwen3_8b",
        "method_key": "A",
        "seed": 42,
        "model": {"key": "qwen3_8b", "name": "Qwen/Qwen3-8B", "hf_id": "Qwen/Qwen3-8B"},
        "data": {"dataset_version": "dashboard_v3"},
        "method": {"name": "prompt_only", "type": "prompt_only"},
        "training": {},
    }
    write_run_metadata(tmp_path, cfg)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "hf-test-secret" not in path.read_text(encoding="utf-8", errors="ignore")


def test_final_and_smoke_output_paths_are_separate(tmp_path):
    cfg = _cfg(tmp_path, model="qwen3_8b", method="A")
    final_path = experiment_dir(cfg, tmp_path)
    smoke_cfg = dict(cfg)
    smoke_cfg["output_root"] = str(tmp_path / "smoke_outputs")
    smoke_cfg["profile"] = "smoke"
    smoke_cfg["run_layout"] = "smoke"
    smoke_path = experiment_dir(smoke_cfg, tmp_path)
    assert final_path == tmp_path / "outputs" / "dashboard_v3" / "qwen3_8b" / "A" / "seed_42"
    assert smoke_path == tmp_path / "smoke_outputs" / "dashboard_v3" / "qwen3_8b" / "A" / "seed_42"
    assert smoke_path != final_path


def test_profile_runner_paths_are_unique_per_model_method_seed(tmp_path):
    paths = {
        runner.profile_run_dir(tmp_path, model, method, seed)
        for model in ("qwen3_1_7b", "qwen3_8b")
        for method in "ABCD"
        for seed in (42, 43, 44)
    }
    assert len(paths) == 2 * 4 * 3


def test_d_resolves_exact_same_model_and_seed_c_adapter(tmp_path):
    cfg = _cfg(tmp_path, model="qwen3_8b", method="D", seed=43)
    assert resolve_adapter_path(cfg, tmp_path) == (
        tmp_path / "outputs" / "dashboard_v3" / "qwen3_8b" / "C" / "seed_43" / "adapter"
    )


def test_adapter_compatibility_rejects_wrong_model_and_seed():
    cfg = {
        "model_key": "qwen3_8b",
        "model": {"hf_id": "Qwen/Qwen3-8B", "key": "qwen3_8b"},
        "seed": 42,
        "data": {"dataset_version": "dashboard_v3"},
    }
    problems = check_adapter_compatibility(
        {"base_model": "Qwen/Qwen3-1.7B", "model_key": "qwen3_1_7b", "seed": 43, "dataset_version": "dashboard_v3"},
        cfg,
    )
    assert any("base model mismatch" in problem for problem in problems)
    assert any("model key mismatch" in problem for problem in problems)
    assert any("seed mismatch" in problem for problem in problems)


def test_all_linear_qlora_is_model_agnostic():
    assert build_lora_kwargs({"target_modules": "all-linear"})["target_modules"] == "all-linear"


def test_recursive_aggregation_discovers_profile_runs(tmp_path):
    run = tmp_path / "dashboard_v3" / "qwen3_8b" / "A" / "seed_42"
    run.mkdir(parents=True)
    (run / "metrics_auto.json").write_text('{"n_predictions": 1}', encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "qwen3_8b_A_seed_42", "experiment_id": "qwen3_8b_A_seed_42",
        "model_key": "qwen3_8b", "model_hf_id": "Qwen/Qwen3-8B",
        "method_key": "A", "method": "prompt_only", "seed": 42,
        "profile": "final", "status": "completed", "dataset_version": "dashboard_v3",
    }), encoding="utf-8")
    rows = collect_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["model_key"] == "qwen3_8b"
    assert rows[0]["method_key"] == "A"
