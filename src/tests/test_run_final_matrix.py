"""Ordering, path and completion logic of the final experiment matrix runner.

The runner is what actually produces the thesis results, so its two guarantees
are tested here without launching a single subprocess: the adapter producer runs
before its consumer, and a stage that already finished is not repeated.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "run_final_matrix", _ROOT / "experiments" / "scripts" / "run_final_matrix.py")
matrix_runner = importlib.util.module_from_spec(_SPEC)
sys.modules["run_final_matrix"] = matrix_runner
_SPEC.loader.exec_module(matrix_runner)


def _write_matrix(path: Path, matrix: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    return path


def _adapter(path: Path, *, config: bool = True, weights: str | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if config:
        (path / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    if weights:
        (path / weights).write_text("WEIGHTS", encoding="utf-8")
    return path


# --- matrix loading --------------------------------------------------------

def test_load_matrix_reads_runs(tmp_path):
    path = _write_matrix(tmp_path / "m.yaml", {
        "model": "qwen2_5_0_5b",
        "seeds": [42, 43],
        "runs": [{"key": "A", "experiment": "E01"}],
    })
    matrix = matrix_runner.load_matrix(path)
    assert matrix["seeds"] == [42, 43]
    assert matrix["runs"][0]["key"] == "A"


def test_load_matrix_fails_clearly_on_a_missing_file(tmp_path):
    with pytest.raises(SystemExit) as exc:
        matrix_runner.load_matrix(tmp_path / "absent.yaml")
    assert "not found" in str(exc.value)


@pytest.mark.parametrize("content", [{"model": "qwen2_5_0_5b"}, {"runs": []}, {}])
def test_load_matrix_rejects_a_matrix_without_runs(tmp_path, content):
    path = _write_matrix(tmp_path / "m.yaml", content)
    with pytest.raises(SystemExit) as exc:
        matrix_runner.load_matrix(path)
    assert "no runs" in str(exc.value)


# --- ordering --------------------------------------------------------------

def test_producers_run_before_consumers_regardless_of_yaml_order():
    """YAML order is documentation; ordered_runs is the guarantee."""
    matrix = {"runs": [
        {"key": "D", "experiment": "E04", "adapter_from": "C"},
        {"key": "A", "experiment": "E01"},
        {"key": "C", "experiment": "E03", "trains_adapter": True},
    ]}
    assert [r["key"] for r in matrix_runner.ordered_runs(matrix)] == ["A", "C", "D"]


def test_ordering_preserves_relative_order_within_each_group():
    matrix = {"runs": [
        {"key": "A", "experiment": "E01"},
        {"key": "B", "experiment": "E02"},
        {"key": "C", "experiment": "E03", "trains_adapter": True},
        {"key": "D", "experiment": "E04", "adapter_from": "C"},
    ]}
    assert [r["key"] for r in matrix_runner.ordered_runs(matrix)] == ["A", "B", "C", "D"]


def test_dangling_adapter_from_is_rejected():
    matrix = {"runs": [
        {"key": "A", "experiment": "E01"},
        {"key": "D", "experiment": "E04", "adapter_from": "Z"},
    ]}
    with pytest.raises(SystemExit) as exc:
        matrix_runner.ordered_runs(matrix)
    message = str(exc.value)
    assert "adapter_from=Z" in message and "D" in message


def test_shipped_final_matrix_puts_c_before_d():
    matrix = matrix_runner.load_matrix(matrix_runner.DEFAULT_MATRIX)
    keys = [r["key"] for r in matrix_runner.ordered_runs(matrix)]
    assert keys.index("C") < keys.index("D")
    consumer = next(r for r in matrix["runs"] if r["key"] == "D")
    assert consumer["adapter_from"] == "C"


# --- path helpers ----------------------------------------------------------

def test_experiment_id_joins_experiment_and_seed():
    assert matrix_runner.experiment_id("E03_qwen0_5b_ft", 43) == "E03_qwen0_5b_ft_43"


def test_run_and_adapter_directories(tmp_path):
    run = matrix_runner.run_dir(tmp_path, "E03_qwen0_5b_ft", 44)
    assert run == tmp_path / "E03_qwen0_5b_ft_44"
    assert matrix_runner.adapter_dir(tmp_path, "E03_qwen0_5b_ft", 44) == run / "adapter"


def test_adapter_directories_differ_per_seed(tmp_path):
    dirs = {matrix_runner.adapter_dir(tmp_path, "E03_qwen0_5b_ft", s) for s in (42, 43, 44)}
    assert len(dirs) == 3


# --- completion checks -----------------------------------------------------

def test_empty_directory_is_not_a_trained_adapter(tmp_path):
    empty = tmp_path / "adapter"
    empty.mkdir()
    assert matrix_runner.adapter_is_trained(empty) is False
    assert matrix_runner.adapter_is_trained(tmp_path / "absent") is False


def test_config_without_weights_is_not_a_trained_adapter(tmp_path):
    assert matrix_runner.adapter_is_trained(_adapter(tmp_path / "adapter")) is False


def test_weights_without_config_is_not_a_trained_adapter(tmp_path):
    path = _adapter(tmp_path / "adapter", config=False,
                    weights="adapter_model.safetensors")
    assert matrix_runner.adapter_is_trained(path) is False


@pytest.mark.parametrize("weights", ["adapter_model.safetensors", "adapter_model.bin"])
def test_config_plus_weights_is_a_trained_adapter(tmp_path, weights):
    path = _adapter(tmp_path / weights, weights=weights)
    assert matrix_runner.adapter_is_trained(path) is True


def test_run_is_complete_is_keyed_on_metrics_auto(tmp_path):
    run = tmp_path / "E01_qwen0_5b_prompt_42"
    run.mkdir()
    assert matrix_runner.run_is_complete(run) is False

    # Predictions alone are not completion — evaluation must have run too.
    (run / "predictions.jsonl").write_text('{"item_id": "a"}\n', encoding="utf-8")
    assert matrix_runner.run_is_complete(run) is False

    (run / "metrics_auto.json").write_text('{"n_predictions": 1}', encoding="utf-8")
    assert matrix_runner.run_is_complete(run) is True


# --- command construction --------------------------------------------------

def test_commands_carry_the_seed_and_extra_overrides():
    train = matrix_runner.train_cmd("E03_qwen0_5b_ft", 43, ["output_root=x"])
    assert "--experiment" in train and "E03_qwen0_5b_ft" in train
    assert "seed=43" in train and train[-1] == "output_root=x"
    assert train[1].endswith("train.py")

    run = matrix_runner.experiment_cmd("E04_qwen0_5b_ft_rag", 43, [])
    assert run[1].endswith("run_experiment.py")
    assert "seed=43" in run


def test_hydra_path_quotes_external_windows_paths():
    assert matrix_runner._hydra_path(r"C:\Users\Researcher\run outputs") == (
        '"C:/Users/Researcher/run outputs"'
    )
    assert matrix_runner._hydra_path("/mnt/scratch/runs") == "/mnt/scratch/runs"


# --- dataset isolation -----------------------------------------------------

def _completed_run(path: Path, dataset: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics_auto.json").write_text('{"n_predictions": 2}', encoding="utf-8")
    (path / "manifest.json").write_text(
        '{"profile": "final", "model_key": "qwen3_8b", "method_key": "A", "seed": 42, '
        f'"status": "completed", "dataset_version": "{dataset}"}}',
        encoding="utf-8",
    )
    return path


def test_a_completed_run_of_another_dataset_is_not_a_cache_hit(tmp_path):
    run = _completed_run(tmp_path / "run", "dashboard_v3")
    kwargs = dict(model="qwen3_8b", method="A", seed=42, profile="final")
    assert matrix_runner._compatible_complete(run, dataset="dashboard_v3", **kwargs) is True
    assert matrix_runner._compatible_complete(run, dataset="dashboard_v4", **kwargs) is False


def test_profile_run_dir_separates_datasets(tmp_path):
    v3 = matrix_runner.profile_run_dir(tmp_path, "qwen3_8b", "C", 42, "dashboard_v3")
    v4 = matrix_runner.profile_run_dir(tmp_path, "qwen3_8b", "C", 42, "dashboard_v4")
    assert v3 == tmp_path / "dashboard_v3" / "qwen3_8b" / "C" / "seed_42"
    assert v4 == tmp_path / "dashboard_v4" / "qwen3_8b" / "C" / "seed_42"
    assert matrix_runner.DEFAULT_DATASET == "dashboard_v4"


def test_dataset_group_override_is_not_duplicated():
    with_caller = matrix_runner._base_overrides(
        profile="final", output_root_arg="out", model="qwen3_8b", method="A", seed=42,
        extra=["data=dashboard_v3"], dataset="dashboard_v3",
    )
    assert with_caller.count("data=dashboard_v3") == 1
    without_caller = matrix_runner._base_overrides(
        profile="final", output_root_arg="out", model="qwen3_8b", method="A", seed=42,
        extra=[], dataset="dashboard_v4",
    )
    assert without_caller[0] == "data=dashboard_v4"
