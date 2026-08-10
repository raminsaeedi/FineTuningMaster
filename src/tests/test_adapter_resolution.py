"""Adapter resolution and compatibility checks (the C -> D hand-over).

Method D consumes the adapter method C trained. Two things must hold or the
comparison is worthless: D must load the adapter of the *same seed*, and a
mismatched adapter must fail loudly instead of being silently reused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.adapter import (
    AdapterError,
    check_adapter_compatibility,
    read_training_metadata,
    resolve_adapter_path,
    validate_adapter,
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _cfg(tmp_path: Path, **overrides) -> dict:
    """A minimal resolved-config stand-in (plain dict, like OmegaConf here)."""
    cfg = {
        "output_root": str(tmp_path / "outputs"),
        "experiment_id": "E04_qwen0_5b_ft_rag_42",
        "seed": 42,
        "model": {"hf_id": BASE_MODEL, "name": "qwen2_5_0_5b"},
        "data": {"dataset_version": "dashboard_v3"},
        "method": {"name": "ft_rag", "type": "fine_tuned_rag",
                   "adapter_path": None, "adapter_source_experiment": None},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


def _make_adapter(
    path: Path,
    *,
    weights: str = "adapter_model.safetensors",
    metadata: dict | None = None,
    config: bool = True,
) -> Path:
    """Create a folder that looks like a saved PEFT adapter."""
    path.mkdir(parents=True, exist_ok=True)
    if config:
        (path / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    if weights:
        (path / weights).write_text("WEIGHTS", encoding="utf-8")
    if metadata is not None:
        (path / "training_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    return path


# --- resolution order ------------------------------------------------------

def test_explicit_adapter_path_wins_over_source_experiment(tmp_path):
    explicit = tmp_path / "somewhere" / "else" / "adapter"
    cfg = _cfg(tmp_path, method={
        "adapter_path": str(explicit),
        "adapter_source_experiment": "E03_qwen0_5b_ft",
    })
    assert resolve_adapter_path(cfg) == explicit


def test_relative_explicit_adapter_path_is_anchored_at_project_root(tmp_path):
    cfg = _cfg(tmp_path, method={"adapter_path": "outputs/manual/adapter"})
    resolved = resolve_adapter_path(cfg, project_root=tmp_path)
    assert resolved == tmp_path / "outputs" / "manual" / "adapter"
    assert resolved.is_absolute()


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_source_experiment_resolves_per_seed(tmp_path, seed):
    """The C -> D pairing guarantee: seed 43 must load the seed-43 adapter."""
    cfg = _cfg(
        tmp_path,
        seed=seed,
        experiment_id=f"E04_qwen0_5b_ft_rag_{seed}",
        method={"adapter_source_experiment": "E03_qwen0_5b_ft"},
    )
    expected = tmp_path / "outputs" / f"E03_qwen0_5b_ft_{seed}" / "adapter"
    assert resolve_adapter_path(cfg) == expected


def test_seeds_never_share_an_adapter_directory(tmp_path):
    paths = {
        seed: resolve_adapter_path(
            _cfg(tmp_path, seed=seed,
                 method={"adapter_source_experiment": "E03_qwen0_5b_ft"})
        )
        for seed in (42, 43, 44)
    }
    assert len(set(paths.values())) == 3


def test_without_source_experiment_it_uses_the_runs_own_directory(tmp_path):
    """Correct for method C, which both trains and consumes its adapter."""
    cfg = _cfg(tmp_path, experiment_id="E03_qwen0_5b_ft_44", seed=44)
    assert resolve_adapter_path(cfg) == (
        tmp_path / "outputs" / "E03_qwen0_5b_ft_44" / "adapter"
    )


def test_relative_output_root_is_anchored_at_project_root(tmp_path):
    cfg = _cfg(tmp_path, output_root="experiments/outputs/final",
               experiment_id="E03_qwen0_5b_ft_42")
    assert resolve_adapter_path(cfg, project_root=tmp_path) == (
        tmp_path / "experiments" / "outputs" / "final"
        / "E03_qwen0_5b_ft_42" / "adapter"
    )


# --- existence validation --------------------------------------------------

def test_missing_adapter_directory_explains_how_to_produce_it(tmp_path):
    cfg = _cfg(tmp_path, seed=43)
    missing = tmp_path / "outputs" / "E03_qwen0_5b_ft_43" / "adapter"

    with pytest.raises(AdapterError) as exc:
        validate_adapter(missing, cfg)

    message = str(exc.value)
    assert str(missing) in message
    assert "seed=43" in message          # names the seed that must be trained
    assert "train.py" in message


def test_missing_adapter_config_is_rejected(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter", config=False)

    with pytest.raises(AdapterError) as exc:
        validate_adapter(adapter, _cfg(tmp_path))
    assert "adapter_config.json" in str(exc.value)


def test_adapter_without_weights_is_rejected(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter", weights="")

    with pytest.raises(AdapterError) as exc:
        validate_adapter(adapter, _cfg(tmp_path))
    assert "no adapter weights" in str(exc.value)


@pytest.mark.parametrize("weights", ["adapter_model.safetensors", "adapter_model.bin"])
def test_either_weight_format_is_accepted(tmp_path, weights):
    adapter = _make_adapter(tmp_path / weights, weights=weights)
    report = validate_adapter(adapter, _cfg(tmp_path))
    assert report["problems"] == []
    assert report["adapter_dir"] == str(adapter)


# --- compatibility ---------------------------------------------------------

def test_compatible_metadata_reports_no_problems(tmp_path):
    metadata = {"base_model": BASE_MODEL, "seed": 42, "dataset_version": "dashboard_v3"}
    assert check_adapter_compatibility(metadata, _cfg(tmp_path)) == []


def test_base_model_mismatch_is_detected(tmp_path):
    metadata = {"base_model": "meta-llama/Llama-3.2-1B", "seed": 42}
    problems = check_adapter_compatibility(metadata, _cfg(tmp_path))
    assert len(problems) == 1 and "base model mismatch" in problems[0]


def test_seed_mismatch_is_detected(tmp_path):
    metadata = {"base_model": BASE_MODEL, "seed": 42}
    problems = check_adapter_compatibility(metadata, _cfg(tmp_path, seed=44))
    assert len(problems) == 1 and "seed mismatch" in problems[0]
    assert "42" in problems[0] and "44" in problems[0]


def test_dataset_version_mismatch_is_detected(tmp_path):
    metadata = {"base_model": BASE_MODEL, "seed": 42, "dataset_version": "dashboard_v2"}
    problems = check_adapter_compatibility(metadata, _cfg(tmp_path))
    assert len(problems) == 1 and "dataset version mismatch" in problems[0]


def test_all_mismatches_are_reported_together(tmp_path):
    metadata = {"base_model": "other/model", "seed": 1, "dataset_version": "dashboard_v1"}
    assert len(check_adapter_compatibility(metadata, _cfg(tmp_path))) == 3


def test_fields_absent_from_metadata_are_skipped_not_flagged(tmp_path):
    """Older adapters predate dataset_version — absence is not a mismatch."""
    assert check_adapter_compatibility({}, _cfg(tmp_path)) == []
    assert check_adapter_compatibility({"seed": 42}, _cfg(tmp_path)) == []
    assert check_adapter_compatibility(
        {"base_model": BASE_MODEL, "seed": 42}, _cfg(tmp_path)
    ) == []


def test_seed_is_compared_numerically_not_as_text(tmp_path):
    assert check_adapter_compatibility({"seed": "42"}, _cfg(tmp_path)) == []


# --- validate_adapter end to end ------------------------------------------

def test_validate_adapter_returns_metadata_when_compatible(tmp_path):
    metadata = {"base_model": BASE_MODEL, "seed": 42, "dataset_version": "dashboard_v3"}
    adapter = _make_adapter(tmp_path / "adapter", metadata=metadata)

    report = validate_adapter(adapter, _cfg(tmp_path))
    assert report["metadata"] == metadata
    assert report["problems"] == []


def test_strict_validation_refuses_a_mismatched_adapter(tmp_path):
    adapter = _make_adapter(
        tmp_path / "adapter", metadata={"base_model": BASE_MODEL, "seed": 42}
    )

    with pytest.raises(AdapterError) as exc:
        validate_adapter(adapter, _cfg(tmp_path, seed=43))
    message = str(exc.value)
    assert "seed mismatch" in message
    assert "method.adapter_path" in message   # names the deliberate override


def test_non_strict_validation_reports_instead_of_raising(tmp_path):
    adapter = _make_adapter(
        tmp_path / "adapter", metadata={"base_model": BASE_MODEL, "seed": 42}
    )

    report = validate_adapter(adapter, _cfg(tmp_path, seed=43), strict=False)
    assert any("seed mismatch" in p for p in report["problems"])


def test_non_strict_validation_still_requires_the_directory(tmp_path):
    """Existence is never downgraded — a missing adapter cannot be reported past."""
    with pytest.raises(AdapterError):
        validate_adapter(tmp_path / "nope", _cfg(tmp_path), strict=False)


# --- metadata reading ------------------------------------------------------

def test_read_training_metadata_tolerates_absent_and_corrupt_files(tmp_path):
    assert read_training_metadata(tmp_path / "nope") == {}

    adapter = _make_adapter(tmp_path / "adapter")
    assert read_training_metadata(adapter) == {}

    (adapter / "training_metadata.json").write_text("{not json", encoding="utf-8")
    assert read_training_metadata(adapter) == {}
