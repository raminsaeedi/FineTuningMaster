"""Focused tests for explicit training checkpoint resume."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.training.sft_trainer import QLoRASFTTrainer


_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "train_script", _ROOT / "experiments" / "scripts" / "train.py"
)
train_script = importlib.util.module_from_spec(_SPEC)
sys.modules["train_script"] = train_script
_SPEC.loader.exec_module(train_script)


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _cfg(tmp_path: Path, **overrides):
    cfg = OmegaConf.create({
        "experiment_name": "E03_qwen0_5b_ft",
        "experiment_id": "E03_qwen0_5b_ft_42",
        "seed": 42,
        "model": {"name": "qwen2_5_0_5b", "hf_id": BASE_MODEL},
        "data": {
            "dataset_version": "dashboard_v3",
            "train_file": str(tmp_path / "train.jsonl"),
        },
        "training": {"type": "qlora_sft", "sft": {"learning_rate": 2.0e-4}},
    })
    for key, value in overrides.items():
        if isinstance(value, dict) and key in cfg:
            for nested_key, nested_value in value.items():
                cfg[key][nested_key] = nested_value
        else:
            cfg[key] = value
    return cfg


def _checkpoint(path: Path, step: int, metadata: dict | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    (path / "optimizer.pt").write_bytes(b"state")
    if metadata is not None:
        (path.parent / "resume_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    return path


def _metadata(tmp_path: Path, **overrides) -> dict:
    metadata = train_script.build_resume_metadata(_cfg(tmp_path), tmp_path)
    metadata.update(overrides)
    return metadata


def test_no_resume_flag_does_not_select_existing_checkpoint(tmp_path):
    _checkpoint(tmp_path / "checkpoints" / "checkpoint-10", 10)

    assert train_script.resolve_resume_checkpoint(
        tmp_path, _cfg(tmp_path), resume=False, project_root=tmp_path
    ) is None


def test_newest_checkpoint_is_selected_numerically(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    _checkpoint(checkpoint_dir / "checkpoint-9", 9)
    _checkpoint(checkpoint_dir / "checkpoint-100", 100)
    _checkpoint(checkpoint_dir / "checkpoint-20", 20)
    metadata = _metadata(tmp_path)
    (checkpoint_dir / "resume_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    selected = train_script.resolve_resume_checkpoint(
        tmp_path, _cfg(tmp_path), resume=True, project_root=tmp_path
    )

    assert selected == checkpoint_dir / "checkpoint-100"


def test_explicit_checkpoint_overrides_automatic_discovery(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    _checkpoint(checkpoint_dir / "checkpoint-100", 100)
    explicit = _checkpoint(checkpoint_dir / "checkpoint-20", 20)
    (checkpoint_dir / "resume_metadata.json").write_text(
        json.dumps(_metadata(tmp_path)), encoding="utf-8"
    )

    selected = train_script.resolve_resume_checkpoint(
        tmp_path, _cfg(tmp_path), resume=True, resume_from=str(explicit),
        project_root=tmp_path
    )

    assert selected == explicit


def test_resume_without_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(train_script.ResumeError, match="No valid checkpoint"):
        train_script.resolve_resume_checkpoint(
            tmp_path, _cfg(tmp_path), resume=True, project_root=tmp_path
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_hf_id", "other/model", "model"),
        ("seed", 43, "seed"),
        ("dataset_version", "dashboard_v2", "dataset"),
    ],
)
def test_incompatible_checkpoint_metadata_is_rejected(tmp_path, field, value, message):
    checkpoint_dir = tmp_path / "checkpoints"
    _checkpoint(checkpoint_dir / "checkpoint-10", 10)
    metadata = _metadata(tmp_path, **{field: value})
    (checkpoint_dir / "resume_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(train_script.ResumeError, match=message):
        train_script.resolve_resume_checkpoint(
            tmp_path, _cfg(tmp_path), resume=True, project_root=tmp_path
        )


def test_legacy_checkpoint_requires_explicit_path(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoints" / "checkpoint-10", 10)

    with pytest.raises(train_script.ResumeError, match="metadata"):
        train_script.resolve_resume_checkpoint(
            tmp_path, _cfg(tmp_path), resume=True, project_root=tmp_path
        )

    assert train_script.resolve_resume_checkpoint(
        tmp_path, _cfg(tmp_path), resume=True, resume_from=str(checkpoint),
        project_root=tmp_path
    ) == checkpoint


def test_partial_metadata_requires_explicit_path(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoints" / "checkpoint-10", 10)
    (tmp_path / "checkpoints" / "resume_metadata.json").write_text(
        json.dumps({"experiment_id": "E03_qwen0_5b_ft_42", "seed": 42}),
        encoding="utf-8",
    )

    with pytest.raises(train_script.ResumeError, match="sufficient compatibility"):
        train_script.resolve_resume_checkpoint(
            tmp_path, _cfg(tmp_path), resume=True, project_root=tmp_path
        )

    assert train_script.resolve_resume_checkpoint(
        tmp_path, _cfg(tmp_path), resume=True, resume_from=str(checkpoint),
        project_root=tmp_path
    ) == checkpoint


def test_manifest_records_resume_fields(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"experiment_id": "E03_42"}), encoding="utf-8")

    updated = train_script.update_resume_manifest(
        tmp_path,
        resumed=True,
        resume_checkpoint=str(tmp_path / "checkpoints" / "checkpoint-10"),
        initial_global_step=10,
        final_global_step=20,
        resume_timestamp="2026-08-12T10:00:00+00:00",
    )

    assert updated["resumed"] is True
    assert updated["resume_checkpoint"].endswith("checkpoint-10")
    assert updated["initial_global_step"] == 10
    assert updated["final_global_step"] == 20
    assert updated["resume_timestamp"] == "2026-08-12T10:00:00+00:00"


def test_trainer_forwards_resume_checkpoint_and_keeps_adapter_path(tmp_path, monkeypatch):
    captured = {}

    class FakeTrainingConfig:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            pass

        def train(self, *args, **kwargs):
            captured["resume_from_checkpoint"] = kwargs.get("resume_from_checkpoint")
            self.state = types.SimpleNamespace(global_step=12)
            return types.SimpleNamespace(metrics={"global_step": 12})

    fake_trl = types.ModuleType("trl")
    fake_trl.SFTConfig = FakeTrainingConfig
    fake_trl.SFTTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    cfg = _cfg(tmp_path)
    trainer = QLoRASFTTrainer(cfg)
    trainer._setup = lambda: None
    trainer.model = object()
    trainer.tokenizer = object()
    trainer._save = lambda path: captured.setdefault("adapter_path", path)

    output_dir = tmp_path / "adapter"
    trainer.train([], None, str(output_dir), resume_from_checkpoint="checkpoint-10")

    assert captured["resume_from_checkpoint"] == "checkpoint-10"
    assert captured["adapter_path"] == output_dir


def test_trainer_without_resume_uses_normal_native_call(tmp_path, monkeypatch):
    captured = {}

    class FakeTrainingConfig:
        def __init__(self, **kwargs):
            pass

    class FakeTrainer:
        def __init__(self, **kwargs):
            pass

        def train(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            self.state = types.SimpleNamespace(global_step=3)
            return types.SimpleNamespace(metrics={})

    fake_trl = types.ModuleType("trl")
    fake_trl.SFTConfig = FakeTrainingConfig
    fake_trl.SFTTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    trainer = QLoRASFTTrainer(_cfg(tmp_path))
    trainer._setup = lambda: None
    trainer.model = object()
    trainer.tokenizer = object()
    trainer._save = lambda path: None
    trainer.train([], None, str(tmp_path / "adapter"))

    assert captured["kwargs"] == {}
