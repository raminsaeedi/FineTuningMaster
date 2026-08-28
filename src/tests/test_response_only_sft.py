"""Optional prompt-completion SFT without changing legacy full-text runs."""

from __future__ import annotations

import sys
import types

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from src.data_pipeline.formatter import split_training_text
from src.training.sft_trainer import QLoRASFTTrainer


def _cfg(*, completion_only_loss: bool):
    return OmegaConf.create(
        {
            "seed": 42,
            "model": {"hf_id": "Qwen/Qwen3-1.7B", "max_seq_length": 4096},
            "data": {},
            "training": {
                "sft": {
                    "completion_only_loss": completion_only_loss,
                    "gradient_checkpointing": False,
                }
            },
        }
    )


class _DummyModel:
    def parameters(self):
        return iter(())

    def named_parameters(self):
        return iter(())


def _run_with_fake_trl(monkeypatch, cfg, train_dataset, eval_dataset=None, *, compatible=True):
    captured = {}

    if compatible:
        class FakeSFTConfig:
            def __init__(self, *, completion_only_loss=None, **kwargs):
                captured["training_args"] = {
                    **kwargs,
                    "completion_only_loss": completion_only_loss,
                }
    else:
        class FakeSFTConfig:
            def __init__(self, *, output_dir):
                self.output_dir = output_dir

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            captured["train_dataset"] = kwargs["train_dataset"]
            captured["eval_dataset"] = kwargs["eval_dataset"]
            self.model = _DummyModel()

        def train(self, **kwargs):
            del kwargs
            self.state = types.SimpleNamespace(global_step=1, log_history=[])
            return types.SimpleNamespace(metrics={})

    fake_trl = types.ModuleType("trl")
    fake_trl.SFTConfig = FakeSFTConfig
    fake_trl.SFTTrainer = FakeSFTTrainer
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    trainer = QLoRASFTTrainer(cfg)
    trainer._setup = lambda: None
    trainer.model = _DummyModel()
    trainer.tokenizer = types.SimpleNamespace(eos_token="<eos>")
    trainer._save = lambda path: None
    trainer.train(train_dataset, eval_dataset, "unused-adapter")
    return captured


def test_formatted_text_splits_losslessly_at_final_json_object():
    text = 'Prompt containing {"schema": true}.\nAssistant:\n{\n  "answer": "ok"\n}<eos>'

    result = split_training_text(text, eos_token="<eos>")

    assert result == {
        "prompt": 'Prompt containing {"schema": true}.\nAssistant:\n',
        "completion": '{\n  "answer": "ok"\n}<eos>',
    }


def test_completion_only_flag_uses_prompt_completion_for_train_and_eval(monkeypatch):
    full_text = 'Prompt\nAssistant:\n{\n  "answer": "ok"\n}<eos>'
    train_dataset = Dataset.from_list([{"text": full_text}])
    eval_dataset = Dataset.from_list([{"text": full_text}])

    captured = _run_with_fake_trl(
        monkeypatch,
        _cfg(completion_only_loss=True),
        train_dataset,
        eval_dataset,
    )

    assert captured["training_args"]["completion_only_loss"] is True
    assert "dataset_text_field" not in captured["training_args"]
    assert captured["train_dataset"][0] == {
        "prompt": "Prompt\nAssistant:\n",
        "completion": '{\n  "answer": "ok"\n}<eos>',
    }
    assert captured["eval_dataset"][0] == captured["train_dataset"][0]


def test_legacy_flag_keeps_text_dataset_and_omits_native_override(monkeypatch):
    train_dataset = Dataset.from_list([{"text": "legacy full text"}])

    captured = _run_with_fake_trl(
        monkeypatch,
        _cfg(completion_only_loss=False),
        train_dataset,
    )

    assert captured["train_dataset"] is train_dataset
    assert captured["training_args"]["completion_only_loss"] is None
    assert captured["training_args"]["dataset_text_field"] == "text"


def test_completion_only_fails_clearly_when_trl_api_is_incompatible(monkeypatch):
    dataset = Dataset.from_list([{"text": 'Prompt\n{\n  "answer": "ok"\n}<eos>'}])

    with pytest.raises(RuntimeError, match="TRL.*completion_only_loss"):
        _run_with_fake_trl(
            monkeypatch,
            _cfg(completion_only_loss=True),
            dataset,
            compatible=False,
        )
