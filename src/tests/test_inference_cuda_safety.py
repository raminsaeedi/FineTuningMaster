"""Regression tests for CUDA-safe HuggingFace inference."""

from __future__ import annotations

import json
import sys
import types

import pytest
import torch

from src.core.schemas import DashboardBrief, GenerationResult
from src.inference.runner import InferenceRunner
from src.methods.base import RAGHFMethod
from src.models.hf_causal import HFCausalModel


class _Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "rendered prompt"

    def __call__(self, prompt, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs):
        return "safe output"


class _SamplingBoundary(torch.nn.Module):
    """Stand-in for Transformers generation at the external model boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, **kwargs):
        if not kwargs.get("remove_invalid_values"):
            raise RuntimeError("CUDA error: device-side assert triggered")
        if not kwargs.get("renormalize_logits"):
            raise RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")
        return torch.cat([input_ids, torch.tensor([[3]], device=input_ids.device)], dim=1)


def test_chat_sanitizes_sampling_scores_before_multinomial():
    model = HFCausalModel({"max_seq_length": 128})
    model.model = _SamplingBoundary()
    model.tokenizer = _Tokenizer()
    model.max_seq_length = 128

    assert model.chat(
        "system", "user", do_sample=True, max_new_tokens=16
    ) == "safe output"


def test_chat_rejects_silent_prompt_truncation():
    model = HFCausalModel({"max_seq_length": 17})
    model.model = _SamplingBoundary()
    model.tokenizer = _Tokenizer()
    model.max_seq_length = 17

    with pytest.raises(ValueError, match="Prompt exceeds"):
        model.chat("system", "user", max_new_tokens=16)


def test_inference_loader_forwards_4bit_quantization(monkeypatch):
    import src.models.hf_causal as hf_causal
    from src.utils import gpu_precision

    captured = {}

    class DummyTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace()

    class FakeAutoLoader:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise AssertionError("patched loader boundary should intercept this call")

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def fake_load(_loader, _name, *, kwargs, logger, component):
        del logger
        captured[component] = kwargs
        return DummyTokenizer() if component == "inference tokenizer" else DummyModel()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_precision, "resolve_inference_dtype", lambda _value: "bfloat16")
    monkeypatch.setattr(hf_causal, "load_pretrained_with_cache_repair", fake_load)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoLoader
    fake_transformers.AutoTokenizer = FakeAutoLoader
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = HFCausalModel(
        {"name": "fake/27b", "max_seq_length": 4096, "dtype": "bfloat16"},
        {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        },
    ).load()

    quantization = captured["inference model"]["quantization_config"]
    assert quantization.load_in_4bit is True
    assert quantization.bnb_4bit_quant_type == "nf4"
    assert quantization.bnb_4bit_compute_dtype == torch.bfloat16
    assert captured["inference model"]["low_cpu_mem_usage"] is True
    assert model.model.training is False


class _CharacterTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(message["content"] for message in messages) + "\nassistant"

    def __call__(self, text, return_tensors=None, **kwargs):
        del kwargs
        ids = [ord(character) for character in text]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": ids}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


def test_rag_context_is_shared_across_all_passages_and_fits_budget():
    method = RAGHFMethod({
        "model": {"max_seq_length": 1800},
        "method": {
            "generate": {"max_new_tokens": 300},
            "retriever": {"top_k": 3},
        },
    })
    method.model = HFCausalModel({"max_seq_length": 1800})
    method.model.tokenizer = _CharacterTokenizer()
    method.model.max_seq_length = 1800
    passages = [
        {
            "source": f"source-{index}",
            "heading": f"heading-{index}",
            "text": chr(64 + index) * 1000,
        }
        for index in range(1, 4)
    ]

    system, truncated = method._fit_system_prompt(passages, "short user brief")

    assert truncated is True
    assert method.model.prompt_token_count(system, "short user brief") <= 1500
    for index, marker in enumerate(("A", "B", "C"), start=1):
        assert f"source-{index}" in system
        assert marker in system
        assert marker * 1000 not in system


class _FatalCudaMethod:
    name = "fatal-cuda"
    config_hash = "test"

    def __init__(self) -> None:
        self.generated: list[str] = []
        self.teardown_calls = 0

    def setup(self) -> None:
        pass

    def generate(self, brief: DashboardBrief) -> GenerationResult:
        self.generated.append(brief.item_id)
        if len(self.generated) == 1:
            raise RuntimeError("CUDA error: device-side assert triggered")
        return GenerationResult(
            item_id=brief.item_id,
            method_name=self.name,
            model_name="fake-model",
            config_hash=self.config_hash,
        )

    def teardown(self) -> None:
        self.teardown_calls += 1


def test_runner_stops_after_first_fatal_cuda_error(tmp_path):
    method = _FatalCudaMethod()
    briefs = [DashboardBrief(item_id="first"), DashboardBrief(item_id="second")]
    runner = InferenceRunner(method, tmp_path / "predictions.jsonl")

    with pytest.raises(RuntimeError, match="device-side assert triggered"):
        runner.run(briefs)

    errors = [json.loads(line) for line in runner.errors_path.read_text().splitlines()]
    assert method.generated == ["first"]
    assert method.teardown_calls == 1
    assert [error["item_id"] for error in errors] == ["first"]
