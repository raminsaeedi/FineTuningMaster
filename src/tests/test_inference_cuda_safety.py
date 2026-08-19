"""Regression tests for CUDA-safe HuggingFace inference."""

from __future__ import annotations

import json

import pytest
import torch

from src.core.schemas import DashboardBrief, GenerationResult
from src.inference.runner import InferenceRunner
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

    assert model.chat("system", "user", do_sample=True) == "safe output"


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
