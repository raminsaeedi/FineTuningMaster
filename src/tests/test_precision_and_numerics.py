"""Hardware-aware precision and non-finite-weight regression tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.utils.numerics import (
    nonfinite_parameter_summary,
    raise_if_nonfinite_parameters,
    validate_checkpoint_weights_finite,
)
from src.utils.precision import (
    align_fp16_trainable_parameters,
    cuda_supports_bfloat16,
    resolve_precision,
)


class _FakeCuda:
    def __init__(self, *, available: bool, capability: tuple[int, int], supported: bool):
        self._available = available
        self._capability = capability
        self._supported = supported

    def is_available(self):
        return self._available

    def get_device_capability(self):
        return self._capability

    def get_device_name(self):
        return "test-gpu"

    def is_bf16_supported(self, including_emulation=False):
        del including_emulation
        return self._supported


def _fake_torch(cuda: _FakeCuda):
    return SimpleNamespace(
        cuda=cuda,
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
    )


def test_bfloat16_falls_back_to_float16_on_p100_like_gpu():
    torch_module = _fake_torch(
        _FakeCuda(available=True, capability=(6, 0), supported=True)
    )

    policy = resolve_precision("bfloat16", torch_module)

    assert cuda_supports_bfloat16(torch_module) is False
    assert policy.effective_dtype == "float16"
    assert policy.fp16 is True
    assert policy.bf16 is False
    assert policy.used_fallback is True


def test_bfloat16_is_kept_on_native_bfloat16_gpu():
    torch_module = _fake_torch(
        _FakeCuda(available=True, capability=(8, 0), supported=True)
    )

    policy = resolve_precision("bfloat16", torch_module)

    assert cuda_supports_bfloat16(torch_module) is True
    assert policy.effective_dtype == "bfloat16"
    assert policy.fp16 is False
    assert policy.bf16 is True


def test_cpu_always_uses_float32():
    torch_module = _fake_torch(
        _FakeCuda(available=False, capability=(0, 0), supported=False)
    )

    policy = resolve_precision("bfloat16", torch_module, bf16=True)

    assert policy.effective_dtype == "float32"
    assert policy.fp16 is False
    assert policy.bf16 is False


def test_fp16_policy_converts_bfloat16_trainable_parameters():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.frozen = torch.nn.Parameter(
                torch.ones(2, dtype=torch.bfloat16), requires_grad=False
            )

    model = Model()
    changed = align_fp16_trainable_parameters(model, torch.float16)

    assert changed == 1
    assert model.lora.dtype == torch.float16
    assert model.frozen.dtype == torch.bfloat16


def test_nonfinite_trainable_weights_are_reported_and_rejected():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.tensor([float("nan"), 1.0]))
            self.base = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    model = Model()
    summary = nonfinite_parameter_summary(model, trainable_only=True)

    assert summary and "lora_A" in summary[0]
    with pytest.raises(FloatingPointError, match="non-finite weights"):
        raise_if_nonfinite_parameters(model, trainable_only=True, context="adapter")


def test_nonfinite_resume_checkpoint_is_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    torch.save(
        {"lora_A": torch.tensor([float("nan"), 0.0])},
        checkpoint / "pytorch_model.bin",
    )

    with pytest.raises(FloatingPointError, match="Resume checkpoint"):
        validate_checkpoint_weights_finite(checkpoint)
