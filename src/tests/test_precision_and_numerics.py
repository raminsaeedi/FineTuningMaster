"""Hardware-aware precision and non-finite-weight regression tests."""

from __future__ import annotations

import pytest

from src.utils.numerics import (
    nonfinite_parameter_summary,
    raise_if_nonfinite_parameters,
    validate_checkpoint_weights_finite,
)
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
