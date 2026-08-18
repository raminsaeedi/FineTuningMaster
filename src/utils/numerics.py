"""Finite-value guards for model weights and resumable checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _nonfinite_tensor_count(tensor: Any) -> int:
    """Return the number of non-finite elements without importing torch eagerly."""
    import torch

    if not (tensor.is_floating_point() or tensor.is_complex()):
        return 0
    finite = torch.isfinite(tensor.detach())
    return int((~finite).sum().item())


def nonfinite_parameter_summary(
    model: Any,
    *,
    trainable_only: bool = False,
    name_contains: str | None = None,
) -> list[str]:
    """Describe non-finite parameters on a model.

    ``name_contains`` is used for PEFT adapter-only checks, while
    ``trainable_only`` is used during QLoRA training. Missing model interfaces
    are tolerated so lightweight test doubles remain usable.
    """
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return []

    summary: list[str] = []
    for name, parameter in named_parameters():
        if trainable_only and not bool(getattr(parameter, "requires_grad", False)):
            continue
        if name_contains and name_contains.lower() not in str(name).lower():
            continue
        bad_count = _nonfinite_tensor_count(parameter)
        if bad_count:
            summary.append(f"{name} ({bad_count}/{parameter.numel()} non-finite)")
    return summary


def raise_if_nonfinite_parameters(
    model: Any,
    *,
    context: str,
    trainable_only: bool = False,
    name_contains: str | None = None,
) -> None:
    """Raise before invalid weights can be saved or used for inference."""
    summary = nonfinite_parameter_summary(
        model,
        trainable_only=trainable_only,
        name_contains=name_contains,
    )
    if summary:
        preview = "; ".join(summary[:8])
        more = f"; plus {len(summary) - 8} more" if len(summary) > 8 else ""
        raise FloatingPointError(
            f"{context} contains non-finite weights: {preview}{more}. "
            "The run is invalid; retrain from a clean checkpoint."
        )


def _walk_tensor_state(value: Any, prefix: str = "") -> list[str]:
    """Find non-finite tensors in a CPU-loaded checkpoint object."""
    import torch

    if torch.is_tensor(value):
        bad_count = _nonfinite_tensor_count(value)
        return [f"{prefix or '<tensor>'} ({bad_count}/{value.numel()} non-finite)"] if bad_count else []
    if isinstance(value, Mapping):
        bad: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            bad.extend(_walk_tensor_state(child, child_prefix))
        return bad
    if isinstance(value, (list, tuple)):
        bad = []
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            bad.extend(_walk_tensor_state(child, child_prefix))
        return bad
    return []


def validate_checkpoint_weights_finite(checkpoint: str | Path) -> None:
    """Reject a resume checkpoint containing NaN/Inf model or adapter weights.

    The function intentionally checks model-weight files only. Optimizer state
    is loaded by the trainer after this point and is not a model artifact; the
    post-train parameter guard still prevents an invalid adapter from being
    saved if a malformed optimizer state propagates non-finite values.
    """
    checkpoint_path = Path(checkpoint)
    candidates = [
        checkpoint_path / "adapter_model.safetensors",
        checkpoint_path / "model.safetensors",
        checkpoint_path / "pytorch_model.bin",
        checkpoint_path / "adapter_model.bin",
    ]
    weight_file = next((path for path in candidates if path.is_file()), None)
    if weight_file is None:
        return

    bad: list[str] = []
    if weight_file.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                bad.extend(_walk_tensor_state(tensor, key))
    else:
        import torch

        state = torch.load(str(weight_file), map_location="cpu")
        bad.extend(_walk_tensor_state(state))

    if bad:
        preview = "; ".join(bad[:8])
        more = f"; plus {len(bad) - 8} more" if len(bad) > 8 else ""
        raise FloatingPointError(
            f"Resume checkpoint {checkpoint_path} contains non-finite model weights: "
            f"{preview}{more}. Start a fresh run instead."
        )


def nonfinite_scalar_items(values: Mapping[str, Any]) -> list[str]:
    """Return metric names whose scalar values are NaN or infinite."""
    bad: list[str] = []
    for key, value in values.items():
        try:
            if not math.isfinite(float(value)):
                bad.append(str(key))
        except (TypeError, ValueError):
            continue
    return bad
