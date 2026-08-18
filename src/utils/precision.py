"""Hardware-aware precision selection shared by training and inference.

Model profiles express a preferred dtype. The actual dtype is selected at
runtime because a profile can be executed on different GPUs. In particular,
Tesla P100 (compute capability 6.0) does not provide native bfloat16 support;
silently loading a Qwen3 profile as bfloat16 can produce non-finite LoRA
weights. The policy below keeps the profile preference on capable hardware and
falls back to float16 on older CUDA devices or float32 on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "float16": "float16",
    "torch.float16": "float16",
    "fp32": "float32",
    "float": "float32",
    "float32": "float32",
    "torch.float32": "float32",
}


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved model and Trainer precision settings."""

    requested_dtype: str
    effective_dtype: Any
    effective_dtype_name: str
    fp16: bool
    bf16: bool
    used_fallback: bool = False
    fallback_reason: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        """Return JSON-safe precision metadata for run artifacts."""
        return {
            "requested_dtype": self.requested_dtype,
            "effective_dtype": self.effective_dtype_name,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
        }


def normalize_dtype_name(value: Any, default: str = "float16") -> str:
    """Normalize common config and torch dtype spellings."""
    raw = str(value or "").strip().lower()
    return _DTYPE_ALIASES.get(raw, default)


def _cuda_available(torch_module: Any) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    try:
        return bool(cuda is not None and cuda.is_available())
    except Exception:
        return False


def cuda_supports_bfloat16(torch_module: Any) -> bool:
    """Return whether the active CUDA device has native bfloat16 support.

    ``torch.cuda.is_bf16_supported`` can report emulation support on some
    versions. Compute capability is therefore used as a hard guard: native
    CUDA bfloat16 starts at major capability 8 (Ampere). The capability check
    also makes the decision stable across PyTorch/CUDA version changes.
    """
    if not _cuda_available(torch_module):
        return False

    cuda = torch_module.cuda
    capability: tuple[int, int] | None = None
    try:
        raw_capability = cuda.get_device_capability()
        capability = (int(raw_capability[0]), int(raw_capability[1]))
    except Exception:
        pass

    if capability is not None and capability[0] < 8:
        return False

    checker = getattr(cuda, "is_bf16_supported", None)
    if callable(checker):
        try:
            # ``including_emulation=False`` is available in current PyTorch.
            supported = checker(including_emulation=False)
        except TypeError:
            try:
                supported = checker()
            except Exception:
                supported = None
        except Exception:
            supported = None
        if supported is not None:
            return bool(supported)

    return capability is not None and capability[0] >= 8


def _device_name(torch_module: Any) -> str:
    try:
        return str(torch_module.cuda.get_device_name())
    except Exception:
        return "unknown CUDA device"


def resolve_precision(
    requested_dtype: Any,
    torch_module: Any,
    *,
    fp16: bool = False,
    bf16: bool = False,
    logger: Any = None,
) -> PrecisionPolicy:
    """Resolve a profile preference into safe model and Trainer settings.

    The model dtype is the default preference. Explicit ``fp16``/``bf16``
    Trainer flags can request a mixed-precision mode as well. If both are
    requested, bfloat16 wins only when the device supports it; otherwise the
    policy deterministically selects float16. CPU always uses float32.
    """
    requested = normalize_dtype_name(requested_dtype)
    on_gpu = _cuda_available(torch_module)

    if not on_gpu:
        fallback = requested != "float32" or bool(fp16) or bool(bf16)
        reason = "CUDA is unavailable; CPU execution uses float32" if fallback else None
        if fallback and logger is not None:
            logger.info("Using float32 because CUDA is unavailable")
        return PrecisionPolicy(
            requested_dtype=requested,
            effective_dtype=torch_module.float32,
            effective_dtype_name="float32",
            fp16=False,
            bf16=False,
            used_fallback=fallback,
            fallback_reason=reason,
        )

    wants_bf16 = bool(bf16) or requested == "bfloat16"
    wants_fp16 = bool(fp16) or requested == "float16"

    if wants_bf16:
        if cuda_supports_bfloat16(torch_module):
            return PrecisionPolicy(
                requested_dtype=requested,
                effective_dtype=torch_module.bfloat16,
                effective_dtype_name="bfloat16",
                fp16=False,
                bf16=True,
            )

        reason = (
            f"CUDA device '{_device_name(torch_module)}' has no native bfloat16 support"
        )
        if logger is not None:
            logger.warning("%s; falling back to float16", reason)
        return PrecisionPolicy(
            requested_dtype=requested,
            effective_dtype=torch_module.float16,
            effective_dtype_name="float16",
            fp16=True,
            bf16=False,
            used_fallback=True,
            fallback_reason=reason,
        )

    if wants_fp16:
        return PrecisionPolicy(
            requested_dtype=requested,
            effective_dtype=torch_module.float16,
            effective_dtype_name="float16",
            fp16=True,
            bf16=False,
        )

    return PrecisionPolicy(
        requested_dtype=requested,
        effective_dtype=torch_module.float32,
        effective_dtype_name="float32",
        fp16=False,
        bf16=False,
    )
