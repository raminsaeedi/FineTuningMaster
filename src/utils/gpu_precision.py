"""Pick AMP and QLoRA compute dtypes from the weakest visible GPU.

The HPC allocator may give L40S / A40 / A100 (bf16 AMP), V100 / P100
(fp16 AMP + fp32 LoRA), or a mix. Mixed jobs must use the intersection of
capabilities so Pascal/Volta never run bf16 AMP. Qwen3 checkpoints declare
bfloat16; LoRA must stay fp32 on those GPUs or GradScaler crashes, and AMP
must stay on or 4-bit fp16 compute overflows the adapters.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# Longest match first so "L40S" does not collapse to "L40".
_GPU_CAPABILITIES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("l40s", (8, 9)),
    ("l40", (8, 9)),
    ("a100", (8, 0)),
    ("a40", (8, 6)),
    ("h200", (9, 0)),
    ("h100", (9, 0)),
    ("v100", (7, 0)),
    ("p100", (6, 0)),
    ("t4", (7, 5)),
)

_BF16_MAJOR = 8
_FP16_MAJOR = 6


@dataclass(frozen=True)
class GpuDevice:
    name: str
    major: int
    minor: int = 0
    index: int = 0

    @property
    def capability(self) -> tuple[int, int]:
        return (self.major, self.minor)


@dataclass(frozen=True)
class PrecisionChoice:
    mode: str
    amp_fp16: bool
    amp_bf16: bool
    compute_dtype: str
    inference_dtype: str
    reason: str
    devices: tuple[str, ...]
    min_capability: Optional[tuple[int, int]] = None

    def as_metadata(self) -> dict[str, Any]:
        """Return JSON-safe precision metadata for run artifacts."""
        return {
            "mode": self.mode,
            "amp_fp16": self.amp_fp16,
            "amp_bf16": self.amp_bf16,
            "compute_dtype": self.compute_dtype,
            "inference_dtype": self.inference_dtype,
            "reason": self.reason,
            "devices": list(self.devices),
            "min_capability": self.min_capability,
        }


def parse_gpu_name(name: str) -> Optional[tuple[int, int]]:
    """Map a marketing / nvidia-smi name to (major, minor), or None if unknown."""
    text = str(name or "").strip().lower()
    if not text or text in {"none", "any", "cpu"}:
        return None
    for token, cap in _GPU_CAPABILITIES:
        if re.search(rf"\b{re.escape(token)}\b", text) or token in text:
            return cap
    return None


def devices_from_names(names: Sequence[str]) -> tuple[GpuDevice, ...]:
    devices: list[GpuDevice] = []
    for index, name in enumerate(names):
        cap = parse_gpu_name(name) or (0, 0)
        devices.append(GpuDevice(name=str(name), major=cap[0], minor=cap[1], index=index))
    return tuple(devices)


def inspect_cuda_devices() -> tuple[GpuDevice, ...]:
    """Query live CUDA devices. Returns () when CUDA is missing or empty."""
    try:
        import torch
    except Exception:
        return ()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return ()
    found: list[GpuDevice] = []
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        try:
            major, minor = torch.cuda.get_device_capability(index)
        except Exception:
            parsed = parse_gpu_name(name)
            if parsed is None:
                logger.warning("Could not read capability for %s; assuming Pascal fp16", name)
                major, minor = 6, 0
            else:
                major, minor = parsed
        found.append(GpuDevice(name=name, major=int(major), minor=int(minor), index=index))
    return tuple(found)


def _from_weakest(min_major: int) -> tuple[str, bool, bool, str, str]:
    if min_major >= _BF16_MAJOR:
        return "bf16", False, True, "bfloat16", "bfloat16"
    if min_major >= _FP16_MAJOR:
        return "fp16", True, False, "float16", "float16"
    return "fp32", False, False, "float32", "float32"


def choose_precision(
    devices: Sequence[GpuDevice],
    *,
    requested: str = "auto",
) -> PrecisionChoice:
    """Select AMP flags from the weakest GPU. ``requested`` is auto|bf16|fp16|fp32."""
    mode_in = str(requested or "auto").strip().lower()
    mode_in = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
    }.get(mode_in, mode_in)
    names = tuple(device.name for device in devices)
    if not devices:
        return PrecisionChoice(
            mode="fp32",
            amp_fp16=False,
            amp_bf16=False,
            compute_dtype="float32",
            inference_dtype="float32",
            reason="no CUDA device; using fp32",
            devices=names,
            min_capability=None,
        )

    weakest = min(devices, key=lambda device: (device.major, device.minor))
    hw_mode, amp_fp16, amp_bf16, compute_dtype, inference_dtype = _from_weakest(weakest.major)
    reason = (
        f"weakest GPU {weakest.name} SM {weakest.major}.{weakest.minor} => {hw_mode}"
    )

    if mode_in in {"", "auto"}:
        mode = hw_mode
    elif mode_in == "fp32":
        mode, amp_fp16, amp_bf16 = "fp32", False, False
        compute_dtype = inference_dtype = "float32"
        reason = f"requested fp32 (hardware would allow {hw_mode})"
    elif mode_in == "fp16":
        if hw_mode == "fp32":
            mode, amp_fp16, amp_bf16 = "fp32", False, False
            compute_dtype = inference_dtype = "float32"
            reason = f"requested fp16 unsupported on {weakest.name}; using fp32"
        else:
            mode, amp_fp16, amp_bf16 = "fp16", True, False
            compute_dtype = inference_dtype = "float16"
            reason = f"requested fp16 AMP on {weakest.name}"
    elif mode_in == "bf16":
        if hw_mode == "bf16":
            mode = "bf16"
            reason = f"requested bf16 on {weakest.name}"
        else:
            mode, amp_fp16, amp_bf16 = hw_mode, amp_fp16, amp_bf16
            reason = (
                f"requested bf16 unsupported on {weakest.name} "
                f"SM {weakest.major}.{weakest.minor}; fallback {mode}"
            )
    else:
        raise ValueError(f"Unknown precision mode {requested!r}; use auto|bf16|fp16|fp32")

    return PrecisionChoice(
        mode=mode,
        amp_fp16=amp_fp16,
        amp_bf16=amp_bf16,
        compute_dtype=compute_dtype,
        inference_dtype=inference_dtype,
        reason=reason,
        devices=names,
        min_capability=weakest.capability,
    )


def _requested_mode(sft_cfg: Optional[Mapping[str, Any]]) -> str:
    if not sft_cfg:
        return "auto"
    try:
        explicit = sft_cfg.get("precision")  # type: ignore[union-attr]
    except AttributeError:
        explicit = getattr(sft_cfg, "precision", None)
    if explicit not in (None, ""):
        return str(explicit).strip().lower()
    return "auto"


def resolve_training_precision(
    sft_cfg: Optional[Mapping[str, Any]] = None,
    *,
    devices: Optional[Sequence[GpuDevice]] = None,
) -> PrecisionChoice:
    """Resolve AMP from config. ``fp16=false`` + ``bf16=false`` means auto, not fp32."""
    if devices is None:
        devices = inspect_cuda_devices()
    requested = _requested_mode(sft_cfg)
    choice = choose_precision(devices, requested=requested)
    logger.info("Training precision: %s (%s)", choice.mode, choice.reason)
    return choice


def lora_param_dtype_name(choice: PrecisionChoice) -> str:
    """Dtype for trainable LoRA tensors so AMP GradScaler can unscale them.

    Qwen3 checkpoints declare ``torch_dtype: bfloat16``. PEFT then creates LoRA
    A/B in bf16 even when this GPU resolved fp16 AMP. Torch's fp16 GradScaler
    cannot unscale bf16 (``_amp_foreach_non_finite_check_and_unscale_cuda``),
    which aborts the first training step on V100/P100. Keep adapters in fp32
    for fp16 AMP (the QLoRA recipe); match bf16 only when AMP is bf16.
    """
    if choice.amp_bf16:
        return "bfloat16"
    return "float32"


def summarize_trainable_dtypes(model: Any) -> dict[str, int]:
    """Count trainable parameter tensors by dtype (for diagnostics)."""
    import torch

    counts: dict[str, int] = {}
    for param in model.parameters():
        if param.requires_grad:
            key = str(param.dtype).replace("torch.", "")
            counts[key] = counts.get(key, 0) + 1
    return counts


def align_trainable_parameters(model: Any, choice: PrecisionChoice) -> int:
    """Cast ``requires_grad`` tensors to :func:`lora_param_dtype_name`."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return 0
    import torch

    target_name = lora_param_dtype_name(choice)
    target = getattr(torch, target_name)
    n_cast = 0
    for param in parameters():
        if param.requires_grad and param.dtype != target:
            param.data = param.data.to(target)
            n_cast += 1
    summary = summarize_trainable_dtypes(model)
    if summary:
        logger.info("Trainable parameter dtypes: %s", summary)
    if n_cast:
        logger.info(
            "Cast %d trainable parameter tensor(s) to %s (%s)",
            n_cast,
            target_name,
            choice.reason,
        )
    return n_cast


def assert_fp16_amp_safe(model: Any, choice: PrecisionChoice) -> None:
    """Refuse to start fp16 GradScaler when any trainable tensor is bfloat16."""
    if not choice.amp_fp16:
        return
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return
    import torch

    bad = [
        name
        for name, param in named_parameters()
        if param.requires_grad and param.dtype == torch.bfloat16
    ]
    if bad:
        sample = ", ".join(bad[:3])
        raise RuntimeError(
            "fp16 GradScaler cannot unscale bfloat16 trainable tensors "
            f"({len(bad)} found, e.g. {sample}). Cast LoRA adapters to float32."
        )


def resolve_inference_dtype(
    preferred: Optional[str] = None,
    *,
    devices: Optional[Sequence[GpuDevice]] = None,
) -> str:
    """Clamp a YAML dtype (e.g. bfloat16) to what the weakest GPU can run."""
    if devices is None:
        devices = inspect_cuda_devices()
    hardware = choose_precision(devices, requested="auto")
    want = str(preferred or "").strip().lower()
    if hardware.mode == "fp32":
        return "float32"
    if want in {"bfloat16", "bf16"} and hardware.mode != "bf16":
        logger.warning(
            "Model dtype bfloat16 is not supported on %s; using %s",
            hardware.devices or "this GPU",
            hardware.inference_dtype,
        )
        return hardware.inference_dtype
    if want in {"float16", "fp16", "half"}:
        return "float16" if hardware.mode != "fp32" else "float32"
    if want in {"float32", "fp32"}:
        return "float32"
    if want in {"bfloat16", "bf16"}:
        return "bfloat16"
    return hardware.inference_dtype
