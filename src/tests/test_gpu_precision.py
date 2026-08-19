"""GPU-adaptive AMP / QLoRA compute dtype — prevents the P100 NaN collapse.

The cluster may allocate L40S, A40, A100, V100, P100, or a mix. Precision must
follow the weakest visible device so bf16 is never used on Pascal/Volta.
"""

from __future__ import annotations

import sys
import types

import pytest
from omegaconf import OmegaConf

from src.utils.gpu_precision import (
    GpuDevice,
    assert_fp16_amp_safe,
    choose_precision,
    devices_from_names,
    lora_param_dtype_name,
    parse_gpu_name,
    resolve_inference_dtype,
    resolve_training_precision,
)


def test_parse_cluster_gpu_names():
    assert parse_gpu_name("NVIDIA L40S") == (8, 9)
    assert parse_gpu_name("NVIDIA A40") == (8, 6)
    assert parse_gpu_name("NVIDIA A100-SXM4-40GB") == (8, 0)
    assert parse_gpu_name("Tesla V100-SXM2-32GB") == (7, 0)
    assert parse_gpu_name("Tesla P100-PCIE-16GB") == (6, 0)
    assert parse_gpu_name("None") is None
    assert parse_gpu_name("Any") is None


def test_p100_selects_fp16_amp():
    choice = choose_precision([GpuDevice("Tesla P100-PCIE-16GB", 6, 0)])
    assert choice.mode == "fp16"
    assert choice.amp_fp16 is True
    assert choice.amp_bf16 is False
    assert choice.compute_dtype == "float16"
    assert choice.inference_dtype == "float16"


def test_v100_selects_fp16_amp_with_fp32_lora():
    choice = choose_precision([GpuDevice("Tesla V100", 7, 0)])
    assert choice.mode == "fp16"
    assert choice.amp_fp16 is True
    assert choice.amp_bf16 is False
    assert choice.compute_dtype == "float16"
    # LoRA must stay fp32: fp16 GradScaler cannot unscale Qwen3 bf16 adapters.
    assert lora_param_dtype_name(choice) == "float32"


def test_a100_keeps_lora_in_bfloat16():
    choice = choose_precision([GpuDevice("NVIDIA A100", 8, 0)])
    assert lora_param_dtype_name(choice) == "bfloat16"


@pytest.mark.parametrize(
    ("name", "major", "minor"),
    [
        ("NVIDIA A100-SXM4-40GB", 8, 0),
        ("NVIDIA A40", 8, 6),
        ("NVIDIA L40S", 8, 9),
    ],
)
def test_ampere_and_ada_select_bf16(name, major, minor):
    choice = choose_precision([GpuDevice(name, major, minor)])
    assert choice.mode == "bf16"
    assert choice.amp_fp16 is False
    assert choice.amp_bf16 is True
    assert choice.compute_dtype == "bfloat16"
    assert choice.inference_dtype == "bfloat16"


def test_mixed_a100_and_p100_uses_fp16_amp():
    choice = choose_precision([
        GpuDevice("NVIDIA A100", 8, 0, index=0),
        GpuDevice("Tesla P100-PCIE-16GB", 6, 0, index=1),
    ])
    assert choice.mode == "fp16"
    assert choice.amp_fp16 is True
    assert choice.amp_bf16 is False
    assert choice.min_capability == (6, 0)


def test_no_gpu_uses_fp32():
    choice = choose_precision(())
    assert choice.mode == "fp32"
    assert choice.amp_fp16 is False
    assert choice.amp_bf16 is False
    assert choice.compute_dtype == "float32"
    assert choice.inference_dtype == "float32"


def test_devices_from_names_matches_dropdown():
    devices = devices_from_names(["L40S", "A40", "A100", "V100", "P100"])
    choice = choose_precision(devices)
    assert choice.mode == "fp16"
    assert [d.name for d in devices] == ["L40S", "A40", "A100", "V100", "P100"]


def test_yaml_fp16_and_bf16_false_is_auto_not_fp32():
    """The C-run bug: both flags false was treated as no AMP on fp16 compute."""
    sft = {"fp16": False, "bf16": False}
    choice = resolve_training_precision(
        sft, devices=[GpuDevice("Tesla P100-PCIE-16GB", 6, 0)]
    )
    assert choice.mode == "fp16"
    assert choice.amp_fp16 is True


def test_auto_on_a100_uses_bf16_even_if_yaml_flags_false():
    choice = resolve_training_precision(
        {"precision": "auto", "fp16": False, "bf16": False},
        devices=[GpuDevice("NVIDIA A100", 8, 0)],
    )
    assert choice.mode == "bf16"


def test_requested_bf16_falls_back_to_fp16_on_p100():
    choice = choose_precision(
        [GpuDevice("Tesla P100", 6, 0)], requested="bf16"
    )
    assert choice.mode == "fp16"
    assert "unsupported" in choice.reason.lower() or "fallback" in choice.reason.lower()


def test_requested_fp32_is_honored_on_a100():
    choice = choose_precision(
        [GpuDevice("NVIDIA A100", 8, 0)], requested="fp32"
    )
    assert choice.mode == "fp32"
    assert choice.amp_fp16 is False
    assert choice.amp_bf16 is False


def test_inference_clamps_yaml_bf16_on_p100():
    dtype = resolve_inference_dtype(
        "bfloat16", devices=[GpuDevice("Tesla P100-PCIE-16GB", 6, 0)]
    )
    assert dtype == "float16"


def test_inference_keeps_yaml_bf16_on_a100():
    dtype = resolve_inference_dtype(
        "bfloat16", devices=[GpuDevice("NVIDIA A100", 8, 0)]
    )
    assert dtype == "bfloat16"


def test_nonfinite_log_reason_detects_nan_grad():
    from src.training.stability import nonfinite_log_reason

    assert nonfinite_log_reason({"loss": 0.4, "grad_norm": 1.0}) is None
    reason = nonfinite_log_reason({"loss": 10.8, "grad_norm": float("nan")})
    assert reason is not None and "grad_norm" in reason
    assert nonfinite_log_reason({"loss": float("inf")}) is not None


def test_abort_callback_raises_on_nan_grad():
    from src.training.stability import AbortOnNonFiniteCallback

    cb = AbortOnNonFiniteCallback()
    state = types.SimpleNamespace(global_step=200)
    with pytest.raises(FloatingPointError, match="grad_norm"):
        cb.on_log(None, state, None, logs={"grad_norm": float("nan"), "loss": 10.8})


def test_abort_callback_supports_transformers_lifecycle_hooks():
    from src.training.stability import AbortOnNonFiniteCallback

    cb = AbortOnNonFiniteCallback()
    control = object()

    assert cb.on_init_end(None, None, control) is control


def test_trainer_applies_fp16_amp_and_max_length_on_p100(tmp_path, monkeypatch):
    from src.training.sft_trainer import QLoRASFTTrainer
    from src.utils import gpu_precision

    captured = {}

    class FakeTrainingConfig:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs

    class DummyModel:
        def parameters(self):
            return iter(())

        def named_parameters(self):
            return iter(())

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["callbacks"] = kwargs.get("callbacks")
            self.model = DummyModel()

        def train(self, *args, **kwargs):
            self.state = types.SimpleNamespace(global_step=1)
            return types.SimpleNamespace(metrics={})

    fake_trl = types.ModuleType("trl")
    fake_trl.SFTConfig = FakeTrainingConfig
    fake_trl.SFTTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setattr(
        gpu_precision,
        "inspect_cuda_devices",
        lambda: (GpuDevice("Tesla P100-PCIE-16GB", 6, 0),),
    )

    cfg = OmegaConf.create({
        "experiment_name": "C",
        "experiment_id": "C_42",
        "seed": 42,
        "model": {"name": "qwen3", "hf_id": "Qwen/Qwen3-1.7B", "max_seq_length": 4096},
        "data": {"dataset_version": "dashboard_v4", "train_file": str(tmp_path / "train.jsonl")},
        "training": {
            "type": "qlora_sft",
            "sft": {"fp16": False, "bf16": False, "learning_rate": 2.0e-4},
        },
    })
    trainer = QLoRASFTTrainer(cfg)
    trainer._setup = lambda: None
    trainer.model = object()
    trainer.tokenizer = object()
    trainer._save = lambda path: None
    trainer.train([], None, str(tmp_path / "adapter"))

    args = captured["training_args"]
    assert args["fp16"] is True
    assert args["bf16"] is False
    assert args["max_length"] == 4096
    callbacks = captured["callbacks"] or []
    callback_types = [type(cb).__name__ for cb in callbacks]
    assert "AbortOnNonFiniteCallback" in callback_types
    control = object()
    assert all(cb.on_init_end(None, None, control) is control for cb in callbacks)


def test_trainer_applies_bf16_on_a100(tmp_path, monkeypatch):
    from src.training.sft_trainer import QLoRASFTTrainer
    from src.utils import gpu_precision

    captured = {}

    class FakeTrainingConfig:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs

    class DummyModel:
        def parameters(self):
            return iter(())

        def named_parameters(self):
            return iter(())

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = DummyModel()

        def train(self, *args, **kwargs):
            self.state = types.SimpleNamespace(global_step=1)
            return types.SimpleNamespace(metrics={})

    fake_trl = types.ModuleType("trl")
    fake_trl.SFTConfig = FakeTrainingConfig
    fake_trl.SFTTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setattr(
        gpu_precision,
        "inspect_cuda_devices",
        lambda: (GpuDevice("NVIDIA A100", 8, 0),),
    )

    cfg = OmegaConf.create({
        "seed": 42,
        "model": {"hf_id": "Qwen/Qwen3-1.7B", "max_seq_length": 4096},
        "data": {},
        "training": {"sft": {"fp16": False, "bf16": False}},
    })
    trainer = QLoRASFTTrainer(cfg)
    trainer._setup = lambda: None
    trainer.model = object()
    trainer.tokenizer = object()
    trainer._save = lambda path: None
    trainer.train([], None, str(tmp_path / "adapter"))

    assert captured["training_args"]["bf16"] is True
    assert captured["training_args"]["fp16"] is False


def test_assert_fp16_amp_safe_rejects_bf16_lora():
    torch = pytest.importorskip("torch")
    choice = choose_precision([GpuDevice("Tesla V100", 7, 0)])
    param = types.SimpleNamespace(requires_grad=True, dtype=torch.bfloat16)
    model = types.SimpleNamespace(named_parameters=lambda: [("lora_A", param)])
    with pytest.raises(RuntimeError, match="bfloat16"):
        assert_fp16_amp_safe(model, choice)
