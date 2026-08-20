"""Regression checks for the reproducible tiny dashboard_v4 derivative."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from src.utils.adapter import check_adapter_compatibility, resolve_adapter_path


_ROOT = Path(__file__).resolve().parents[2]
_BUILDER_PATH = _ROOT / "experiments" / "scripts" / "build_dashboard_v4_tiny.py"
_SPEC = importlib.util.spec_from_file_location("build_dashboard_v4_tiny", _BUILDER_PATH)
builder = importlib.util.module_from_spec(_SPEC)
sys.modules["build_dashboard_v4_tiny"] = builder
_SPEC.loader.exec_module(builder)


def test_tiny_dataset_integrity_and_parent_lineage():
    builder.verify(
        source=_ROOT / "data" / "frozen" / "dashboard_v4",
        out=_ROOT / "data" / "frozen" / "dashboard_v4_tiny",
    )


def test_sports_config_can_reuse_dashboard_tiny_adapter(tmp_path):
    cfg = {
        "output_root": "runs",
        "run_layout": "final",
        "seed": 42,
        "model_key": "qwen2_5_0_5b",
        "method_key": "C",
        "model": {"hf_id": "Qwen/Qwen2.5-0.5B-Instruct", "key": "qwen2_5_0_5b"},
        "training": {},
        "data": {
            "dataset_version": "sports_v4_tiny",
            "adapter_dataset_version": "dashboard_v4_tiny",
        },
        "method": {},
    }
    resolved = resolve_adapter_path(cfg, tmp_path)
    assert resolved == (
        tmp_path / "runs" / "dashboard_v4_tiny" / "qwen2_5_0_5b" / "C" / "seed_42" / "adapter"
    )
    metadata = {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_key": "qwen2_5_0_5b",
        "seed": 42,
        "dataset_version": "dashboard_v4_tiny",
    }
    assert check_adapter_compatibility(metadata, cfg) == []
