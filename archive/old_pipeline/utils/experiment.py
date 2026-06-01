"""
Experiment management utilities.

Handles: experiment ID generation, output directory creation,
config snapshot saving, seed setting, and experiment listing.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def make_experiment_id(config: dict) -> str:
    """
    Build a deterministic, human-readable experiment ID:
      {algorithm}_{model_short}_{YYYYMMDD}_{HHMMSS}_{hash4}

    Example: qlora_qwen05b_20250513_143022_a3f1

    The 4-character hash is derived from the full config dict, so that two
    runs with identical configs (minus timestamp) share the same hash prefix —
    useful for identifying duplicates.
    """
    algorithm = config.get("algorithm", {}).get("name", "unknown")
    model_name = config.get("model", {}).get("name", "unknown")
    model_short = _model_short_name(model_name)

    # Stable hash of the config (exclude timestamp-sensitive keys)
    config_for_hash = {k: v for k, v in config.items() if k not in ("meta",)}
    config_str = json.dumps(config_for_hash, sort_keys=True, default=str)
    hash4 = hashlib.md5(config_str.encode()).hexdigest()[:4]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{algorithm}_{model_short}_{timestamp}_{hash4}"


def setup_experiment(config: dict) -> tuple[Path, dict]:
    """
    Prepare the experiment directory and freeze the config.

    Steps:
      1. Generate experiment_id
      2. Create output directory tree under paths.outputs_root
      3. Save config_snapshot.yaml
      4. Set random seeds
      5. Return (experiment_dir, config_with_resolved_absolute_paths)
    """
    from utils.config_loader import resolve_paths

    config = resolve_paths(config)

    experiment_id = make_experiment_id(config)
    outputs_root = Path(config["paths"]["outputs_root"])
    experiment_dir = outputs_root / experiment_id

    _create_experiment_dirs(experiment_dir)
    save_config_snapshot(config, experiment_dir, experiment_id)
    # NOTE: set_seeds is NOT called here. Call it in the pipeline script
    # AFTER dataset loading (pyarrow/DLL init) to avoid Windows 0xC0000005.

    # Inject the experiment_id and dir into config for downstream use
    config.setdefault("_runtime", {})
    config["_runtime"]["experiment_id"] = experiment_id
    config["_runtime"]["experiment_dir"] = str(experiment_dir)

    return experiment_dir, config


def set_seeds(seed: int) -> None:
    """Set Python random, NumPy, and PyTorch seeds for full reproducibility.

    NOTE: torch.cuda.manual_seed_all is intentionally omitted here.
    Calling it before dataset loading (pyarrow/DLL init) causes a Windows
    0xC0000005 access violation due to CUDA/DLL loading order.
    torch.manual_seed covers the default CUDA device — sufficient for single-GPU.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_config_snapshot(
    config: dict, experiment_dir: Path, experiment_id: str | None = None
) -> None:
    """Write a frozen copy of the merged config to experiment_dir/config_snapshot.yaml."""
    snapshot = dict(config)
    if experiment_id:
        snapshot.setdefault("_runtime", {})["experiment_id"] = experiment_id
    snapshot_path = experiment_dir / "config_snapshot.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_experiment_config(experiment_dir: str | Path) -> dict:
    """Read config_snapshot.yaml from a completed experiment directory."""
    path = Path(experiment_dir) / "config_snapshot.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config_snapshot.yaml found in {experiment_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_experiments(outputs_root: str = "outputs/experiments") -> list[dict]:
    """
    Scan outputs_root for completed experiment directories and return a summary list.

    Each entry contains:
      experiment_id, algorithm, model, timestamp, has_adapter, has_predictions, has_metrics
    """
    root = Path(outputs_root)
    if not root.exists():
        return []

    experiments = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        try:
            cfg = load_experiment_config(exp_dir)
        except FileNotFoundError:
            continue

        experiments.append({
            "experiment_id": exp_dir.name,
            "algorithm": cfg.get("algorithm", {}).get("name", "?"),
            "model": cfg.get("model", {}).get("name", "?"),
            "lora_r": cfg.get("lora", {}).get("r"),
            "learning_rate": cfg.get("training", {}).get("learning_rate"),
            "num_epochs": cfg.get("training", {}).get("num_train_epochs"),
            "has_adapter": (exp_dir / "final_adapter").exists(),
            "has_predictions": (exp_dir / "predictions").exists(),
            "has_metrics": (exp_dir / "metrics.json").exists(),
        })

    return experiments


def load_metrics(experiment_dir: str | Path) -> dict | None:
    """Return the metrics.json dict for an experiment, or None if not present."""
    path = Path(experiment_dir) / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics(metrics: dict, experiment_dir: str | Path) -> None:
    """Write or merge metrics into experiment_dir/metrics.json."""
    path = Path(experiment_dir) / "metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        # Deep-merge new metrics into existing
        existing = _deep_merge_dicts(existing, metrics)
        metrics = existing
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_short_name(model_name: str) -> str:
    """Convert a HuggingFace model name to a compact identifier."""
    name = model_name.lower().split("/")[-1]
    mapping = {
        "qwen2.5-0.5b-instruct": "qwen05b",
        "qwen2.5-1.5b-instruct": "qwen15b",
        "qwen2.5-3b-instruct":   "qwen3b",
        "qwen2.5-7b-instruct":   "qwen7b",
    }
    return mapping.get(name, name.replace(".", "").replace("-", "_")[:12])


def _create_experiment_dirs(experiment_dir: Path) -> None:
    for subdir in ("checkpoints", "final_adapter", "predictions", "logs"):
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    import copy
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
