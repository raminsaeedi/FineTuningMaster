"""
Hierarchical config loader for the fine-tuning pipeline.

Resolution order (each layer overrides the previous):
  1. configs/base.yaml           — universal defaults
  2. configs/models/*.yaml       — model-specific overrides
  3. configs/experiments/*.yaml  — experiment-specific overrides
  4. CLI --override key=value    — dot-notation runtime overrides

Usage:
  from utils.config_loader import load_config
  config = load_config(
      base_config="configs/base.yaml",
      model_config="configs/models/qwen_0_5b.yaml",
      experiment_config="configs/experiments/qlora.yaml",
      cli_overrides={"training.learning_rate": 5e-4, "lora.r": 32},
  )
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(
    base_config: str = "configs/base.yaml",
    model_config: str | None = None,
    experiment_config: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    project_root: str | None = None,
) -> dict:
    """
    Compose configuration from up to four layers and return a single merged dict.

    Parameters
    ----------
    base_config:        Path to the universal defaults YAML (relative to project_root).
    model_config:       Optional path to model-specific overrides YAML.
    experiment_config:  Optional path to experiment-specific overrides YAML.
    cli_overrides:      Dict of dot-notation key → value pairs from argparse
                        (e.g. {"training.learning_rate": 5e-4, "lora.r": 32}).
    project_root:       Root of the project. If None, auto-detected from this file.
    """
    root = _resolve_root(project_root)

    config = _load_yaml(root / base_config)

    if model_config:
        config = _deep_merge(config, _load_yaml(root / model_config))

    if experiment_config:
        config = _deep_merge(config, _load_yaml(root / experiment_config))

    if cli_overrides:
        config = _apply_cli_overrides(config, cli_overrides)

    # Store the resolved project root so downstream code can anchor paths.
    config.setdefault("paths", {})["project_root"] = str(root)

    return config


def resolve_paths(config: dict) -> dict:
    """
    Rewrite relative paths in config['paths'] and config['data'] to absolute,
    anchored to config['paths']['project_root'].  Returns an updated copy.
    """
    config = copy.deepcopy(config)
    root = Path(config["paths"]["project_root"])

    for key, value in config.get("paths", {}).items():
        if key == "project_root":
            continue
        if value and not Path(value).is_absolute():
            config["paths"][key] = str(root / value)

    for key, value in config.get("data", {}).items():
        if value and isinstance(value, str) and not Path(value).is_absolute():
            config["data"][key] = str(root / value)

    return config


def parse_cli_overrides(override_list: list[str] | None) -> dict[str, Any]:
    """
    Convert a list of "key=value" strings (from argparse nargs='*') to a dict.

    Keys use dot-notation to reference nested config fields:
      "training.learning_rate=5e-4"  →  {"training.learning_rate": 5e-4}
      "lora.r=32"                    →  {"lora.r": 32}

    Values are cast to int, float, bool, or left as str automatically.
    """
    if not override_list:
        return {}
    result: dict[str, Any] = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not in 'key=value' format.")
        key, raw_value = item.split("=", 1)
        result[key.strip()] = _cast_value(raw_value.strip())
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_root(project_root: str | None) -> Path:
    if project_root:
        return Path(project_root).resolve()
    # Auto-detect: this file lives at <root>/utils/config_loader.py
    return Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_cli_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    """Apply dot-notation overrides to the config dict in-place (on a copy)."""
    config = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = config
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return config


def _cast_value(raw: str) -> Any:
    """Cast a CLI string value to the most appropriate Python type."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if raw.lower() in ("null", "none", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw
