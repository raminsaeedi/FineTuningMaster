"""Adapter resolution and compatibility validation for methods C and D.

Method D (ft_rag) reuses the adapter that method C (ft) trained. Nothing used to
tie the two together: ``adapter_path`` interpolated to the *running* experiment's
own folder, so E04 pointed at a directory no training step ever writes, and a
seed-42 adapter could be loaded into a seed-44 run without a warning.

This module fixes both halves of that problem:

* :func:`resolve_adapter_path` derives the adapter folder from the experiment
  that produced it (``method.adapter_source_experiment``), keyed by seed, so
  ``C seed 43`` feeds ``D seed 43`` automatically.
* :func:`validate_adapter` compares the adapter's ``training_metadata.json``
  against the config that is about to consume it and raises on any mismatch of
  base model, seed or dataset version.

Both are pure path/JSON logic and import neither torch nor peft.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

# Files a saved PEFT adapter folder must contain to be usable.
_REQUIRED_ADAPTER_FILES = ("adapter_config.json",)
_ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")

TRAINING_METADATA_FILENAME = "training_metadata.json"


class AdapterError(RuntimeError):
    """Raised when an adapter is missing, unusable, or incompatible."""


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read one key from a dict / OmegaConf node, tolerant of missing keys."""
    if cfg is None:
        return default
    try:
        value = cfg.get(key, default)
    except AttributeError:
        value = getattr(cfg, key, default)
    return default if value is None else value


def _nested_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    cur = cfg
    for key in dotted.split("."):
        cur = _get(cur, key)
        if cur is None:
            return default
    return cur


def resolve_adapter_path(cfg: Any, project_root: Optional[Path] = None) -> Path:
    """Return the adapter folder this run should load.

    Resolution order, most explicit first:

    1. ``method.adapter_path`` when set — an operator override always wins.
    2. ``method.adapter_source_experiment`` — the adapter is looked up under
       ``<output_root>/<source_experiment>_<seed>/adapter``. This is how D finds
       the C run of the *same* seed.
    3. The running experiment's own folder — correct for method C itself, which
       both trains and consumes its adapter.
    """
    output_root = str(_get(cfg, "output_root", "experiments/outputs/experiments"))
    seed = int(_get(cfg, "seed", 42))

    explicit = _nested_get(cfg, "method.adapter_path")
    if explicit:
        path = Path(str(explicit))
        return path if path.is_absolute() else _abs(path, project_root)

    source_experiment = _nested_get(cfg, "method.adapter_source_experiment")
    if source_experiment:
        experiment_id = f"{source_experiment}_{seed}"
    else:
        experiment_id = str(
            _get(cfg, "experiment_id", _get(cfg, "experiment_name", "default"))
        )

    root = Path(output_root)
    if not root.is_absolute():
        root = _abs(root, project_root)
    return root / experiment_id / "adapter"


def _abs(path: Path, project_root: Optional[Path]) -> Path:
    base = project_root if project_root is not None else Path.cwd()
    return Path(base) / path


def read_training_metadata(adapter_dir: Path) -> dict:
    """Return the adapter's ``training_metadata.json``, or ``{}`` if unreadable."""
    meta_path = Path(adapter_dir) / TRAINING_METADATA_FILENAME
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _expected_base_model(cfg: Any) -> Optional[str]:
    return _nested_get(cfg, "model.hf_id") or _nested_get(cfg, "model.name")


def check_adapter_compatibility(metadata: Mapping[str, Any], cfg: Any) -> list[str]:
    """Return a list of human-readable mismatches between adapter and config.

    An empty list means compatible. Fields the adapter does not record are
    skipped rather than guessed — older adapters predate ``dataset_version``.
    """
    problems: list[str] = []

    expected_model = _expected_base_model(cfg)
    actual_model = metadata.get("base_model")
    if expected_model and actual_model and str(actual_model) != str(expected_model):
        problems.append(
            f"base model mismatch: adapter was trained on '{actual_model}', "
            f"this run uses '{expected_model}'"
        )

    expected_seed = _get(cfg, "seed")
    actual_seed = metadata.get("seed")
    if expected_seed is not None and actual_seed is not None:
        if int(actual_seed) != int(expected_seed):
            problems.append(
                f"seed mismatch: adapter was trained with seed {actual_seed}, "
                f"this run uses seed {expected_seed}"
            )

    expected_dataset = _nested_get(cfg, "data.dataset_version")
    actual_dataset = metadata.get("dataset_version")
    if expected_dataset and actual_dataset and str(actual_dataset) != str(expected_dataset):
        problems.append(
            f"dataset version mismatch: adapter was trained on "
            f"'{actual_dataset}', this run uses '{expected_dataset}'"
        )

    return problems


def validate_adapter(adapter_dir: Path, cfg: Any, *, strict: bool = True) -> dict:
    """Check the adapter exists and matches ``cfg``; return its metadata.

    Raises :class:`AdapterError` with an actionable message when the folder is
    missing, is not a PEFT adapter, or was trained under different conditions.
    ``strict=False`` downgrades the *compatibility* check to a returned report
    (existence is always enforced) — used by the preflight, which reports rather
    than aborts.
    """
    adapter_dir = Path(adapter_dir)

    if not adapter_dir.exists():
        raise AdapterError(
            f"Adapter not found: {adapter_dir}\n"
            f"Method C (fine-tuning) must run before method D for the same seed. "
            f"Train it with:\n"
            f"    python experiments/scripts/train.py --experiment E03_qwen0_5b_ft "
            f"--override seed={_get(cfg, 'seed', 42)}"
        )

    missing = [n for n in _REQUIRED_ADAPTER_FILES if not (adapter_dir / n).exists()]
    if missing:
        raise AdapterError(
            f"{adapter_dir} is not a usable PEFT adapter folder "
            f"(missing: {', '.join(missing)})."
        )

    if not any((adapter_dir / n).exists() for n in _ADAPTER_WEIGHT_FILES):
        raise AdapterError(
            f"{adapter_dir} contains no adapter weights "
            f"(expected one of: {', '.join(_ADAPTER_WEIGHT_FILES)})."
        )

    metadata = read_training_metadata(adapter_dir)
    problems = check_adapter_compatibility(metadata, cfg)

    if problems and strict:
        bullets = "\n".join(f"  - {p}" for p in problems)
        raise AdapterError(
            f"Adapter at {adapter_dir} is incompatible with this run:\n{bullets}\n"
            f"Refusing to silently reuse a mismatched adapter. Train the matching "
            f"adapter, or set method.adapter_path explicitly if the reuse is "
            f"intentional."
        )

    return {"adapter_dir": str(adapter_dir), "metadata": metadata, "problems": problems}
