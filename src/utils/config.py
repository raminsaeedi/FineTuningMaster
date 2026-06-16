"""Hydra config composition behind a simple function.

The scripts call ``load_cfg(experiment=..., overrides=[...])`` and get a fully
composed OmegaConf config back. This hides Hydra from the entry points so the
professor never has to learn Hydra syntax — ``scripts/train.py --experiment X``
is all that is needed. Hydra-style ``key=value`` overrides are still accepted
for power users.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "src" / "config"


def load_cfg(experiment: Optional[str] = None, overrides: Optional[List[str]] = None):
    """Compose the config; optionally select an experiment and apply overrides."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    all_overrides: List[str] = []
    if experiment:
        all_overrides.append(f"+experiment={experiment}")
    if overrides:
        all_overrides.extend(overrides)

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=all_overrides)

    # Attach a stable hash of the resolved config (used for provenance + caching).
    from src.utils.config_hash import hash_config

    OmegaConf.set_struct(cfg, False)
    cfg.config_hash = hash_config(cfg)
    return cfg
