"""Stable hashing of a resolved config, for caching and reproducibility.

The hash is taken over the fully resolved YAML so that two runs with identical
effective configuration produce the same hash, regardless of how the config was
composed.
"""

from __future__ import annotations

import hashlib
from typing import Any


def hash_config(cfg: Any) -> str:
    """Return a short, stable sha256 prefix of the resolved config."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            serial = OmegaConf.to_yaml(cfg, resolve=True)
        else:
            serial = str(cfg)
    except Exception:
        serial = str(cfg)
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()[:12]
