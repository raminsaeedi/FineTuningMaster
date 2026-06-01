"""Run directory + reproducibility artifacts.

Every run writes a self-describing folder so it can be reproduced or audited
later. The artifact contract (also what the professor sends back after training)
is:

    outputs/experiments/<experiment_id>/
        config_snapshot.yaml   # fully resolved config at run time
        config_hash.txt        # stable hash of that config
        git_hash.txt           # repo commit the run was launched from
        env.txt                # `pip freeze` of the environment
        adapter/               # (training) saved LoRA adapter + tokenizer
        logs/                  # log files
        predictions*.jsonl     # (inference) cached predictions
        metrics_auto.json      # (evaluation) computed metrics
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils.config_hash import hash_config
from src.utils.git import get_git_hash


def experiment_dir(cfg: Any, project_root: Path) -> Path:
    """Resolve ``<output_root>/<experiment_id>`` as an absolute path."""
    output_root = str(cfg.get("output_root", "outputs/experiments"))
    experiment_id = str(cfg.get("experiment_id", cfg.get("experiment_name", "default")))
    root = Path(output_root)
    if not root.is_absolute():
        root = project_root / root
    return root / experiment_id


def setup_run_dir(cfg: Any, project_root: Path) -> Path:
    """Create the run directory (and logs subdir) and return it."""
    exp_dir = experiment_dir(cfg, project_root)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    return exp_dir


def _pip_freeze() -> str:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        return out.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"# pip freeze unavailable: {exc}\n"


def write_run_metadata(exp_dir: Path, cfg: Any) -> None:
    """Write config snapshot, config hash, git hash and the environment."""
    exp_dir.mkdir(parents=True, exist_ok=True)

    try:
        from omegaconf import OmegaConf

        snapshot = OmegaConf.to_yaml(cfg, resolve=True)
    except Exception:
        snapshot = str(cfg)
    (exp_dir / "config_snapshot.yaml").write_text(snapshot, encoding="utf-8")
    (exp_dir / "config_hash.txt").write_text(hash_config(cfg) + "\n", encoding="utf-8")
    (exp_dir / "git_hash.txt").write_text(get_git_hash() + "\n", encoding="utf-8")
    (exp_dir / "env.txt").write_text(_pip_freeze(), encoding="utf-8")
