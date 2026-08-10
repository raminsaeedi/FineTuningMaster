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

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.core.constants import REPORT_SCHEMA_VERSION
from src.utils.config_hash import hash_config
from src.utils.git import get_git_hash, is_git_dirty

# Known data-file config keys -> short labels for manifest hashing.
_DATA_FILE_KEYS = {
    "train": "train_file",
    "val": "val_file",
    "test": "test_file",
    "paraphrased": "paraphrased_file",
    "missing_info": "missing_info_file",
}


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


def _nested_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from a dict / OmegaConf config, tolerant of missing keys."""
    cur = cfg
    for key in dotted.split("."):
        try:
            cur = cur.get(key) if hasattr(cur, "get") else getattr(cur, key)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def _sha256_file(path: Path) -> Optional[str]:
    """Return the sha256 of a file, or ``None`` if it cannot be read."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _data_file_hashes(cfg: Any) -> dict:
    """sha256 of each known data file (best-effort; ``None`` when absent).

    Paths are resolved relative to the current working directory when not
    absolute (runs are launched from the project root). Never raises.
    """
    out: dict = {}
    for label, key in _DATA_FILE_KEYS.items():
        raw = _nested_get(cfg, f"data.{key}")
        if not raw:
            out[label] = None
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = Path.cwd() / path
        out[label] = _sha256_file(path) if path.exists() else None
    return out


def _kb_provenance(cfg: Any) -> dict:
    """Identify the knowledge base a RAG run retrieved from.

    Without this, a B/D run is not reproducible: the retrieved context depends on
    the KB contents, which are rebuilt locally and gitignored. Returns nulls for
    non-RAG methods, which have no retriever config.
    """
    chunks_raw = _nested_get(cfg, "method.retriever.chunks_path")
    if not chunks_raw:
        return {"kb_version": None, "chunks_path": None, "chunks_sha256": None}

    chunks_path = Path(str(chunks_raw))
    if not chunks_path.is_absolute():
        chunks_path = Path.cwd() / chunks_path

    kb_version = None
    manifest_path = chunks_path.parent / "kb_manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                kb_version = json.load(f).get("kb_version")
        except Exception:
            kb_version = None

    return {
        "kb_version": kb_version,
        "chunks_path": str(chunks_raw),
        "chunks_sha256": _sha256_file(chunks_path) if chunks_path.exists() else None,
    }


def _adapter_provenance(cfg: Any) -> dict:
    """Record which adapter this run loads and what it was trained on.

    Methods A and B load none. For C and D the resolved path plus the adapter's
    own recorded training metadata is what lets a reader confirm afterwards that
    D really consumed the C adapter of the same seed.
    """
    method_type = str(_nested_get(cfg, "method.type", ""))
    if "fine_tuned" not in method_type:
        return {"adapter_path": None, "adapter_training_metadata": None}

    try:
        from src.utils.adapter import read_training_metadata, resolve_adapter_path

        adapter_dir = resolve_adapter_path(cfg)
        return {
            "adapter_path": str(adapter_dir),
            "adapter_training_metadata": read_training_metadata(adapter_dir) or None,
        }
    except Exception:
        return {"adapter_path": None, "adapter_training_metadata": None}


def write_manifest(exp_dir: Path, cfg: Any) -> dict:
    """Write a single machine-readable record identifying the run.

    One ``manifest.json`` ties together what the other provenance files describe
    separately (method/model/seed/hashes/timestamp), so a run can be indexed
    without parsing the YAML snapshot. Phase-1 adds additive provenance fields
    (``report_schema_version``, ``eval_tier``, ``layer_status``, data-file hashes);
    the release pass adds dataset version, KB and adapter provenance, and a
    dirty-tree flag so an uncommitted working copy is visible in the record.
    """
    manifest = {
        "experiment_id": str(cfg.get("experiment_id", "")),
        "experiment_name": str(cfg.get("experiment_name", "")),
        "method": str(_nested_get(cfg, "method.name", "")),
        "model": str(_nested_get(cfg, "model.name", "")),
        "model_hf_id": _nested_get(cfg, "model.hf_id"),
        "seed": int(cfg.get("seed", 42)),
        "dataset_version": _nested_get(cfg, "data.dataset_version"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": hash_config(cfg),
        "git_hash": get_git_hash(),
        "git_dirty": is_git_dirty(),
        # --- additive Phase-1 reporting provenance ---
        "report_schema_version": REPORT_SCHEMA_VERSION,
        # Predictions are scored against the synthetic gold; the independent
        # layers (L1 human-effectiveness, L4 human) are pending.
        "eval_tier": str(cfg.get("eval_tier", "internal-synthetic")),
        "layer_status": {
            "L1": "pending",          # set-valued human-effectiveness scorer not implemented
            "L2": "implemented",      # schema/format/robustness metrics exist
            "L3": "pending-data",     # Tableau Census not acquired
            "L4": "pending-ratings",  # human ratings not collected
        },
        "data_file_sha256": _data_file_hashes(cfg),
        "knowledge_base": _kb_provenance(cfg),
        "adapter": _adapter_provenance(cfg),
    }
    with (exp_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def finalize_manifest(exp_dir: Path, status: str = "completed") -> Optional[dict]:
    """Stamp completion time and duration onto an existing manifest.

    ``write_manifest`` runs before the work starts, so on its own the record says
    nothing about whether the run finished or how long it took. Called at the end
    of a run; a no-op (returning ``None``) when the manifest is absent, so a
    partial run never fails here.
    """
    manifest_path = Path(exp_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return None

    finished = datetime.now(timezone.utc)
    manifest["finished_utc"] = finished.isoformat(timespec="seconds")
    manifest["status"] = status
    started_raw = manifest.get("created_utc")
    if started_raw:
        try:
            started = datetime.fromisoformat(str(started_raw))
            manifest["duration_seconds"] = round((finished - started).total_seconds(), 1)
        except Exception:
            manifest["duration_seconds"] = None

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def write_run_metadata(exp_dir: Path, cfg: Any) -> None:
    """Write config snapshot, config hash, git hash, environment and manifest."""
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
    write_manifest(exp_dir, cfg)
