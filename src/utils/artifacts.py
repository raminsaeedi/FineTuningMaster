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
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.core.constants import REPORT_SCHEMA_VERSION
from src.inference.batching import batching_provenance
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
    """Resolve a run directory, preserving the legacy layout by default.

    Profile-aware runs use the collision-proof thesis layout:
    ``<root>/<dataset>/<model>/<method>/seed_<n>``.
    """
    output_root = str(cfg.get("output_root", "outputs/experiments"))
    experiment_id = str(cfg.get("experiment_id", cfg.get("experiment_name", "default")))
    root = Path(output_root)
    if not root.is_absolute():
        root = project_root / root
    layout = str(cfg.get("run_layout", cfg.get("profile", "legacy")) or "legacy")
    if layout in {"final", "smoke"}:
        dataset = str(_nested_get(cfg, "data.dataset_version", "dashboard_v3"))
        model_key = str(
            cfg.get("model_key")
            or _nested_get(cfg, "model.key")
            or _nested_get(cfg, "model.name", "model")
        )
        method_key = str(
            cfg.get("method_key")
            or _nested_get(cfg, "method.key")
            or _nested_get(cfg, "method.name", "method")
        )
        seed = int(cfg.get("seed", 42))
        return root / dataset / model_key / method_key / f"seed_{seed}"
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


def _plain(value: Any) -> Any:
    """Convert OmegaConf containers to JSON-safe plain values."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _redact(value: Any, key: str = "") -> Any:
    """Remove accidental secret-like config fields from shipped artifacts."""
    lowered = key.lower()
    secret_key = (
        lowered in {"token", "hf_token", "secret", "password", "api_key"}
        or "secret" in lowered
        or "password" in lowered
        or "api_key" in lowered
    )
    if secret_key:
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key) for v in value]
    return value


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


def _kb_manifest_hash(cfg: Any) -> Optional[str]:
    chunks_raw = _nested_get(cfg, "method.retriever.chunks_path")
    if not chunks_raw:
        return None
    chunks_path = Path(str(chunks_raw))
    if not chunks_path.is_absolute():
        chunks_path = Path.cwd() / chunks_path
    manifest_path = chunks_path.parent / "kb_manifest.json"
    return _sha256_file(manifest_path) if manifest_path.exists() else None


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


def _source_run_provenance(adapter: dict) -> dict:
    """Read the producing C manifest without changing the legacy adapter shape."""
    raw_path = adapter.get("adapter_path")
    if not raw_path:
        return {"source_c_run_id": None, "adapter_manifest_hash": None}
    adapter_dir = Path(str(raw_path))
    manifest_path = adapter_dir.parent / "manifest.json"
    if not manifest_path.exists():
        return {"source_c_run_id": None, "adapter_manifest_hash": None}
    source_id = None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, dict):
                source_id = payload.get("run_id") or payload.get("experiment_id")
    except Exception:
        source_id = None
    return {
        "source_c_run_id": source_id,
        "adapter_manifest_hash": _sha256_file(manifest_path),
    }


def _model_key(cfg: Any) -> str:
    return str(
        cfg.get("model_key")
        or _nested_get(cfg, "model.key")
        or _nested_get(cfg, "model.name", "")
    )


def _method_key(cfg: Any) -> str:
    return str(
        cfg.get("method_key")
        or _nested_get(cfg, "method.key")
        or _nested_get(cfg, "method.name", "")
    )


def _hardware_provenance() -> dict:
    result = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        result["pytorch"] = getattr(torch, "__version__", None)
        result["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            result["gpu_name"] = torch.cuda.get_device_name(device)
            result["gpu_total_memory_bytes"] = int(getattr(props, "total_memory", 0))
    except Exception:
        result["pytorch"] = None
        result["cuda_version"] = None
        result["cuda_available"] = False
    return result


def _package_versions() -> dict:
    names = ("transformers", "peft", "trl", "accelerate", "bitsandbytes", "datasets")
    return {
        name: (importlib.metadata.version(name) if _distribution_exists(name) else None)
        for name in names
    }


def _distribution_exists(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
    except Exception:
        return False


def cache_identity(cfg: Any) -> dict:
    """Identity used to decide whether cached inference is reusable."""
    data_hashes = _data_file_hashes(cfg)
    kb = _kb_provenance(cfg)
    return {
        "dataset_version": _nested_get(cfg, "data.dataset_version"),
        "dataset_hashes": data_hashes,
        "model_key": _model_key(cfg),
        "model_hf_id": _nested_get(cfg, "model.hf_id") or _nested_get(cfg, "model.name"),
        "model_revision": _nested_get(cfg, "model.revision"),
        "method": _method_key(cfg),
        "seed": int(cfg.get("seed", 42)),
        "training_config_hash": hash_config(_plain(_nested_get(cfg, "training", {}))),
        "inference_config_hash": hash_config({
            "generate": _plain(_nested_get(cfg, "method.generate", {})),
            "eval": _plain(_nested_get(cfg, "eval", {})),
        }),
        "kb_hash": kb.get("chunks_sha256"),
        "kb_manifest_hash": _kb_manifest_hash(cfg),
    }


def write_manifest(exp_dir: Path, cfg: Any) -> dict:
    """Write a single machine-readable record identifying the run.

    One ``manifest.json`` ties together what the other provenance files describe
    separately (method/model/seed/hashes/timestamp), so a run can be indexed
    without parsing the YAML snapshot. Phase-1 adds additive provenance fields
    (``report_schema_version``, ``eval_tier``, ``layer_status``, data-file hashes);
    the release pass adds dataset version, KB and adapter provenance, and a
    dirty-tree flag so an uncommitted working copy is visible in the record.
    """
    data_hashes = _data_file_hashes(cfg)
    adapter = _adapter_provenance(cfg)
    identity = cache_identity(cfg)
    method_type = str(_nested_get(cfg, "method.type", ""))
    source = (
        _source_run_provenance(adapter)
        if "fine_tuned_rag" in method_type
        else {"source_c_run_id": None, "adapter_manifest_hash": None}
    )
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "experiment_id": str(cfg.get("experiment_id", "")),
        "run_id": str(cfg.get("run_id", cfg.get("experiment_id", ""))),
        "experiment_name": str(cfg.get("experiment_name", "")),
        "method": str(_nested_get(cfg, "method.name", "")),
        "method_key": _method_key(cfg),
        "model": str(_nested_get(cfg, "model.name", "")),
        "model_key": _model_key(cfg),
        "model_hf_id": _nested_get(cfg, "model.hf_id"),
        "model_revision": _nested_get(cfg, "model.revision"),
        "model_parameters_billions": _nested_get(cfg, "model.size_billions"),
        "chat_template": _redact(_plain(_nested_get(cfg, "model.chat_template", {}))),
        "seed": int(cfg.get("seed", 42)),
        "dataset_version": _nested_get(cfg, "data.dataset_version"),
        "created_utc": created,
        "start_utc": created,
        "profile": str(cfg.get("profile", "legacy")),
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
        "data_file_sha256": data_hashes,
        "dataset_hashes": data_hashes,
        "knowledge_base": _kb_provenance(cfg),
        "adapter": adapter,
        "source_c_run_id": source["source_c_run_id"],
        "adapter_path": adapter.get("adapter_path"),
        "adapter_manifest_hash": source["adapter_manifest_hash"],
        "training_config": _redact(_plain(_nested_get(cfg, "training", {}))),
        "inference_config": _redact(_plain(_nested_get(cfg, "method.generate", {}))),
        # Which inference regime produced this run: sequential (the default) or
        # the opt-in batched throughput mode, whose per-item outputs are not
        # comparable with a sequential run. Never absent, so a reader can always
        # tell the two apart.
        "inference_batching": batching_provenance(cfg),
        "rag_config": _redact(_plain(_nested_get(cfg, "method.retriever", {}))),
        "kb_hashes": {
            "chunks_sha256": _kb_provenance(cfg).get("chunks_sha256"),
            "manifest_sha256": _kb_manifest_hash(cfg),
        },
        "cache_identity": identity,
        "cache_identity_hash": hash_config(identity),
        "hardware": _hardware_provenance(),
        "package_versions": _package_versions(),
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
    manifest["end_utc"] = manifest["finished_utc"]
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


def update_manifest(exp_dir: Path, updates: dict) -> dict:
    """Merge additive provenance fields into an existing run manifest."""
    manifest_path = Path(exp_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Run manifest is not valid JSON: {manifest_path}") from exc

    manifest.update(dict(updates))
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def write_run_metadata(exp_dir: Path, cfg: Any) -> None:
    """Write config snapshot, config hash, git hash, environment and manifest."""
    exp_dir.mkdir(parents=True, exist_ok=True)

    try:
        from omegaconf import OmegaConf

        snapshot_value = _redact(_plain(cfg))
        snapshot = OmegaConf.to_yaml(snapshot_value, resolve=True)
    except Exception:
        snapshot = json.dumps(_redact(_plain(cfg)), indent=2, default=str)
    (exp_dir / "config_snapshot.yaml").write_text(snapshot, encoding="utf-8")
    (exp_dir / "config_hash.txt").write_text(hash_config(cfg) + "\n", encoding="utf-8")
    (exp_dir / "git_hash.txt").write_text(get_git_hash() + "\n", encoding="utf-8")
    (exp_dir / "env.txt").write_text(_pip_freeze(), encoding="utf-8")
    manifest = write_manifest(exp_dir, cfg)
    (exp_dir / "cache_identity.json").write_text(
        json.dumps(manifest["cache_identity"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (exp_dir / "dataset_hashes.json").write_text(
        json.dumps(manifest["dataset_hashes"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (exp_dir / "kb_hashes.json").write_text(
        json.dumps(manifest["kb_hashes"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
