"""Fine-tune the model — the single entry point for the GPU machine.

See the "how to train" section of the
README. In short, install the training requirements and run ONE command:

    python scripts/train.py --experiment E03_qwen0_5b_ft

This writes everything needed to reproduce and use the run to:

    outputs/experiments/<experiment_id>/

Send that whole folder back. Hydra is used under the hood; you do not need to
know it. Use --debug for a fast sanity run (few samples, 1 epoch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import src.training  # noqa: F401,E402  (registers trainers under TRAINERS)
from src.core.registry import TRAINERS  # noqa: E402
from src.data_pipeline.dataset import load_gold_items  # noqa: E402
from src.data_pipeline.formatter import format_training_example  # noqa: E402
from src.utils.artifacts import (  # noqa: E402
    setup_run_dir,
    update_manifest,
    write_run_metadata,
)
from src.utils.config import load_cfg  # noqa: E402
from src.utils.config_hash import hash_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.seed import set_seeds  # noqa: E402
from src.utils.adapter import resolve_adapter_output_path  # noqa: E402
from src.models.hf_utils import (  # noqa: E402
    chat_template_kwargs,
    from_pretrained_kwargs,
    model_identifier,
    safe_model_access_error,
)


class ResumeError(RuntimeError):
    """Raised when a requested checkpoint cannot be used safely."""


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_CHECKPOINT_MARKERS = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _resolve_path(raw: Any, project_root: Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_resume_metadata(cfg: Any, project_root: Optional[Path] = None) -> dict:
    """Build compatibility metadata for checkpoints created by this trainer."""
    project_root = project_root or _PROJECT_ROOT
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    train_file = data_cfg.get("train_file")
    train_path = _resolve_path(train_file, project_root) if train_file else None
    dataset_hash = _sha256_file(train_path) if train_path else None
    model_name = model_cfg.get("name")
    model_hf_id = model_cfg.get("hf_id") or model_name
    return {
        "metadata_version": 1,
        "created_utc": _utc_now(),
        "experiment": str(cfg.get("experiment_name", "")),
        "experiment_name": str(cfg.get("experiment_name", "")),
        "experiment_id": str(cfg.get("experiment_id", "")),
        "model": str(model_name or ""),
        "model_hf_id": str(model_hf_id or ""),
        "model_key": model_cfg.get("key") or cfg.get("model_key"),
        "model_revision": model_cfg.get("revision"),
        "seed": int(cfg.get("seed", 42)),
        "dataset_version": data_cfg.get("dataset_version"),
        "dataset_hash": dataset_hash,
        "train_file": str(train_file) if train_file else None,
        "train_file_sha256": dataset_hash,
        "training_config_hash": hash_config(training_cfg),
        "config_hash": hash_config(cfg),
    }


def _checkpoint_number(path: Path) -> Optional[int]:
    match = _CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def is_valid_checkpoint(path: Path) -> bool:
    """Return whether path has native Trainer state and checkpoint payload."""
    path = Path(path)
    if not path.is_dir() or _checkpoint_number(path) is None:
        return False
    state_path = path / "trainer_state.json"
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("global_step"), (int, float)):
            return False
    except (OSError, ValueError, TypeError):
        return False
    return any((path / marker).exists() for marker in _CHECKPOINT_MARKERS)


def find_latest_checkpoint(checkpoint_root: Path) -> Optional[Path]:
    """Find newest valid checkpoint by numeric step, not filename ordering."""
    checkpoint_root = Path(checkpoint_root)
    candidates = [
        path for path in checkpoint_root.glob("checkpoint-*")
        if is_valid_checkpoint(path)
    ]
    return max(candidates, key=lambda path: _checkpoint_number(path) or -1) if candidates else None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _metadata_candidates(checkpoint: Path) -> list[Path]:
    return [
        checkpoint / "resume_metadata.json",
        checkpoint.parent / "resume_metadata.json",
        checkpoint.parent.parent / "resume_metadata.json",
    ]


def read_resume_metadata(checkpoint: Path) -> dict:
    for metadata_path in _metadata_candidates(Path(checkpoint)):
        if not metadata_path.exists():
            continue
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def checkpoint_global_step(checkpoint: Path) -> int:
    try:
        state = json.loads((Path(checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
        return int(state.get("global_step", 0) or 0)
    except (OSError, ValueError, TypeError):
        return 0


def _metadata_problems(actual: dict, expected: dict) -> list[str]:
    problems = []
    actual_experiment = actual.get("experiment_name") or actual.get("experiment")
    if actual_experiment and actual_experiment != expected["experiment_name"]:
        problems.append(
            f"experiment mismatch: checkpoint is '{actual_experiment}', "
            f"run is '{expected['experiment_name']}'"
        )
    if actual.get("experiment_id") and actual["experiment_id"] != expected["experiment_id"]:
        problems.append(
            f"experiment id mismatch: checkpoint is '{actual['experiment_id']}', "
            f"run is '{expected['experiment_id']}'"
        )

    actual_model = actual.get("model_hf_id") or actual.get("base_model")
    expected_model = expected["model_hf_id"]
    if actual_model and str(actual_model) != str(expected_model):
        problems.append(
            f"model mismatch: checkpoint is '{actual_model}', run is '{expected_model}'"
        )
    actual_model_name = actual.get("model")
    if actual_model_name and actual_model_name not in {
        expected["model"], expected["model_hf_id"]
    }:
        problems.append(
            f"model config mismatch: checkpoint is '{actual_model_name}', "
            f"run is '{expected['model']}'"
        )

    if actual.get("seed") is not None:
        try:
            seed_mismatch = int(actual["seed"]) != int(expected["seed"])
        except (TypeError, ValueError):
            problems.append(f"invalid checkpoint seed: {actual['seed']}")
        else:
            if seed_mismatch:
                problems.append(
                    f"seed mismatch: checkpoint is {actual['seed']}, run is {expected['seed']}"
                )
    if (
        actual.get("dataset_version") is not None
        and actual.get("dataset_version") != expected["dataset_version"]
    ):
        problems.append(
            f"dataset version mismatch: checkpoint is '{actual['dataset_version']}', "
            f"run is '{expected['dataset_version']}'"
        )

    actual_dataset_hash = actual.get("dataset_hash") or actual.get("train_file_sha256")
    if actual_dataset_hash and actual_dataset_hash != expected["dataset_hash"]:
        problems.append("dataset hash mismatch between checkpoint and run")
    if (
        actual.get("training_config_hash")
        and actual["training_config_hash"] != expected["training_config_hash"]
    ):
        problems.append("training configuration hash mismatch between checkpoint and run")
    return problems


def _metadata_missing_fields(metadata: dict) -> list[str]:
    """Return compatibility fields absent from checkpoint provenance."""
    missing = []
    if not (
        metadata.get("experiment_id")
        or metadata.get("experiment_name")
        or metadata.get("experiment")
    ):
        missing.append("experiment identity")
    if not (metadata.get("model_hf_id") or metadata.get("base_model") or metadata.get("model")):
        missing.append("model")
    if "seed" not in metadata or metadata.get("seed") is None:
        missing.append("seed")
    if "dataset_version" not in metadata:
        missing.append("dataset version")
    if not ("dataset_hash" in metadata or "train_file_sha256" in metadata):
        missing.append("dataset hash")
    if not metadata.get("training_config_hash"):
        missing.append("training configuration hash")
    return missing


def resolve_resume_checkpoint(
    exp_dir: Path,
    cfg: Any,
    *,
    resume: bool = False,
    resume_from: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve and validate explicit or automatic checkpoint resume."""
    if not resume and not resume_from:
        return None

    project_root = project_root or _PROJECT_ROOT
    exp_dir = Path(exp_dir)
    checkpoint_root = exp_dir / "checkpoints"
    explicit = bool(resume_from)
    if resume_from:
        checkpoint = _resolve_path(resume_from, project_root)
        if not is_valid_checkpoint(checkpoint):
            raise ResumeError(f"Invalid resume checkpoint: {checkpoint}")
    else:
        checkpoint = find_latest_checkpoint(checkpoint_root)
        if checkpoint is None:
            raise ResumeError(f"No valid checkpoint found under {checkpoint_root}")

    metadata = read_resume_metadata(checkpoint)
    expected = build_resume_metadata(cfg, project_root)
    expected_root = checkpoint_root
    if not metadata:
        if not explicit or not _is_within(checkpoint, expected_root):
            raise ResumeError(
                f"Resume checkpoint lacks compatibility metadata: {checkpoint}. "
                "Use an explicit checkpoint from this run or create a new run."
            )
    else:
        missing_fields = _metadata_missing_fields(metadata)
        if missing_fields and (not explicit or not _is_within(checkpoint, expected_root)):
            raise ResumeError(
                f"Resume checkpoint lacks sufficient compatibility metadata: {checkpoint}. "
                f"Missing {', '.join(missing_fields)}. "
                "Use an explicit checkpoint from this run or create a new run."
            )
        problems = _metadata_problems(metadata, expected)
        if problems:
            raise ResumeError(
                f"Resume checkpoint incompatible: {checkpoint}\n"
                + "\n".join(f"- {problem}" for problem in problems)
            )

    return checkpoint


def _write_resume_metadata(exp_dir: Path, metadata: dict) -> None:
    exp_dir = Path(exp_dir)
    checkpoint_root = exp_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for path in (exp_dir / "resume_metadata.json", checkpoint_root / "resume_metadata.json"):
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def update_resume_manifest(
    exp_dir: Path,
    *,
    resumed: bool,
    resume_checkpoint: Optional[str],
    initial_global_step: int,
    final_global_step: Optional[int],
    resume_timestamp: Optional[str],
) -> dict:
    """Record resume provenance without replacing existing manifest fields."""
    return update_manifest(exp_dir, {
        "resumed": bool(resumed),
        "resume_checkpoint": resume_checkpoint,
        "initial_global_step": int(initial_global_step),
        "final_global_step": final_global_step,
        "resume_timestamp": resume_timestamp,
    })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune with the configured trainer")
    p.add_argument("--experiment", required=True,
                   help="Experiment config name, e.g. E03_qwen0_5b_ft")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Optional Hydra-style overrides, e.g. training.sft.learning_rate=1e-4")
    p.add_argument("--debug", action="store_true",
                   help="Fast sanity run: 10 samples, 1 epoch")
    p.add_argument("--resume", action="store_true",
                   help="Resume from newest valid checkpoint in this run")
    p.add_argument("--resume-from", default=None, metavar="CHECKPOINT",
                   help="Resume from this explicit Trainer checkpoint path")
    return p.parse_args()


def _apply_debug(cfg) -> None:
    cfg.training.sft.num_train_epochs = 1
    cfg.training.sft.logging_steps = 1
    cfg.training.sft.save_steps = 5
    cfg.data.max_samples = 10


def load_and_format_train_dataset(cfg, debug: bool):
    """Load the processed train split and format it into a 'text' column.

    Done BEFORE any CUDA call (Windows DLL load-order safety).
    """
    from datasets import Dataset
    from transformers import AutoTokenizer

    name = model_identifier(cfg.model)
    tokenizer_kwargs = from_pretrained_kwargs(
        cfg.model,
        cache_dir=cfg.model.get("cache_dir"),
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
    except Exception as exc:
        raise safe_model_access_error(name, exc) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_file = cfg.data.get("train_file")
    train_path = Path(train_file)
    if not train_path.is_absolute():
        train_path = _PROJECT_ROOT / train_path
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. Run `python scripts/build_data.py` first."
        )

    items = load_gold_items(train_path)
    if debug:
        items = items[:10]
    max_samples = cfg.data.get("max_samples")
    if max_samples:
        items = items[: int(max_samples)]

    rows = [
        {
            "text": format_training_example(
                it.brief,
                it.recommendation,
                tokenizer,
                chat_template_kwargs(cfg.model),
            )
        }
        for it in items
    ]
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(experiment=args.experiment, overrides=args.override)
    if args.debug:
        _apply_debug(cfg)

    exp_dir = setup_run_dir(cfg, _PROJECT_ROOT)
    adapter_dir = resolve_adapter_output_path(cfg, _PROJECT_ROOT)
    # Checkpoints follow the adapter root so OUTPUT_MODEL_PATH is a complete,
    # relocatable training destination. With the default config this is the
    # historical exp_dir/checkpoints location.
    resume_root = adapter_dir.parent
    resume_checkpoint = resolve_resume_checkpoint(
        resume_root,
        cfg,
        resume=args.resume,
        resume_from=args.resume_from,
        project_root=_PROJECT_ROOT,
    )
    resume_metadata = build_resume_metadata(cfg, _PROJECT_ROOT)
    _write_resume_metadata(resume_root, resume_metadata)
    logger = setup_logging(
        level=str(cfg.get("log_level", "INFO")),
        log_file=str(exp_dir / "logs" / "train.log"),
    )

    logger.info("=" * 60)
    logger.info("Experiment : %s", cfg.get("experiment_id"))
    logger.info("Trainer    : %s", cfg.training.get("type"))
    logger.info("Model      : %s", cfg.model.get("name"))
    logger.info("Output dir : %s", exp_dir)
    logger.info("=" * 60)

    # Dataset first (before CUDA), then provenance, then seeds.
    logger.info("Loading and formatting training data…")
    train_dataset = load_and_format_train_dataset(cfg, debug=args.debug)
    logger.info("Training examples: %d", len(train_dataset))

    write_run_metadata(exp_dir, cfg)
    resumed = resume_checkpoint is not None
    initial_global_step = checkpoint_global_step(resume_checkpoint) if resume_checkpoint else 0
    resume_timestamp = _utc_now() if resumed else None
    update_resume_manifest(
        exp_dir,
        resumed=resumed,
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
        initial_global_step=initial_global_step,
        final_global_step=None,
        resume_timestamp=resume_timestamp,
    )
    set_seeds(int(cfg.get("seed", 42)))

    trainer_cls = TRAINERS.get(str(cfg.training.type))
    trainer = trainer_cls(cfg)
    if resume_checkpoint:
        trainer.train(
            train_dataset,
            None,
            str(adapter_dir),
            resume_from_checkpoint=str(resume_checkpoint),
        )
    else:
        trainer.train(train_dataset, None, str(adapter_dir))
    final_global_step = int(getattr(trainer, "final_global_step", 0) or 0)
    update_resume_manifest(
        exp_dir,
        resumed=resumed,
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
        initial_global_step=initial_global_step,
        final_global_step=final_global_step,
        resume_timestamp=resume_timestamp,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Experiment : {cfg.get('experiment_id')}")
    print(f"  Send back this whole folder:\n    {exp_dir}")
    print("  It contains: adapter/, config_snapshot.yaml, config_hash.txt,")
    print("               env.txt, git_hash.txt, logs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
