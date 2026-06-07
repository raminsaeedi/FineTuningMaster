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
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import src.training  # noqa: F401,E402  (registers trainers under TRAINERS)
from src.core.registry import TRAINERS  # noqa: E402
from src.data_pipeline.dataset import load_gold_items  # noqa: E402
from src.data_pipeline.formatter import format_training_example  # noqa: E402
from src.utils.artifacts import setup_run_dir, write_run_metadata  # noqa: E402
from src.utils.config import load_cfg  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.seed import set_seeds  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune with the configured trainer")
    p.add_argument("--experiment", required=True,
                   help="Experiment config name, e.g. E03_qwen0_5b_ft")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Optional Hydra-style overrides, e.g. training.sft.learning_rate=1e-4")
    p.add_argument("--debug", action="store_true",
                   help="Fast sanity run: 10 samples, 1 epoch")
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

    name = cfg.model.get("hf_id") or cfg.model.get("name")
    tokenizer = AutoTokenizer.from_pretrained(
        name, trust_remote_code=True, cache_dir=cfg.model.get("cache_dir")
    )
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
        {"text": format_training_example(it.brief, it.recommendation, tokenizer)}
        for it in items
    ]
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(experiment=args.experiment, overrides=args.override)
    if args.debug:
        _apply_debug(cfg)

    exp_dir = setup_run_dir(cfg, _PROJECT_ROOT)
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
    set_seeds(int(cfg.get("seed", 42)))

    trainer_cls = TRAINERS.get(str(cfg.training.type))
    trainer = trainer_cls(cfg)
    adapter_dir = exp_dir / "adapter"
    trainer.train(train_dataset, None, str(adapter_dir))

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
