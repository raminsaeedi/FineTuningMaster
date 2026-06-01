"""
pipeline/train.py — Stage 2: Fine-tune with the selected algorithm.

Usage:
    python pipeline/train.py \\
        --base-config configs/base.yaml \\
        --model-config configs/models/qwen_0_5b.yaml \\
        --experiment-config configs/experiments/qlora.yaml \\
        [--override training.num_train_epochs=5] \\
        [--override lora.r=32] \\
        [--debug]   # 10 samples, 1 epoch — for quick smoke tests

On completion the experiment directory is printed. Pass it to
pipeline/inference.py to run predictions.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked from any directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from algorithms import get_algorithm
from utils.config_loader import load_config, parse_cli_overrides
from utils.experiment import save_metrics, set_seeds, setup_experiment
from utils.helpers import format_training_example, load_jsonl, print_device_info, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 — fine-tune with selected algorithm")
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--model-config", required=True,
                   help="e.g. configs/models/qwen_0_5b.yaml")
    p.add_argument("--experiment-config", required=True,
                   help="e.g. configs/experiments/qlora.yaml")
    p.add_argument("--override", nargs="*", default=[],
                   metavar="KEY=VALUE",
                   help="Dot-notation config overrides, e.g. training.learning_rate=5e-4")
    p.add_argument("--debug", action="store_true",
                   help="Debug mode: 10 samples, 1 epoch — fast sanity check")
    return p.parse_args()


def load_and_format_dataset(config: dict, debug: bool = False):
    """Load JSONL data and produce a HuggingFace Dataset with a 'text' column."""
    from datasets import Dataset
    from transformers import AutoTokenizer

    model_name = config["model"]["name"]
    cache_dir = config["model"].get("cache_dir")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_file = config["data"]["train_file"]
    if not Path(train_file).exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            "Run pipeline/prepare_data.py first."
        )

    raw = load_jsonl(train_file)

    if debug:
        raw = raw[:10]
        logging.getLogger(__name__).warning("DEBUG MODE: using only 10 samples")

    max_samples = config["data"].get("max_samples")
    if max_samples:
        raw = raw[:int(max_samples)]

    algorithm_name = config.get("algorithm", {}).get("name", "qlora")

    if algorithm_name == "orpo":
        # ORPO requires preference pairs {prompt, chosen, rejected}
        rows = _format_orpo_examples(raw, tokenizer)
    else:
        rows = [
            {"text": format_training_example(item["brief"], item["recommendation"], tokenizer)}
            for item in raw
        ]

    return Dataset.from_list(rows)


def _format_orpo_examples(raw: list[dict], tokenizer) -> list[dict]:
    """
    Format examples as ORPO preference pairs.

    Chosen  = the ground-truth structured recommendation (valid JSON).
    Rejected = the same response with the first 2 required keys removed
               (simulates an incomplete/bad output).

    NOTE: In a real experiment, rejected responses should be actual bad model
    outputs or deliberately degraded examples.  This provides a functional
    placeholder that can be replaced with real preference data.
    """
    import json
    from utils.helpers import REQUIRED_KEYS, format_instruction_prompt

    rows = []
    keys = list(REQUIRED_KEYS)
    for item in raw:
        prompt = format_instruction_prompt(item["brief"], tokenizer)
        chosen = json.dumps(item["recommendation"], ensure_ascii=False)
        # Create rejected by dropping the last 2 required keys
        bad_rec = {k: v for k, v in item["recommendation"].items() if k not in keys[-2:]}
        rejected = json.dumps(bad_rec, ensure_ascii=False)
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return rows


def main() -> None:
    args = parse_args()
    cli = parse_cli_overrides(args.override or [])

    config = load_config(
        base_config=args.base_config,
        model_config=args.model_config,
        experiment_config=args.experiment_config,
        cli_overrides=cli,
    )

    if args.debug:
        config["training"]["num_train_epochs"] = 1
        config["training"]["save_steps"] = 5
        config["training"]["logging_steps"] = 1
        config["data"]["max_samples"] = 10

    # Set up experiment directory and frozen config snapshot
    experiment_dir, config = setup_experiment(config)
    experiment_id = config["_runtime"]["experiment_id"]

    # Configure logging to write to experiment directory
    log_file = str(experiment_dir / "logs" / "train.log")
    logger = setup_logging(
        log_level=config.get("meta", {}).get("log_level", "INFO"),
        log_file=log_file,
    )

    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_id}")
    logger.info(f"Algorithm : {config['algorithm']['name']}")
    logger.info(f"Model     : {config['model']['name']}")
    logger.info(f"Output dir: {experiment_dir}")
    logger.info("=" * 60)

    # Load dataset BEFORE any torch.cuda call — avoids Windows 0xC0000005 (pyarrow/CUDA DLL order)
    logger.info("Loading and formatting training dataset…")
    train_dataset = load_and_format_dataset(config, debug=args.debug)
    logger.info(f"Training dataset: {len(train_dataset)} examples")

    # Set seeds and log device info after pyarrow DLLs are already loaded
    set_seeds(config.get("meta", {}).get("seed", 42))
    print_device_info()

    # Dispatch to algorithm
    algorithm_name = config["algorithm"]["name"]
    AlgorithmClass = get_algorithm(algorithm_name)
    finetuner = AlgorithmClass(config, experiment_dir)

    metrics = finetuner.run(train_dataset)

    # Persist training metrics into experiment metrics.json
    metrics_payload = {
        "experiment_id": experiment_id,
        "algorithm": algorithm_name,
        "model": config["model"]["name"],
        "training": metrics,
    }
    save_metrics(metrics_payload, experiment_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Experiment ID : {experiment_id}")
    print(f"  Loss          : {metrics.get('train_loss', 'N/A')}")
    print(f"  Runtime       : {metrics.get('train_runtime', 0):.0f}s")
    print(f"  Output dir    : {experiment_dir}")
    print(f"\nNext step:")
    print(f"  python pipeline/inference.py --experiment-dir {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
