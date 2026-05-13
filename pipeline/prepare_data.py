"""
pipeline/prepare_data.py — Stage 1: Generate and format dataset.

Generates train/val/test JSONL splits plus two robustness test variants:
  - test_paraphrased.jsonl  : same briefs with rephrased field text
  - test_missing_info.jsonl : briefs with 1–2 optional fields removed

Usage:
    python pipeline/prepare_data.py \\
        --base-config configs/base.yaml \\
        --experiment-config configs/experiments/qlora.yaml \\
        [--model-config configs/models/qwen_0_5b.yaml] \\
        [--output-dir data/] \\
        [--force]

The script is idempotent: it skips generation if files already exist
unless --force is passed.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked from any directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config_loader import load_config, parse_cli_overrides
from utils.helpers import save_jsonl, setup_logging

logger = setup_logging()


# ---------------------------------------------------------------------------
# Brief paraphrase helpers
# ---------------------------------------------------------------------------

_GOAL_PREFIXES = [
    "Track and monitor",
    "Gain insights into",
    "Provide visibility into",
    "Enable stakeholders to monitor",
    "Analyse and report on",
]

_AUDIENCE_SUFFIXES = [
    "and decision-makers",
    "across the organisation",
    "in the relevant business unit",
    "",
    "who need actionable insights",
]


def _paraphrase_brief(brief: dict, rng: random.Random) -> dict:
    """
    Apply light textual perturbations to a brief to test robustness.

    Changes: rephrases business_goals prefix, appends to target_audience,
    varies the data_context phrasing.  Core KPIs and industry are kept.
    """
    p = copy.deepcopy(brief)

    # Rephrase business_goals
    goal = p.get("business_goals", "")
    # Strip any existing prefix and re-wrap
    for prefix in ("Monitor ", "Track ", "Analyse ", "Analyze "):
        if goal.startswith(prefix):
            goal = goal[len(prefix):]
            break
    new_prefix = rng.choice(_GOAL_PREFIXES)
    p["business_goals"] = f"{new_prefix} {goal[0].lower() + goal[1:]}"

    # Append noise to target_audience
    suffix = rng.choice(_AUDIENCE_SUFFIXES)
    if suffix:
        p["target_audience"] = p.get("target_audience", "") + " " + suffix

    # Vary data_context wording
    orig_ctx = p.get("data_context", "")
    if "Updated" in orig_ctx:
        p["data_context"] = orig_ctx.replace("Updated", "Refreshed")
    elif "Refreshed" in orig_ctx:
        p["data_context"] = orig_ctx.replace("Refreshed", "Synchronized")

    return p


def _drop_fields_brief(brief: dict, rng: random.Random) -> dict:
    """
    Remove 1–2 optional fields from a brief to simulate incomplete input.

    Required fields preserved: title, kpis, industry.
    Optional fields that may be removed: target_audience, business_goals,
    data_context, update_frequency, user_expertise.
    """
    optional = ["target_audience", "business_goals", "data_context",
                 "update_frequency", "user_expertise"]
    present_optional = [f for f in optional if f in brief]
    n_drop = rng.randint(1, min(2, len(present_optional)))
    to_drop = rng.sample(present_optional, n_drop)

    p = copy.deepcopy(brief)
    for field in to_drop:
        del p[field]
    return p


def generate_perturbation_variants(
    test_data: list[dict], seed: int
) -> tuple[list[dict], list[dict]]:
    """
    Produce paraphrased and missing-info variants of the test set.

    Returns (paraphrased_data, missing_info_data), each with the same
    number of examples as test_data.  The 'recommendation' field is
    preserved unchanged — only the 'brief' is perturbed.
    """
    rng = random.Random(seed + 9999)
    paraphrased = []
    missing_info = []
    for item in test_data:
        paraphrased.append({
            "brief": _paraphrase_brief(item["brief"], rng),
            "recommendation": item["recommendation"],
        })
        missing_info.append({
            "brief": _drop_fields_brief(item["brief"], rng),
            "recommendation": item["recommendation"],
        })
    return paraphrased, missing_info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 — generate and format dataset")
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--model-config", default=None)
    p.add_argument("--experiment-config", default=None)
    p.add_argument("--override", nargs="*", default=[],
                   metavar="KEY=VALUE",
                   help="Dot-notation config overrides, e.g. dataset_generation.num_train_samples=200")
    p.add_argument("--output-dir", default=None,
                   help="Override paths.data_dir")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if files already exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cli = parse_cli_overrides(args.override or [])
    if args.output_dir:
        cli["paths.data_dir"] = args.output_dir

    config = load_config(
        base_config=args.base_config,
        model_config=args.model_config,
        experiment_config=args.experiment_config,
        cli_overrides=cli,
    )

    gen_cfg = config.get("dataset_generation", {})
    num_train = gen_cfg.get("num_train_samples", 80)
    num_val   = gen_cfg.get("num_val_samples",   10)
    num_test  = gen_cfg.get("num_test_samples",  10)
    seed      = gen_cfg.get("seed", 42)
    gen_perturb = gen_cfg.get("generate_perturbations", True)

    data_dir = Path(config["paths"].get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file  = data_dir / "train.jsonl"
    val_file    = data_dir / "val.jsonl"
    test_file   = data_dir / "test.jsonl"
    para_file   = data_dir / "test_paraphrased.jsonl"
    miss_file   = data_dir / "test_missing_info.jsonl"

    files_exist = all(f.exists() for f in [train_file, val_file, test_file])
    if files_exist and not args.force:
        logger.info(
            "Dataset files already exist. Use --force to regenerate. Skipping."
        )
        return

    # Import generate_dataset from the project root
    from generate_dataset import generate_dataset  # type: ignore

    logger.info(f"Generating dataset: {num_train} train / {num_val} val / {num_test} test")
    train_data, val_data, test_data = generate_dataset(num_train, num_val, num_test, seed)

    save_jsonl(train_data, str(train_file))
    save_jsonl(val_data,   str(val_file))
    save_jsonl(test_data,  str(test_file))

    if gen_perturb:
        logger.info("Generating robustness perturbation variants…")
        paraphrased, missing_info = generate_perturbation_variants(test_data, seed)
        save_jsonl(paraphrased,  str(para_file))
        save_jsonl(missing_info, str(miss_file))
        logger.info(f"Saved perturbation variants: {para_file}, {miss_file}")

    logger.info(f"Dataset written to {data_dir.resolve()}")
    print(f"\nDataset ready in {data_dir}/")
    print(f"  train.jsonl            : {len(train_data)} examples")
    print(f"  val.jsonl              : {len(val_data)} examples")
    print(f"  test.jsonl             : {len(test_data)} examples")
    if gen_perturb:
        print(f"  test_paraphrased.jsonl : {len(paraphrased)} examples")
        print(f"  test_missing_info.jsonl: {len(missing_info)} examples")


if __name__ == "__main__":
    main()
