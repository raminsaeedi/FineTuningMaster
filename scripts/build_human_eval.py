"""Build the blind human-evaluation set + balanced rater assignment.

    python scripts/build_human_eval.py \
        --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag \
        --n-items 60 --n-raters 6 --ratings-per-output 3

Reads each experiment's cached predictions + the gold test briefs, selects items
present in every method, and writes:
    results/human_eval/items.jsonl     (brief + each system's output per item)
    results/human_eval/assignment.json (per-rater blind task lists)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import load_gold_items
from src.evaluation.human.assignment import build_assignment, build_eval_items
from src.utils.artifacts import experiment_dir
from src.utils.config import load_cfg
from src.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build human-eval set + assignment")
    p.add_argument("--experiments", nargs="+", required=True,
                   help="Experiment names, one per method")
    p.add_argument("--n-items", type=int, default=60)
    p.add_argument("--n-raters", type=int, default=6)
    p.add_argument("--rater-ids", nargs="*", default=None,
                   help="Explicit rater ids (overrides --n-raters)")
    p.add_argument("--ratings-per-output", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/human_eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    method_to_predictions = {}
    briefs_by_id = None
    for name in args.experiments:
        cfg = load_cfg(experiment=name)
        method = str(cfg.method.name)
        exp_dir = experiment_dir(cfg, _PROJECT_ROOT)
        pred_path = exp_dir / "predictions.jsonl"
        if not pred_path.exists():
            raise SystemExit(f"Missing predictions for {name}: {pred_path}. Run inference first.")
        method_to_predictions[method] = read_jsonl(pred_path)

        if briefs_by_id is None:
            test_file = Path(str(cfg.data.test_file))
            if not test_file.is_absolute():
                test_file = _PROJECT_ROOT / test_file
            briefs_by_id = {
                it.item_id: it.brief.model_dump(mode="json") for it in load_gold_items(test_file)
            }

    items = build_eval_items(method_to_predictions, briefs_by_id, n_items=args.n_items, seed=args.seed)
    if not items:
        raise SystemExit("No items common to all methods — check that every method ran on the same test set.")

    methods = list(method_to_predictions)
    item_ids = [it["item_id"] for it in items]
    raters = args.rater_ids or [f"rater_{i:02d}" for i in range(1, args.n_raters + 1)]
    assignment = build_assignment(item_ids, methods, raters, args.ratings_per_output, args.seed)

    out_dir = _PROJECT_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(items, out_dir / "items.jsonl")
    (out_dir / "assignment.json").write_text(json.dumps(assignment, indent=2), encoding="utf-8")

    print("=" * 56)
    print("HUMAN-EVAL SET BUILT")
    print("=" * 56)
    print(f"  Methods            : {methods}")
    print(f"  Items              : {len(items)}")
    print(f"  Raters             : {raters}")
    print(f"  Ratings per output : {args.ratings_per_output}")
    print(f"  Total ratings      : {assignment['config']['n_units'] * args.ratings_per_output}")
    print(f"  Per-rater load     : {assignment['load']}")
    print(f"  Output dir         : {out_dir}")
    print("\nNext: python scripts/run_human_eval.py")
    print("=" * 56)


if __name__ == "__main__":
    main()
