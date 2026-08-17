"""Build one immutable blind human-evaluation study.

Primary Professor layout::

    python experiments/scripts/build_human_eval.py \
        --dataset dashboard_v4 --model qwen3_8b --seed 42 \
        --n-items 40 --n-raters 6 --ratings-per-output 3

The four prediction files are resolved automatically from
``experiments/outputs/final/<dataset>/<model>/<A-D>/seed_<seed>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.human.pipeline import (  # noqa: E402
    DEFAULT_OUTPUTS_ROOT,
    HumanEvaluationError,
    build_study,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blind human-evaluation study")
    parser.add_argument("--dataset", default="dashboard_v4")
    parser.add_argument("--model", default=None, help="Final model key, e.g. qwen3_8b")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outputs-root", default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--n-items", type=int, default=40)
    parser.add_argument("--n-raters", type=int, default=6)
    parser.add_argument("--rater-ids", nargs="+", default=None)
    parser.add_argument("--ratings-per-output", type=int, default=3)
    parser.add_argument("--assignment-seed", type=int, default=42)
    parser.add_argument("--item-list", default=None, help="Optional CSV/JSONL item-ID list")
    parser.add_argument("--test-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out-dir", default=None)
    # Retain old spelling as a clear migration path. It is deliberately not
    # used for the final path: callers must place Professor outputs in A-D dirs.
    parser.add_argument("--experiments", nargs="+", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.experiments:
        raise SystemExit(
            "Legacy --experiments path resolution is deprecated. "
            "Use --dataset, --model, --seed with the Professor A/B/C/D layout."
        )
    if args.model is None or args.seed is None:
        raise SystemExit("--model and --seed are required for the final Professor layout.")
    try:
        result = build_study(
            project_root=_PROJECT_ROOT,
            dataset=args.dataset,
            model=args.model,
            seed=args.seed,
            outputs_root=args.outputs_root,
            n_items=args.n_items,
            n_raters=args.n_raters,
            rater_ids=args.rater_ids,
            ratings_per_output=args.ratings_per_output,
            assignment_seed=args.assignment_seed,
            item_list=args.item_list,
            test_file=args.test_file,
            out_dir=args.out_dir,
        )
    except HumanEvaluationError as exc:
        raise SystemExit(str(exc)) from exc

    manifest = result["manifest"]
    assignment = result["assignment"]
    print("HUMAN-EVAL STUDY BUILT")
    print(f"  study type        : {manifest['study_type']}")
    print(f"  dataset/model/seed: {manifest['dataset']} / {manifest['model']} / {manifest['seed']}")
    print(f"  methods           : {', '.join(manifest['methods'])}")
    print(f"  items             : {manifest['n_items']}")
    print(f"  outputs           : {manifest['total_expected_outputs']}")
    print(f"  raters            : {manifest['n_raters']}")
    print(f"  ratings/output    : {manifest['ratings_per_output']}")
    print(f"  expected ratings  : {manifest['total_expected_ratings']}")
    print(f"  rater load        : {assignment['load']}")
    print(f"  study directory   : {result['study_dir']}")
    print("  rating files      : ratings/")
    print("  analysis files    : analysis/")


if __name__ == "__main__":
    main()
