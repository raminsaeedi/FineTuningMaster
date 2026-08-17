"""Validate ratings and compute thesis-ready human-evaluation statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.human.pipeline import (  # noqa: E402
    HumanEvaluationError,
    IncompleteStudyError,
    run_analysis,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ratings and compute human-evaluation statistics")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Pilot/debug analysis only")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    study_dir = Path(args.study_dir)
    if not study_dir.is_absolute():
        study_dir = _PROJECT_ROOT / study_dir
    try:
        result = run_analysis(
            study_dir=study_dir.resolve(),
            project_root=_PROJECT_ROOT,
            allow_incomplete=args.allow_incomplete,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    except (HumanEvaluationError, IncompleteStudyError) as exc:
        raise SystemExit(str(exc)) from exc

    completion = result["completion"]
    print("HUMAN-EVAL ANALYSIS COMPLETE")
    print(f"  ratings           : {completion['received_ratings']} / {completion['expected_ratings']}")
    print(f"  completion        : {completion['completion_percentage']}%")
    print(f"  analysis directory: {result['analysis_dir']}")
    print("  outputs           : rating_completion, irr_alphas, system_means, per_item_scores,")
    print("                      human_stats, human_chart_acceptability, human_eval_summary")


if __name__ == "__main__":
    main()
