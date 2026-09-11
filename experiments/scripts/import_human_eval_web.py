"""Import web-collected ratings back into a human-evaluation study.

Accepts either the CSV exported from the collector spreadsheet or the JSON
backup files that raters can download from the app (or a directory of both)::

    python experiments/scripts/import_human_eval_web.py \
        --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42 \
        --input ~/Downloads/ratings.csv

Ratings are unblinded with the key written by the exporter and stored as
``ratings/<rater>.jsonl``, the same format the local Streamlit app writes, so
``compute_irr.py`` can run immediately afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.human.pipeline import HumanEvaluationError  # noqa: E402
from src.evaluation.human.web import import_web_ratings  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import web ratings into a study")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="CSV/JSON files or directories containing them",
    )
    parser.add_argument("--key", default=None, help="Defaults to <study-dir>/web/unblinding_key.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing rating files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    study_dir = Path(args.study_dir)
    if not study_dir.is_absolute():
        study_dir = _PROJECT_ROOT / study_dir
    inputs = []
    for raw in args.input:
        path = Path(raw)
        inputs.append(path if path.is_absolute() else _PROJECT_ROOT / path)
    key_path = Path(args.key) if args.key else None
    if key_path is not None and not key_path.is_absolute():
        key_path = _PROJECT_ROOT / key_path

    try:
        report = import_web_ratings(
            study_dir=study_dir.resolve(),
            inputs=inputs,
            key_path=key_path,
            dry_run=args.dry_run,
        )
    except HumanEvaluationError as exc:
        raise SystemExit(str(exc)) from exc

    print("WEB RATINGS IMPORTED" + (" (dry run)" if args.dry_run else ""))
    print(f"  records read      : {report['records_read']}")
    print(f"  accepted          : {report['accepted']}")
    print(f"  rejected          : {report['rejected']}")
    print(f"  duplicate sends   : {report['duplicate_submissions']}")
    print(f"  rows after merge  : {report['total_rows']} / {report['expected_total']}"
          f" ({report['completion_percentage']}%)")
    for rater, count in report["rows_per_rater"].items():
        expected = report["expected_per_rater"].get(rater, 0)
        marker = "ok" if count == expected else "incomplete"
        print(f"    {rater}: {count}/{expected} {marker}")
    missing = [
        rater for rater in report["expected_per_rater"] if rater not in report["rows_per_rater"]
    ]
    for rater in missing:
        print(f"    {rater}: 0/{report['expected_per_rater'][rater]} nothing received")
    if report["rejected_rows"]:
        print("  first rejections  :")
        for entry in report["rejected_rows"][:10]:
            print(f"    {entry['rater_id'] or '?'} {entry['unit_token'] or '?'}: {entry['problem']}")
    if not args.dry_run:
        print("  next step         : python experiments/scripts/compute_irr.py --study-dir <study-dir>")


if __name__ == "__main__":
    main()
