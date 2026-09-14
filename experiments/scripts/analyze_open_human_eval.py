"""Analyze the three private openv1 Sheet tabs exported as CSV (not demo tabs)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.human.open_analysis import analyze_exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-dir", "participants", "assignments", "ratings", "out-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = analyze_exports(study_dir=args.study_dir, participant_csv=args.participants,
                             assignment_csv=args.assignments, rating_csv=args.ratings, out_dir=args.out_dir)
    print(f"Validated {result['participants']} participants, {result['submitted_pages']} rating pages")
    print(f"Coverage target met: {result['coverage_target_met']}; descriptive exports: {args.out_dir}")


if __name__ == "__main__":
    main()
