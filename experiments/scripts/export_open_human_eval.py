"""Build the single-link, variable-rater study without publishing private data."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.human.open_study import build_open_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--spreadsheet-id", required=True)
    args = parser.parse_args()
    result = build_open_study(source_dir=args.source_dir.resolve(), out_dir=args.out_dir.resolve(),
                              project_root=ROOT, spreadsheet_id=args.spreadsheet_id)
    print(f"Built {result['study_id']}: 8 tasks/person, 32 outputs, variable participant count")
    print(f"Private bundle: {args.out_dir.resolve() / 'Code.gs'}")


if __name__ == "__main__":
    main()
