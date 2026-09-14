"""Import ratings from the online human-evaluation collector or backups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.human.pipeline import HumanEvaluationError  # noqa: E402
from src.evaluation.human.web import import_web_ratings  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import and validate online human-evaluation ratings"
    )
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--input", nargs="+", required=True, dest="inputs")
    parser.add_argument("--key", default=None, help="Optional private unblinding-key path")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved
    return resolved.resolve()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = import_web_ratings(
            study_dir=_resolve(args.study_dir),
            inputs=[_resolve(path) for path in args.inputs],
            key_path=_resolve(args.key),
            dry_run=args.dry_run,
        )
    except (HumanEvaluationError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    print("HUMAN-EVAL WEB RATINGS VALIDATED" if args.dry_run else "HUMAN-EVAL WEB RATINGS IMPORTED")
    print(f"  records read      : {report['records_read']}")
    print(f"  accepted          : {report['accepted']}")
    print(f"  rejected          : {report['rejected']}")
    print(f"  duplicate submits : {report['duplicate_submissions']}")
    print(f"  stored ratings    : {report['total_rows']} / {report['expected_total']}")
    print(f"  completion        : {report['completion_percentage']}%")
    if report["rejected"]:
        print("  rejected details  : see web/import_report.json after a non-dry-run import")


if __name__ == "__main__":
    main()
