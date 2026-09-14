"""Export one blind human-evaluation study as a deployable static website."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.human.pipeline import HumanEvaluationError  # noqa: E402
from src.evaluation.human.web import build_web_bundle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a blind human-evaluation study as a static website"
    )
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--contact-email", default="")
    parser.add_argument("--institution", default="")
    parser.add_argument("--retention-months", type=int, default=12)
    parser.add_argument(
        "--ethics-status",
        default="approved for use in this thesis",
        help="Short participant-facing ethics statement",
    )
    parser.add_argument("--salt", default=None)
    parser.add_argument("--presentation-seed", type=int, default=20250911)
    parser.add_argument(
        "--skip-source-verification",
        action="store_true",
        help="Debug only: do not verify that source predictions and metadata are unchanged",
    )
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
        result = build_web_bundle(
            study_dir=_resolve(args.study_dir),
            project_root=_PROJECT_ROOT,
            out_dir=_resolve(args.out_dir),
            endpoint=args.endpoint.strip(),
            contact_email=args.contact_email.strip(),
            institution=args.institution.strip(),
            retention_months=args.retention_months,
            ethics_status=args.ethics_status.strip(),
            base_url=args.base_url.strip(),
            salt=args.salt,
            presentation_seed=args.presentation_seed,
            verify_sources=not args.skip_source_verification,
        )
    except HumanEvaluationError as exc:
        raise SystemExit(str(exc)) from exc

    print("HUMAN-EVAL WEB BUNDLE EXPORTED")
    print(f"  study             : {result['study_id']}")
    print(f"  public study ID   : {result['public_study_id']}")
    print(f"  items / units     : {result['n_items']} / {result['n_units']}")
    print(f"  raters            : {result['n_raters']}")
    print(f"  endpoint          : {result['endpoint'] or 'browser-only backup mode'}")
    print(f"  public site       : {result['site_dir']}")
    print(f"  PRIVATE key       : {result['key_path']}")
    print(f"  PRIVATE links     : {result['links_path']}")
    print("  rater links:")
    for rater, link in result["links"].items():
        print(f"    {rater}: {link}")
    print("  machine result:")
    print(json.dumps({key: str(value) for key, value in result.items() if key != "links"}))


if __name__ == "__main__":
    main()
