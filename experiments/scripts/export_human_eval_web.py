"""Export one human-evaluation study as a deployable static rating website.

Typical use::

    python experiments/scripts/export_human_eval_web.py \
        --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42 \
        --base-url https://<user>.github.io/<repo> \
        --endpoint https://script.google.com/macros/s/<id>/exec \
        --contact-email you@example.com

The deployable files are written to ``<study-dir>/web/site`` and the local
unblinding key to ``<study-dir>/web/unblinding_key.json``. Publish the site
directory only; the key must never be uploaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.human.pipeline import HumanEvaluationError  # noqa: E402
from src.evaluation.human.web import build_web_bundle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a blind rating study as a static website")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--out-dir", default=None, help="Defaults to <study-dir>/web")
    parser.add_argument("--base-url", default="", help="Public URL of the deployed site, used for rater links")
    parser.add_argument("--endpoint", default="", help="Collector URL; empty means download-only collection")
    parser.add_argument("--contact-email", default="", help="Shown to raters if something goes wrong")
    parser.add_argument("--salt", default=None, help="Reuse a previous salt to keep tokens and links stable")
    parser.add_argument("--presentation-seed", type=int, default=20250911)
    parser.add_argument(
        "--skip-source-verification",
        action="store_true",
        help="Skip re-hashing the source prediction files (use only when the runs are not on this machine)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    study_dir = Path(args.study_dir)
    if not study_dir.is_absolute():
        study_dir = _PROJECT_ROOT / study_dir
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = _PROJECT_ROOT / out_dir
    try:
        result = build_web_bundle(
            study_dir=study_dir.resolve(),
            project_root=_PROJECT_ROOT,
            out_dir=out_dir,
            endpoint=args.endpoint,
            contact_email=args.contact_email,
            base_url=args.base_url,
            salt=args.salt,
            presentation_seed=args.presentation_seed,
            verify_sources=not args.skip_source_verification,
        )
    except HumanEvaluationError as exc:
        raise SystemExit(str(exc)) from exc

    print("WEB BUNDLE EXPORTED")
    print(f"  study             : {result['study_id']}")
    print(f"  export id         : {result['export_id']}")
    print(f"  items / units     : {result['n_items']} / {result['n_units']}")
    print(f"  raters            : {result['n_raters']}")
    print(f"  bundle size       : {result['bundle_bytes'] / 1024:.0f} KiB")
    print(f"  collector endpoint: {result['endpoint'] or '(none - raters download their file)'}")
    print(f"  publish this      : {result['site_dir']}")
    print(f"  keep this private : {result['key_path']}")
    print(f"  rater links       : {result['links_path']}")
    for rater, url in result["links"].items():
        print(f"    {rater}: {url}")


if __name__ == "__main__":
    main()
