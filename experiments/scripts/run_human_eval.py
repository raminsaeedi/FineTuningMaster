"""Launch Streamlit rating app for one human-evaluation study."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP = _PROJECT_ROOT / "src" / "evaluation" / "human" / "streamlit_app.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the blind human-evaluation app")
    parser.add_argument("--study-dir", default=None)
    # Compatibility alias for old launch commands. It now means study dir and
    # no longer permits a separate generic ratings directory.
    parser.add_argument("--eval-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8501)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_dir = args.study_dir or args.eval_dir
    if not raw_dir:
        raise SystemExit("--study-dir is required.")
    study_dir = Path(raw_dir)
    if not study_dir.is_absolute():
        study_dir = _PROJECT_ROOT / study_dir
    study_dir = study_dir.resolve()
    required = [study_dir / "study_manifest.json", study_dir / "items.jsonl", study_dir / "assignment.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Study directory is incomplete: missing {missing}")
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit('streamlit is not installed. Run: pip install -e ".[human]"')

    env = dict(os.environ)
    env["HUMAN_EVAL_STUDY_DIR"] = str(study_dir)
    env["HUMAN_EVAL_DIR"] = str(study_dir)
    env["HUMAN_RATINGS_DIR"] = str(study_dir / "ratings")
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", str(args.port)]
    raise SystemExit(subprocess.run(cmd, env=env, cwd=str(_PROJECT_ROOT)).returncode)


if __name__ == "__main__":
    main()
