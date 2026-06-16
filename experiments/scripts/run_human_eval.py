"""Launch the Streamlit human-rating app.

    python scripts/run_human_eval.py

Thin wrapper around `streamlit run` that points the app at the eval set and the
ratings output directory. Requires the human extra: `pip install -e ".[human]"`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP = _PROJECT_ROOT / "src" / "evaluation" / "human" / "streamlit_app.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch the human-eval Streamlit app")
    p.add_argument("--eval-dir", default="experiments/results/human_eval")
    p.add_argument("--ratings-dir", default="experiments/results/human_ratings")
    p.add_argument("--port", type=int, default=8501)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = dict(os.environ)
    env["HUMAN_EVAL_DIR"] = str((_PROJECT_ROOT / args.eval_dir).resolve())
    env["HUMAN_RATINGS_DIR"] = str((_PROJECT_ROOT / args.ratings_dir).resolve())

    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit("streamlit is not installed. Run: pip install -e \".[human]\"")

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP),
           "--server.port", str(args.port)]
    subprocess.run(cmd, env=env, cwd=str(_PROJECT_ROOT))


if __name__ == "__main__":
    main()
