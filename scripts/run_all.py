"""Run several experiments across several seeds (local infer + eval).

    python scripts/run_all.py --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag \
        E03_qwen0_5b_ft E04_qwen0_5b_ft_rag --seeds 42 43 44

Each (experiment, seed) pair is run as an isolated subprocess via
run_experiment.py, so a failure in one cell does not abort the rest. Inference is
cached per item, so re-running only fills what is missing. For method C/D, the
trained adapter must already be present at each run's adapter_path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run experiments across seeds")
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment = str(_PROJECT_ROOT / "scripts" / "run_experiment.py")

    summary = []
    for exp in args.experiments:
        for seed in args.seeds:
            print(f"\n{'=' * 60}\nRUN: {exp} (seed={seed})\n{'=' * 60}")
            cmd = [sys.executable, run_experiment, "--experiment", exp,
                   "--override", f"seed={seed}"]
            rc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT)).returncode
            summary.append((exp, seed, "ok" if rc == 0 else f"FAILED({rc})"))

    print("\n" + "=" * 60)
    print("RUN-ALL SUMMARY")
    print("=" * 60)
    for exp, seed, status in summary:
        print(f"  {exp:28} seed={seed}  {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
