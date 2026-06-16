"""Compute automatic metrics for one experiment.

    python scripts/eval_auto.py --experiment E01_qwen0_5b_prompt

Reads cached predictions, joins them to the gold references by item id, runs the
metrics listed in the eval config plus robustness, and writes metrics_auto.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.pipeline.runner import ExperimentRunner  # noqa: E402
from src.utils.config import load_cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute automatic metrics")
    p.add_argument("--experiment", required=True)
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(experiment=args.experiment, overrides=args.override)
    runner = ExperimentRunner(cfg, _PROJECT_ROOT)
    payload = runner.run_eval()

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(json.dumps(payload["metrics"], indent=2, default=str))
    print("=" * 60)
    print(f"Saved to: {runner.exp_dir / 'metrics_auto.json'}")


if __name__ == "__main__":
    main()
