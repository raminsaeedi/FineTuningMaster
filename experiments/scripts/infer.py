"""Run inference for one experiment (local, CPU-friendly).

Loads the composed config, resolves the method from the registry, and runs
cached inference over the test split into the experiment folder. Does NOT import
the training stack.

    python scripts/infer.py --experiment E01_qwen0_5b_prompt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.pipeline.runner import ExperimentRunner  # noqa: E402
from src.utils.artifacts import setup_run_dir, write_run_metadata  # noqa: E402
from src.utils.config import load_cfg  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference for an experiment")
    p.add_argument("--experiment", required=True)
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(experiment=args.experiment, overrides=args.override)
    exp_dir = setup_run_dir(cfg, _PROJECT_ROOT)
    setup_logging(level=str(cfg.get("log_level", "INFO")),
                  log_file=str(exp_dir / "logs" / "infer.log"))
    write_run_metadata(exp_dir, cfg)

    runner = ExperimentRunner(cfg, _PROJECT_ROOT)
    runner.run_inference()
    print(f"\nPredictions written under: {exp_dir}")
    print(f"Next: python scripts/eval_auto.py --experiment {args.experiment}")


if __name__ == "__main__":
    main()
