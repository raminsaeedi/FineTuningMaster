"""Run one experiment end-to-end locally: infer -> eval.

    python scripts/run_experiment.py --experiment E01_qwen0_5b_prompt

For method C (ft), the adapter folder is expected at the experiment's
adapter_path (produced by scripts/train.py on the GPU machine and copied into
outputs/experiments/<experiment_id>/adapter). This script never imports the
training stack.
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
from src.utils.artifacts import setup_run_dir, write_run_metadata  # noqa: E402
from src.utils.config import load_cfg  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an experiment end-to-end (infer + eval)")
    p.add_argument("--experiment", required=True)
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(experiment=args.experiment, overrides=args.override)
    exp_dir = setup_run_dir(cfg, _PROJECT_ROOT)
    setup_logging(level=str(cfg.get("log_level", "INFO")),
                  log_file=str(exp_dir / "logs" / "run.log"))
    write_run_metadata(exp_dir, cfg)

    runner = ExperimentRunner(cfg, _PROJECT_ROOT)
    payload = runner.run()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(json.dumps(payload["metrics"], indent=2, default=str))
    print("=" * 60)
    print(f"Artifacts in: {exp_dir}")


if __name__ == "__main__":
    main()
