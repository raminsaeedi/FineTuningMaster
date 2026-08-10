"""Run the full A/B/C/D x seeds experiment matrix with one command.

    python experiments/scripts/run_final_matrix.py

This is the entry point for the GPU machine. It reads the matrix definition from
src/config/matrix/final.yaml and, for each seed:

    1. trains the method C adapter (if that seed has none yet);
    2. runs A, B, C and D end-to-end (inference + evaluation);
    3. records which adapter D consumed, so C seed 43 provably feeds D seed 43.

Everything is resumable: a run that already produced ``metrics_auto.json`` is
skipped, and inference itself is cached per item, so re-issuing the same command
after an interruption continues rather than restarts.

Each (stage, seed) pair runs as an isolated subprocess, so one failure does not
abort the rest of the matrix; the exit code reflects whether anything failed.

Nothing here hard-codes a model: `model:` in the matrix file (or --model) selects
a config from src/config/model/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402

DEFAULT_MATRIX = _PROJECT_ROOT / "src" / "config" / "matrix" / "final.yaml"


# ----------------------------------------------------------------------------
# Matrix definition
# ----------------------------------------------------------------------------
def load_matrix(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Matrix config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f) or {}
    if not matrix.get("runs"):
        raise SystemExit(f"{path} defines no runs.")
    return matrix


def _run_by_key(matrix: dict, key: str) -> Optional[dict]:
    for run in matrix["runs"]:
        if str(run.get("key")) == str(key):
            return run
    return None


def ordered_runs(matrix: dict) -> list[dict]:
    """Return runs with adapter producers before their consumers.

    Order in the YAML is documentation; this is the guarantee. A run that
    declares ``adapter_from`` cannot start before the run it names.
    """
    runs = list(matrix["runs"])
    producers = [r for r in runs if not r.get("adapter_from")]
    consumers = [r for r in runs if r.get("adapter_from")]
    for consumer in consumers:
        if _run_by_key(matrix, consumer["adapter_from"]) is None:
            raise SystemExit(
                f"Run {consumer.get('key')} declares adapter_from="
                f"{consumer['adapter_from']}, which is not defined in the matrix."
            )
    return producers + consumers


# ----------------------------------------------------------------------------
# Paths and completion checks
# ----------------------------------------------------------------------------
def experiment_id(experiment: str, seed: int) -> str:
    return f"{experiment}_{seed}"


def run_dir(output_root: Path, experiment: str, seed: int) -> Path:
    return output_root / experiment_id(experiment, seed)


def adapter_dir(output_root: Path, experiment: str, seed: int) -> Path:
    return run_dir(output_root, experiment, seed) / "adapter"


def adapter_is_trained(path: Path) -> bool:
    """A usable adapter has a config plus weights, not just a directory."""
    if not (path / "adapter_config.json").exists():
        return False
    return any(
        (path / n).exists() for n in ("adapter_model.safetensors", "adapter_model.bin")
    )


def run_is_complete(path: Path) -> bool:
    return (path / "metrics_auto.json").exists()


# ----------------------------------------------------------------------------
# Subprocess execution
# ----------------------------------------------------------------------------
def _execute(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print("  " + " ".join(cmd[1:]))
    started = time.time()
    rc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT)).returncode
    elapsed = time.time() - started
    print(f"  -> {'ok' if rc == 0 else f'FAILED (exit {rc})'} in {elapsed / 60:.1f} min")
    return rc


def train_cmd(experiment: str, seed: int, overrides: list[str]) -> list[str]:
    script = str(_PROJECT_ROOT / "experiments" / "scripts" / "train.py")
    cmd = [sys.executable, script, "--experiment", experiment, "--override", f"seed={seed}"]
    cmd.extend(overrides)
    return cmd


def experiment_cmd(experiment: str, seed: int, overrides: list[str]) -> list[str]:
    script = str(_PROJECT_ROOT / "experiments" / "scripts" / "run_experiment.py")
    cmd = [sys.executable, script, "--experiment", experiment, "--override", f"seed={seed}"]
    cmd.extend(overrides)
    return cmd


# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the final A/B/C/D x seeds experiment matrix",
    )
    p.add_argument("--matrix", default=str(DEFAULT_MATRIX),
                   help="Matrix definition (default: src/config/matrix/final.yaml)")
    p.add_argument("--model", default=None,
                   help="Override the model config name from the matrix file")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Override the seed list from the matrix file")
    p.add_argument("--only", nargs="+", default=None, metavar="KEY",
                   help="Run only these matrix keys, e.g. --only C D")
    p.add_argument("--output-root", default=None,
                   help="Override the matrix output root")
    p.add_argument("--force", action="store_true",
                   help="Re-run stages that already completed")
    p.add_argument("--skip-training", action="store_true",
                   help="Never train; assume adapters already exist (fails clearly if not)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without running anything")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Extra Hydra overrides applied to every run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_matrix(Path(args.matrix))

    model = args.model or matrix.get("model")
    seeds = args.seeds or [int(s) for s in matrix.get("seeds", [42])]
    output_root_rel = args.output_root or matrix.get(
        "output_root", "experiments/outputs/final"
    )
    output_root = Path(output_root_rel)
    if not output_root.is_absolute():
        output_root = _PROJECT_ROOT / output_root

    runs = ordered_runs(matrix)
    if args.only:
        wanted = {str(k) for k in args.only}
        runs = [r for r in runs if str(r.get("key")) in wanted]
        if not runs:
            raise SystemExit(f"No matrix runs match --only {args.only}")

    # Overrides shared by every subprocess. output_root must be identical across
    # stages or D would look for C's adapter in the wrong tree.
    common: list[str] = [f"output_root={output_root_rel}"]
    if model:
        common.append(f"model={model}")
    common.extend(args.override)

    print("=" * 70)
    print("FINAL EXPERIMENT MATRIX")
    print("=" * 70)
    print(f"  matrix       : {args.matrix}")
    print(f"  model        : {model}")
    print(f"  seeds        : {seeds}")
    print(f"  output root  : {output_root}")
    plan = ", ".join(f"{r.get('key')}={r.get('experiment')}" for r in runs)
    print(f"  runs         : {plan}")
    print("=" * 70)

    summary: list[dict[str, Any]] = []
    failures = 0

    for seed in seeds:
        for run in runs:
            key = str(run.get("key"))
            experiment = str(run.get("experiment"))
            overrides = list(common)

            # ---- training stage (method C) ------------------------------
            if run.get("trains_adapter"):
                target = adapter_dir(output_root, experiment, seed)
                if adapter_is_trained(target) and not args.force:
                    print(f"\n[SKIP] adapter already trained: {target}")
                    summary.append({"stage": f"train:{key}", "seed": seed, "status": "skipped"})
                elif args.skip_training:
                    print(f"\n[SKIP] --skip-training set; expecting adapter at {target}")
                    summary.append({"stage": f"train:{key}", "seed": seed, "status": "skipped"})
                else:
                    cmd = train_cmd(experiment, seed, overrides)
                    if args.dry_run:
                        print(f"\n[DRY-RUN] {' '.join(cmd[1:])}")
                        summary.append({"stage": f"train:{key}", "seed": seed, "status": "dry-run"})
                    else:
                        rc = _execute(cmd, f"TRAIN {key} ({experiment}) seed={seed}")
                        failures += rc != 0
                        summary.append({
                            "stage": f"train:{key}", "seed": seed,
                            "status": "ok" if rc == 0 else f"FAILED({rc})",
                        })
                        if rc != 0:
                            # Without an adapter, C and D for this seed cannot run.
                            print(f"  training failed for seed {seed}; "
                                  f"dependent runs will be skipped.")
                            continue

            # ---- adapter wiring (method D) ------------------------------
            if run.get("adapter_from"):
                producer = _run_by_key(matrix, str(run["adapter_from"]))
                producer_experiment = str(producer.get("experiment"))
                source = adapter_dir(output_root, producer_experiment, seed)
                # In a dry run the producing stage has not executed, so an absent
                # adapter is expected rather than an error.
                if not args.dry_run and not adapter_is_trained(source):
                    msg = (f"missing adapter for seed {seed} at {source} "
                           f"(run {run['adapter_from']} must train first)")
                    print(f"\n[FAIL] {key} seed={seed}: {msg}")
                    summary.append({"stage": key, "seed": seed, "status": "FAILED(no-adapter)"})
                    failures += 1
                    continue
                # Name the producing experiment explicitly rather than relying on
                # the config default, so the link is recorded in the snapshot.
                overrides = overrides + [
                    f"method.adapter_source_experiment={producer_experiment}"
                ]

            # ---- inference + evaluation ---------------------------------
            target_dir = run_dir(output_root, experiment, seed)
            if run_is_complete(target_dir) and not args.force:
                print(f"\n[SKIP] already complete: {target_dir}")
                summary.append({"stage": key, "seed": seed, "status": "skipped"})
                continue

            cmd = experiment_cmd(experiment, seed, overrides)
            if args.dry_run:
                print(f"\n[DRY-RUN] {' '.join(cmd[1:])}")
                summary.append({"stage": key, "seed": seed, "status": "dry-run"})
                continue

            rc = _execute(cmd, f"RUN {key} ({experiment}) seed={seed}")
            failures += rc != 0
            summary.append({
                "stage": key, "seed": seed,
                "status": "ok" if rc == 0 else f"FAILED({rc})",
            })

    print("\n" + "=" * 70)
    print("MATRIX SUMMARY")
    print("=" * 70)
    for row in summary:
        print(f"  {row['stage']:<12} seed={row['seed']:<5} {row['status']}")
    print("=" * 70)

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "matrix_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"model": model, "seeds": seeds, "output_root": str(output_root),
                 "runs": summary, "failures": failures},
                f, indent=2, default=str,
            )
        print(f"Summary written to {summary_path}")

    if failures:
        print(f"\n{failures} stage(s) failed. Re-run the same command to resume.")
        raise SystemExit(1)

    print("\nAll stages completed. Next:")
    print("  python experiments/scripts/aggregate_results.py "
          f"--outputs-root {output_root_rel}")
    print("  python experiments/scripts/package_results.py")


if __name__ == "__main__":
    main()
