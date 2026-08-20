"""Run the tiny dashboard_v4 pipeline on one small Kaggle GPU.

Default run trains Qwen2.5-0.5B with QLoRA on 100 examples, evaluates 50
in-domain examples for A/B/C/D, then evaluates A/C on a separate 50-example
sports holdout.  It verifies every requested result folder is complete.

Examples:

    python experiments/scripts/run_tiny_v4_kaggle.py --dry-run
    python experiments/scripts/run_tiny_v4_kaggle.py
    python experiments/scripts/run_tiny_v4_kaggle.py --methods A C
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DATA = _PROJECT_ROOT / "experiments" / "scripts" / "build_dashboard_v4_tiny.py"
BUILD_KB = _PROJECT_ROOT / "experiments" / "scripts" / "build_kb.py"
RUN_MATRIX = _PROJECT_ROOT / "experiments" / "scripts" / "run_final_matrix.py"
RUN_EXPERIMENT = _PROJECT_ROOT / "experiments" / "scripts" / "run_experiment.py"
TINY_DIR = _PROJECT_ROOT / "data" / "frozen" / "dashboard_v4_tiny"
KB_CHUNKS = _PROJECT_ROOT / "data" / "knowledge_base" / "chunks.jsonl"

DATASET = "dashboard_v4_tiny"
SPORTS_DATASET = "sports_v4_tiny"
METHOD_EXPERIMENT = {
    "A": "E01_qwen0_5b_prompt",
    "B": "E02_qwen0_5b_rag",
    "C": "E03_qwen0_5b_ft",
    "D": "E04_qwen0_5b_ft_rag",
}
VALID_METHODS = tuple(METHOD_EXPERIMENT)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _PROJECT_ROOT / value


def _run(command: list[str], label: str, *, dry_run: bool) -> int:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print(" ".join(command))
    if dry_run:
        return 0
    started = time.monotonic()
    result = subprocess.run(command, cwd=str(_PROJECT_ROOT))
    elapsed = time.monotonic() - started
    print(f"{label}: {'ok' if result.returncode == 0 else f'FAILED({result.returncode})'} in {elapsed / 60:.1f} min")
    return result.returncode


def _shared_overrides(args: argparse.Namespace) -> list[str]:
    return [
        f"model.max_seq_length={args.max_seq_length}",
        f"method.generate.max_new_tokens={args.max_new_tokens}",
        "method.generate.do_sample=false",
        f"training.sft.num_train_epochs={args.train_epochs}",
        f"training.sft.per_device_train_batch_size={args.batch_size}",
        f"training.sft.gradient_accumulation_steps={args.gradient_accumulation}",
        "training.sft.gradient_checkpointing=true",
        "training.sft.logging_steps=1",
        f"training.sft.save_steps={args.save_steps}",
        "training.sft.save_total_limit=1",
    ]


def _run_dir(output_root: Path, dataset: str, model: str, method: str, seed: int) -> Path:
    return output_root / dataset / model / method / f"seed_{seed}"


def _verify_run(
    output_root: Path,
    *,
    dataset: str,
    model: str,
    method: str,
    seed: int,
    expected_items: int,
) -> list[str]:
    run_dir = _run_dir(output_root, dataset, model, method, seed)
    required = (
        "predictions.jsonl",
        "metrics_auto.json",
        "manifest.json",
        "config_snapshot.yaml",
        "config_hash.txt",
        "cache_identity.json",
    )
    problems = [f"{dataset}/{method}: missing {name}" for name in required if not (run_dir / name).exists()]
    if problems:
        return problems

    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics_auto.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{dataset}/{method}: invalid manifest or metrics JSON: {exc}"]

    if manifest.get("status") != "completed":
        problems.append(f"{dataset}/{method}: manifest status is {manifest.get('status')!r}")
    if manifest.get("dataset_version") != dataset:
        problems.append(
            f"{dataset}/{method}: manifest dataset is {manifest.get('dataset_version')!r}, expected {dataset!r}"
        )
    coverage = metrics.get("coverage") or {}
    if coverage.get("n_requested") != expected_items:
        problems.append(
            f"{dataset}/{method}: requested {coverage.get('n_requested')}, expected {expected_items}"
        )
    if coverage.get("n_predictions") != expected_items or coverage.get("n_missing") != 0:
        problems.append(
            f"{dataset}/{method}: incomplete coverage {coverage.get('n_predictions')}/{coverage.get('n_requested')}"
        )
    variants = metrics.get("variant_coverage") or {}
    for name, variant in variants.items():
        if variant.get("n_missing"):
            problems.append(f"{dataset}/{method}: {name} has missing predictions")
    return problems


def _require_known_methods(values: Iterable[str], option: str) -> list[str]:
    methods = [value.upper() for value in values]
    unknown = sorted(set(methods) - set(VALID_METHODS))
    if unknown:
        raise SystemExit(f"{option}: unknown method(s): {', '.join(unknown)}")
    return list(dict.fromkeys(methods))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle-sized dashboard_v4 full pipeline")
    parser.add_argument("--model", default="qwen2_5_0_5b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--sports-methods", nargs="+", default=None)
    parser.add_argument("--skip-sports", action="store_true")
    parser.add_argument("--output-root", default="experiments/outputs/kaggle_tiny_v4")
    parser.add_argument("--results-dir", default="experiments/results/kaggle_tiny_v4")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.methods = _require_known_methods(args.methods, "--methods")
    if args.sports_methods is None:
        args.sports_methods = [method for method in ("A", "C") if method in args.methods]
    else:
        args.sports_methods = _require_known_methods(args.sports_methods, "--sports-methods")
    if any(method in {"C", "D"} for method in args.sports_methods) and "C" not in args.methods:
        raise SystemExit("sports Method C/D needs dashboard Method C training; include C in --methods")
    return args


def main() -> None:
    args = parse_args()
    output_root = _resolve(args.output_root)
    results_dir = _resolve(args.results_dir)
    failures: list[str] = []

    if not TINY_DIR.exists():
        if _run([sys.executable, str(BUILD_DATA)], "BUILD TINY DATASET", dry_run=args.dry_run) != 0:
            raise SystemExit("tiny dataset build failed")
    elif _run([sys.executable, str(BUILD_DATA), "--verify"], "VERIFY TINY DATASET", dry_run=args.dry_run) != 0:
        raise SystemExit("tiny dataset verification failed")

    requested_methods = [*args.methods, *args.sports_methods]
    if any(method in {"B", "D"} for method in requested_methods) and not KB_CHUNKS.exists():
        if _run([sys.executable, str(BUILD_KB)], "BUILD RAG KNOWLEDGE BASE", dry_run=args.dry_run) != 0:
            raise SystemExit("knowledge-base build failed")

    matrix_command = [
        sys.executable,
        str(RUN_MATRIX),
        "--profile", "tiny",
        "--model", args.model,
        "--dataset", DATASET,
        "--methods", *args.methods,
        "--seed", str(args.seed),
        "--output-root", str(output_root),
        "--results-dir", str(results_dir),
        "--override", *_shared_overrides(args),
    ]
    if "D" in args.methods:
        matrix_command.append("--with-dependencies")
    if args.resume:
        matrix_command.append("--resume")
    if args.force:
        matrix_command.append("--force")
    if args.dry_run:
        matrix_command.append("--dry-run")

    if _run(matrix_command, "RUN DASHBOARD_V4_TINY", dry_run=args.dry_run) != 0:
        failures.append("dashboard_v4_tiny matrix failed")

    if not args.dry_run:
        for method in args.methods:
            failures.extend(_verify_run(
                output_root,
                dataset=DATASET,
                model=args.model,
                method=method,
                seed=args.seed,
                expected_items=50,
            ))

    if not args.skip_sports:
        for method in args.sports_methods:
            if args.resume and not args.force and not args.dry_run:
                existing_problems = _verify_run(
                    output_root,
                    dataset=SPORTS_DATASET,
                    model=args.model,
                    method=method,
                    seed=args.seed,
                    expected_items=50,
                )
                if not existing_problems:
                    print(f"[SKIP] SPORTS_V4_TINY {method} already complete")
                    continue
            command = [
                sys.executable,
                str(RUN_EXPERIMENT),
                "--experiment", METHOD_EXPERIMENT[method],
                "--override",
                f"data={SPORTS_DATASET}",
                f"output_root={output_root.as_posix()}",
                "profile=tiny",
                "run_layout=final",
                f"model={args.model}",
                f"model_key={args.model}",
                f"method_key={method}",
                f"experiment_name={args.model}_{method}_sports",
                f"experiment_id={SPORTS_DATASET}_{args.model}_{method}_seed_{args.seed}",
                f"seed={args.seed}",
                *_shared_overrides(args),
            ]
            if _run(command, f"RUN SPORTS_V4_TINY {method}", dry_run=args.dry_run) != 0:
                failures.append(f"sports_v4_tiny {method} failed")
                continue
            if not args.dry_run:
                failures.extend(_verify_run(
                    output_root,
                    dataset=SPORTS_DATASET,
                    model=args.model,
                    method=method,
                    seed=args.seed,
                    expected_items=50,
                ))

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET,
        "sports_dataset": SPORTS_DATASET,
        "model": args.model,
        "seed": args.seed,
        "methods": args.methods,
        "sports_methods": [] if args.skip_sports else args.sports_methods,
        "dry_run": args.dry_run,
        "failures": failures,
    }
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "tiny_run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    if failures:
        print("TINY_V4_PIPELINE_FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PASS_TINY_V4_KAGGLE_PIPELINE" if not args.dry_run else "PASS_TINY_V4_KAGGLE_DRY_RUN")


if __name__ == "__main__":
    main()
