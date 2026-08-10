"""End-to-end pipeline smoke test on Qwen2.5-0.5B-Instruct.

    python experiments/scripts/run_smoke.py

Proves the machine can execute the whole pipeline -- A, B, C and D, inference,
parsing, schema evaluation, metrics and provenance artifacts -- before anyone
spends GPU hours on the real matrix.

The PASS criterion is functional completion, NOT model quality. A 0.5B model
trained for a handful of steps produces poor recommendations; that is expected
and irrelevant here.

Held-out data is never touched. Evaluation runs against a tiny deterministic
slice of the *validation* split, written to the smoke output directory; the
frozen test split and the 40-item human-evaluation file are not read. Training
uses a small slice of the frozen train split.

Everything lands under outputs/smoke/dashboard_v3_qwen0_5b/ so smoke artifacts
can never be mistaken for thesis results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SMOKE_ROOT_REL = "outputs/smoke/dashboard_v3_qwen0_5b"
PASS_TOKEN = "PASS_QWEN_0_5B_END_TO_END_SMOKE"

SOURCE_VAL = _PROJECT_ROOT / "data" / "frozen" / "dashboard_v3" / "val.jsonl"

STAGES = [
    ("A", "E01_qwen0_5b_prompt", "prompt-only"),
    ("B", "E02_qwen0_5b_rag", "RAG"),
    ("C", "E03_qwen0_5b_ft", "QLoRA fine-tuned"),
    ("D", "E04_qwen0_5b_ft_rag", "QLoRA + RAG"),
]

# Artifacts every completed stage must have produced.
REQUIRED_ARTIFACTS = [
    "predictions.jsonl",
    "metrics_auto.json",
    "manifest.json",
    "config_snapshot.yaml",
    "config_hash.txt",
]


def build_eval_slice(n_items: int, out_path: Path) -> int:
    """Write the first ``n_items`` validation records, ordered by item id.

    Sorting makes the slice identical on every machine and every run, so a smoke
    failure is reproducible rather than a function of file order.
    """
    if not SOURCE_VAL.exists():
        raise SystemExit(
            f"Validation split not found: {SOURCE_VAL}\n"
            f"Run: python experiments/scripts/check_experiment_release.py"
        )

    records = []
    with SOURCE_VAL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records.sort(key=lambda r: str(r.get("item_id", "")))
    selected = records[:n_items]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(selected)


def _run(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    started = time.time()
    rc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT)).returncode
    print(f"  -> {'ok' if rc == 0 else f'FAILED (exit {rc})'} "
          f"in {time.time() - started:.0f}s")
    return rc


def verify_stage(stage_dir: Path, stage: str, expected_seed: int) -> list[str]:
    """Return the problems found in one finished stage directory."""
    problems = []

    for name in REQUIRED_ARTIFACTS:
        if not (stage_dir / name).exists():
            problems.append(f"{stage}: missing {name}")

    manifest_path = stage_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("seed") != expected_seed:
            problems.append(
                f"{stage}: manifest seed {manifest.get('seed')} != {expected_seed}"
            )
        if not manifest.get("dataset_version"):
            problems.append(f"{stage}: manifest records no dataset_version")
        if manifest.get("status") != "completed":
            problems.append(f"{stage}: manifest status is {manifest.get('status')!r}")
        if stage in ("C", "D") and not (manifest.get("adapter") or {}).get("adapter_path"):
            problems.append(f"{stage}: manifest records no adapter path")
        if stage in ("B", "D") and not (manifest.get("knowledge_base") or {}).get("chunks_sha256"):
            problems.append(f"{stage}: manifest records no knowledge-base hash")

    metrics_path = stage_dir / "metrics_auto.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not payload.get("n_predictions"):
            problems.append(f"{stage}: no predictions were scored")
        metrics = payload.get("metrics") or {}
        if "schema_compliance" not in metrics:
            problems.append(f"{stage}: schema evaluation did not run")

    predictions = stage_dir / "predictions.jsonl"
    if predictions.exists():
        lines = [ln for ln in predictions.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            problems.append(f"{stage}: predictions.jsonl is empty")
        else:
            first = json.loads(lines[0])
            if "raw_text" not in first:
                problems.append(f"{stage}: predictions carry no raw model output")
            if first.get("seed") != expected_seed:
                problems.append(f"{stage}: prediction seed != {expected_seed}")

    return problems


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end pipeline smoke test (Qwen 0.5B)")
    p.add_argument("--n-eval-items", type=int, default=3,
                   help="Validation records to evaluate on (default: 3)")
    p.add_argument("--n-train-items", type=int, default=8,
                   help="Train records for the smoke fine-tune (default: 8)")
    p.add_argument("--max-steps", type=int, default=2,
                   help="Optimizer steps for the smoke fine-tune (default: 2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", nargs="+", default=None, metavar="STAGE",
                   help="Run only these stages, e.g. --only A B")
    p.add_argument("--keep", action="store_true",
                   help="Keep existing smoke outputs instead of starting clean")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    smoke_root = _PROJECT_ROOT / SMOKE_ROOT_REL

    if smoke_root.exists() and not args.keep:
        import shutil

        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)

    eval_slice = smoke_root / "smoke_eval_items.jsonl"
    n_eval = build_eval_slice(args.n_eval_items, eval_slice)
    eval_slice_rel = eval_slice.relative_to(_PROJECT_ROOT).as_posix()

    print("=" * 70)
    print("QWEN 0.5B END-TO-END SMOKE TEST")
    print("=" * 70)
    print(f"  eval slice   : {n_eval} records from val.jsonl (held-out test NOT used)")
    print(f"  train slice  : {args.n_train_items} records from train.jsonl")
    print(f"  train steps  : {args.max_steps}")
    print(f"  seed         : {args.seed}")
    print(f"  output root  : {smoke_root}")
    print("=" * 70)

    # Shared overrides. Pointing test_file at the validation slice keeps the
    # frozen test split untouched; the perturbation variants are disabled so the
    # smoke stays short (robustness is exercised by the real matrix).
    common = [
        f"output_root={SMOKE_ROOT_REL}",
        f"seed={args.seed}",
        f"data.test_file={eval_slice_rel}",
        "data.paraphrased_file=null",
        "data.missing_info_file=null",
        f"data.max_samples={n_eval}",
    ]

    stages = STAGES
    if args.only:
        wanted = {s.upper() for s in args.only}
        stages = [s for s in STAGES if s[0] in wanted]

    run_experiment = str(_PROJECT_ROOT / "experiments" / "scripts" / "run_experiment.py")
    train_script = str(_PROJECT_ROOT / "experiments" / "scripts" / "train.py")

    results: list[tuple[str, str]] = []
    problems: list[str] = []

    for stage, experiment, description in stages:
        # Method C trains the adapter that both C and D consume.
        if stage == "C":
            train_cmd = [
                sys.executable, train_script, "--experiment", experiment,
                "--override",
                f"output_root={SMOKE_ROOT_REL}",
                f"seed={args.seed}",
                f"data.max_samples={args.n_train_items}",
                "training.sft.num_train_epochs=1",
                # max_steps is not part of the training config, so it must be
                # appended rather than overridden.
                f"+training.sft.max_steps={args.max_steps}",
                "training.sft.logging_steps=1",
                "training.sft.save_steps=1000",
            ]
            rc = _run(train_cmd, f"SMOKE TRAIN {stage} ({experiment})")
            if rc != 0:
                results.append((f"train:{stage}", f"FAILED({rc})"))
                problems.append(f"{stage}: training failed")
                continue
            results.append((f"train:{stage}", "ok"))

            adapter = smoke_root / f"{experiment}_{args.seed}" / "adapter"
            if not (adapter / "adapter_config.json").exists():
                problems.append(f"{stage}: adapter was not saved to {adapter}")
                continue
            print(f"  adapter saved: {adapter}")

        overrides = list(common)
        if stage == "D":
            overrides.append("method.adapter_source_experiment=E03_qwen0_5b_ft")

        cmd = [sys.executable, run_experiment, "--experiment", experiment,
               "--override"] + overrides
        rc = _run(cmd, f"SMOKE {stage} ({experiment}) -- {description}")
        results.append((stage, "ok" if rc == 0 else f"FAILED({rc})"))
        if rc != 0:
            problems.append(f"{stage}: run_experiment exited {rc}")
            continue

        stage_dir = smoke_root / f"{experiment}_{args.seed}"
        problems.extend(verify_stage(stage_dir, stage, args.seed))

    print("\n" + "=" * 70)
    print("SMOKE SUMMARY")
    print("=" * 70)
    for name, status in results:
        print(f"  {name:<12} {status}")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        print("=" * 70)
        print("SMOKE FAILED")
        raise SystemExit(1)

    print("=" * 70)
    print(PASS_TOKEN)
    print("=" * 70)
    print(f"Smoke artifacts: {smoke_root}")
    print("These are pipeline checks only and are NOT thesis results.")


if __name__ == "__main__":
    main()
