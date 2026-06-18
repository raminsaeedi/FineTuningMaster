"""Remote GPU entry point — one command for the professor or any GPU server.

Modes
-----
  smoke   Validate pipeline end-to-end without a GPU.
          Runs data prep + Method-A inference (5 items, quick eval). No training.
          Use this first to confirm the environment works.

  train   QLoRA fine-tune the selected model on a GPU, package the adapter.

  full    train + E01-E04 inference/eval across all seeds + aggregate + ZIP.

Examples
--------
  python experiments/scripts/run_remote.py --mode smoke
  python experiments/scripts/run_remote.py --mode train
  python experiments/scripts/run_remote.py --mode full --seeds 42 43 44
  python experiments/scripts/run_remote.py --mode full \\
      --model-config qwen2_5_0_5b --cache-dir /mnt/data/hf_cache
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS = _PROJECT_ROOT / "experiments" / "scripts"
_PY = sys.executable

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Default experiment names ─────────────────────────────────────────────────
_TRAIN_EXP = "E03_qwen0_5b_ft"
_ALL_EXPS = [
    "E01_qwen0_5b_prompt",
    "E02_qwen0_5b_rag",
    "E03_qwen0_5b_ft",
    "E04_qwen0_5b_ft_rag",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remote GPU orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", required=True, choices=["smoke", "train", "full"],
                   help="smoke=validate only, train=QLoRA only, full=train+eval+ZIP")
    p.add_argument("--train-experiment", default=_TRAIN_EXP,
                   help=f"Training experiment config (default: {_TRAIN_EXP})")
    p.add_argument("--experiments", nargs="+", default=_ALL_EXPS,
                   help="Experiment configs for inference in full mode")
    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="Seeds for experiment runs (default: 42)")
    p.add_argument("--output-dir", default=None,
                   help="Override output_root (default: experiments/outputs/experiments)")
    p.add_argument("--cache-dir", default=None,
                   help="HuggingFace model cache directory (useful on HPC/shared storage)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap dataset items per split (None = full dataset)")
    p.add_argument("--smoke-items", type=int, default=5,
                   help="Number of items for smoke inference (default: 5)")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


def _run(cmd: list, label: str, timings: dict) -> None:
    """Run a subprocess, record elapsed time, abort on failure."""
    print(f"\n  $ {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    rc = subprocess.run([str(c) for c in cmd], cwd=str(_PROJECT_ROOT)).returncode
    elapsed = time.perf_counter() - t0
    timings[label] = round(elapsed, 1)
    status = "done" if rc == 0 else f"FAILED (exit {rc})"
    print(f"  --> {label}: {status} in {elapsed:.0f}s")
    if rc != 0:
        raise SystemExit(f"\n[ABORT] Step '{label}' failed. See output above.")


def _outputs_root(args: argparse.Namespace) -> Path:
    if args.output_dir:
        p = Path(args.output_dir)
        return p if p.is_absolute() else _PROJECT_ROOT / p
    return _PROJECT_ROOT / "experiments" / "outputs" / "experiments"


def _adapter_path(outputs_root: Path, train_exp: str, seed: int) -> Path:
    """Adapter is written to <outputs_root>/<train_exp>_<seed>/adapter."""
    return outputs_root / f"{train_exp}_{seed}" / "adapter"


def _build_overrides(args: argparse.Namespace, extra: list[str] | None = None) -> list[str]:
    ov: list[str] = []
    if args.output_dir:
        ov.append(f"output_root={args.output_dir}")
    if args.cache_dir:
        ov.append(f"model.cache_dir={args.cache_dir}")
    if args.max_samples:
        ov.append(f"data.max_samples={args.max_samples}")
    if extra:
        ov.extend(extra)
    return ov


# ── Stages ────────────────────────────────────────────────────────────────────

def validate_environment(mode: str) -> None:
    _sep("Environment check")

    v = sys.version_info
    print(f"  Python        : {v.major}.{v.minor}.{v.micro}")
    if v < (3, 10):
        raise SystemExit("Python >= 3.10 required.")

    checks = {
        "torch":          "torch",
        "transformers":   "transformers",
        "hydra-core":     "hydra",
        "pydantic":       "pydantic",
        "scikit-learn":   "sklearn",
        "pandas":         "pandas",
    }
    training_checks = {
        "peft":           "peft",
        "trl":            "trl",
        "bitsandbytes":   "bitsandbytes",
        "accelerate":     "accelerate",
        "datasets":       "datasets",
    }

    if mode != "smoke":
        checks.update(training_checks)

    missing = []
    for pkg, import_name in checks.items():
        try:
            __import__(import_name)
            print(f"  {pkg:20} OK")
        except ImportError:
            missing.append(pkg)
            print(f"  {pkg:20} MISSING")

    if missing:
        raise SystemExit(
            f"\nMissing packages: {missing}\n"
            "Run: pip install -r requirements-train.txt"
        )

    import torch  # noqa: PLC0415
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  CUDA          : {name}  ({vram_gb:.1f} GB VRAM)")
    else:
        msg = "  CUDA          : NOT available"
        if mode in ("train", "full"):
            raise SystemExit(
                f"{msg}\n"
                "  Training and full-mode require a CUDA GPU.\n"
                "  Use --mode smoke to validate the environment without a GPU."
            )
        print(f"{msg} — OK for smoke mode")


def prepare_data(timings: dict) -> None:
    _sep("Data preparation")

    processed = _PROJECT_ROOT / "data" / "processed"
    if (processed / "train.jsonl").exists():
        print("  data/processed/ found — skipping build_data.")
    else:
        _run([_PY, _SCRIPTS / "build_data.py"], "build_data", timings)

    chunks = _PROJECT_ROOT / "data" / "knowledge_base" / "chunks.jsonl"
    if chunks.exists():
        print("  knowledge_base/chunks.jsonl found — skipping build_kb.")
    else:
        _run([_PY, _SCRIPTS / "build_kb.py"], "build_kb", timings)


def run_smoke(args: argparse.Namespace, timings: dict) -> None:
    _sep(f"Smoke inference — E01 ({args.smoke_items} items, quick eval)")
    ov = _build_overrides(args, extra=[
        f"data.max_samples={args.smoke_items}",
        "eval=quick",
    ])
    cmd = [_PY, _SCRIPTS / "run_experiment.py",
           "--experiment", "E01_qwen0_5b_prompt",
           "--override"] + ov
    _run(cmd, "smoke_inference", timings)


def run_train(args: argparse.Namespace, timings: dict) -> None:
    _sep(f"Training — {args.train_experiment}")
    ov = _build_overrides(args)
    cmd = [_PY, _SCRIPTS / "train.py",
           "--experiment", args.train_experiment]
    if ov:
        cmd += ["--override"] + ov
    _run(cmd, "train", timings)


def run_experiments(args: argparse.Namespace, timings: dict,
                    outputs_root: Path) -> None:
    _sep(f"Inference + evaluation — {len(args.experiments)} experiments × {len(args.seeds)} seeds")
    for exp in args.experiments:
        for seed in args.seeds:
            label = f"infer+eval:{exp}:seed{seed}"
            ov = _build_overrides(args, extra=[f"seed={seed}"])

            # E04 (FT+RAG) uses the adapter trained for E03.
            # Resolve the path explicitly so Hydra interpolation finds it.
            is_ft_method = any(x in exp for x in ("E03", "E04", "_ft"))
            if is_ft_method and "E04" in exp:
                ap = _adapter_path(outputs_root, args.train_experiment, seed)
                if not ap.exists():
                    print(f"  WARNING: adapter not found at {ap} — E04 will likely fail.")
                ov.append(f"method.adapter_path={ap}")

            cmd = [_PY, _SCRIPTS / "run_experiment.py",
                   "--experiment", exp,
                   "--override"] + ov
            _run(cmd, label, timings)


def run_aggregate(outputs_root: Path, results_dir: Path, timings: dict) -> None:
    _sep("Aggregating results")
    cmd = [
        _PY, _SCRIPTS / "aggregate_results.py",
        "--outputs-root", str(outputs_root),
        "--out-dir", str(results_dir),
    ]
    _run(cmd, "aggregate", timings)


def package_zip(outputs_root: Path, results_dir: Path, timings: dict) -> Path:
    _sep("Packaging ZIP")

    total = round(sum(timings.values()), 1)
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "timings_seconds": timings,
        "total_seconds": total,
        "total_minutes": round(total / 60, 1),
    }
    summary_path = _PROJECT_ROOT / "experiments" / "outputs" / "runtime_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  runtime_summary.json written ({total/60:.1f} min total)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = _PROJECT_ROOT / f"thesis_results_{ts}.zip"

    dirs_to_pack = [
        (outputs_root,  "outputs"),
        (results_dir,   "results"),
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src_dir, arc_prefix in dirs_to_pack:
            if not src_dir.exists():
                continue
            for f in sorted(src_dir.rglob("*")):
                if f.is_file():
                    zf.write(f, Path(arc_prefix) / f.relative_to(src_dir))
        zf.write(summary_path, "runtime_summary.json")

    size_mb = zip_path.stat().st_size / 1e6
    print(f"  Created: {zip_path.name}  ({size_mb:.1f} MB)")
    return zip_path


def print_summary(timings: dict, zip_path: Path | None) -> None:
    _sep("Runtime summary")
    total = sum(timings.values())
    for label, secs in timings.items():
        print(f"  {label:45}  {secs/60:5.1f} min")
    print(f"  {'─' * 52}")
    print(f"  {'TOTAL':45}  {total/60:5.1f} min")
    if zip_path:
        print(f"\n  Output ZIP : {zip_path.name}")
        print(f"  Copy back  : scp user@server:$(pwd)/{zip_path.name} .")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    timings: dict[str, float] = {}
    outputs_root = _outputs_root(args)
    results_dir = _PROJECT_ROOT / "experiments" / "results"

    validate_environment(args.mode)
    prepare_data(timings)

    if args.mode == "smoke":
        run_smoke(args, timings)
        print_summary(timings, None)
        print("\n  Smoke passed. Environment is ready for GPU runs.")
        return

    run_train(args, timings)

    if args.mode == "full":
        run_experiments(args, timings, outputs_root)
        run_aggregate(outputs_root, results_dir, timings)

    zip_path = package_zip(outputs_root, results_dir, timings)
    print_summary(timings, zip_path)


if __name__ == "__main__":
    main()
