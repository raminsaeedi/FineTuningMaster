"""Full-pipeline GPU-machine preflight (NO training / NO inference).

Read-only readiness checks for the FULL final run
(dataset build -> train -> synthetic diagnostics -> benchmark inference -> score).
Prints PASS/WARN/FAIL and exits non-zero on any HARD failure.

CUDA policy:
  - default (local lightweight review): CUDA unavailable is a WARN.
  - `--require-cuda` (the full GPU run, passed by run_supervisor_full_gpu.ps1):
    CUDA unavailable is a HARD FAIL, so the pipeline aborts before training/inference.

    python experiments/scripts/preflight_supervisor_full_gpu.py                 # local review
    python experiments/scripts/preflight_supervisor_full_gpu.py --require-cuda   # GPU run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

ADAPTER_DEFAULT = "experiments/outputs/experiments/E03_qwen0_5b_ft_42/adapter"
KB_CHUNKS = "data/knowledge_base/chunks.jsonl"
BENCHMARK = "data/eval/benchmark_v1.jsonl"
BENCHMARK_INFER = "data/eval/benchmark_v1_infer.jsonl"
TRAIN_FILE = "data/frozen/dashboard_v2/train.jsonl"
VAL_FILE = "data/frozen/dashboard_v2/val.jsonl"
TEST_FILE = "data/frozen/dashboard_v2/internal_test.jsonl"
TRAIN_ROOT = "experiments/outputs/experiments"   # default output root (matches run_supervisor_full_gpu.ps1)
BENCH_ROOT = "experiments/outputs/benchmark_v1"
MIN_TRAIN = 1000  # below this the frozen set is still the sample; build step must run first

_hard_fail = 0
_warn = 0


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def warn(msg: str) -> None:
    global _warn
    _warn += 1
    print(f"  WARN  {msg}")


def fail(msg: str) -> None:
    global _hard_fail
    _hard_fail += 1
    print(f"  FAIL  {msg}")


def _count(rel: str) -> int:
    p = _PROJECT_ROOT / rel
    if not p.exists():
        return -1
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def check_imports() -> None:
    try:
        import src.pipeline.runner  # noqa: F401
        import src.evaluation.l1_independent  # noqa: F401
        from src.data_pipeline.dataset import load_gold_items  # noqa: F401
        ok("project imports (pipeline, l1_independent, dataset)")
    except Exception as exc:  # noqa: BLE001
        fail(f"project imports failed: {exc} (run `pip install -e .`)")


def check_cuda(require: bool) -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        (fail if require else warn)(f"torch not importable: {exc}")
        return
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = "unknown device"
        ok(f"CUDA available: {name}")
    elif require:
        fail("CUDA unavailable but --require-cuda set: the full GPU pipeline needs a GPU. "
             "Aborting before training/inference.")
    else:
        warn("CUDA unavailable — acceptable for local lightweight review only; the full run "
             "must be launched on the supervisor GPU machine.")


def check_training_data() -> None:
    n_train, n_val, n_test = _count(TRAIN_FILE), _count(VAL_FILE), _count(TEST_FILE)
    if n_train < 0:
        warn(f"training set not built yet: {TRAIN_FILE} missing — build it with "
             "`python experiments/scripts/generate_dataset_v2.py --n 2000` then "
             "`freeze_dataset_v2.py` (the .ps1 does this automatically).")
        return
    msg = f"frozen v2 counts — train={n_train}, val={n_val}, internal_test={n_test}"
    if n_train < MIN_TRAIN:
        warn(f"{msg}. Train < {MIN_TRAIN}: this is still the SAMPLE; the build step will "
             "produce the full 1500-2000 set before training.")
    else:
        ok(msg)


def check_benchmark_files() -> None:
    for rel in (BENCHMARK, BENCHMARK_INFER):
        if (_PROJECT_ROOT / rel).exists():
            ok(f"exists: {rel}")
        elif rel == BENCHMARK_INFER:
            fail(f"missing: {rel} — build with "
                 "`python experiments/scripts/prepare_benchmark_infer.py`")
        else:
            fail(f"missing: {rel}")


def check_briefs_nonempty() -> None:
    path = _PROJECT_ROOT / BENCHMARK_INFER
    if not path.exists():
        warn(f"skip brief check — {BENCHMARK_INFER} not present yet")
        return
    try:
        from src.data_pipeline.dataset import load_gold_items
        items = load_gold_items(path)
        empty = sum(1 for it in items
                    if not (it.brief.users and it.brief.kpis and it.brief.goals and it.brief.columns))
        if items and empty == 0:
            ok(f"{BENCHMARK_INFER} loads {len(items)} items, 0 empty briefs")
        else:
            fail(f"{BENCHMARK_INFER}: {empty}/{len(items)} empty briefs (wrapper build broken)")
    except Exception as exc:  # noqa: BLE001
        fail(f"could not load {BENCHMARK_INFER}: {exc}")


def check_kb() -> None:
    if (_PROJECT_ROOT / KB_CHUNKS).exists():
        ok(f"KB chunks present: {KB_CHUNKS}")
    else:
        warn(f"KB chunks missing: {KB_CHUNKS} — build with "
             "`python experiments/scripts/build_kb.py` (needed for methods B and D)")


def check_outputs_writable() -> None:
    for rel in (TRAIN_ROOT, BENCH_ROOT):
        root = _PROJECT_ROOT / rel
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".preflight_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            ok(f"output folder writable: {rel}")
        except Exception as exc:  # noqa: BLE001
            fail(f"output folder not writable ({rel}): {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Full GPU preflight (no train/infer)")
    ap.add_argument("--require-cuda", action="store_true",
                    help="treat CUDA-unavailable as a hard fail (use for the full GPU run)")
    args = ap.parse_args()

    print("=" * 60)
    print("SUPERVISOR FULL-GPU PREFLIGHT (no training, no inference)")
    print(f"mode: {'REQUIRE-CUDA (full run)' if args.require_cuda else 'lightweight review'}")
    print("=" * 60)
    check_imports()
    check_cuda(args.require_cuda)
    check_training_data()
    check_benchmark_files()
    check_briefs_nonempty()
    check_kb()
    check_outputs_writable()
    print("-" * 60)
    print(f"Result: {_hard_fail} hard failure(s), {_warn} warning(s).")
    if _hard_fail:
        print("Preflight FAILED — resolve FAIL items before the full run.")
        sys.exit(1)
    print("Preflight OK — see docs/project/SUPERVISOR_FULL_GPU_RUNBOOK.md.")


if __name__ == "__main__":
    main()
