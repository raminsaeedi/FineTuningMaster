"""Lightweight GPU-machine preflight for the O3 benchmark inference (NO model inference).

Read-only checks that the environment is ready before the supervisor spends GPU time.
Prints PASS/WARN/FAIL per check and exits non-zero if any HARD check fails.

    python experiments/scripts/preflight_supervisor_gpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

ADAPTER = "experiments/outputs/E03_qwen0_5b_ft_42/adapter"
KB_CHUNKS = "data/knowledge_base/chunks.jsonl"
BENCHMARK = "data/eval/benchmark_v1.jsonl"
BENCHMARK_INFER = "data/eval/benchmark_v1_infer.jsonl"
OUT_ROOT = "experiments/outputs/benchmark_v1"

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


def check_imports() -> None:
    try:
        import src.pipeline.runner  # noqa: F401
        import src.evaluation.l1_independent  # noqa: F401
        from src.data_pipeline.dataset import load_gold_items  # noqa: F401
        ok("project imports (pipeline, l1_independent, dataset)")
    except Exception as exc:  # noqa: BLE001
        fail(f"project imports failed: {exc} (run `pip install -e .`)")


def check_cuda() -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        fail(f"torch not importable: {exc}")
        return
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = "unknown device"
        ok(f"CUDA available: {name}")
    else:
        warn("CUDA NOT available — this box is CPU-only; run the benchmark on a GPU machine")


def check_benchmark_files() -> None:
    for rel in (BENCHMARK, BENCHMARK_INFER):
        if (_PROJECT_ROOT / rel).exists():
            ok(f"exists: {rel}")
        elif rel == BENCHMARK_INFER:
            fail(f"missing: {rel} — build it with "
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


def check_adapter() -> None:
    d = _PROJECT_ROOT / ADAPTER
    cfg = d / "adapter_config.json"
    weights = list(d.glob("adapter_model.*")) if d.exists() else []
    if d.is_dir() and cfg.exists() and weights:
        ok(f"FT adapter present: {ADAPTER}")
    else:
        fail(f"FT adapter missing/incomplete at {ADAPTER} — required for methods C (ft) and D (ft_rag)")


def check_kb() -> None:
    if (_PROJECT_ROOT / KB_CHUNKS).exists():
        ok(f"KB chunks present: {KB_CHUNKS}")
    else:
        warn(f"KB chunks missing: {KB_CHUNKS} — build with "
             "`python experiments/scripts/build_kb.py` (needed for methods B and D)")


def check_output_writable() -> None:
    root = _PROJECT_ROOT / OUT_ROOT
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        ok(f"output folder writable: {OUT_ROOT}")
    except Exception as exc:  # noqa: BLE001
        fail(f"output folder not writable ({OUT_ROOT}): {exc}")


def main() -> None:
    print("=" * 60)
    print("SUPERVISOR GPU PREFLIGHT (no model inference)")
    print("=" * 60)
    check_imports()
    check_cuda()
    check_benchmark_files()
    check_briefs_nonempty()
    check_adapter()
    check_kb()
    check_output_writable()
    print("-" * 60)
    print(f"Result: {_hard_fail} hard failure(s), {_warn} warning(s).")
    if _hard_fail:
        print("Preflight FAILED — resolve the FAIL items before running benchmark inference.")
        sys.exit(1)
    print("Preflight OK — ready for O3 benchmark inference (see SUPERVISOR_GPU_RUNBOOK.md).")


if __name__ == "__main__":
    main()
