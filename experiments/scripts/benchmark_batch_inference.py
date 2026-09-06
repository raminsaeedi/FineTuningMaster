"""Smoke benchmark: sequential vs. batched inference, on the same loaded model.

Answers the only two questions that decide whether batching may be used for a
thesis run:

1. **How much faster is it?** Wall time and peak GPU memory for the same items,
   measured in one process with one model load.
2. **Are the outputs identical?** Every item's raw text is compared byte-for-byte
   against the sequential run. Batching is safe for a scientific run *only* if
   this reports EXACT MATCH — and with ``do_sample=true`` it will not, because
   the rows of a batch share one RNG stream.

The script never touches a run directory: it drives the method object directly,
writes no ``predictions*.jsonl``, and puts its report under
``experiments/outputs/benchmarks/``.

Smoke benchmark (small, local, ~minutes on CPU)::

    python experiments/scripts/benchmark_batch_inference.py \
        --experiment E01_qwen0_5b_prompt --model qwen2_5_0_5b \
        --n-items 8 --batch-size 4 --max-new-tokens 32 --greedy

Full validation before enabling the feature (20-item fixture)::

    python experiments/scripts/benchmark_batch_inference.py \
        --experiment E01_qwen0_5b_prompt --model qwen2_5_0_5b \
        --n-items 20 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import src.methods  # noqa: E402,F401  (registers methods under METHODS)
from src.core.registry import METHODS  # noqa: E402
from src.data_pipeline.dataset import load_gold_items  # noqa: E402
from src.utils.config import load_cfg  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", default="E01_qwen0_5b_prompt",
                   help="Experiment config to benchmark (A/B/C/D all work)")
    p.add_argument("--model", default=None, help="Model profile override, e.g. qwen2_5_0_5b")
    p.add_argument("--batch-size", type=int, default=4, help="Batched-run batch size")
    p.add_argument("--n-items", type=int, default=20,
                   help="Items taken from the head of the test split (fixed slice)")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Shorten generation for a quick smoke benchmark")
    p.add_argument("--greedy", action="store_true",
                   help="Force do_sample=false, isolating numerical differences "
                        "from sampling-stream differences")
    p.add_argument("--data-file", default=None, help="Override the item source JSONL")
    p.add_argument("--out", default=None, help="Report path (JSON)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args(argv)


def _cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _reset_peak() -> None:
    torch_cuda = _cuda()
    if torch_cuda is not None:
        torch_cuda.cuda.reset_peak_memory_stats()


def _memory() -> dict:
    torch_cuda = _cuda()
    if torch_cuda is None:
        return {"cuda": False, "peak_allocated_mb": None, "peak_reserved_mb": None}
    return {
        "cuda": True,
        "peak_allocated_mb": round(torch_cuda.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mb": round(torch_cuda.cuda.max_memory_reserved() / 2**20, 1),
    }


def _run_sequential(method: Any, briefs: list) -> tuple[list[str], float]:
    method.inference_batch_size = 1
    _reset_peak()
    start = time.perf_counter()
    texts = [method.generate(brief).raw_text for brief in briefs]
    return texts, (time.perf_counter() - start)


def _run_batched(method: Any, briefs: list, batch_size: int) -> tuple[list[str], float]:
    method.inference_batch_size = batch_size
    _reset_peak()
    start = time.perf_counter()
    texts: list[str] = []
    for offset in range(0, len(briefs), batch_size):
        outcomes = method.generate_batch(briefs[offset : offset + batch_size])
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            texts.append(outcome.raw_text)
    return texts, (time.perf_counter() - start)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be >= 2; there is nothing to compare at 1.")

    overrides = list(args.override)
    if args.model:
        overrides.append(f"model={args.model}")
    if args.max_new_tokens:
        overrides.append(f"method.generate.max_new_tokens={args.max_new_tokens}")
    if args.greedy:
        overrides.append("method.generate.do_sample=false")
    # The benchmark itself is the validation step, so it acknowledges the mode
    # on the operator's behalf — inside this process only, and without writing
    # any prediction file.
    overrides += [
        f"+method.inference.batch_size={args.batch_size}",
        "+method.inference.allow_nonequivalent_batching=true",
    ]
    cfg = load_cfg(experiment=args.experiment, overrides=overrides)

    data_file = Path(args.data_file) if args.data_file else Path(str(cfg.data.test_file))
    if not data_file.is_absolute():
        data_file = _PROJECT_ROOT / data_file
    items = load_gold_items(data_file)[: args.n_items]
    if not items:
        raise SystemExit(f"No items found in {data_file}")
    briefs = [item.brief for item in items]

    method = METHODS.get(str(cfg.method.name))(cfg)
    print(f"Loading method '{cfg.method.name}' ({cfg.model.name}) …")
    method.setup()
    try:
        print(f"Sequential pass: {len(briefs)} items, batch_size=1 …")
        sequential_texts, sequential_s = _run_sequential(method, briefs)
        sequential_memory = _memory()

        print(f"Batched pass:    {len(briefs)} items, batch_size={args.batch_size} …")
        batched_texts, batched_s = _run_batched(method, briefs, args.batch_size)
        batched_memory = _memory()
    finally:
        method.teardown()

    mismatches = [
        {
            "index": index,
            "item_id": items[index].item_id,
            "sequential_chars": len(sequential_texts[index]),
            "batched_chars": len(batched_texts[index]),
        }
        for index in range(len(briefs))
        if sequential_texts[index] != batched_texts[index]
    ]
    identical = not mismatches

    report = {
        "experiment": args.experiment,
        "model": str(cfg.model.get("name", "")),
        "method": str(cfg.method.name),
        "data_file": str(data_file),
        "n_items": len(briefs),
        "item_ids": [item.item_id for item in items],
        "batch_size": args.batch_size,
        "do_sample": bool(cfg.method.generate.get("do_sample", True)),
        "max_new_tokens": int(cfg.method.generate.get("max_new_tokens", 0)),
        "sequential": {
            "wall_time_s": round(sequential_s, 3),
            "items_per_s": round(len(briefs) / sequential_s, 4) if sequential_s else None,
            "memory": sequential_memory,
        },
        "batched": {
            "wall_time_s": round(batched_s, 3),
            "items_per_s": round(len(briefs) / batched_s, 4) if batched_s else None,
            "memory": batched_memory,
        },
        "speedup_x": round(sequential_s / batched_s, 3) if batched_s else None,
        "outputs_identical": identical,
        "n_mismatched_items": len(mismatches),
        "mismatches": mismatches[:20],
        "verdict": (
            "EXACT MATCH — batched outputs are byte-identical on this fixture"
            if identical
            else "OUTPUTS DIFFER — batching changes results; not safe for thesis runs"
        ),
    }

    out_path = Path(args.out) if args.out else (
        _PROJECT_ROOT / "experiments" / "outputs" / "benchmarks"
        / f"batch_inference_{args.experiment}_bs{args.batch_size}_n{len(briefs)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("BATCH INFERENCE SMOKE BENCHMARK")
    print("=" * 70)
    print(f"  items                : {len(briefs)} from {data_file.name}")
    print(f"  do_sample            : {report['do_sample']}")
    print(f"  sequential           : {report['sequential']['wall_time_s']} s")
    print(f"  batched (bs={args.batch_size})       : {report['batched']['wall_time_s']} s")
    print(f"  speedup              : {report['speedup_x']}x")
    print(f"  peak GPU MB seq/batch: {sequential_memory['peak_allocated_mb']} / "
          f"{batched_memory['peak_allocated_mb']}")
    print(f"  identical outputs    : {identical} ({len(mismatches)} mismatched)")
    print(f"  {report['verdict']}")
    print(f"  report               : {out_path}")
    print("=" * 70)
    return 0 if identical else 2


if __name__ == "__main__":
    raise SystemExit(main())
