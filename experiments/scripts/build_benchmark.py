"""Build the independent evaluation benchmark `data/eval/benchmark_v1.jsonl`.

EVALUATION-ONLY (benchmark lock): the output must never be used for training,
augmentation, prompt optimization, retriever tuning, hyperparameter tuning, or model
selection. See docs/evaluation/benchmark_dataset_construction.md.

    python experiments/scripts/build_benchmark.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.benchmark_build import build_items
from src.utils.io import write_jsonl

OUT = "data/eval/benchmark_v1.jsonl"


def main() -> None:
    items = build_items(_PROJECT_ROOT)
    write_jsonl(items, _PROJECT_ROOT / OUT)

    by_source = Counter(it["source_type"] for it in items)
    by_task = Counter(it["task_type"] for it in items)
    auto = sum(1 for it in items if it["suitable_for_auto_scoring"])
    print("=" * 56)
    print("BENCHMARK v1 BUILT (evaluation-only)")
    print("=" * 56)
    print(f"  items            : {len(items)}")
    print(f"  by source_type   : {dict(by_source)}")
    print(f"  by task_type     : {dict(by_task)}")
    print(f"  auto-scorable    : {auto} / {len(items)}")
    print(f"  written          : {OUT}")
    print("  NOTE: run validate_benchmark.py for the full report.")
    print("=" * 56)


if __name__ == "__main__":
    main()
