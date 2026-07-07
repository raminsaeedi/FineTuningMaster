"""B1 — build the inference-compatible wrapper for benchmark_v1 (data-prep only).

`data/eval/benchmark_v1.jsonl` stores FLAT `DashboardBrief` items (no `brief`/
`recommendation` wrapper), so `dataset.load_gold_items` would read empty briefs.
This one-off transform wraps each item into the standard gold-record shape so the
inference pipeline can consume it:

    {item_id = benchmark_id, brief = {users, goals, kpis, columns, constraints},
     recommendation = {}}

`recommendation` is intentionally EMPTY — the benchmark has no chart gold and must
never be used as training labels. EVALUATION-ONLY (benchmark lock). No model runs.

    python experiments/scripts/prepare_benchmark_infer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.io import read_jsonl, write_jsonl

SRC = "data/eval/benchmark_v1.jsonl"
OUT = "data/eval/benchmark_v1_infer.jsonl"
_BRIEF_FIELDS = ("users", "goals", "kpis", "columns", "constraints")


def to_infer_record(item: dict) -> dict:
    item_id = item.get("benchmark_id") or item.get("item_id", "")
    brief = {"item_id": item_id}
    brief["users"] = item.get("users", "")
    brief["goals"] = list(item.get("goals", []))
    brief["kpis"] = list(item.get("kpis", []))
    brief["columns"] = list(item.get("columns", []))
    brief["constraints"] = item.get("constraints")
    return {"item_id": item_id, "brief": brief, "recommendation": {}}


def main() -> None:
    src = _PROJECT_ROOT / SRC
    items = read_jsonl(src)
    records = [to_infer_record(it) for it in items]
    write_jsonl(records, _PROJECT_ROOT / OUT)
    print(f"Wrote {len(records)} inference records -> {OUT} (eval-only; empty recommendation)")


if __name__ == "__main__":
    main()
