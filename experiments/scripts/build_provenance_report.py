"""Derive and report dataset provenance/lineage (derive-only, no record mutation).

Writes experiments/results/dataset_provenance_report.{json,md}. Makes explicit which
items are safe for independent evaluation and which are internal diagnostics only.

    python experiments/scripts/build_provenance_report.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.frozen_validation import read_jsonl_strict
from src.data_pipeline.provenance import derive_benchmark, derive_real_brief, derive_synthetic
from src.utils.io import read_json, write_json

FROZEN = ("train.jsonl", "val.jsonl", "internal_test.jsonl")
FROZEN_DIR = "data/frozen/dashboard_v2"


def _dataset_version() -> str:
    p = _PROJECT_ROOT / FROZEN_DIR / "hashes.json"
    if p.exists():
        try:
            return read_json(p).get("generator_version", "dashboard_v2")
        except Exception:
            return "dashboard_v2"
    return "dashboard_v2"


def main() -> None:
    version = _dataset_version()
    rows = []

    for fname in FROZEN:
        recs, _ = read_jsonl_strict(_PROJECT_ROOT / FROZEN_DIR / fname)
        rows += [derive_synthetic(r, fname, version) for r in recs]

    bench, _ = read_jsonl_strict(_PROJECT_ROOT / "data/eval/benchmark_v1.jsonl")
    rows += [derive_benchmark(r, "benchmark_v1") for r in bench]

    real, _ = read_jsonl_strict(_PROJECT_ROOT / "data/eval/real_briefs_v1.jsonl")
    rows += [derive_real_brief(r, "real_briefs_v1") for r in real]

    write_json({"n": len(rows), "items": rows},
               _PROJECT_ROOT / "experiments/results/dataset_provenance_report.json")

    by_source = Counter(r["source_type"] for r in rows)
    by_use = Counter(r["intended_use"] for r in rows)
    by_lineage = Counter(r["label_lineage_id"] for r in rows)
    n_synth = sum(1 for r in rows if r["is_synthetic"])
    n_safe = sum(1 for r in rows if r["independent_eval_safe"])

    def _c(counter):
        return "\n".join(f"- `{k}`: {v}" for k, v in counter.most_common())

    md = [
        "# Dataset Provenance & Lineage Report", "",
        f"Derived (no record mutation) for {len(rows)} items. Dataset version tag: `{version}`.", "",
        "## By source_type", "", _c(by_source), "",
        "## By intended_use", "", _c(by_use), "",
        "## By label lineage", "", _c(by_lineage), "",
        "## Safety summary", "",
        f"- synthetic items (internal-diagnostic only): {n_synth}",
        f"- items safe for INDEPENDENT evaluation: {n_safe} "
        f"(benchmark_v1 + real_briefs_v1)",
        f"- items NOT safe for independent claims (synthetic, generator lineage): {len(rows) - n_safe}", "",
        "> Independent-eval-safe items carry a non-generator label lineage "
        "(`literature_L1`, `manual_expert`, or `none`). Synthetic items share the "
        "`synthetic_generator:TASK_CHART` lineage and are internal diagnostics only.",
    ]
    (_PROJECT_ROOT / "experiments/results/dataset_provenance_report.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(f"Provenance report: {len(rows)} items, {n_safe} independent-eval-safe -> "
          "experiments/results/dataset_provenance_report.md")


if __name__ == "__main__":
    main()
