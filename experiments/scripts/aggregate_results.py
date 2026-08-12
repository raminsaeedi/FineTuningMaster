"""Aggregate per-run metrics into the final comparison artifacts.

Walks ``outputs/experiments/`` for every run's ``metrics_auto.json`` and writes,
under ``results/``:

    comparison_table.csv   one row per run (flattened metrics + provenance)
    comparison_seeds.csv   per (model, method): mean / std across seeds
    final_report.md        human-readable summary tables for the thesis

This is the missing aggregation step: the per-run metrics already exist, but
nothing previously collected them into a table or pooled the seeds.

    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --outputs-root outputs/experiments --out-dir results
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.aggregator import (  # noqa: E402
    aggregate,
    build_multi_seed_summary,
)

# (substring in flattened column name, short label) for the report table.
_REPORT_COLS = [
    ("n_predictions", "n"),
    ("json_parse", "json_parse%"),
    ("schema_validity_rate", "schema_valid%"),   # corrected: full Pydantic validity
    ("completeness_score", "complete"),           # corrected: non-empty keys
    ("top_1_accuracy", "top1%"),                  # over all items; parse-fail = wrong
    ("n_parse_failures", "n_fail"),
    ("top_3_valid", "top3_ok"),
    ("top_3_accuracy", "top3%"),                  # None unless top3_support_rate >= 0.8
    ("top_3_support_rate", "top3_support"),       # frac of items with 3 distinct recs
    ("n_with_3_recs", "n_3rec"),
    ("n_with_alternatives", "n_alt"),
    ("macro_f1", "macro_f1"),
    ("latency", "latency_s"),
    ("paraphrase_consistency", "para_stab%"),
    ("paraphrase_accuracy", "para_acc%"),
    ("missing_info_clarification_rate", "clarify%"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate experiment metrics into a comparison table + report")
    p.add_argument("--outputs-root", default="experiments/outputs/experiments")
    p.add_argument("--out-dir", default="experiments/results")
    return p.parse_args()


def _fmt(v) -> str:
    if v is None or v is pd.NA:
        return "NA"
    if isinstance(v, float) and math.isnan(v):
        return "NA"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def _md_table(rows: list[dict], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _select(df):
    """Map available flattened columns to short report labels (order preserved)."""
    chosen = []  # (actual_col, label)
    for needle, label in _REPORT_COLS:
        match = next((c for c in df.columns if needle in c), None)
        if match is not None:
            chosen.append((match, label))
    return chosen


def main() -> None:
    args = parse_args()
    outputs_root = (_PROJECT_ROOT / args.outputs_root) if not Path(args.outputs_root).is_absolute() else Path(args.outputs_root)
    out_dir = (_PROJECT_ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate(outputs_root, out_dir / "comparison_table.csv")
    if df.empty:
        raise SystemExit(
            f"No metrics_auto.json found under {outputs_root}. "
            f"Run experiments first (scripts/run_experiment.py / run_all.py)."
        )

    id_cols = [c for c in ("experiment_id", "method", "model", "seed") if c in df.columns]
    selected = _select(df)

    # ── Per-run report table ────────────────────────────────────────────────
    run_cols = [c for c in ("experiment_id", "method", "model", "seed") if c in df.columns]
    run_cols += [actual for actual, _ in selected]
    run_label = {actual: label for actual, label in selected}
    run_rows = [
        {run_label.get(c, c): row.get(c) for c in run_cols}
        for row in df.to_dict(orient="records")
    ]
    run_header = [run_label.get(c, c) for c in run_cols]

    comparison_table = "# Per-run comparison table\n\n" + _md_table(run_rows, run_header) + "\n"
    (out_dir / "comparison_table.md").write_text(comparison_table, encoding="utf-8")

    # ── Per-(model, method) mean/std across seeds ───────────────────────────
    group_keys = [
        c for c in ("model", "method", "dataset_version", "test_file_sha256")
        if c in df.columns
    ]
    seeds_md = "_No (model, method) grouping available._"
    if group_keys and selected:
        num_cols = [actual for actual, _ in selected]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        agg = df.groupby(group_keys, dropna=False)[num_cols].agg(["mean", "std", "count"])
        agg.columns = [f"{run_label[c]}_{stat}" for c, stat in agg.columns]
        agg = agg.reset_index()
        agg.to_csv(out_dir / "comparison_seeds.csv", index=False)
        seeds_cols = group_keys + [c for c in agg.columns if c not in group_keys]
        seeds_rows = agg.to_dict(orient="records")
        seeds_md = _md_table(seeds_rows, seeds_cols)

    multi_seed = build_multi_seed_summary(df)
    multi_seed.to_csv(out_dir / "multi_seed_summary.csv", index=False)
    multi_seed_columns = list(multi_seed.columns)
    multi_seed_rows = multi_seed.to_dict(orient="records")
    multi_seed_md = (
        "# Multi-seed descriptive summary\n\n"
        "Raw seed values remain visible. Seeds are repeated runs, not pooled items.\n\n"
        + (_md_table(multi_seed_rows, multi_seed_columns) if multi_seed_rows else "_No complete runs found._")
        + "\n"
    )
    (out_dir / "multi_seed_summary.md").write_text(multi_seed_md, encoding="utf-8")

    report = (
        "# Experiment comparison report\n\n"
        f"Generated from `{args.outputs_root}` — {len(df)} run(s).\n\n"
        "## Per-run metrics\n\n"
        f"{_md_table(run_rows, run_header)}\n\n"
        "## Across seeds (mean / std per model+method)\n\n"
        f"{seeds_md}\n\n"
        "## Multi-seed metric summary\n\n"
        f"See `multi_seed_summary.csv` and `multi_seed_summary.md`; raw seed values are retained and "
        "prediction rows are never pooled across seeds.\n\n"
        "> `top1%` is over ALL items with a reference (parse failures count as "
        "wrong; see `n_fail`). `top3_ok=False` means fewer than 80% of items "
        "carried 3 distinct ordered recommendations (`top3_support`), so `top3%` "
        "is reported as empty - see `src/evaluation/metrics/topk_accuracy.py`.\n\n"
        "## Evidence tier & limitations (read before citing)\n\n"
        "- **INTERNAL SYNTHETIC DIAGNOSTICS ONLY.** `top1%`, `top3%` and `macro_f1` here "
        "are scored against the synthetic generator's own labels (shared "
        "`KEYWORD_TASK -> task_type -> TASK_CHART` lineage). High values reflect "
        "reproduction of the generator rule, **not** real dashboard-design quality. See "
        "`experiments/results/rule_leakage_report.md`.\n"
        "- **Independent chart-selection** is in `experiments/results/l1_independent_results.md` "
        "(covered items only, coverage reported) — do not substitute the circular `top1%` for it.\n"
        "- **Usefulness / actionability / real-dashboard quality:** NOT supported here (require "
        "human evaluation; no ratings collected yet).\n"
        "- Per-run tables preserve each seed independently. The multi-seed summary "
        "reports descriptive seed variability only; with three seeds, confidence "
        "intervals are intentionally omitted and significance claims are not supported "
        "from across-seed summaries alone.\n"
    )
    (out_dir / "final_report.md").write_text(report, encoding="utf-8")

    print("=" * 56)
    print("AGGREGATION COMPLETE")
    print("=" * 56)
    print(f"  Runs aggregated : {len(df)}")
    print(f"  comparison_table.csv : {out_dir / 'comparison_table.csv'}")
    print(f"  comparison_table.md  : {out_dir / 'comparison_table.md'}")
    if group_keys and selected:
        print(f"  comparison_seeds.csv : {out_dir / 'comparison_seeds.csv'}")
    print(f"  multi_seed_summary.csv: {out_dir / 'multi_seed_summary.csv'}")
    print(f"  multi_seed_summary.md : {out_dir / 'multi_seed_summary.md'}")
    print(f"  final_report.md      : {out_dir / 'final_report.md'}")
    print("=" * 56)


if __name__ == "__main__":
    main()
