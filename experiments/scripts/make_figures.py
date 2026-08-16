"""Generate the thesis figures from the aggregated result tables.

This script reads ONLY the aggregation output produced by
``aggregate_results.py`` -- it never loads a model, never runs inference and
never touches the frozen dataset. It is therefore cheap, deterministic and safe
to re-run at any time::

    python experiments/scripts/make_figures.py --dataset dashboard_v4

Input (written by the experiment runner):

    experiments/results/final/<dataset>/comparison_table.csv
        one row per run: model_key, method_key, seed + flattened metrics

Output:

    experiments/results/final/<dataset>/figures/
        F1_accuracy_by_method.(png|pdf)      top-1 accuracy, mean +- SD over seeds
        F2_schema_quality.(png|pdf)          schema validity and completeness
        F3_seed_variability.(png|pdf)        every seed drawn separately
        F4_robustness.(png|pdf)              paraphrase / missing-info behaviour
        F5_latency.(png|pdf)                 average latency per item (ms)
        figure_data.csv                      the exact numbers behind the bars
        README.md                            what each figure shows, regenerated

Every figure follows the same conventions:

* one panel per metric, models on the x-axis, methods A/B/C/D as grouped bars;
* the bar height is the mean across seeds, the error bar is the sample standard
  deviation across seeds (n is printed in the caption; with three seeds this is
  descriptive spread, not a confidence interval);
* a metric that no run recorded is skipped rather than drawn empty, and the
  skipped figure is listed in the console output;
* PNG (for slides) and PDF (vector, for the thesis) are written side by side.

The plotting style is intentionally plain matplotlib with no seaborn dependency
and no custom colour cycle, so the figures render identically on any machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on a GPU server
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

METHOD_ORDER = ["A", "B", "C", "D"]
METHOD_LABELS = {
    "A": "A: prompt-only",
    "B": "B: RAG",
    "C": "C: QLoRA",
    "D": "D: QLoRA+RAG",
}

# (figure key, title, y-axis label, [(column substring, panel title), ...])
FIGURES: list[tuple[str, str, str, list[tuple[str, str]]]] = [
    (
        "F1_accuracy_by_method",
        "Chart-selection accuracy by method",
        "accuracy",
        [("top_1_accuracy", "Top-1 accuracy"), ("macro_f1", "Macro F1")],
    ),
    (
        "F2_schema_quality",
        "Output-format quality by method",
        "rate",
        [
            ("schema_validity_rate", "Schema validity"),
            ("completeness_score", "Completeness"),
            ("json_parse", "JSON parse rate"),
        ],
    ),
    (
        "F4_robustness",
        "Robustness by method",
        "rate",
        [
            ("paraphrase_consistency", "Paraphrase consistency"),
            ("paraphrase_accuracy", "Paraphrase accuracy"),
            ("missing_info_clarification_rate", "Missing-info clarification"),
        ],
    ),
    (
        "F5_latency",
        "Average generation latency",
        "milliseconds per item",
        [("latency", "Latency")],
    ),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the thesis figures from aggregated results")
    parser.add_argument("--dataset", default="dashboard_v4",
                        help="Dataset the runs used (default: dashboard_v4)")
    parser.add_argument("--profile", default="final", choices=("final", "smoke"))
    parser.add_argument("--results-dir", default=None,
                        help="Default: experiments/results/<profile>/<dataset>")
    parser.add_argument("--out-dir", default=None,
                        help="Default: <results-dir>/figures")
    parser.add_argument("--format", nargs="+", default=["png", "pdf"],
                        help="Output formats (default: png pdf)")
    return parser.parse_args(argv)


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _column(df: pd.DataFrame, needle: str) -> str | None:
    """First flattened metric column containing ``needle`` (aggregator naming)."""
    return next((c for c in df.columns if needle in c), None)


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the aggregated table to model_key / method_key / seed columns."""
    work = df.copy()
    if "model_key" not in work.columns and "model" in work.columns:
        work["model_key"] = work["model"]
    if "method_key" not in work.columns and "method" in work.columns:
        work["method_key"] = work["method"]
    for column in ("model_key", "method_key"):
        if column not in work.columns:
            raise SystemExit(f"Aggregated table has no '{column}' column; re-run aggregate_results.py")
    work["method_key"] = work["method_key"].astype(str).str.upper().str[:1]
    return work


def _grouped(work: pd.DataFrame, column: str) -> pd.DataFrame:
    """mean / std / n per (model, method) for one metric column."""
    values = pd.to_numeric(work[column], errors="coerce")
    frame = work.assign(_value=values).dropna(subset=["_value"])
    if frame.empty:
        return frame
    grouped = frame.groupby(["model_key", "method_key"])["_value"].agg(["mean", "std", "count"])
    return grouped.reset_index()


def _bar_panel(ax, stats: pd.DataFrame, models: list[str], title: str, ylabel: str) -> None:
    width = 0.8 / max(len(METHOD_ORDER), 1)
    positions = range(len(models))
    for index, method in enumerate(METHOD_ORDER):
        subset = stats[stats["method_key"] == method].set_index("model_key")
        means = [subset["mean"].get(model, float("nan")) for model in models]
        errors = [subset["std"].get(model, 0.0) or 0.0 for model in models]
        offsets = [position + index * width - 0.4 + width / 2 for position in positions]
        ax.bar(offsets, means, width=width, yerr=errors, capsize=3,
               label=METHOD_LABELS[method])
    ax.set_xticks(list(positions))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    # Rates and accuracies are non-negative; anchoring at 0 keeps bar heights
    # comparable across figures instead of auto-zooming on tiny differences.
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)


def _save(fig, out_dir: Path, name: str, formats: list[str]) -> list[Path]:
    written = []
    for suffix in formats:
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _seed_variability(work: pd.DataFrame, column: str, out_dir: Path,
                      formats: list[str]) -> list[Path]:
    """One marker per seed, so run-to-run spread is visible, not averaged away."""
    values = pd.to_numeric(work[column], errors="coerce")
    frame = work.assign(_value=values).dropna(subset=["_value"])
    if frame.empty:
        return []
    models = sorted(frame["model_key"].unique())
    fig, ax = plt.subplots(figsize=(1.8 * max(len(models), 3) + 3, 4))
    for index, method in enumerate(METHOD_ORDER):
        subset = frame[frame["method_key"] == method]
        if subset.empty:
            continue
        x = [models.index(model) + (index - 1.5) * 0.15 for model in subset["model_key"]]
        ax.scatter(x, subset["_value"], label=METHOD_LABELS[method], s=36)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Per-seed results (each point is one seed)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "F3_seed_variability", formats)


README_TEMPLATE = """# Figures for `{dataset}`

Generated by `experiments/scripts/make_figures.py` from
`{source}`. Regenerate at any time (no GPU needed):

```bash
python experiments/scripts/make_figures.py --dataset {dataset}
```

Bars show the mean over seeds; error bars are the sample standard deviation
across seeds ({seeds} seed(s) present). Each point in F3 is one seed. Numbers
behind the bars: `figure_data.csv`.

| Figure | Shows |
|---|---|
| `F1_accuracy_by_method` | Top-1 chart-selection accuracy and macro F1, per model, methods A-D |
| `F2_schema_quality` | Schema validity, completeness and JSON parse rate |
| `F3_seed_variability` | Top-1 accuracy of every individual run (seed spread) |
| `F4_robustness` | Paraphrase consistency/accuracy and missing-info clarification |
| `F5_latency` | Average generation latency per item |

Metrics with no data in this run are skipped: {skipped}

Caveat carried over from the evaluation code: `top-1`/`macro F1` are scored
against the dataset's own labels and are internal diagnostics, not evidence of
real dashboard-design quality. See `final_report.md` next to this folder.
"""


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = _resolve(args.results_dir or f"experiments/results/{args.profile}/{args.dataset}")
    out_dir = _resolve(args.out_dir) if args.out_dir else results_dir / "figures"
    source = results_dir / "comparison_table.csv"
    if not source.exists():
        raise SystemExit(
            f"No aggregated table at {source}. Run the experiments first "
            f"(./run_professor.sh --dataset {args.dataset})."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    work = _tidy(pd.read_csv(source))
    models = sorted(work["model_key"].astype(str).unique())
    seeds = sorted(work["seed"].unique()) if "seed" in work.columns else []

    written: list[Path] = []
    skipped: list[str] = []
    tidy_rows: list[pd.DataFrame] = []

    for name, title, ylabel, panels in FIGURES:
        available = [(needle, label, _column(work, needle)) for needle, label in panels]
        available = [(needle, label, column) for needle, label, column in available if column]
        panel_stats = []
        for needle, label, column in available:
            stats = _grouped(work, column)
            if not stats.empty:
                panel_stats.append((label, stats))
                tidy_rows.append(stats.assign(figure=name, metric=label))
            else:
                skipped.append(f"{name}:{label}")
        if not panel_stats:
            skipped.append(name)
            continue
        fig, axes = plt.subplots(
            1, len(panel_stats),
            figsize=(max(4.5 * len(panel_stats), 6), 4),
            squeeze=False,
        )
        for ax, (label, stats) in zip(axes[0], panel_stats):
            _bar_panel(ax, stats, models, label, ylabel)
        axes[0][-1].legend(fontsize=8)
        fig.suptitle(f"{title} - {args.dataset}")
        written.extend(_save(fig, out_dir, name, args.format))

    accuracy_column = _column(work, "top_1_accuracy")
    if accuracy_column:
        written.extend(_seed_variability(work, accuracy_column, out_dir, args.format))
    else:
        skipped.append("F3_seed_variability")

    if tidy_rows:
        pd.concat(tidy_rows, ignore_index=True).to_csv(out_dir / "figure_data.csv", index=False)

    (out_dir / "README.md").write_text(
        README_TEMPLATE.format(
            dataset=args.dataset,
            source=source.relative_to(_PROJECT_ROOT).as_posix()
            if source.is_relative_to(_PROJECT_ROOT) else source,
            seeds=len(seeds) or "unknown",
            skipped=", ".join(sorted(set(skipped))) or "none",
        ),
        encoding="utf-8",
    )

    print("=" * 56)
    print("FIGURES COMPLETE")
    print("=" * 56)
    print(f"  runs plotted : {len(work)}")
    print(f"  models       : {', '.join(models)}")
    print(f"  seeds        : {', '.join(str(seed) for seed in seeds) or 'n/a'}")
    print(f"  files        : {len(written)} in {out_dir}")
    if skipped:
        print(f"  skipped      : {', '.join(sorted(set(skipped)))} (no data)")
    print("=" * 56)


if __name__ == "__main__":
    main()
