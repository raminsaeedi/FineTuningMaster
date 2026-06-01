"""
pipeline/compare.py — Stage 5: Cross-experiment comparison table.

Reads config_snapshot.yaml and metrics.json from every experiment
directory under outputs_root and renders a sorted comparison table.

Usage:
    python pipeline/compare.py \\
        [--outputs-root outputs/experiments] \\
        [--experiments exp_id_1 exp_id_2 ...]  # filter to specific IDs \\
        [--output-format table|csv|json]        # default: table \\
        [--output-file results/comparison.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.experiment import list_experiments, load_experiment_config, load_metrics
from utils.helpers import setup_logging

logger = setup_logging()

# Columns included in the comparison table, in display order
_COLUMNS = [
    ("experiment_id",         "Experiment ID"),
    ("algorithm",             "Algorithm"),
    ("model_short",           "Model"),
    ("lora_r",                "LoRA r"),
    ("learning_rate",         "LR"),
    ("num_epochs",            "Epochs"),
    # Base model metrics
    ("base_json_parse",       "Base JSON%"),
    ("base_schema_valid",     "Base Schema%"),
    ("base_completeness",     "Base Complete"),
    ("base_top1",             "Base Top-1"),
    ("base_latency_avg",      "Base Lat(s)"),
    # Fine-tuned model metrics
    ("ft_json_parse",         "FT JSON%"),
    ("ft_schema_valid",       "FT Schema%"),
    ("ft_completeness",       "FT Complete"),
    ("ft_top1",               "FT Top-1"),
    ("ft_latency_avg",        "FT Lat(s)"),
    # Robustness
    ("ft_paraphrase_cons",    "FT Para.Cons"),
    ("ft_missing_valid",      "FT Miss.Valid"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _model_short(model_name: str) -> str:
    name = model_name.lower().split("/")[-1]
    mapping = {
        "qwen2.5-0.5b-instruct": "qwen05b",
        "qwen2.5-1.5b-instruct": "qwen15b",
        "qwen2.5-3b-instruct":   "qwen3b",
    }
    return mapping.get(name, name[:12])


def _get_nested(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


def load_comparison_rows(
    outputs_root: str,
    experiment_ids: list[str] | None = None,
) -> list[dict]:
    """
    Build a flat comparison row for each experiment that has metrics.json.
    """
    root = Path(outputs_root)
    if not root.exists():
        logger.error(f"outputs_root not found: {root}")
        return []

    rows = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if experiment_ids and exp_dir.name not in experiment_ids:
            continue

        metrics = load_metrics(exp_dir)
        if metrics is None:
            logger.debug(f"Skipping {exp_dir.name} — no metrics.json")
            continue

        try:
            cfg = load_experiment_config(exp_dir)
        except FileNotFoundError:
            cfg = {}

        base = _get_nested(metrics, "eval", "base_model") or {}
        ft   = _get_nested(metrics, "eval", "finetuned_model") or {}

        row = {
            "experiment_id":      exp_dir.name,
            "algorithm":          cfg.get("algorithm", {}).get("name", "?"),
            "model_short":        _model_short(cfg.get("model", {}).get("name", "?")),
            "lora_r":             cfg.get("lora", {}).get("r"),
            "learning_rate":      cfg.get("training", {}).get("learning_rate"),
            "num_epochs":         cfg.get("training", {}).get("num_train_epochs"),

            "base_json_parse":    _get_nested(base, "schema", "json_parse_rate"),
            "base_schema_valid":  _get_nested(base, "schema", "schema_validity_rate"),
            "base_completeness":  _get_nested(base, "schema", "completeness_score"),
            "base_top1":          _get_nested(base, "chart_type", "chart_top1_accuracy"),
            "base_latency_avg":   _get_nested(base, "latency", "avg_latency_s"),

            "ft_json_parse":      _get_nested(ft, "schema", "json_parse_rate"),
            "ft_schema_valid":    _get_nested(ft, "schema", "schema_validity_rate"),
            "ft_completeness":    _get_nested(ft, "schema", "completeness_score"),
            "ft_top1":            _get_nested(ft, "chart_type", "chart_top1_accuracy"),
            "ft_latency_avg":     _get_nested(ft, "latency", "avg_latency_s"),

            "ft_paraphrase_cons": _get_nested(ft, "robustness", "paraphrase_consistency"),
            "ft_missing_valid":   _get_nested(ft, "robustness", "missing_info_validity_rate"),
        }
        rows.append(row)

    # Sort: fine-tuned schema_validity_rate descending, nulls last
    rows.sort(
        key=lambda r: (r["ft_schema_valid"] is None, -(r["ft_schema_valid"] or 0))
    )
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        # Small values (e.g. learning rates) render in scientific notation
        return f"{value:.2e}" if abs(value) < 0.01 and value != 0.0 else f"{value:.2f}"
    return str(value)


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "No experiments with metrics found."

    col_keys   = [c[0] for c in _COLUMNS]
    col_labels = [c[1] for c in _COLUMNS]

    # Compute column widths
    widths = [len(h) for h in col_labels]
    for row in rows:
        for i, key in enumerate(col_keys):
            widths[i] = max(widths[i], len(_fmt(row.get(key))))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    hdr = "|" + "|".join(f" {h:<{w}} " for h, w in zip(col_labels, widths)) + "|"

    lines = [sep, hdr, sep]
    for row in rows:
        line = "|" + "|".join(
            f" {_fmt(row.get(k)):<{w}} " for k, w in zip(col_keys, widths)
        ) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def render_csv(rows: list[dict]) -> str:
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[c[0] for c in _COLUMNS], extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def render_json(rows: list[dict]) -> str:
    return json.dumps(rows, indent=2, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 5 — cross-experiment comparison")
    p.add_argument("--outputs-root", default="outputs/experiments")
    p.add_argument("--experiments", nargs="*", default=None,
                   help="Filter to specific experiment IDs")
    p.add_argument("--output-format", choices=["table", "csv", "json"], default="table")
    p.add_argument("--output-file", default=None,
                   help="Save output to this file in addition to printing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_comparison_rows(args.outputs_root, args.experiments)

    if not rows:
        print("No experiments with metrics found in:", args.outputs_root)
        return

    if args.output_format == "table":
        output = render_table(rows)
    elif args.output_format == "csv":
        output = render_csv(rows)
    else:
        output = render_json(rows)

    print(output)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
