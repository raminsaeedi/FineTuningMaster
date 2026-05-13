"""
pipeline/evaluate.py — Stage 4: Compute and save evaluation metrics.

Reads prediction files from {experiment_dir}/predictions/ and computes
the full metric set aligned to the thesis evaluation protocol:

  A. Schema compliance  — json_parse_rate, schema_validity_rate,
                          completeness_score, field_coverage
  B. Chart-type metrics — top-1/top-3 accuracy, macro-F1
                          (only when reference labels exist)
  C. Robustness         — paraphrase_consistency, missing_info_validity_rate
                          (only when perturbation prediction files exist)
  D. Latency            — avg, p50, p95

Results are merged into {experiment_dir}/metrics.json.

Usage:
    python pipeline/evaluate.py \\
        --experiment-dir outputs/experiments/qlora_qwen05b_20250513_143022_a3f1 \\
        [--mode standard|robustness|all]   (default: all)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.experiment import load_experiment_config, load_metrics, save_metrics
from utils.helpers import load_jsonl, setup_logging

logger = setup_logging()

REQUIRED_FIELDS = [
    "context_summary",
    "kpi_task_chart_mapping",
    "layout_hierarchy",
    "labels_scales_colors",
    "interactions",
    "design_rationales",
]


# ---------------------------------------------------------------------------
# Schema compliance metrics
# ---------------------------------------------------------------------------

def compute_schema_metrics(results: list[dict]) -> dict:
    """
    Compute schema-compliance metrics over a list of prediction rows.

    Returns: json_parse_rate, schema_validity_rate, completeness_score,
             field_coverage (per-field presence rate).
    """
    n = len(results)
    if n == 0:
        return {}

    json_ok = sum(1 for r in results if r.get("valid_json"))
    schema_ok = sum(1 for r in results if r.get("valid_schema"))
    completeness_sum = sum(r.get("completeness", 0.0) for r in results)

    field_counts: dict[str, int] = {f: 0 for f in REQUIRED_FIELDS}
    for r in results:
        parsed = r.get("parsed") or {}
        for field in REQUIRED_FIELDS:
            if field in parsed:
                field_counts[field] += 1

    return {
        "json_parse_rate":     round(100.0 * json_ok / n, 2),
        "schema_validity_rate": round(100.0 * schema_ok / n, 2),
        "completeness_score":  round(completeness_sum / n, 4),
        "field_coverage":      {f: round(c / n, 4) for f, c in field_counts.items()},
    }


# ---------------------------------------------------------------------------
# Chart-type metrics (classification)
# ---------------------------------------------------------------------------

def _normalise_chart(s: str) -> str:
    return s.lower().strip()


def compute_chart_metrics(results: list[dict]) -> dict:
    """
    Compute Top-1/Top-3 accuracy and macro-F1 over chart-type predictions.

    Only computed for rows where chart_types_reference is non-empty.
    Returns null fields if no reference labels are present.
    """
    labelled = [
        r for r in results
        if r.get("chart_types_reference") and r.get("chart_types_predicted") is not None
    ]

    if not labelled:
        return {
            "chart_top1_accuracy": None,
            "chart_top3_accuracy": None,
            "chart_macro_f1":      None,
        }

    top1_hits = 0
    top3_hits = 0
    all_true: list[str] = []
    all_pred: list[str] = []

    for r in labelled:
        refs  = [_normalise_chart(c) for c in r["chart_types_reference"]]
        preds = [_normalise_chart(c) for c in r["chart_types_predicted"]]

        # Primary reference is the first chart type in the ground truth
        primary_ref = refs[0] if refs else ""
        top1_hits += 1 if (preds and preds[0] == primary_ref) else 0
        top3_hits += 1 if any(p == primary_ref for p in preds[:3]) else 0

        # For F1: pair up by position (primary KPI → first chart)
        all_true.append(primary_ref)
        all_pred.append(preds[0] if preds else "")

    n = len(labelled)
    top1 = round(100.0 * top1_hits / n, 2)
    top3 = round(100.0 * top3_hits / n, 2)

    # Macro-F1 with sklearn if available, else simple implementation
    macro_f1 = _macro_f1(all_true, all_pred)

    return {
        "chart_top1_accuracy": top1,
        "chart_top3_accuracy": top3,
        "chart_macro_f1":      macro_f1,
    }


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float | None:
    try:
        from sklearn.metrics import f1_score
        return round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    except ImportError:
        pass
    # Fallback: manual macro-F1
    classes = set(y_true)
    if not classes:
        return None
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return round(sum(f1s) / len(f1s), 4)


# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------

def compute_latency_metrics(results: list[dict]) -> dict:
    latencies = sorted(r.get("latency_s", 0.0) for r in results)
    n = len(latencies)
    if n == 0:
        return {}
    avg = sum(latencies) / n
    p50 = latencies[int(0.50 * n)]
    p95 = latencies[min(int(0.95 * n), n - 1)]
    return {
        "avg_latency_s": round(avg, 3),
        "p50_latency_s": round(p50, 3),
        "p95_latency_s": round(p95, 3),
    }


# ---------------------------------------------------------------------------
# Robustness metrics
# ---------------------------------------------------------------------------

def compute_robustness_metrics(
    original_results: list[dict],
    paraphrase_results: list[dict] | None,
    missing_info_results: list[dict] | None,
) -> dict:
    """
    Compute consistency under perturbation.

    paraphrase_consistency: fraction of pairs where both original AND
    paraphrased outputs have valid_schema == True (or both == False).

    missing_info_validity_rate: schema_validity_rate on missing-info variant.
    """
    paraphrase_consistency = None
    if paraphrase_results and len(paraphrase_results) == len(original_results):
        consistent = sum(
            1 for o, p in zip(original_results, paraphrase_results)
            if o.get("valid_schema") == p.get("valid_schema")
        )
        paraphrase_consistency = round(consistent / len(original_results), 4)

    missing_validity = None
    if missing_info_results:
        n = len(missing_info_results)
        ok = sum(1 for r in missing_info_results if r.get("valid_schema"))
        missing_validity = round(ok / n, 4) if n > 0 else None

    return {
        "paraphrase_consistency":    paraphrase_consistency,
        "missing_info_validity_rate": missing_validity,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4 — compute evaluation metrics")
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--mode", choices=["standard", "robustness", "all"], default="all")
    return p.parse_args()


def _load_predictions(predictions_dir: Path, stem: str) -> list[dict] | None:
    path = predictions_dir / f"{stem}.jsonl"
    if path.exists():
        return load_jsonl(str(path))
    return None


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    config = load_experiment_config(experiment_dir)
    predictions_dir = experiment_dir / "predictions"

    existing_metrics = load_metrics(experiment_dir) or {}

    eval_results: dict[str, Any] = existing_metrics.get("eval", {})

    for model_stem in ("base_model", "finetuned_model"):
        results = _load_predictions(predictions_dir, model_stem)
        if results is None:
            logger.warning(f"No predictions found for {model_stem} — skipping")
            continue

        logger.info(f"Evaluating {model_stem} ({len(results)} examples)…")

        model_eval: dict[str, Any] = eval_results.get(model_stem, {})

        if args.mode in ("standard", "all"):
            model_eval["schema"]     = compute_schema_metrics(results)
            model_eval["chart_type"] = compute_chart_metrics(results)
            model_eval["latency"]    = compute_latency_metrics(results)
            model_eval["rag"]        = {"unsupported_claim_rate": None}

        if args.mode in ("robustness", "all"):
            para_results  = _load_predictions(predictions_dir, f"{model_stem}_paraphrased")
            miss_results  = _load_predictions(predictions_dir, f"{model_stem}_missing_info")
            model_eval["robustness"] = compute_robustness_metrics(
                results, para_results, miss_results
            )

        eval_results[model_stem] = model_eval

    payload = dict(existing_metrics)
    payload["eval"] = eval_results
    save_metrics(payload, experiment_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for model_stem in ("base_model", "finetuned_model"):
        me = eval_results.get(model_stem, {})
        schema = me.get("schema", {})
        latency = me.get("latency", {})
        print(f"\n  [{model_stem}]")
        print(f"    json_parse_rate      : {schema.get('json_parse_rate', 'N/A')}%")
        print(f"    schema_validity_rate : {schema.get('schema_validity_rate', 'N/A')}%")
        print(f"    completeness_score   : {schema.get('completeness_score', 'N/A')}")
        ct = me.get("chart_type", {})
        print(f"    chart_top1_accuracy  : {ct.get('chart_top1_accuracy', 'N/A')}")
        print(f"    avg_latency_s        : {latency.get('avg_latency_s', 'N/A')}")
    print("=" * 60)
    print(f"\nMetrics saved to: {experiment_dir / 'metrics.json'}")
    print(f"\nNext step:")
    print(f"  python pipeline/compare.py --outputs-root outputs/experiments")


if __name__ == "__main__":
    main()
