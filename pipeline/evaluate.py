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

from evaluation import (
    compute_chart_metrics,
    compute_latency_metrics,
    compute_robustness_metrics,
    compute_schema_metrics,
)
from utils.experiment import load_experiment_config, load_metrics, save_metrics
from utils.helpers import load_jsonl, setup_logging

logger = setup_logging()


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
