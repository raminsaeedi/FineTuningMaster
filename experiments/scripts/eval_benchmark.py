"""B2 — score benchmark_v1 predictions (offline, no model inference).

Reads per-method `predictions.jsonl` under a predictions root and scores each
method's primary chart against `benchmark_v1.jsonl`'s independent
`acceptable_chart_types` (covered items only). Also reports reference-free schema
metrics. Writes experiments/results/benchmark_v1_eval.{json,md}.

This scorer uses NO synthetic gold labels. It is only meaningful AFTER a benchmark
inference pass has produced predictions (approval-gated); with no predictions it
reports coverage with all-wrong covered accuracy (parse failures) and says so.

    python experiments/scripts/eval_benchmark.py --predictions-root experiments/outputs/benchmark_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.schemas import GenerationResult
from src.evaluation.l1_independent import score_benchmark
from src.evaluation.metrics.schema_compliance import SchemaCompliance
from src.inference.postprocess import reparse
from src.utils.io import read_jsonl, write_json

BENCHMARK = "data/eval/benchmark_v1.jsonl"


def _load_predictions(run_dir: Path):
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        return None
    return [reparse(GenerationResult(**r)) for r in read_jsonl(path)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score benchmark_v1 predictions (offline)")
    p.add_argument("--predictions-root", default="experiments/outputs/benchmark_v1")
    p.add_argument("--benchmark", default=BENCHMARK)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = (_PROJECT_ROOT / args.predictions_root) if not Path(args.predictions_root).is_absolute() else Path(args.predictions_root)
    benchmark_items = read_jsonl(_PROJECT_ROOT / args.benchmark)

    per_method = {}
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    for run_dir in run_dirs:
        results = _load_predictions(run_dir)
        if results is None:
            continue
        chart = score_benchmark(results, benchmark_items)
        schema = SchemaCompliance().compute(results, None)
        per_method[run_dir.name] = {
            "benchmark_chart_acceptability": chart,
            "schema_compliance": {k: schema[k] for k in
                                  ("json_parse_rate", "schema_validity_rate", "completeness_score")},
        }

    payload = {"benchmark": args.benchmark, "predictions_root": str(args.predictions_root),
               "n_benchmark_items": len(benchmark_items), "per_method": per_method}
    write_json(payload, _PROJECT_ROOT / "experiments/results/benchmark_v1_eval.json")

    lines = ["# Benchmark v1 Evaluation (independent, non-circular)", "",
             "> **Tier 3 — benchmark_v1 evaluation.** Covered items only; parse errors/missing "
             "primary chart = wrong; uncovered items excluded from accuracy but counted in "
             "coverage. Uses independent `acceptable_chart_types` (literature_L1/manual_expert), "
             "NOT synthetic generator labels. Do NOT read as layout/usefulness/quality.", "",
             f"Benchmark: `{args.benchmark}` ({len(benchmark_items)} items). "
             f"Predictions root: `{args.predictions_root}`.", ""]
    if not per_method:
        lines += ["_No predictions found — run the (approval-gated) benchmark inference first._"]
    else:
        lines += ["| run | coverage | covered_acc | parse_fail | json_parse% | schema_valid% |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, m in per_method.items():
            c = m["benchmark_chart_acceptability"]
            s = m["schema_compliance"]
            lines.append(f"| {name} | {c['coverage_rate']} ({c['n_covered']}/{c['n_total']}) | "
                         f"{c['covered_accuracy']} | {c['parse_failures']} | "
                         f"{s['json_parse_rate']} | {s['schema_validity_rate']} |")
        lines += ["", "> Forbidden: usefulness/actionability/real-quality claims (need human "
                  "eval); significance/variance (single seed); reuse of benchmark_v1 for any "
                  "training/tuning/selection."]

    out = _PROJECT_ROOT / "experiments/results/benchmark_v1_eval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Benchmark eval: {len(per_method)} method run(s) scored -> {out}")


if __name__ == "__main__":
    main()
