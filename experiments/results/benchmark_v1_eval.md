# Benchmark v1 Evaluation (independent, non-circular)

> **Tier 3 — benchmark_v1 evaluation.** Covered items only; parse errors/missing primary chart = wrong; uncovered items excluded from accuracy but counted in coverage. Uses independent `acceptable_chart_types` (literature_L1/manual_expert), NOT synthetic generator labels. Do NOT read as layout/usefulness/quality.

Benchmark: `data/eval/benchmark_v1.jsonl` (30 items). Predictions root: `experiments/outputs/benchmark_v1_smoke`.

| run | coverage | covered_acc | parse_fail | json_parse% | schema_valid% |
| --- | --- | --- | --- | --- | --- |
| E01_smoke__benchmark_v1_42 | 0.7333 (22/30) | 0.0455 | 21 | 50.0 | 0.0 |

> Forbidden: usefulness/actionability/real-quality claims (need human eval); significance/variance (single seed); reuse of benchmark_v1 for any training/tuning/selection.