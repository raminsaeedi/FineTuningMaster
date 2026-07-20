# Benchmark Dataset Construction (`benchmark_v1`)

How `data/eval/benchmark_v1.jsonl` is built and why it is defensible as an **independent**
evaluation set for comparing methods A/B/C/D. Field contract:
[`benchmark_v1_schema.yaml`](../../data/eval/benchmark_v1_schema.yaml).

## Benchmark lock (non-negotiable)
`data/eval/benchmark_v1.jsonl` is **evaluation-only**. It must **never** be used for training,
augmentation, prompt optimization, retriever tuning, hyperparameter tuning, or model
selection. This lock is restated in the leakage report and the thesis claim policy, and is
checked by `experiments/scripts/validate_benchmark.py`.

## Independence from the synthetic generator
The construct-validity risk in this project is **rule leakage**: the synthetic generator labels
charts via `KEYWORD_TASK → task_type → TASK_CHART[task_type][0]`. To be an independent test,
`benchmark_v1` keeps a **disjoint label lineage**:
- **`task_type`** is assigned from documented analytical intent using
  [`task_crosswalk.yaml`](../../data/eval/task_crosswalk.yaml), an independent crosswalk
  authored from analytical-task taxonomies (Amar/Eagan/Stasko 2005; Brehmer & Munzner 2013;
  Munzner 2014) — **not** copied from the generator.
- **`acceptable_chart_types`** are **set-valued**. For task types covered by the independent
  L1 literature table ([`l1_chart_effectiveness_v1.csv`](../../data/eval/l1_chart_effectiveness_v1.csv),
  Saket 2019 + Kim & Heer 2018) the set is taken from that table (`label_source=literature_L1`).
  For the three task types not covered by L1 (`composition`, `part_to_whole`, `flow`) a
  documented expert set from `task_crosswalk.yaml` is used (`label_source=manual_expert`).
  **Never `TASK_CHART`.**

## Sources and evidence strength
- **`real_public`** — verified public dashboard briefs reused from
  `data/eval/real_briefs/items.jsonl` (provenance in
  [`real_briefs_provenance.md`](../datasets/real_briefs_provenance.md)). **Strong evidence.**
- **`realistic_manual`** — author-drafted realistic scenarios, clearly marked, with
  `source_reference="author-drafted realistic scenario (no external source)"`. **Weaker
  evidence**; used to broaden domain/task coverage.

Evidence strength: **strong** = `real_public` + `literature_L1`; otherwise **weak**.

## Auto-scoring vs human-eval suitability
- `suitable_for_auto_scoring = true` **iff** the item's `task_type` is covered by the L1 table.
  Only these items feed the independent L1 scorer (`eval_l1_independent.py`), so automatic
  scoring always uses independent literature sets.
- `suitable_for_human_eval = true` for all items. Items whose task_type is L1-uncovered are
  **human-eval-only** (their expert `acceptable_chart_types` are reviewer reference, not an
  auto-scoring key).

## Build procedure
`experiments/scripts/build_benchmark.py` (core logic in
`src/data_pipeline/benchmark_build.py`):
1. Load verified `real_public` briefs; load the `realistic_manual` seed list.
2. Assign `task_type` per item from its primary goal via `task_crosswalk.yaml`.
3. Set `acceptable_chart_types` from the L1 table if covered (`literature_L1`) else the expert
   set (`manual_expert`); set `suitable_for_auto_scoring` accordingly.
4. Write `data/eval/benchmark_v1.jsonl` (30–50 items).

Validation and the ten required checks: `experiments/scripts/validate_benchmark.py` →
[`benchmark_dataset_report.md`](../../experiments/results/benchmark_dataset_report.md).

## What this benchmark can and cannot support
- **Can:** independent, non-circular comparison of methods' **chart selection** on covered
  items (once predictions on these briefs exist — an inference pass gated behind explicit
  approval).
- **Cannot:** validate layout, styling, interaction, rationale quality, or overall
  usefulness — those require human evaluation. High synthetic Top-1 accuracy must never be
  presented as evidence of real dashboard-design quality.
