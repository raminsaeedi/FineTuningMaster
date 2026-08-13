# Training Data Mapping (builders)

How the synthetic generator and the external sources (ChartGPT, nvBench / nvBench 2.0, Quda) map
into the project training schema (`src/core/schemas.py` `GoldItem`). Builders live in
[`src/data_pipeline/builders/`](../../src/data_pipeline/builders) (**confirmed path** — the project
package is `src/data_pipeline/`, there is no `src/data/`).

## Scientific framing (four questions)

- **What does this produce?** Training/augmentation data in the `GoldItem` schema, tagged by source
  and usage tier. It is **not** a metric.
- **With which data?** Synthetic generator (full dashboards, functional now) + ChartGPT/nvBench/Quda
  (partial signals; **stubs**, not ingested yet).
- **Separation rule (verbatim):** _"No dataset artifact, label set, or label-generation lineage is
  used both for training/augmentation and final independent evaluation gold."_
- **How to interpret?** Builders do raw→schema mapping only; dedup/leakage and split policy are
  explicit and auditable, so a reviewer can confirm no eval item leaks into training.

## Builder architecture (minimal)

- **Selection is a plain dict**, not a registry: `BUILDERS = {"synthetic": ..., "chartgpt": ...}` with
  `get_builder(name)` in `builders/__init__.py`. (A registry was deliberately **avoided** — explicit
  selection is enough.)
- `base.py` — `BaseBuilder` ABC (`source`, `usage_tier`, `load_raw`, `to_gold_items`, `_tag`),
  `DataNotProvisionedError`, and `trainval_split`.
- `synthetic_builder.py` — functional; wraps `synth_generator.generate_dataset`.
- `chartgpt_builder.py`, `nvbench_builder.py`, `quda_builder.py` — **stubs**: each documents its
  expected raw format + mapping contract + validation, and raises `DataNotProvisionedError`.
- `leakage.py` — simple duplicate/leakage check (below).
- **No existing files were modified**; no Hydra config or build script added yet (deferred to ingestion).

## Per-source mapping

| Source              | Status         | Brief fields                      | DesignOutput                                                                                                                   | task_type                                       | chart_type                                   | Confidence                      |
| ------------------- | -------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- | -------------------------------------------- | ------------------------------- |
| Synthetic generator | **functional** | all                               | full (already in schema)                                                                                                       | from generator                                  | from `TASK_CHART`                            | n/a (canonical full dashboards) |
| nvBench / 2.0       | stub           | kpis, columns (DB schema), goals  | `kpi_chart_mapping[0]` (chart, encoding); 2.0 → `alternatives` (set-valued) + `rationales` (reasoning); minimal layout/styling | inferred (low)                                  | source label → `ChartType`                   | med–high (chart/encoding)       |
| ChartGPT            | stub           | kpis, columns (data table), goals | `kpi_chart_mapping[0]` (chart, encoding); minimal layout/styling                                                               | inferred (low)                                  | source label → `ChartType`                   | medium                          |
| Quda                | stub           | kpis/goals (NL query)             | `kpi_chart_mapping[0].task_type` (real label)                                                                                  | **Quda label (high)** via `task_crosswalk.yaml` | **derived** via `TASK_CHART` (training-only) | high task_type, derived chart   |

Partial sources map to **single-KPI mini-dashboards** (a 1-KPI brief → a 1-chart valid recommendation,
with minimal-but-valid `layout`/`styling`, empty `interactions`). Chart labels outside `ChartType` (17)
are **dropped and logged**. `task_type` reuse [`data/eval/task_crosswalk.yaml`](../../data/eval/task_crosswalk.yaml)
— consistent with Task 3 (Quda is tagged train/aug there).

## Usage tiers & "never final gold unless justified"

- All four sources are `usage_tier = "train_aug"` (stamped in `brief.extra.source` / `.usage_tier`).
- External sources are **train/val only, never test, never eval gold** (`trainval_split`).
- The synthetic **test** split is **internal and circular** (its task→chart labels come from the
  generator's own fixed rule). It is usable only for **limited L2 / format / robustness checks**,
  never the main chart-quality validity claim.
- **Final independent gold is unchanged:** L1 human-effectiveness ([`human_effectiveness_gold.csv`](../../data/eval/human_effectiveness_gold.csv)),
  real briefs ([`real_briefs/items.jsonl`](../../data/eval/real_briefs/items.jsonl)), and human evaluation.
- **"Unless justified" bar:** an external set may be used for evaluation **only** for narrow,
  non-circular format/legality checks, explicitly labeled, with the circularity caveat — default is **no**.

## Leakage prevention (simple first version)

`leakage.py` provides two checks, then drops + reports collisions (no silent truncation):

1. **exact `item_id`** collision, and
2. **normalized text fingerprint** collision (lowercased; whitespace- and order-normalized
   `users` + `goals` + `kpis` + column names).

`filter_against(candidates, reference)` returns `(kept, dropped_report)`. Intended use: filter every
augmentation pool against each evaluation set (synthetic test split, L1 gold, real briefs) before
training. No fuzzy/shingled near-duplicate matching yet (deferred).

## Deterministic splits

- **Synthetic:** unchanged `assign_split` (MD5 of content-based `item_id`, 80/10/10). The synthetic
  builder computes `item_id` from the **untagged** brief, so ids match `build_data` exactly.
- **External:** `trainval_split` = `assign_split` with the `test` bucket remapped to `train`
  (train/val only). Content-based ids mean adding sources never reshuffles existing items.

## Implemented now vs. deferred

- **Now:** this doc; `builders/` package; functional `SyntheticBuilder`; documented stubs for
  ChartGPT / nvBench(+2.0) / Quda; `trainval_split`; simple `leakage` check.
- **Deferred to a later ingestion task:** download/parse external data (ChartGPT + Quda are
  cite-and-ask; nvBench MIT / nvBench 2.0 CC-BY-4.0), implement each stub's mapping, add chart-label→
  `ChartType` tables, wire a `build_augmentation` step + Hydra config, and run the leakage filter on
  real augmentation pools.
