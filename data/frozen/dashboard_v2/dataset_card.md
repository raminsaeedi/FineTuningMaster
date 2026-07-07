# Dataset Card — dashboard_v2 (frozen)

## What this is
`dashboard_v2` is **source-conditioned synthetic** data for the dashboard-design
recommendation task — **not** a real, human-labeled dataset. Briefs and reference
recommendations are machine-generated (offline, deterministic) and conditioned on
documented domain source templates; chart-task labels follow a principled,
literature-backed mapping (Cleveland & McGill 1984; Few 2006; Munzner 2014).

See the full plan: [`docs/datasets/DATASET_V2_IMPLEMENTATION_PLAN.md`](../../../docs/datasets/DATASET_V2_IMPLEMENTATION_PLAN.md)
and the evaluation protocol: [`docs/evaluation/evaluation_protocol.md`](../../../docs/evaluation/evaluation_protocol.md).

## Files
| File | Role | Trainable? |
| --- | --- | --- |
| `train.jsonl` | SFT corpus | **YES** (gradient updates) |
| `val.jsonl` | validation | **VALIDATION-ONLY** (no gradient updates) |
| `internal_test.jsonl` | format/robustness diagnostic | **NO** |
| `test_paraphrased.jsonl` | robustness variant | **NO** |
| `test_missing_info.jsonl` | robustness variant | **NO** |
| `../../eval/l1_chart_effectiveness_v1.csv` | independent L1 gold | **NEVER** |
| `../../eval/real_briefs_v1.jsonl` | external briefs (no labels) | **NEVER** |
| `raw_batches/*.jsonl` | pre-freeze candidate staging | — |

Stored `split` values are `train` / `val` / `test`; the `test` bucket is written
to `internal_test.jsonl` for schema/loader compatibility.

## Build pipeline
```
generate_dataset_v2.py   -> raw_batches/            (candidates, offline sample mode)
freeze_dataset_v2.py     -> train/val/internal_test (validated, hash-split, deduped)
build_perturbations_v2.py-> test_paraphrased/test_missing_info
validate_frozen_dataset.py -> validation_report.md + hashes.json
```

## Provenance
Each item stores `brief.extra.source_id`, `brief.extra.source_ref` and
`brief.extra.generator_version`. Generation parameters are in
[`generation_spec.yaml`](generation_spec.yaml); file integrity is in
[`hashes.json`](hashes.json) (SHA256).

## Intended use vs prohibited use
- **Intended:** train/validate the four methods; measure JSON/schema validity,
  completeness and robustness (L2) on the diagnostic splits.
- **Prohibited:** using any synthetic split as a **primary chart-quality claim**
  (train and test share the generator's `task_type -> chart_type` lineage, so this
  is circular). Independent chart-selection claims use the L1 literature gold; the
  validity anchor for usefulness is human evaluation (L4).

## Known limitations
- Synthetic content; not human-authored.
- Circular for chart choice → internal diagnostic only.
- `real_briefs_v1.jsonl` currently seeds 10 verified external briefs and must be
  expanded to 30–40 before final evaluation.
