# Dataset Leakage Report

Training records checked: 21. Near-duplicate threshold: 0.8 (char-3gram Jaccard).

## Active trainable dataset (authoritative)

The check scopes `train` to the **only files the current v2 pipeline trains on** (per `src/config/data/dashboard_v2.yaml`: `train_file` + `val_file`). Legacy synthetic v1 files (`data/processed/*`, `data/gold.jsonl`) are **superseded and never trained on**, so they are intentionally excluded here — including them would conflate two generator generations and produce misleading cross-generation overlaps.

- `data/frozen/dashboard_v2/train.jsonl = 18`
- `data/frozen/dashboard_v2/val.jsonl = 3`
- **total active train+val = 21**

**Overall: RULE_LEAKAGE**

| check | severity | n | detail |
| --- | --- | --- | --- |
| `exact_item_id::train~benchmark_v1` | no_issue | 0 |  |
| `exact_brief::train~benchmark_v1` | no_issue | 0 |  |
| `near_duplicate::train~benchmark_v1` | no_issue | 0 |  |
| `label_lineage::train~benchmark_v1` | no_issue | 0 |  |
| `exact_item_id::train~real_briefs_v1` | no_issue | 0 |  |
| `exact_brief::train~real_briefs_v1` | no_issue | 0 |  |
| `near_duplicate::train~real_briefs_v1` | no_issue | 0 |  |
| `label_lineage::train~real_briefs_v1` | no_issue | 0 |  |
| `exact_item_id::train~internal_test` | no_issue | 0 |  |
| `exact_brief::train~internal_test` | no_issue | 0 |  |
| `near_duplicate::train~internal_test` | no_issue | 0 |  |
| `label_lineage::train~internal_test` | rule_leakage | 1 | ['synthetic_generator'] |
| `same_source::train~benchmark_v1` | no_issue | 0 |  |

> `label_lineage::train~benchmark_v1` is `no_issue` because benchmark labels use `literature_L1`/`manual_expert`, disjoint from the generator's `synthetic_generator` lineage — this is the rule-leakage guard for the benchmark.
> `label_lineage::train~internal_test` is expected to be `rule_leakage` (both are synthetic-generator lineage) — the internal test is diagnostic only, never an independent claim.
> Some benchmark `acceptable_chart_types` sets happen to match the generator's `TASK_CHART` set for the same task_type (see `benchmark_dataset_report.md` §10). This is **informational overlap only, not leakage**: those labels are sourced from the independent L1 literature table / documented expert judgment, not from the synthetic generator.