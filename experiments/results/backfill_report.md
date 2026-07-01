# Backfill report

> This is a backfilled legacy internal-synthetic diagnostic report. It is not the final thesis-valid independent evaluation report. L1 human-effectiveness, L3 realism, and L4 human evaluation are pending.

## Provenance (legacy carry-forward)
- **Point values** (parse/schema/completeness/top-1/macro-F1) in `metrics.json`: re-presented from each run's `metrics_auto.json` (legacy, pre-Task-7; internal-synthetic diagnostic). No fresh metric numbers were computed.
- **`eval_per_item.jsonl`**: run-time STORED fields only (`parse_error`, `parsed`, `predicted_primary_chart`, `n_distinct_recs`).
- **not_available (deferred to a corrected re-scoring task):** per-item `schema_valid`, `completeness`, `gold_primary_chart`, `synthetic_top1_correct`, and **all CIs** — computing them now would pair legacy values with current-code results.

## Processed runs
| run | status | n_items | metrics_auto | refs_on_disk | manifest_keys_added | raw_byte_identical |
| --- | --- | --- | --- | --- | --- | --- |
| E01_qwen0_5b_prompt_42 | legacy-carry-forward | 50 | True | True | - | True |
| E02_qwen0_5b_rag_42 | legacy-carry-forward | 50 | True | True | - | True |
| E03_qwen0_5b_ft_42 | legacy-carry-forward | 50 | True | True | - | True |
| E04_qwen0_5b_ft_rag_42 | legacy-carry-forward | 50 | True | True | - | True |

## Skipped (left untouched)
- `experiments`: missing config_snapshot.yaml and/or predictions.jsonl (left untouched; possibly a nested/stray folder)

## Missing artifacts among processed runs (exact paths)
- none

## Inputs for the deferred corrected re-scoring task
- References (`data/processed/test.jsonl` etc.): presence per run is shown as `refs_on_disk` above; they are intentionally NOT used here (using them would produce corrected numbers).
- Per-run `predictions_paraphrased.jsonl` / `predictions_missing_info.jsonl` are absent in the flat runs, so robustness re-derivation is not possible in a later re-scoring either unless those variant predictions are regenerated.
