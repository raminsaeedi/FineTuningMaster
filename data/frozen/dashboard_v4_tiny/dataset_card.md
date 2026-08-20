# Dataset Card — dashboard_v4_tiny_1

## Purpose

`dashboard_v4_tiny_1` is a development-only, reproducible Kaggle/GPU smoke dataset derived from frozen `dashboard_v4_1`. It is not a replacement for the thesis dataset and must not be used for final thesis claims.

## Splits

- Train: 100 records from parent `train.jsonl`.
- Validation: 50 records from parent `val.jsonl`.
- Canonical test: 50 records from parent `test.jsonl`.
- Sports test: 50 cross-domain records selected from explicit sports database identifiers across the parent dataset. It is excluded from tiny Train/Validation and is not the parent dataset's canonical test split.

Sampling uses seed `20260820`, random selection inside chart-type strata, and a minimum of three examples per available chart type. When a source split contains fewer than three items of a chart type, all available items are retained. The parent canonical test has only five chart types; therefore the tiny canonical test has only those five chart types. Train and Validation cover all fourteen chart types present in their parent splits.

## Integrity

All record content is copied from the immutable parent dataset. Only `split` is relabeled to `sports_test` for the cross-domain evaluation file. `manifest.json`, `hashes.json`, and the files in `reports/` record lineage, deterministic sampling, validation, distributions, leakage checks, and SHA-256 hashes.

## Human evaluation

`human_eval_test_items_50.csv` and `human_eval_sports_test_items_50.csv` are blank reviewer templates. They contain input briefs and source-backed evidence, but no human ratings. Fill them only after model predictions are generated.

## Use

Train only with `train.jsonl`; use `val.jsonl` for development validation; use `test.jsonl` for the small in-domain check; use `sports_test.jsonl` only for the separate cross-domain diagnostic.
