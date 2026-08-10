# nvBench Train+Validation — Pre-Freeze Completeness Audit

Audit status: **PASS_DATASET_READY_FOR_FREEZE**  
Audited: Train `1281` + Validation `264` = `1545`  
Scope: Train + Validation only; offline checks; no enrichment API calls

## Results

- Train count: `1281`
- Validation count: `264`
- total audited: `1545`
- complete before repair: `1545/1545`
- records requiring repair: `0`
- successfully repaired during audit: `0`
- remaining invalid: `0`
- schema-valid rate: `1545/1545` (`100%`)
- enrichment-field completeness: `1545/1545` (`100%`)
- immutable/source-backed violations: `0`
- duplicate item IDs: `0`

Field completeness: `users 1545/1545`, `context_summary 1545/1545`, `layout 1545/1545`,
`styling 1545/1545`, `interactions 1545/1545`, `rationales 1545/1545`.

Lineage: all six enrichment fields are `llm_generated` from
`deepseek-v4-flash-sovereign` with `reasoning_effort=xhigh` and `temperature=0`.
Source-backed analytical fields remain unchanged.

Each record contains non-null, non-empty, schema-valid `users`, `context_summary`, `layout`,
`styling`, `interactions`, and `rationales`. Existing Phase-3 content checks and existing nvBench
source-semantic checks found no content contradictions.

The prior audit found blank `immutable_fingerprint` metadata on 1534 records. Before freeze,
those values were recomputed using the canonical `src.data_pipeline.enrichment.immutable_fingerprint`
implementation. Normalization changed metadata only; source-backed projections and all six
enrichment values remained byte-equivalent after JSON round-trip comparison.

## Leakage and held-out data

- Train/Test leakage: `0`
- Validation/Test leakage: `0`
- Train/Validation source-group leakage: `0`
- Test count: `274`; Test untouched and not processed during enrichment
- Test records processed during enrichment: `0`
- human_eval_test_items_40 count: `40`; preserved separately and not processed during enrichment
- `human_eval_test_items_40` processed during enrichment: `0`

Final package hashes: SHA-256 generated and immediately verified in `../hashes.json`.

Test and human-evaluation artifacts were preserved separately and were not modified.
