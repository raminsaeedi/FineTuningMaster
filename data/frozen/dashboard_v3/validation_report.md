# Frozen dashboard_v3 — Validation Report

Dataset: nvBench

## `train.jsonl`
- records: 1281
- JSON parse errors: 0
- schema/enum/non-empty problems: 0
- duplicate item_id: 0
- duplicate brief fingerprint: 0 (diagnostic)
- sha256: `f9ace854c64f89e2a26d2767ad2754b4951316fbcceec4a8e9374e4624fac222`

## `val.jsonl`
- records: 264
- JSON parse errors: 0
- schema/enum/non-empty problems: 0
- duplicate item_id: 0
- duplicate brief fingerprint: 0 (diagnostic)
- sha256: `8394dcdb75d53de19fcd71c4c688e05e162b895d44a920f3cbb678f37ba2999d`

## `test.jsonl`
- records: 274
- JSON parse errors: 0
- schema/enum/non-empty problems: 0
- duplicate item_id: 0
- duplicate brief fingerprint: 0 (diagnostic)
- sha256: `e2df055d0a75c25f53a88cb830a5b6d66411fa179413f352f45ca2f6873829d5`

## Distributions (train + val + test)

### Domain

_(none)_

### task_type

- `comparison`: 1379
- `part_to_whole`: 242
- `trend`: 109
- `composition`: 76
- `correlation`: 13

### chart_type

- `bar`: 1379
- `pie`: 242
- `line`: 109
- `stacked_bar`: 76
- `scatter`: 13

## Leakage check (train/val vs test + real_briefs)

- item_id overlap: 0
- fingerprint overlap: 0

## Result

**Hard checks: PASS**

> `train.jsonl` and `val.jsonl` are the only trainable files. `test.jsonl`, the human-evaluation CSV, and all reports are NEVER used for training.