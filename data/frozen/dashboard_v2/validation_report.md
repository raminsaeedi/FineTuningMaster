# Frozen Dataset v2 — Validation Report

Generator: `v2.0-sample`

## `train.jsonl`
- records: 18
- JSON parse errors: 0
- schema/enum/non-empty problems: 0
- duplicate item_id: 0
- duplicate brief fingerprint: 0 (diagnostic)
- sha256: `01fbffd0cdbcdd0e782db6f43ccf517f9f0708d65e0329146dd578d7a4e58a32`

## `val.jsonl`
- records: 3
- JSON parse errors: 0
- schema/enum/non-empty problems: 0
- duplicate item_id: 0
- duplicate brief fingerprint: 0 (diagnostic)
- sha256: `25cd0137a40d5b38ddd9f075dc0c83052b025e2064c4e7e0d5cb37f37d8e661c`

## `internal_test.jsonl`
- records: 3
- JSON parse errors: 0
- schema/enum/non-empty problems: 0
- duplicate item_id: 0
- duplicate brief fingerprint: 0 (diagnostic)
- sha256: `1bc80e37a327f2379a977f91bd7674348392e053d5876efa9adbf1793778b0a6`

## `test_paraphrased.jsonl`
- records: 3
- JSON parse errors: 0
- schema/enum/non-empty problems: 0
- duplicate item_id: 0
- duplicate brief fingerprint: 0 (diagnostic)
- sha256: `9dd6147fc11253edc12026232dc00e4e027fb1deb33f8bacb708d4dc0c4d3d88`

## `test_missing_info.jsonl`
- records: 3
- JSON parse errors: 0
- schema/enum/non-empty problems: 0
- duplicate item_id: 0
- duplicate brief fingerprint: 0 (diagnostic)
- sha256: `43be2bfde2aea7a3f56b62e7208730ba51ca7c5fc60ec94389436f1ed62e4e47`

## Distributions (train + val + internal_test)

### Domain

- `E-Commerce`: 3
- `Finance & Banking`: 3
- `Healthcare`: 3
- `SaaS / Software`: 3
- `Energy & Utilities`: 2
- `HR & People Analytics`: 2
- `Logistics & Supply Chain`: 2
- `Manufacturing`: 2
- `Marketing & Advertising`: 2
- `Retail`: 2

### task_type

- `comparison`: 26
- `trend`: 18
- `part_to_whole`: 14
- `distribution`: 13
- `deviation`: 7
- `flow`: 7
- `ranking`: 4
- `correlation`: 3
- `composition`: 2

### chart_type

- `bar`: 37
- `line`: 18
- `donut`: 14
- `histogram`: 13
- `sankey`: 7
- `scatter`: 3
- `stacked_bar`: 2

## Leakage check (train/val vs internal_test + real_briefs)

- item_id overlap: 0
- fingerprint overlap: 0

## Result

**Hard checks: PASS**

> `train.jsonl` and `val.jsonl` are the only trainable files. `internal_test.jsonl`, the perturbation sets, `data/eval/l1_chart_effectiveness_v1.csv` and `data/eval/real_briefs_v1.jsonl` are NEVER used for training.