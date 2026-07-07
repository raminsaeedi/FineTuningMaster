# Independent L1 Chart-Selection Results

L1 gold (independent, literature): `data/eval/l1_chart_effectiveness_v1.csv` (Saket 2019 + Kim & Heer 2018).
Predictions gold join: `data/processed/test.jsonl` (cached v1 synthetic test).

> **Keying case — diagnostic/limited.** These cached predictions are on the SYNTHETIC v1 test set, whose `task_type` is generator-derived. The chart set is independent (literature), but the task label shares the generator lineage, so these L1 numbers are a diagnostic, not a fully independent claim. Independent L1 requires `benchmark_v1` predictions (approval-gated inference).

> **L1 limitation.** L1 validates only chart-selection *acceptability for covered task types*. It does NOT validate layout, styling, interaction, rationale, or overall dashboard-design quality. Uncovered items are excluded from accuracy (never counted correct); coverage is reported below.

| method | coverage_rate | n_covered | n_uncovered | covered_accuracy |
| --- | --- | --- | --- | --- |
| prompt_only | 0.7236 | 144 | 55 | 0.2222 |
| rag | 0.7236 | 144 | 55 | 0.1667 |
| ft | 0.7236 | 144 | 55 | 0.1944 |
| ft_rag | 0.7236 | 144 | 55 | 0.2778 |

## Per-method detail (per task_type accuracy on covered items)

### prompt_only
- coverage_rate=0.7236 (covered 144 of 199 gold KPI entries; uncovered 55)
- covered_accuracy=0.2222
  - `comparison`: 15/60 = 0.25
  - `correlation`: 0/7 = 0.0
  - `deviation`: 6/21 = 0.2857
  - `distribution`: 5/22 = 0.2273
  - `ranking`: 1/5 = 0.2
  - `trend`: 5/29 = 0.1724
- uncovered task types (excluded): {'composition': 10, 'flow': 10, 'part_to_whole': 35}

### rag
- coverage_rate=0.7236 (covered 144 of 199 gold KPI entries; uncovered 55)
- covered_accuracy=0.1667
  - `comparison`: 10/60 = 0.1667
  - `correlation`: 0/7 = 0.0
  - `deviation`: 7/21 = 0.3333
  - `distribution`: 1/22 = 0.0455
  - `ranking`: 2/5 = 0.4
  - `trend`: 4/29 = 0.1379
- uncovered task types (excluded): {'composition': 10, 'flow': 10, 'part_to_whole': 35}

### ft
- coverage_rate=0.7236 (covered 144 of 199 gold KPI entries; uncovered 55)
- covered_accuracy=0.1944
  - `comparison`: 17/60 = 0.2833
  - `correlation`: 0/7 = 0.0
  - `deviation`: 3/21 = 0.1429
  - `distribution`: 4/22 = 0.1818
  - `ranking`: 0/5 = 0.0
  - `trend`: 4/29 = 0.1379
- uncovered task types (excluded): {'composition': 10, 'flow': 10, 'part_to_whole': 35}

### ft_rag
- coverage_rate=0.7236 (covered 144 of 199 gold KPI entries; uncovered 55)
- covered_accuracy=0.2778
  - `comparison`: 19/60 = 0.3167
  - `correlation`: 0/7 = 0.0
  - `deviation`: 6/21 = 0.2857
  - `distribution`: 10/22 = 0.4545
  - `ranking`: 1/5 = 0.2
  - `trend`: 4/29 = 0.1379
- uncovered task types (excluded): {'composition': 10, 'flow': 10, 'part_to_whole': 35}
