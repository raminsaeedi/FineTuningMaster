# Benchmark Dataset Report — `benchmark_v1`

> EVALUATION-ONLY (benchmark lock): never use for training, augmentation, prompt optimization, retriever tuning, hyperparameter tuning, or model selection.

**Hard checks: PASS**

## 1. Item count

- items: 30
- JSON parse errors: 0
- schema-invalid items: 0

### 2. Domain distribution

- `Finance`: 4
- `Retail`: 4
- `Logistics & Supply Chain`: 3
- `Marketing`: 3
- `SaaS / Software`: 3
- `E-Commerce`: 2
- `Energy & Utilities`: 2
- `HR & People Analytics`: 2
- `Healthcare`: 2
- `Manufacturing`: 2
- `IT Finance`: 1
- `IT Ops`: 1
- `Procurement`: 1

### 3. Task-type distribution

- `deviation`: 6
- `ranking`: 5
- `comparison`: 4
- `part_to_whole`: 4
- `correlation`: 3
- `composition`: 2
- `distribution`: 2
- `flow`: 2
- `trend`: 2

### 4. Chart-label distribution (acceptable_chart_types)

- `bar`: 25
- `table`: 19
- `line`: 8
- `scatter`: 7
- `area`: 4
- `donut`: 4
- `grouped_bar`: 4
- `pie`: 4
- `treemap`: 4
- `heatmap`: 3
- `box`: 2
- `histogram`: 2
- `sankey`: 2
- `stacked_bar`: 2

## 5. Chart-type coverage

- covered chart types (14): ['area', 'bar', 'box', 'donut', 'grouped_bar', 'heatmap', 'histogram', 'line', 'pie', 'sankey', 'scatter', 'stacked_bar', 'table', 'treemap']
- not covered (3): ['gauge', 'kpi_card', 'map']

## 6-7. Auto-scorable vs human-eval

- auto-scorable (task_type L1-covered): 22
- human-eval suitable: 30
- human-eval-ONLY (not auto-scorable): 8

## 8. Evidence strength

- strong (real_public + literature_L1): 9
- weak (realistic_manual or manual_expert): 21

## 9. Source/provenance leakage vs training

- training briefs checked: 512
- benchmark items colliding with training (fingerprint): 0 []

## 10. Label-lineage leakage vs synthetic generator

- all labels sourced from literature_L1/manual_expert (never generator): True
- literature_L1 items mismatching the independent L1 table: 0 []
- acceptable sets identical to the generator's TASK_CHART set (informational): 10 ['bm_v1_002', 'bm_v1_006', 'bm_v1_009', 'bm_v1_014', 'bm_v1_017', 'bm_v1_018', 'bm_v1_023', 'bm_v1_024', 'bm_v1_025', 'bm_v1_029']

> `task_type` is assigned independently via `data/eval/task_crosswalk.yaml`; `acceptable_chart_types` come from the independent L1 literature table (covered tasks) or documented expert judgment (uncovered tasks). Neither is derived from `TASK_CHART`/`KEYWORD_TASK`. Overlap with the generator set is possible but not derivation; identity is reported above for transparency.
