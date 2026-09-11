# Chapter 4 Figure Specification

Status: specification only. No unverified image has been inserted into the manuscript.

## Figure 4.1 — Dataset lineage and evaluation boundary

Purpose: show the construction path from the registered nvBench archive to the frozen `dashboard_v4` package and the exact `dashboard_v4_1` manifest revision. The figure must make the mixed field-level provenance visible and must show that the held-out test and the 40 human-evaluation items bypass enrichment, augmentation, and semantic repair.

### Nodes and labels

1. **Registered external source**
   - Label: `nvBench-main.zip`
   - Sub-label: `main` branch; no pinned upstream commit
   - Counts: 7,247 top-level visualization objects; 25,762 local natural-language query records; 25,750 published NL/VIS pairs from 750 tables and 105 domains
   - Source: `data/raw_external/nvbench/source_manifest.json`, `source_revision`, `dataset_stats`
   - Integrity label: `SHA-256 2c95244aca93aaca689fc954f8ae228c6c17fd47c81e1d7b265c4191cb012e4c`

2. **Source-faithful extraction and validation**
   - Label: `query/schema/SQL/VQL/chart evidence`
   - Sub-label: field resolution, aggregation, constraints, encoding, task inference, quality rules
   - Source: `src/data_pipeline/builders/nvbench_builder.py`, `src/data_pipeline/nvbench_quality.py`, `src/config/data/nvbench_mapping.yaml`, and `src/config/data/nvbench_quality_rules.yaml`

3. **Quality selection and group-safe split**
   - Label: `Tier A selection; seed 42; group-safe split`
   - Counts: 1,819 source-grounded modeling records; 784/167/167 unique source groups in train/validation/test
   - Sub-label: 1,281 train; 264 validation; 274 held-out test
   - Source: `data/frozen/dashboard_v3/manifest.json`, `data/frozen/dashboard_v3/dataset_card.md`, and the `nvbench_large_v2` selection artifacts

4. **v3 train/validation enrichment branch**
   - Label: `1,545 train/validation records`
   - Sub-label: six writable presentation fields; `deepseek-v4-flash-sovereign`; temperature 0.0; requested/configured `xhigh`
   - Writable fields: `users`, `context_summary`, `layout`, `styling`, `interactions`, `rationales`
   - Sub-label: source-backed analytical fields protected; `not_gold=true`
   - Source: `data/frozen/dashboard_v3/manifest.json` and `data/frozen/dashboard_v3/reports/human_enrichment_r1.csv`

5. **Frozen source-grounded predecessor**
   - Label: `dashboard_v3`
   - Counts: 1,281 train; 264 validation; 274 test; 40 test-derived human-evaluation items
   - Sub-label: test and human-evaluation artifacts are held out from the enrichment input
   - Source: `data/frozen/dashboard_v3/manifest.json`

6. **Controlled v4 augmentation branch**
   - Label: `2,915 candidate attempts → 2,000 accepted generated records`
   - Sub-label: `gpt-5.6-luna`; `codex_agent`; seed 42; batch size 40; 50 batches
   - Counts: 1,651 generated train; 349 generated validation; 0 generated test
   - Sub-label: `source=llm_generated`; `not_gold=true`
   - Source: `data/staging/dashboard_v4/run_20260814T010356Z/reports/generation_report.json`

7. **v4.1 semantic repair branch**
   - Label: `six-field semantic repair`
   - Sub-label: `gpt-5.6-luna`; `codex_agent_context_aware`; 2,000 repaired records
   - Protected fields: `goals`, `kpis`, `columns`, `constraints`, `task_type`, `chart_type`, `encoding`
   - Repaired fields: `users`, `context_summary`, `layout`, `styling`, `interactions`, `rationales`
   - Source: `data/frozen/dashboard_v4/manifest.json` and `data/frozen/dashboard_v4/reports/repair_report.json`

8. **Final frozen package**
   - Label: operational identity `dashboard_v4`
   - Sub-label: exact frozen manifest revision `dashboard_v4_1`
   - Counts: 2,932 train; 613 validation; 274 held-out test; 3,819 modeling records; 40 test-derived human-evaluation rows
   - Sub-label: v3-derived and v4/v4.1 generated records remain distinguishable by lineage
   - Source: `src/config/data/dashboard_v4.yaml` and `data/frozen/dashboard_v4/manifest.json`

9. **Held-out evaluation branch**
   - Label: `test bypasses enrichment, generation, and repair`
   - Counts: 274 source-grounded test records; 40 test-derived human-evaluation items
   - Sub-label: the 40 rows are an evaluation instrument, not a fourth modeling split
   - Source: `data/frozen/dashboard_v3/manifest.json`, `data/frozen/dashboard_v4/manifest.json`, and `data/frozen/dashboard_v4/reports/leakage_report.json`

### Arrows

- Node 1 → Node 2: `registered bytes → parsed source evidence`
- Node 2 → Node 3: `source-faithful transformation → Tier A quality filtering and group-safe selection`
- Node 3 → Node 4: `train + validation only; 1,545 records`
- Node 3 → Node 5: `freeze source-grounded v3 package`
- Node 4 → Node 5: `six-field enrichment; test bypasses`
- Node 5 → Node 6: `v3 train + validation only; controlled augmentation`
- Node 6 → Node 7: `presentation-layer semantic repair; analytical specification protected`
- Node 5 + Node 7 → Node 8: `preserved v3 records + repaired generated records`
- Node 3 → Node 9: `held-out test and 40-item human-evaluation file bypass all later generation stages`
- Node 9 → Node 8: `test is packaged in final release but is not trainable`

### Visual encoding

Use blue for registered/source-grounded evidence, gray for deterministic transformation and validation, orange for LLM-generated or LLM-repaired content, and red outline for held-out evaluation artifacts. Use solid arrows for data flow and a dashed bypass arrow from the group-safe split to the held-out evaluation branch. Do not use `docs/project/dataset_overview.png`; that image describes the historical synthetic dataset and is not evidence for this figure.

### Caption

**Figure 4.1:** Construction lineage of the final dashboard-design dataset. The source-grounded `dashboard_v3` records are extracted from the registered nvBench archive, selected with quality and group-disjointness controls, and enriched only in train and validation. The controlled `dashboard_v4` augmentation adds 2,000 generated train/validation records, and `dashboard_v4_1` repairs six presentation fields without changing the generated analytical specification. The 274-record test and the 40 test-derived human-evaluation items bypass enrichment, augmentation, and repair and remain evaluation-only.

### Source files for the figure

`data/raw_external/nvbench/source_manifest.json`; `src/config/data/dashboard_v4.yaml`; `data/frozen/dashboard_v3/manifest.json`; `data/frozen/dashboard_v4/manifest.json`; `data/frozen/dashboard_v4/reports/repair_report.json`; `data/frozen/dashboard_v4/reports/leakage_report.json`; `data/staging/dashboard_v4/run_20260814T010356Z/reports/generation_report.json`.
