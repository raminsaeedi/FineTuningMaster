# Dataset Card — dashboard_v4_1

## Scope

`dashboard_v4_1` is a semantic-repair revision of the immutable `dashboard_v4` release. It preserves the original dashboard_v3 Train/Validation records, all generated structural/source fields, and the parent Test and Human-Evaluation files. It repairs generated users, context summaries, layouts, styling, interactions, and rationales from each record's own brief and encoding.

## Counts

- dashboard_v3 Train: 1281
- dashboard_v3 Validation: 264
- Generated Train: 1651
- Generated Validation: 349
- Final Train: 2932
- Final Validation: 613
- Test: 274
- Human Evaluation: 40

## Repair provenance

Generated records retain `source=llm_generated`, carry parent lineage to `dashboard_v4`, and record `dashboard_v4_1-semantic-repair-v1`, `gpt-5.6-luna`, and `codex_agent_context_aware`. They remain AI-generated records and are not nvBench gold, human gold, or expert gold.

## Semantic guarantees

The corrected generated fields are anchored to each record's goals, KPIs, columns, constraints, task, chart, and encoding. Interactions use existing columns only; rationales explain the actual chart and encoding without asserting unobserved trends, correlations, outcomes, or business facts; styling states its accessibility and color semantics; layouts match the number and order of mapped visual components.

See `reports/repair_report.json`, the before/after semantic audits, `manifest.json`, and `hashes.json`.
