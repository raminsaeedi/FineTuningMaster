# Sources Table

Consolidated inventory of the data/label sources used across training and evaluation, with
their lineage and intended use. Referenced by `docs/evaluation/evaluation_protocol.md`.

| source | kind | used for | label lineage | independent of generator? | license / usage |
| --- | --- | --- | --- | --- | --- |
| Source-conditioned synthetic generator (`synth_generator_v2.py`) | synthetic | **train/val** + internal diagnostics | `TASK_CHART` rule (deterministic) | **No** (circular) | project-internal |
| L1 chart-effectiveness (`data/eval/l1_chart_effectiveness_v1.csv`) | literature | **independent L1** chart-selection scoring | Saket 2019; Kim & Heer 2018 | **Yes** | cited literature values |
| Real briefs (`data/eval/real_briefs/items.jsonl`, `real_briefs_v1.jsonl`) | real public | independent eval (format/grounding/human) | none (no chart labels) | **Yes** | see `real_briefs_provenance.md` |
| Benchmark (`data/eval/benchmark_v1.jsonl`) | real_public + realistic_manual | **independent benchmark** (chart selection) | `literature_L1` / `manual_expert` via `task_crosswalk.yaml` | **Yes** | per-item `license_or_usage_note`; eval-only lock |
| Task crosswalk (`data/eval/task_crosswalk.yaml`) | expert/literature | benchmark `task_type` labeling | Amar/Eagan/Stasko 2005; Brehmer & Munzner 2013; Munzner 2014 | **Yes** | authored, project-internal |
| Guideline KB (`data/knowledge_base/guidelines/*.md`) | curated | RAG retrieval (grounding) | n/a | n/a | curated guideline text |
| Tableau Public Census (L3 realism) | external corpus | **pending** realism comparison | n/a | Yes | **not acquired** (OSF terms to confirm) |
| Human ratings (L4) | human | usefulness/quality (validity anchor) | human raters | Yes | **not collected yet** |

## Lineage rule
No source is used both for training/augmentation **and** as independent evaluation gold.
Training = synthetic generator (+ future augmentation). Independent gold = L1 literature,
real briefs, `benchmark_v1` (disjoint labeling), and human ratings. The synthetic **test**
split is internal-diagnostic only (circular for chart choice).

## Pending / to confirm
- Tableau Public Census corpus acquisition + license (L3).
- Expansion of real briefs to 30–40 with pinned provenance snapshots.
- Human ratings collection (L4).
