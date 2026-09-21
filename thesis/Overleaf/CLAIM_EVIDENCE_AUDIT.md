# Claim–Evidence Audit

| Thesis claim | Location | Primary project evidence | Scientific support | Status |
|---|---|---|---|---|
| The executed matrix contains 32 complete runs. | 6.1 | Final run manifests and completeness audit under `experiments/outputs/final/dashboard_v4` | Artifact fact | VERIFIED |
| The evaluated profiles are Qwen3-1.7B, Qwen3-8B, Qwen3.8-27B, and OLMo-2-0425-1B-Instruct. | 6.2 | Run manifests and resolved configurations | Official model reports/cards | VERIFIED |
| Seed coverage is 42/43/44 for Qwen3-1.7B and OLMo, and seed 42 for Qwen3-8B and Qwen3.8-27B. | 6.1 | Final run identities and manifests | Random-seed motivation cited in Chapter 6 | VERIFIED |
| The frozen split contains 2,932 train, 613 validation, 274 test, and 40 human-pool records. | 4.8–4.10 | `data/frozen/dashboard_v4/manifest.json` | Dataset-source references in Chapter 4 | VERIFIED |
| Dashboard-level fields are LLM-generated enrichment, not expert gold. | 4.7; 8.9.1 | Builder, repair report, field provenance, and `not_gold` markers | Dataset-documentation literature | VERIFIED |
| Retrieval uses three project documents split into 41 deterministic chunks. | 5.6; 6.3 | Knowledge-base manifest, chunk file, and run hashes | RAG references in Chapters 2–5 | VERIFIED |
| QLoRA uses rank 16, alpha 32, dropout 0.05, NF4 double quantization, three epochs, and learning rate 2e-4. | 5.7; 6.4 | Training configurations and adapter manifests | Hu et al. (2022); Dettmers et al. (2023) | VERIFIED |
| Text-level C–A is positive in all eight available blocks (+0.36 to +10.58 percentage points) and significant in seven. | 7.3; 8.3 | Paired-statistics outputs and per-item outcomes | Statistical procedures in Chapters 6–7 | VERIFIED |
| B–A is slightly positive at 1.7B, neutral for OLMo chart choice, and negative at 8B/27B. | 7.3; 8.2 | Paired statistics and per-item predictions | Interpretation bounded by the RAG literature | VERIFIED |
| Most larger-model retrieval disagreements are pie-to-donut substitutions linked to a guideline/reference conflict. | 7.3; 8.2 | Confusion counts, retrieved chunks, and 28/36 item trace | Task-dependent chart-choice literature | VERIFIED |
| D–C is negative in 11 of 12 blocks and significant in eight; the 27B exception is +0.73 percentage points and non-significant. | 7.3; 8.3 | Paired-statistics output | Statistical procedures in Chapter 6 | VERIFIED |
| The strict reference schema has a 0/274 construction ceiling. | 7.1; 8.3 | Strict-schema ceiling audit | Implementation diagnostic | VERIFIED |
| The final human study contains 81 pages for 32 outputs, ten complete sessions, and 11 missing dimension judgements. | 6.8; 7.8; 8.9.5 | Frozen `openv1` export, final descriptive analysis, and input hashes | van der Lee et al. (2019); Krippendorff (2018) | VERIFIED |
| Retrieval-only has the highest equal-brief means for chart appropriateness, rationale quality, and overall usefulness in the exploratory sample. | 7.8; 8.4; 9.2 | Frozen ratings export and equal-brief aggregation | Descriptive human evidence; no inferential claim | VERIFIED |
| Submission date, supervisor, industry tutor, declaration, and AI-use disclosure are populated. | Front matter | Researcher-provided metadata and final frontmatter | Not applicable | VERIFIED |

## Audit rules

- Machine-generated artifacts override planning prose where they conflict.
- Historical metric names do not license an independent quality claim; the thesis uses “agreement with the frozen reference.”
- Post-hoc scoring layers are labelled diagnostic.
- Single-seed profiles receive no seed-level uncertainty estimate.
- Latency remains descriptive because hardware differs.
- Human findings remain exploratory and sample-bounded; no population-level, expert-validity, rendered-dashboard, or business-impact claim is made.
