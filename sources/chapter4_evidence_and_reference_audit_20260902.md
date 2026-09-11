# Chapter 4 Evidence and Reference Audit

Audit date: 2026-09-02

Purpose: compact evidence pack for rewriting Chapter 4. It does not replace the frozen manifests, hashes, reports, or source code. If this file conflicts with a frozen artifact, the frozen artifact and a direct file check take priority, and the conflict must be reported.

## Evidence precedence

1. `src/config/data/dashboard_v4.yaml` for the operational dataset identity and active paths.
2. `data/frozen/dashboard_v4/manifest.json`, `hashes.json`, `dataset_card.md`, and `reports/` for the current materialized revision.
3. `data/frozen/dashboard_v3/manifest.json`, `hashes.json`, `dataset_card.md`, and `reports/` for the preserved source-grounded predecessor.
4. `data/raw_external/nvbench/source_manifest.json` and the local nvBench README for the external source registration.
5. `docs/datasets/DATASET_CONSTRUCTION_HISTORY_AND_METHODOLOGY.md` and implementation files for reconstruction and interpretation.
6. Existing thesis prose is a draft, not an authority for facts.

## Evidence ledger

The following IDs are stable within this audit and bind the project-specific claims in the manuscript to the underlying artifacts. “Automated/local check” means that the artifact was read or recomputed in the project workspace. Author or supervisor confirmation of each proposition remains pending; the IDs do not replace that human review.

| ID | Artifact and locator | Main claims supported |
| --- | --- | --- |
| E001 | `src/config/data/dashboard_v4.yaml`, keys `name`, `dataset_version`, `frozen_manifest_version`, `schema_version`, `frozen_dir`, and `not_for_training` | Operational `dashboard_v4` identity, exact `dashboard_v4_1` revision, schema, paths, and evaluation exclusions |
| E002 | `data/frozen/dashboard_v4/manifest.json` and `data/frozen/dashboard_v4/dataset_card.md`, manifest keys `dataset_version`, `parent_dataset_version`, `status`, `counts`, `lineage`, `repair_scope`, `checks`, and `reports` | v4.1 revision, counts, mixed lineage, repair/protected fields, release checks, and package description |
| E003 | `data/frozen/dashboard_v4/hashes.json`, `files.*.sha256` | Current v4 file hashes and hash algorithm |
| E004 | `data/frozen/dashboard_v4/reports/validation_report.json`, keys `counts`, `schema_invalid_count`, `semantic_invalid_count`, and `checks` | Final v4 validation status and zero invalid generated records |
| E005 | `data/frozen/dashboard_v4/reports/repair_report.json`, keys `repaired`, `rejected_or_regenerated`, `repaired_field_counts`, `most_common_original_problems`, and `protected_fields` | v4.1 repair model, scope, counts, and pre-repair issue categories |
| E006 | `data/frozen/dashboard_v4/reports/semantic_audit_before.json` and `semantic_audit_after.json`, keys `records`, `records_with_issues`, `issue_counts`, and `field_issue_counts` | Structural-versus-semantic audit findings before and after repair |
| E007 | `data/frozen/dashboard_v4/reports/duplicate_report.json`, keys `near_duplicate_threshold`, `all_train_val`, and `generated` | Duplicate and near-duplicate controls |
| E008 | `data/frozen/dashboard_v4/reports/leakage_report.json`, keys `generation_input_policy`, `generated_test_overlap`, `generated_human_eval_overlap`, and `dashboard_v3_unchanged` | Test and human-evaluation isolation during augmentation |
| E009 | `data/frozen/dashboard_v4/reports/distribution_report.json`, keys `generated`, `final_train`, and `final_val` | Record-level and mapping-level distribution counts |
| E010 | `data/frozen/dashboard_v3/manifest.json` and `data/frozen/dashboard_v3/dataset_card.md`, manifest keys `counts`, `source_grounded_fields`, `deterministically_derived_fields`, `llm_generated_enrichment_fields`, `enrichment_generation`, `human_enrichment_r1`, `validation`, and `checks` | v3 composition, field lineage, enrichment settings, R1 pilot, isolation checks, and package description |
| E011 | `data/frozen/dashboard_v3/hashes.json`, `files.*.sha256` | Preserved v3 artifact hashes and the direct human-evaluation hash |
| E012 | `data/frozen/dashboard_v3/reports/validation_report.json`, `leakage_report.json`, and `human_enrichment_r1.csv` | v3 validation, leakage boundary, and enrichment pilot evidence |
| E013 | `data/raw_external/nvbench/source_manifest.json`, keys `source_revision`, `license`, and `dataset_stats` | Repository, branch status, archive digest, license evidence, and local/published source counts |
| E014 | `data/raw_external/nvbench/extracted/nvBench-main/README.md`, `License` section | Upstream MIT license statement |
| E015 | `data/staging/dashboard_v4/run_20260814T010356Z/reports/generation_report.json`, top-level generation and `base_train_val_profile`/`generated_profile` keys | v4 generation settings, attempts, acceptance, split allocation, and coverage counts |
| E016 | `docs/datasets/DATASET_CONSTRUCTION_HISTORY_AND_METHODOLOGY.md`, sections on pilots, quality tiers, enrichment, v4 generation, repair, and limitations | Procedural reconstruction for details not represented compactly in the manifests |
| E017 | `src/data_pipeline/builders/nvbench_builder.py`, `src/data_pipeline/nvbench_quality.py`, `src/data_pipeline/nvbench_large_v2.py`, `src/config/data/nvbench_mapping.yaml`, and `src/config/data/nvbench_quality_rules.yaml` | Source-faithful extraction, quality rules, identifiers, chart/task mapping, selection, and group-safe splitting |
| E018 | `src/config/data/dashboard_v3.yaml`, `src/data_pipeline/enrichment.py`, and `src/data_pipeline/enrichment_full.py` | v3 enrichment scope, immutable projection, field-level lineage, validation, retry, and freeze preparation |
| E019 | `experiments/scripts/generate_dashboard_v4.py` and `experiments/scripts/repair_dashboard_v4_semantics.py` | v4 generation and v4.1 repair procedures, protected fields, and split restrictions |
| E020 | `docs/thesis/references.bib` plus the linked arXiv, ACL Anthology, and Crossref records listed below | Bibliographic metadata, DOI/URL checks, and the author-year citations used in the chapter |

## Artifact-backed project facts

### Source registration

- Source: original nvBench repository, `https://github.com/TsinghuaDatabaseGroup/nvBench`.
- Downloaded reference: `main`; no upstream commit was pinned.
- Local archive: `nvBench-main.zip`.
- Archive SHA-256: `2c95244aca93aaca689fc954f8ae228c6c17fd47c81e1d7b265c4191cb012e4c`.
- Local license evidence: MIT statement in the extracted nvBench README; source manifest status `confirmed`.
- Local source manifest counts: 7,247 top-level visualization objects and 25,762 natural-language query records.
- Published nvBench dataset count: 25,750 natural-language/visualization pairs, 750 tables, and 105 domains.
- The difference between 25,762 local query records and 25,750 published pairs must be stated as a source-count distinction. It must not be silently normalized.
- nvBench 2.0 was inspected separately but is not part of the final `dashboard_v3` or `dashboard_v4` lineage.

### Source-grounded `dashboard_v3`

- Modeling records: 1,819.
- Train: 1,281; validation: 264; held-out test: 274.
- Unique source groups: 784 train, 167 validation, 167 test.
- Split and selection seed: 42.
- Final source-grounded chart distribution: 1,379 bar, 109 line, 242 pie, 13 scatter, and 76 stacked bar records.
- Test chart distribution: 208 bar, 36 pie, 17 line, 11 stacked bar, and 2 scatter records.
- The test is in-domain and source-group-disjoint. It is not an external benchmark.
- Character 3-gram Jaccard near-duplicate threshold: 0.8.
- Final Tier-A admission threshold: quality score at least 90 and no mandatory failure.
- Quality-score weights: source fidelity 30, KPI validity 20, chart suitability 25, constraint completeness 15, and database-profile support 10.
- Final quality-pool rebuild: 21,244 technically valid records; 12,147 Tier A, 9,064 Tier B, and 33 Tier C.
- Earlier strict v3 candidate report: 20,986 technically accepted and 4,776 rejected. The repository does not contain a complete row-level reconciliation for the later increase of 258 records. This is a documented rebuild boundary.

### v3 constrained presentation enrichment

- Only 1,545 train/validation records were enriched: 1,281 train and 264 validation.
- Six writable fields: `users`, `context_summary`, `layout`, `styling`, `interactions`, and `rationales`.
- Source-backed analytical content, identifiers, split, KPI/chart mapping, encoding, constraints, and provenance remained immutable.
- Recorded model: `deepseek-v4-flash-sovereign`.
- Recorded temperature: 0.0.
- `xhigh` is the requested/configured reasoning effort. Do not claim that the provider independently proved an internal reasoning level.
- Technical sample: 10/10 accepted.
- Pilot: 29/30 accepted; one rationale disagreement was rejected.
- Full first pass: 1,534 accepted and 11 rejected. Offline revalidation resolved ten; one targeted retry resolved the last item. Final reconciled count: 1,545/1,545.
- Human R1 gate: 29/30 accepted. This is a small enrichment quality gate, not full expert annotation.
- Test and human-evaluation items were not enriched.

### Controlled `dashboard_v4` augmentation

- Purpose: broaden task/chart coverage while preserving the v3 test boundary.
- Generation model: `gpt-5.6-luna`.
- Generation mode: `codex_agent`.
- Generation specification: `dashboard_v4-generation-v1`.
- Seed: 42; batch size: 40; batches: 50.
- Candidate attempts: 2,915; accepted: 2,000; rejected: 915.
- The generation report records all 915 rejections as near-duplicate rejections at the 0.8 threshold.
- Generated train: 1,651; generated validation: 349; generated test: 0.
- Generated records are `source=llm_generated` and `not_gold=true`.
- Generated feature coverage: 1,215 records with filters, 1,373 with grouping, 1,229 with temporal information, and 842 with multiple KPI mappings.
- Generated records cover nine primary task families and 14 chart types.
- Use `data/staging/dashboard_v4/run_20260814T010356Z/reports/generation_report.json` for record-level primary task/chart counts. The frozen `distribution_report.json` also counts mapping instances and can sum above 2,000 because multi-KPI records contain more than one mapping.

### `dashboard_v4_1` semantic repair

- Operational package/configuration identity: `dashboard_v4`.
- Exact frozen manifest revision inside that package: `dashboard_v4_1`.
- Parent release identity: `dashboard_v4`.
- Repair version: `dashboard_v4_1-semantic-repair-v1`.
- Repair model: `gpt-5.6-luna`.
- Repair mode: `codex_agent_context_aware`.
- Repaired fields: `users`, `context_summary`, `layout`, `styling`, `interactions`, and `rationales`.
- Protected fields: `goals`, `kpis`, `columns`, `constraints`, `task_type`, `chart_type`, and `encoding`.
- Repaired records: 2,000; rejected or regenerated: 0.
- `users` changed in 1,819 records. Each of the other five presentation fields changed in all 2,000 records.
- Final audit: zero schema-invalid and zero semantic-invalid generated records under the repository rules.
- “Semantically clean” means passing the repository audit. It does not mean independent expert certification.

### Final composition

| Partition or artifact | Preserved nvBench-derived v3 | AI-generated v4/v4.1 | Total | Role |
| --- | ---: | ---: | ---: | --- |
| Train | 1,281 | 1,651 | 2,932 | Trainable |
| Validation | 264 | 349 | 613 | Validation/model selection only |
| Held-out test | 274 | 0 | 274 | Evaluation only |
| Modeling total | 1,819 | 2,000 | 3,819 | A/B/C/D study |
| Human-evaluation item file | 40 test-derived items | 0 | 40 | Separate evaluation instrument |

- The 40 human-evaluation rows are selected from the 274 test records. They are not 40 additional unique modeling examples and must not be added to 3,819 as if they formed an independent split.
- No completed final human ratings are evidenced by the dataset artifacts. The 29/30 result belongs to the v3 enrichment pilot, not to the final 40-item comparative human study.
- The final dataset has mixed provenance. The 1,545 preserved v3 train/validation records have source-grounded analytical fields and six LLM-generated presentation fields. The 2,000 added records are AI-generated supervision, including their analytical specifications, and later received a six-field semantic repair. The 274 test records remain source-grounded nvBench lineage.

## Arithmetic and consistency checks

The following checks were recomputed from the frozen manifests and the final split table:

- `1,281 + 264 + 274 = 1,819` — pass.
- `1,651 + 349 = 2,000` — pass.
- `1,281 + 1,651 = 2,932` — pass.
- `264 + 349 = 613` — pass.
- `2,932 + 613 + 274 = 3,819` — pass.
- `2,932 + 613 = 3,545` train-plus-validation records — pass.
- The 40 human-evaluation rows are a subset of the 274 test records; an independent file check found 40 overlapping item IDs and no additional modeling records.
- Direct SHA-256 checks found zero mismatches for six listed v3 files and 14 listed v4 files. The conflicting nested v4 manifest value for the human-evaluation hash is retained as an audit issue and is not used in the manuscript.

## Claims intentionally omitted from the manuscript

The chapter does not claim that the generated records are human or expert gold, that the final 40-item human-evaluation study has produced ratings, that the held-out test is an external benchmark, or that the v4 task/chart distribution represents real dashboard distributions. It also does not quote the stale nested human-evaluation hash, state that the nvBench branch was commit-pinned, or treat temperature 0.0 as a guarantee of provider-independent reproducibility. These omissions follow directly from the evidence boundaries above.

## Human-verification status

The current audit provides automated/local evidence bindings and independently checked bibliographic links. The accountable author or supervisor must still open the cited project artifacts, confirm the locators and interpretations, review the final English prose, and approve the chapter before it is treated as part of the submitted thesis.

## Direct hash verification performed by the audit

Direct `Get-FileHash -Algorithm SHA256` checks on 2026-09-02 matched the hashes files for all current train, validation, test, and human-evaluation artifacts. This is an automated/local check; author or supervisor confirmation remains pending.

| File | Records/rows | SHA-256 |
| --- | ---: | --- |
| `data/frozen/dashboard_v3/train.jsonl` | 1,281 | `202b28dd673e4937b5d50b154dae96e0cae7bdbe9eb6aca06b76f9f489a1d648` |
| `data/frozen/dashboard_v3/val.jsonl` | 264 | `dd16d3a2d02c9f4d1e4e84e547272116d87224366c3000deea299ae88fde3456` |
| `data/frozen/dashboard_v3/test.jsonl` | 274 | `e2df055d0a75c25f53a88cb830a5b6d66411fa179413f352f45ca2f6873829d5` |
| `data/frozen/dashboard_v3/human_eval_test_items_40.csv` | 40 | `2c336ee26398e8e487991df7e1c6e31385787c555228962d77e811f9a920cb8a` |
| `data/frozen/dashboard_v4/train.jsonl` | 2,932 | `c618488d92016e6c67925c2cbf021fb860a9e37ad6322107d535413f0bca84b1` |
| `data/frozen/dashboard_v4/val.jsonl` | 613 | `78847c3fe42a0a7221ead9b8970cfa17e51b4d9e79e103e4c4f15961e6462356` |
| `data/frozen/dashboard_v4/test.jsonl` | 274 | `e2df055d0a75c25f53a88cb830a5b6d66411fa179413f352f45ca2f6873829d5` |
| `data/frozen/dashboard_v4/human_eval_test_items_40.csv` | 40 | `2c336ee26398e8e487991df7e1c6e31385787c555228962d77e811f9a920cb8a` |

Known conflict: `data/frozen/dashboard_v4/manifest.json` contains a nested `base_dashboard_v3_sha256.dashboard_v3_human_eval` value beginning with `fa64...`, while the actual v3/v4 files and both `hashes.json` files resolve to `2c336e...`. Do not quote the conflicting nested manifest value as authoritative. Report the conflict in the writing audit if exact human-evaluation hashes are discussed. Do not mutate a frozen release silently.

## Figure warning

`docs/project/dataset_overview.png` describes a historical synthetic 100-brief dataset and includes chart types that do not describe the final `dashboard_v4_1` package. It must not be used as a figure for the final Chapter 4.

## Core scholarly references checked by the audit

The metadata below was checked against arXiv, ACL Anthology, or Crossref on 2026-09-02. The links and metadata are suitable for author review, but final human verification is still pending.

1. Luo, Y., Tang, J., and Li, G. (2021). *nvBench: A Large-Scale Synthesized Dataset for Cross-Domain Natural Language to Visualization Task*. arXiv:2112.12926. https://arxiv.org/abs/2112.12926
2. Luo, Y., Tang, N., Li, G., Chai, C., Li, W., and Qin, X. (2021). *Synthesizing Natural Language to Visualization (NL2VIS) Benchmarks from NL2SQL Benchmarks*. SIGMOD 2021, 1235–1247. https://doi.org/10.1145/3448016.3457261
3. Yu, T. et al. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*. EMNLP 2018, 3911–3921. https://aclanthology.org/D18-1425/
4. Gebru, T. et al. (2021). *Datasheets for Datasets*. Communications of the ACM, 64(12), 86–92. https://doi.org/10.1145/3458723
5. Pushkarna, M., Zaldivar, A., and Kjartansson, O. (2022). *Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI*. FAccT 2022, 1776–1826. https://doi.org/10.1145/3531146.3533231
6. Lee, K. et al. (2022). *Deduplicating Training Data Makes Language Models Better*. ACL 2022, 8424–8445. https://aclanthology.org/2022.acl-long.577/
7. Wang, Y. et al. (2023). *Self-Instruct: Aligning Language Models with Self-Generated Instructions*. ACL 2023, 13484–13508. https://aclanthology.org/2023.acl-long.754/
8. Ribeiro, M. T., Wu, T., Guestrin, C., and Singh, S. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. ACL 2020, 4902–4912. https://aclanthology.org/2020.acl-main.442/
9. van der Lee, C. et al. (2019). *Best Practices for the Human Evaluation of Automatically Generated Text*. INLG 2019, 355–368. https://doi.org/10.18653/v1/W19-8643

The corresponding BibTeX keys already exist in `docs/thesis/references.bib`: `luo2021nvbenchdataset`, `luo2021nvbenchsigmod`, `yu2018spider`, `gebru2021datasheets`, `pushkarna2022datacards`, `lee2022dedup`, `wang2023selfinstruct`, `ribeiro2020checklist`, and `vanderlee2019human`. The DOI for `vanderlee2019human` was added after verification against ACL Anthology.

Visualization references already verified for Chapters 2 and 3 may be reused only where they directly support the distinction between source fidelity and design suitability: `cleveland1984graphical`, `mackinlay1986apt`, `brehmer2013typology`, `saket2019taskbased`, `kim2018taskdata`, and `moritz2019draco`.

## Chapter 4 citation cross-check

The chapter uses the author-year style established in Chapters 2 and 3. Its in-text citations map to the following BibTeX keys: `brehmer2013typology`, `cleveland1984graphical`, `gebru2021datasheets`, `kim2018taskdata`, `lee2022dedup`, `luo2021nvbenchsigmod`, `luo2021nvbenchdataset`, `luo2025nvbench2`, `mackinlay1986apt`, `moritz2019draco`, `pushkarna2022datacards`, `ribeiro2020checklist`, `saket2019taskbased`, `vanderlee2019human`, `wang2023selfinstruct`, and `yu2018spider`. No cited work is missing from `docs/thesis/references.bib`, and the chapter’s reference block contains only these cited works.

The repository citation helper parsed all 77 BibTeX entries with zero syntax errors and zero duplicate keys. Its manuscript cross-check reported zero detected citations because it recognizes LaTeX commands and Pandoc `@key` citations, whereas this thesis draft follows the requested plain author-year style. The author-year-to-BibTeX mapping above was therefore checked separately; the helper’s unused-reference warnings concern the other chapters’ bibliography entries, not missing references in Chapter 4.

Independent Crossref lookups returned HTTP 200 and matched the title and publication year for the 15 DOI-bearing works used in this chapter. The nvBench dataset record was checked against arXiv, and the Spider, deduplication, Self-Instruct, CheckList, and human-evaluation records were checked against ACL Anthology. This confirms that the links in the chapter’s reference block are resolvable primary-source links; final author verification remains pending.
