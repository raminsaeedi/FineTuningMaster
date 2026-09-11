# Evidence and Reference Audit — Chapters 1, 6, 7, 8, 9 (plus corrections to 2, 4, 5)

Date: 2026-09-11 (revised the same day, after the Qwen3-8B A and B re-runs arrived)
Dataset: `dashboard_v4` (frozen revision `dashboard_v4_1`)
Runs covered: **32** complete runs
Scope: the five newly written chapters, the corrections applied to the existing
chapters, the bibliography clean-up, and the re-analysis after the Qwen3-8B
matrix was completed.

This audit separates **project facts** (what was implemented and executed, taken
from repository artifacts) from **scientific claims** (why a choice is
defensible, taken from primary literature). It is an AI-assisted pre-check and
does not replace a final human verification of every source or the supervisor's
approval of the interpretation.

---

## 1. What changed in this pass

| Change | Files |
| --- | --- |
| Chapter 1 written | `docs/thesis/chapters/Chapter_1_Introduction.md` |
| Chapter 6 written | `docs/thesis/chapters/Chapter_6_Experimental_Setup_and_Evaluation_Protocol.md` |
| Chapter 7 written | `docs/thesis/chapters/Chapter_7_Results.md` |
| Chapter 8 written | `docs/thesis/chapters/Chapter_8_Discussion_Error_Analysis_and_Threats_to_Validity.md` |
| Chapter 9 written | `docs/thesis/chapters/Chapter_9_Conclusion_and_Future_Work.md` |
| Model matrix corrected (Llama 3.1 and Qwen3-14B removed, Qwen3.8-27B and OLMo 2 added and justified) | `Chapter_5_System_Design_and_Methods.md` §5.1, §5.3.1, §5.7.1, §5.13.3 |
| Knowledge-base identity corrected (executed hash vs repository hash) | `Chapter_5_...md` §5.6.1, Table 5.5 |
| Effective per-run training values distinguished from the configuration defaults | `Chapter_5_...md` §5.7.3 |
| Retriever justification extended with the alternatives actually considered | `Chapter_5_...md` §5.6.2 |
| Adaptation-algorithm comparison table added | `Chapter_2_Background.md` §2.5.3 |
| Rejected source corpora named with explicit criteria | `Chapter_4_...md` §4.2.1, Table 4.1 |
| All chapters converted from author-year prose citations to `\cite{key}` | all nine chapters |
| Per-chapter reference lists made generated rather than hand-written | `experiments/scripts/build_chapter_reference_lists.py` |
| 21 fabricated or placeholder bibliography entries removed; 19 verified entries added | `docs/thesis/references.bib` |
| Two chapter files renamed for consistency | `Chapter_4_...`, `Chapter_5_...` |
| `.gitignore` adjusted so the cited result artifacts can be committed | `.gitignore` |
| New analysis script producing the statistics the chapters cite | `experiments/scripts/analyze_final_results.py` |
| Revision after the Qwen3-8B A/B re-runs: matrix completed (30 to 32 runs), Chapters 6-9 rewritten against the new artifacts | `Chapter_6...`, `Chapter_7...`, `Chapter_8...`, `Chapter_9...` |
| Per-contrast code-state label added alongside the block-level label | `analyze_final_results.py`; reported in Ch. 6 section 6.5.1 and Ch. 7 Tables 7.5-7.6 |
| Top-3 agreement removed from the thesis entirely (see section 4.4) | all chapters |

---

## 2. Project evidence ledger

| ID | Source and locator | Fact checked | Used in |
| --- | --- | --- | --- |
| P001 | `experiments/outputs/final/dashboard_v4/*/*/seed_*/manifest.json` (32 files) | Model key, method key, seed, model identifier and revision, git commit, config hash, dataset hashes, knowledge-base identity, adapter metadata, hardware, duration, status | Ch. 6 §6.1–6.5, Ch. 7 all |
| P002 | `.../metrics_auto.json` (32 files) | JSON parse rate, strict schema validity, encoding-object rate, completeness, field coverage, top-1, top-3 support, macro-F1 and per-class F1, confusion matrix, robustness, grounding, latency | Ch. 7 §7.1–7.7 |
| P003 | `.../eval_per_item.jsonl` (32 files, 274 rows each, `variant = original`) | Per-item parse flag, schema flag, completeness, predicted and reference primary chart, top-1 outcome | Ch. 7 §7.1–7.2, Ch. 8 §8.3 |
| P004 | `.../predictions.jsonl` (32 files) | Raw model text, parse error, retrieved documents, prompt token counts, RAG truncation flag, per-item latency | Ch. 7 §7.2, §7.5; Ch. 8 §8.2–8.3 |
| P005 | `data/frozen/dashboard_v4/manifest.json` and `hashes.json` | Frozen revision `dashboard_v4_1`, split counts 2,932 / 613 / 274 / 40, `generated_not_gold`, protected fields, byte-identity checks, file digests | Ch. 6 §6.3, Ch. 7, Ch. 8 |
| P006 | `data/frozen/dashboard_v4/test.jsonl` | Reference primary chart and task distribution: bar 208, pie 36, line 17, stacked_bar 11, scatter 2 | Ch. 7 §7.2.4 |
| P007 | `src/core/strict_response_schema.py` + validation run over P006 | 0 of 274 reference records satisfy the strict schema; violations are `extra_forbidden` on 13 encoding keys | Ch. 6 §6.6.3, Ch. 7 §7.1, Ch. 8 §8.6 |
| P008 | `data/knowledge_base/kb_manifest.json`, `chunks.jsonl`, `guidelines/*.md` | 3 documents, 41 chunks, heading-based chunking, min 5 words; repository digest `cad3bb0c…`; CRLF→LF conversion reproduces the executed digest `19fbbfa4…` | Ch. 5 §5.6.1, Ch. 6 §6.3.1 |
| P009 | `data/knowledge_base/guidelines/chart_selection_guidelines.md`, heading "Donut / Pie Chart" | Verbatim guidance "Prefer donut over pie: Donut charts are easier to read because the center can show a total value." | Ch. 7 §7.2.3, Ch. 8 §8.4 |
| P010 | `src/data_pipeline/perturbations.py`, `experiments/scripts/build_perturbations_v3.py` | Paraphrase = fixed synonym table over free-text fields only; missing-info = drop constraints, last KPI, last column; deterministic, no LLM, no randomness | Ch. 6 §6.3.2 |
| P011 | `src/evaluation/metrics/robustness.py` | Definitions of paraphrase consistency, paraphrase accuracy and delta, clarification rate (regex detector), missing-info schema rate | Ch. 6 §6.6.8, Ch. 7 §7.4 |
| P012 | `src/evaluation/metrics/schema_compliance.py` | Three-level definition: `json_parse_rate`, `required_keys_rate`, `schema_validity_rate`; completeness requires non-empty | Ch. 6 §6.6.1–6.6.4 |
| P013 | `src/evaluation/stats/*` | Cochran's Q, exact McNemar, Holm correction, percentile bootstrap (10,000 resamples), Friedman, Wilcoxon, effect sizes | Ch. 6 §6.7 |
| P014 | `experiments/results/final/dashboard_v4/statistics/*` (generated in this pass) | Provenance table, aggregate table, paired tests, contrasts, error analysis, chart confusions, format-tolerant re-scoring, strict-schema ceiling | Ch. 7, Ch. 8 |
| P015 | `docs/evaluation/HPC_RESULTS_CONSOLIDATION_AUDIT.md` (2026-09-09) | The earlier Qwen3-8B A/B duplicate-ID defect, with log timestamps. Superseded for every reported table by the clean re-runs in P001-P004 | Ch. 6 §6.1.1, Ch. 8 §8.6 |
| P016 | `experiments/results/human_eval/` | Study item file and rater assignment exist; no `ratings/` content | Ch. 6 §6.8, Ch. 7 §7.8, Ch. 9 §9.3 |
| P017 | `docs/evaluation/human_eval_plan.md` | Rubric of six 1–5 dimensions, 40 × 4 × 3 design, six raters, blinding, manifest contents, planned analysis | Ch. 6 §6.8 |
| P018 | `src/config/matrix/final.yaml` | Still lists `qwen3_14b` and `llama3_1_8b`; conflicts with the executed matrix | Reported as an open decision, **not** used as a fact in any chapter |

### 2.1 Derived facts computed during this audit

All were computed by `experiments/scripts/analyze_final_results.py` or by direct
verification and are reproducible.

| Derived fact | Value | Method |
| --- | --- | --- |
| Completeness and identifier uniqueness | 32 of 32 runs pass all six checks; 274 unique ids in each of the three prediction files | direct count over `predictions*.jsonl` |
| Reference records passing the strict schema | 0 / 274 | Pydantic validation of `StrictDesignOutput` over `test.jsonl` |
| Recomputed JSON-object recovery vs reported `json_parse_rate` | identical in all 32 runs | `extract_json_dict` over stored raw text |
| Trainable parameters per adapter | 12,058,624 (OLMo) / 17,432,576 (1.7B) / 43,646,976 (8B) / 116,727,808 (27B) | `training_metadata.json` |
| Trainable share of total parameters | 0.81 % / 1.00 % / 0.53 % / 0.43 % | computed from the same file |
| Inference GPU wall-clock, 32 runs | about 162 h | sum of `duration_seconds` |
| Training GPU wall-clock, 8 adapters | ≈ 23 h | sum of `training_duration_seconds` |
| Code-state comparability, 8 model-by-seed blocks | 0 homogeneous, 3 mixed, 5 unknown | git hashes in P001 |
| Code-state comparability, 40 individual contrasts | 7 homogeneous, 14 mixed, 19 unknown | git hashes in P001, evaluated per pair |
| The homogeneous contrasts | B minus A at `qwen3_8b`/42, `qwen3_8_27b`/42 and `olmo2_1_49b`/42; D minus C at `qwen3_8b`/42, `qwen3_8_27b`/42, `olmo2_1_49b`/42 and `olmo2_1_49b`/43 | `method_contrasts.csv` |
| Truncated responses (outermost object never closed) | 0 (27B, all conditions) up to 274 / 274 (OLMo B) | suffix check over stored raw text |
| Pie-to-donut substitutions at 27B | 5 (A) against 28 (B) of 36 pie items | `chart_confusions.csv` |
| Pie-to-donut substitutions at 8B | 0 (A) against 28 (B), and 29 under D | `chart_confusions.csv` |
| Qwen3-8B truncated responses | 0 (A), 0 (B), 36 (C), 69 (D) | closing-brace check over stored raw text |
| Knowledge-base digest reconciliation | `sha256(local_bytes.replace(CRLF, LF))` = executed digest | direct computation |

---

## 3. Claim-to-evidence ledger for the new chapters

| ID | Claim | Evidence | Status |
| --- | --- | --- | --- |
| C101 | Thirty-two runs are complete under the six stated file checks | P001, P002 | Verified directly, including identifier uniqueness |
| C102 | An earlier Qwen3-8B A/B export was rejected for duplicated conflicting item ids, and both conditions were re-inferred from a single process | P015 for the defect; P001-P004 for the clean re-runs | Verified; the reported tables use only the re-runs |
| C102b | Every model has a complete A-D matrix at seed 42 | P001 | Verified |
| C103 | No runs exist for 8B/27B seeds 43–44 | P001 | Verified by absence |
| C104 | All runs consumed identical train/val/test digests | P001, P005 | Verified |
| C105 | Effective training batch is 8 for all adapters, reached differently per device | P001 | Verified |
| C106 | Max training sequence length was 1,024 for two profiles and 4,096 for two | P001 | Verified |
| C107 | Two Qwen3-1.7B adapters trained in fp16 on a Tesla T4 | P001 | Verified |
| C108 | Strict schema validity has a construction ceiling of 0 % | P007 | Verified by direct validation |
| C109 | Pipeline top-1 conflates chart decision, contract validity and truncation | P003, P004, P014 | Verified by the three-level re-scoring |
| C110 | Text-level C − A is +0.4 to +10.6 pp, significant in 6 of 8 blocks | P014 (`method_contrasts.csv`) | Verified |
| C111 | Text-level D − C is negative in 11 of 12 blocks, significant in 8 | P014 | Verified |
| C112 | The 27B B minus A effect of -8.76 pp is dominated by +23 pie-to-donut substitutions, and the 8B effect of -9.49 pp by +28 | P002, P009, P014 | Verified at both scales; both contrasts are code-state homogeneous; the causal reading is bounded in the text |
| C112b | The Qwen3-8B C minus A contrast changes sign between the pipeline metric (-7.66 pp) and the text level (+4.38 pp) | P014 | Verified |
| C112c | Fine-tuning lowers Qwen3-8B macro-F1 from 0.558 to 0.530 while raising text-level agreement; per-class F1 falls on `line` (1.00 to 0.67) and `stacked_bar` (0.59 to 0.15) | P002 | Verified |
| C113 | OLMo truncation is a budget artifact, not malformed output | P004 | Verified by inspecting stored prefixes and the closing-brace check |
| C114 | Every fine-tuned condition produced zero items with more than one distinct recommendation, while Qwen3-8B A and B produced 274 and Qwen3.8-27B A and B produced 266 and 267 | P002 | Verified |
| C116 | Clarification rate is 0.0 % in every fine-tuned condition | P002 | Verified |
| C117 | Adapter pairing C→D verified for all 8 D runs (6 by field, 2 by path + config hash) | P001 | Verified with the stated distinction |
| C118 | No human ratings exist | P016 | Verified by absence |
| C119 | The independent effectiveness scorer is not implemented | P003 (`l1_covered = not_applicable` everywhere) | Verified |
| C120 | Retrieval relevance is unavailable for want of qrels | `src/config/eval/full.yaml`, P002 | Verified |
| C121 | Latency is confounded by five GPU types | P001 | Verified; the text restricts the interpretation accordingly |
| C122 | Seven of forty individual contrasts are code-state homogeneous, and no C minus A contrast is among them | P001, P014 | Verified |

---

## 4. External reference audit

Every reference below was checked against a primary source during this pass.
Metadata was taken from the publisher page, the ACL Anthology entry, the arXiv
abstract page, or the official model card — not from a search snippet.

### 4.1 References added in this pass

| Key | Source verified at | What it supports | Verified metadata |
| --- | --- | --- | --- |
| `sarikaya2019dashboards` | [10.1109/TVCG.2018.2864903](https://doi.org/10.1109/TVCG.2018.2864903), dblp record | Breadth of the dashboard design space | TVCG 25(1), 682–692, 2019 |
| `olmo2025furious` | [arXiv:2501.00656](https://arxiv.org/abs/2501.00656) | OLMo 2 family, open training data/code/checkpoints | Full author list confirmed from the arXiv listing |
| `allenai2025olmo2instruct1b` | [model card](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) and its `config.json` | Checkpoint identity, Apache-2.0, SFT→DPO→RLVR pipeline, MHA (16/16 heads), 4,096 positions, hidden 2,048, 16 layers, vocab 100,352 | Confirmed |
| `lambert2025tulu3` | [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) | Tülu 3 post-training recipe referenced by the OLMo card | Confirmed |
| `qwen2026qwen38` | [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) citation block | The citation the model card itself asks for | Reproduced verbatim from the card |
| `qwen2026qwen3827bcard` | same page | 27B parameters, 64 layers, hidden 5,120, hybrid gated-DeltaNet / gated-attention stack, 262,144 native context, Apache-2.0, revision `1d4bf0f2…` | Confirmed |
| `mcnemar1947` | [10.1007/BF02295996](https://doi.org/10.1007/BF02295996) | Paired binary test | Psychometrika 12(2), 153–157 |
| `cochran1950` | [10.1093/biomet/37.3-4.256](https://doi.org/10.1093/biomet/37.3-4.256) | Omnibus test for matched binary outcomes | Biometrika 37(3-4), 256–266 |
| `holm1979` | Scandinavian Journal of Statistics record | Family-wise error control | 6(2), 65–70 |
| `efron1979bootstrap` | [10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552) | Bootstrap confidence intervals | Ann. Statist. 7(1), 1–26 |
| `wilcoxon1945` | [10.2307/3001968](https://doi.org/10.2307/3001968) | Planned ordinal paired test | Biometrics Bulletin 1(6), 80–83 |
| `friedman1937` | [10.1080/01621459.1937.10503522](https://doi.org/10.1080/01621459.1937.10503522) | Planned omnibus ordinal test | JASA 32(200), 675–701 |
| `krippendorff2018content` | SAGE catalogue entry | Planned inter-rater reliability measure | 4th ed., 2018, ISBN 9781506395661 |
| `sparckjones1972idf` | [10.1108/eb026526](https://doi.org/10.1108/eb026526) | Justification of IDF term weighting | J. Documentation 28(1), 11–21 |
| `robertson2009bm25` | [10.1561/1500000019](https://doi.org/10.1561/1500000019) | BM25 as a considered alternative | FnTIR 3(4), 333–389 |
| `karpukhin2020dpr` | [ACL Anthology 2020.emnlp-main.550](https://aclanthology.org/2020.emnlp-main.550/) | Dense retrieval as a considered alternative | EMNLP 2020, 6769–6781 |
| `tam2024formatrestrictions` | [ACL Anthology 2024.emnlp-industry.91](https://aclanthology.org/2024.emnlp-industry.91/) | Format restrictions change model output | EMNLP 2024 Industry Track, 1218–1236; BibTeX taken from the Anthology page |
| `sclar2024formatspread` | [arXiv:2310.11324](https://arxiv.org/abs/2310.11324), ICLR 2024 proceedings page | Sensitivity to superficial prompt format | ICLR 2024 |
| `willard2023guided` | [arXiv:2307.09702](https://arxiv.org/abs/2307.09702) | Constrained decoding as an alternative to training for format compliance | Confirmed |
| `bui2025seeds` | [arXiv:2503.07329](https://arxiv.org/abs/2503.07329) | Random seeds cause measurable variation in fine-tuned LLMs | **Author correction**: the previous entry credited "Zhou, Hao"; the actual first author is **Nghia Bui**. Accepted at IJCNLP 2025 |

### 4.2 Entries removed as unverifiable or fabricated

The following 21 entries were deleted from `references.bib`. Fourteen of them
carried placeholder authors of the form `{Author, A. and others}`, which is a
reliable signature of a generated citation. None of them was cited by any
chapter, so no text needed rewriting.

`MethodsRecord2026`, `StructuredInference2024`, `AdversarialStructured2020`,
`FlexibleStructured2020`, `Pauk2026`, `Pinecone2023Chunking`, `FedRAG2026`,
`ControlTokenDPR2024`, `SMELLM2025`, `FinLoRA2025`,
`GreenCodeSummarization2025`, `VizQLoRA2023`, `Lee2025FinetuneRAG`,
`LLMGuidelines2026`, `GreenAIReview2024`, `GreenAIEnsembles2024`,
`AllenAI_OLMo2`, `AllenAI_OLMo`, `AllenAI_OLMo2_1B`, `An2024Qwen2.5`,
`Zhou2025Seeds`.

The last five were replaced by correctly attributed entries
(`olmo2025furious`, `allenai2025olmo2instruct1b`, `bui2025seeds`); the Qwen2.5
entry was dropped because the smoke-test model is not part of the reported
comparison and is therefore not cited.

### 4.3 Bibliography audit result

| Check | Result |
| --- | --- |
| Total entries | 77 |
| Distinct `\cite` keys used across the nine chapters | 77 |
| Total `\cite` occurrences | 218 |
| Cite keys missing from the bibliography | 0 |
| Bibliography entries never cited | 0 |
| Duplicate citation keys | 0 |
| Duplicate DOIs | 0 |
| Leftover author-year prose citations | 0 |

Reproduce with:

```bash
python experiments/scripts/build_chapter_reference_lists.py --check
```

---

### 4.4 Scope decision recorded in this pass

**Top-3 agreement was removed from the thesis.** After the Qwen3-8B re-runs,
`top_3_valid` is true in exactly two of thirty-two runs (`qwen3_8b` A and B,
with support rates of 94.89 % and 96.72 %) and false everywhere else, including
every fine-tuned condition of every model. A metric available for two runs and
for no trained condition cannot carry an A/B/C/D comparison, and reporting it as
a fragment would invite a question the evidence cannot answer. Following the
metric-selection rule that a metric without a clear interpretation may be
omitted, it was removed from Chapters 5, 6, 7, 8 and 9. The underlying
behavioural observation - that every fine-tuned condition stops emitting
alternatives - is retained in Ch. 7 section 7.2.5 and Ch. 8 section 8.3.4,
because it is a property of the systems rather than an artifact of the metric.

## 5. Conflicts found between project sources

Rule: where two project sources disagree, the newer machine-readable run
artifact wins, and the conflict is reported rather than silently resolved.

| Conflict | Resolution in the text | Still open |
| --- | --- | --- |
| `src/config/matrix/final.yaml` lists `qwen3_14b` and `llama3_1_8b`; no runs exist for either | The chapters describe the executed matrix only. The config file was **not** modified | Yes — needs a decision about whether the approved plan changed (see `docs/thesis/OPEN_ITEMS_AND_TODO.md` §5.2) |
| Knowledge-base digest differs between repository and runs | Both reported; cause identified as CRLF vs LF; the executed value is authoritative | No — cause proven |
| `qlora_default.yaml` states batch 2 × accumulation 4 and sequence length 4,096; several runs used 1 × 8 and 1,024 | Chapter 6 reports the per-run manifest values and states the discrepancy | No |
| Older per-item records lack `json_object_extracted` and `encoding_object_valid` | Both were recomputed uniformly from the stored raw text; the recomputed JSON figure matches the reported `json_parse_rate` in all 32 runs | No |
| `docs/thesis/proposal/thesis_outline.md` describes a Qwen2.5-0.5B synthetic-generator study | Treated as an outdated document and not used as an authority | No |

---

## 6. Facts that still require the author's confirmation

1. **The approved experiment plan.** Does it still require Qwen3-14B and Llama 3.1 8B? The answer decides whether the thesis is complete or whether a scope change must be documented.
2. **The AI-use declaration.** Tools, purposes, whether the university requires a formal statement, and where it must appear. Nothing was written for this.
3. **The human-evaluation raters.** Whether six independent raters are available, or whether a pilot with fewer raters is the realistic plan.
4. **The Qwen3.8-27B model card details.** The parameter count, revision and architecture were verified from the public model card in September 2026. If the supervisor requires a peer-reviewed source rather than a vendor card, no such source currently exists for this model, and the citation would need a note.
5. **Thesis language and template.** All chapters are in English. If the university template requires a different chapter numbering, section style or a mandatory "Chapter Summary", the files will need adjusting.

---

## 7. Reviewer-style self-check performed on the new chapters

| Question | Answer |
| --- | --- |
| Is every factual project claim traceable to an artifact? | Yes; the ledger above lists the locator for each |
| Is every scientific claim cited, and does the source support it? | Yes; each added reference was opened and checked, not matched by title |
| Are results confined to Chapter 7? | Yes; Chapters 1, 5 and 6 state hypotheses, protocol and expected contrasts only |
| Are unfinished experiments described as unfinished? | Yes; Ch. 6 §6.1.1, Ch. 7 §7.8, Ch. 9 §9.3 |
| Is LLM-generated supervision described transparently? | Yes; Ch. 4 and repeated in Ch. 6, 8 |
| Are seeds described according to real coverage? | Yes; three seeds for two models, one seed for two, never pooled |
| Are Llama and Qwen3-14B removed from the methodology? | Yes; verified by grep across all chapters |
| Are abbreviations defined on first use? | KPI, LLM, PEFT, LoRA, QLoRA, RAG, NF4, SFT, DPO, RLVR, MHA, GQA — checked |
| Are the tables internally consistent with the artifacts? | Yes; every numeric cell of Chapter 7 Tables 7.1, 7.3, 7.4, 7.5, 7.6, 7.7 and 7.11 was checked programmatically against the artifacts (32 + 32 + 16 + 32 + 32 + 32 + 16 values) with zero mismatches, plus spot checks on latency, robustness and per-class F1 |
| Is there duplicated text across chapters? | Overlap is limited to deliberate one-sentence recalls with a forward or backward reference |
| Are limitations stated without weakening the work unnecessarily? | Limitations are concentrated in Ch. 6 §6.6, Ch. 8 §8.6–8.7 and Ch. 9 §9.3 rather than repeated everywhere |
| Is the removal of a planned metric disclosed? | Yes; section 4.4 of this audit records the decision and its reason. The thesis itself does not discuss the metric, which is the intended outcome of the scope decision |
| What is the hardest question an examiner could ask? | "Your headline metric was wrong — why should I trust the rest?" Chapter 8 §8.2 answers it directly: the pre-specified metric is reported unchanged, the decomposition is labelled post-hoc, and it was only possible because raw outputs were retained. The second hardest is "nineteen of forty contrasts have an unrecorded code state, and none of your fine-tuning contrasts is clean" — answered in §8.7.1 by stating the limitation rather than defending it, and partly mitigated by the fact that the two contrasts carrying the main retrieval finding are homogeneous |
