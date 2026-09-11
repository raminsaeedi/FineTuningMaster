# Chapter 5 Evidence and Reference Audit

Date: 2026-09-07

This file records the evidence used for the revised Chapter 5 manuscript. It separates project facts from claims supported by external literature. The audit is an AI-assisted pre-check; it is not a substitute for a final human verification of every source and for the supervisor's approval of the interpretation.

## 1. Scope and writing decisions

The chapter describes the implemented system and the planned comparison of Methods A–D. It does not report experiment results. Project-specific statements are checked against source code, YAML configuration, frozen manifests, or evaluation documentation. General methodological statements are linked to primary papers or official specifications. The chapter uses the repository's author-year citation style and keeps the prose in simple scientific English.

The previous draft contained literal LaTeX citation commands with doubled backslashes, mixed German words, an outdated OLMo-based model description, and several placeholder references. These elements were removed from the new manuscript. The final matrix is described from the current configuration, and no placeholder reference is cited in Chapter 5.

## 2. Project evidence ledger

| Evidence ID | Source and locator | Fact checked | Used in |
| --- | --- | --- | --- |
| E001 | <code>src/config/matrix/final.yaml</code>, lines 13–50 | Dataset identity, final model keys, smoke model, available seeds, method order, and C-to-D adapter dependency | Sections 5.1 and 5.3 |
| E002 | Model YAML files under <code>src/config/model</code>, lines 4–13 | Model identifiers, nominal sizes, context lengths, dtypes, revision fields, token requirements, and chat-template settings | Section 5.3 |
| E003 | <code>src/core/schemas.py</code>, lines 21–125 | Task and chart vocabularies, dashboard brief fields, output fields, and generation-result fields | Section 5.2 |
| E004 | <code>src/core/prompts.py</code>, lines 19–93 | Shared system prompt, six output keys, allowed vocabularies, field instructions, and chat messages | Section 5.4 |
| E005 | <code>src/data_pipeline/formatter.py</code>, lines 14–73 | Reuse of the prompt builder for training, JSON target serialization, EOS handling, and prompt-completion support | Sections 5.4 and 5.7 |
| E006 | Method YAML files under <code>src/config/method</code> | Method names, generation settings, TF-IDF top-k, 4-bit inference settings, and D's adapter source | Sections 5.1, 5.3, 5.4, 5.5, 5.8, and 5.10 |
| E007 | <code>src/config/training/qlora_default.yaml</code>, lines 3–45 | QLoRA rank, alpha, dropout, all-linear target, NF4, double quantization, SFT schedule, validation, and full-sequence loss | Section 5.7 |
| E008 | <code>src/training/sft_trainer.py</code>, lines 78–98, 166–257, 269–391, and 516–523 | DoRA/RSLoRA flags, k-bit preparation, PEFT attachment, validation dataset use, best-checkpoint metadata, and adapter persistence | Sections 5.7 and 5.9 |
| E009 | <code>src/methods/base.py</code>, lines 65–365; <code>src/methods/ft_rag.py</code>, lines 24–30 | Shared HF method path, retrieval query fields, context fitting, result fields, and D adapter validation | Sections 5.4, 5.6, 5.8, and 5.10 |
| E010 | <code>src/retrievers/tfidf.py</code>, lines 23–60; <code>data/knowledge_base/kb_manifest.json</code>, lines 2–35 | TF-IDF implementation, positive-score filtering, three documents, 41 chunks, chunking rule, and knowledge-base hash | Section 5.6 |
| E011 | <code>src/inference/postprocess.py</code>, lines 1–188; <code>src/evaluation/metrics/schema_compliance.py</code>, lines 1–128; <code>src/core/strict_response_schema.py</code>, lines 1–84 | JSON extraction, limited normalization, raw strict validation, completeness, and strict response schema | Sections 5.4, 5.10, and 5.11 |
| E012 | <code>src/evaluation/metrics/grounding.py</code>, lines 1–147; <code>src/evaluation/metrics/retrieval_relevance.py</code>, lines 1–170 | Lexical-proxy and optional semantic grounding, retrieved-document requirements, and qrels-gated retrieval metrics | Sections 5.6 and 5.11 |
| E013 | <code>src/evaluation/reporting.py</code>, lines 1–15 and 250–300; <code>docs/evaluation/evaluation_protocol.md</code>, lines 1–155 | Internal-circular label, pending L1/L3/L4 layers, robustness caveats, and reporting boundaries | Section 5.11 and Section 5.13 |
| E014 | <code>src/config/eval/full.yaml</code>; <code>src/evaluation/metrics/topk_accuracy.py</code>, lines 1–120 | Configured metric families, top-3 support threshold, and treatment of parse failures | Section 5.11 |
| E015 | <code>data/frozen/dashboard_v4/manifest.json</code>, lines 2–70; <code>data/frozen/dashboard_v4/hashes.json</code>, lines 2–32 | Frozen dataset version, counts, generated-not-gold flag, protected fields, byte-identity checks, and hashes | Sections 5.1, 5.7, 5.12, and 5.13 |
| E016 | <code>src/utils/adapter.py</code>, <code>src/utils/config_hash.py</code>, and <code>src/utils/seed.py</code> | Adapter compatibility checks, configuration-hash identity, seed recording, and the documented CUDA seeding limitation | Sections 5.3, 5.7, and 5.12 |

## 3. Claim-to-evidence ledger

| Claim ID | Manuscript claim | Evidence | Status |
| --- | --- | --- | --- |
| C001 | Methods A–D form a two-factor comparison of retrieval and task adaptation | E001, E006, E009 | Supported by implementation |
| C002 | The project maps a dashboard brief to six structured output fields | E003, E004 | Supported by implementation |
| C003 | The final model profiles are Qwen3 1.7B/8B/14B and Llama 3.1 8B; Qwen2.5 0.5B is smoke-only | E001, E002 | Supported by current configuration |
| C004 | The common generation settings are 512 new tokens, temperature 0.1, top-p 0.9, sampling, and repetition penalty 1.15 | E006 | Supported by current configuration |
| C005 | RAG uses three frozen Markdown sources, 41 chunks, TF-IDF, positive scores, and top-k 3 | E006, E009, E010 | Supported by implementation and manifest |
| C006 | RAG queries use users, goals, and KPIs and omit the target recommendation | E009 | Supported by implementation |
| C007 | QLoRA uses 4-bit NF4, double quantization, rank 16, alpha 32, all-linear targets, and the stated SFT settings | E007, E008 | Supported by current configuration and trainer |
| C008 | Validation is loaded and used for epoch evaluation and best-checkpoint selection | E007, E008 | Supported by configuration and trainer |
| C009 | Method D reuses and validates the compatible C adapter for the same model, dataset, and seed | E001, E006, E009, E016 | Supported by implementation |
| C010 | DoRA and RSLoRA are optional trainer/configuration paths; GaLore is a separate full-model path | E008 and the corresponding files under <code>src/config/training</code> | Supported by repository structure |
| C011 | Raw output is retained and strict schema metrics are computed from the raw object rather than lenient repairs | E011 | Supported by implementation |
| C012 | Grounding defaults to a lexical proxy and semantic grounding is opt-in; retrieval relevance needs qrels | E012 | Supported by implementation |
| C013 | Synthetic chart agreement is labelled internal-circular and human-effectiveness/usefulness layers are pending | E013, E014 | Supported by evaluation protocol |
| C014 | The frozen dataset has 2,932 training, 613 validation, 274 test, and 40 human-evaluation items | E015 | Supported by manifest |
| C015 | The current model revisions are null and therefore not immutable upstream commits | E002 | Supported by current configuration; limitation retained |

## 4. External reference audit

The following references are cited in Chapter 5. The listed links were checked against primary proceedings, official specifications, or the authors' archival paper pages. The same works are present in <code>docs/thesis/references.bib</code>; the Qwen3 entry was added as <code>yang2025qwen3</code>, and the ACL URL was added to <code>ribeiro2020checklist</code>.

| Reference | Source checked | Use in Chapter 5 |
| --- | --- | --- |
| Vaswani et al. (2017) | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Autoregressive Transformer background |
| Wei et al. (2022) | [arXiv:2109.01652](https://arxiv.org/abs/2109.01652) | Instruction-tuning motivation |
| Mackinlay (1986) | [DOI:10.1145/22949.22950](https://doi.org/10.1145/22949.22950) | Data, task, and graphical-encoding relation |
| Munzner (2014) | [CRC Press record](https://doi.org/10.1201/b17511) | Visualization-analysis framing |
| Kim and Heer (2018) | [DOI:10.1111/cgf.13409](https://doi.org/10.1111/cgf.13409) | Independent effectiveness layer |
| Saket et al. (2019) | [DOI:10.1109/TVCG.2018.2829750](https://doi.org/10.1109/TVCG.2018.2829750) | Independent effectiveness layer |
| Hu et al. (2022) | [OpenReview record](https://openreview.net/forum?id=nZeVKeeFYf9) | LoRA definition |
| Dettmers et al. (2023) | [NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html) | QLoRA definition and resource rationale |
| Lewis et al. (2020) | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | RAG definition |
| Zhang et al. (2023) | [arXiv:2303.10512](https://arxiv.org/abs/2303.10512) | AdaLoRA alternative |
| Liu et al. (2024) | [PMLR record](https://proceedings.mlr.press/v235/liu24bn.html) | DoRA alternative |
| Kalajdzievski (2023) | [arXiv:2312.03732](https://arxiv.org/abs/2312.03732) | RSLoRA alternative |
| Zhao et al. (2024) | [PMLR record](https://proceedings.mlr.press/v235/zhao24s.html) | GaLore alternative |
| Geng et al. (2025) | [arXiv:2501.10868](https://arxiv.org/abs/2501.10868) | Structured-output and constrained-decoding caveat |
| JSON Schema (2022) | [Official Draft 2020-12 specification](https://json-schema.org/draft/2020-12) | General schema-validation reference |
| Ribeiro et al. (2020) | [ACL record](https://aclanthology.org/2020.acl-main.442/) | Behavioral robustness testing |
| van der Lee et al. (2019) | [ACL record](https://aclanthology.org/W19-8643/) | Human-evaluation reporting |
| Yang et al. (2025) | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) | Qwen3 family reference |

## 5. Citation risks deliberately avoided

The old Chapter 5 cited several entries with placeholder authors or incomplete metadata, including <code>StructuredInference2024</code>, <code>FedRAG2026</code>, <code>ControlTokenDPR2024</code>, <code>SMELLM2025</code>, <code>FinLoRA2025</code>, <code>GreenCodeSummarization2025</code>, <code>VizQLoRA2023</code>, and <code>Lee2025FinetuneRAG</code>. None of these keys is cited in the revised chapter. The old Qwen2.5 key is also not used to describe Qwen3. The entries remain in the shared bibliography because they may belong to earlier project notes; removing them would affect material outside this chapter and was not necessary for the Chapter 5 rewrite.

The project-specific implementation record is treated as evidence for implementation facts, not as a substitute for peer-reviewed support of general scientific claims. The chapter does not claim that the local knowledge base is an independent gold standard, that the grounding metric proves faithfulness, or that automatic chart agreement proves dashboard usefulness.

## 6. Human verification still required before submission

Before the thesis is submitted, the author should open each primary source, confirm the bibliographic metadata and the exact claim locator, and verify that the final LaTeX bibliography style renders the links and author names correctly. The author should also record immutable model revisions for the final runs, confirm the actual executed seed coverage, and update the implementation-status statements if L1, L3, or L4 becomes available. These checks are intentionally visible rather than silently assumed.
