# Methods Implementation and Reproducibility Record

## Scope, evidence policy, and status vocabulary

This record documents the implemented methods, data contracts, training procedure, retrieval pipeline, inference path, evaluation safeguards, provenance, and reproducibility state of the current Master Thesis repository. It is an implementation record, not a results chapter. It reports what the repository actually defines or stores. It does not infer completed experiments from configuration files, smoke tests, partial output folders, or historical prose.

The inspection basis was the repository source tree, Hydra and OmegaConf configurations, frozen dataset manifests and hashes, knowledge-base manifests, run manifests, saved adapter metadata, existing reports, tests, and the repository bibliography. The record is current as of 2026-08-19. No training run, inference run, API call, data-generation job, or Git operation was performed while preparing this record.

The evidence hierarchy used here is:

1. Executed code and machine-readable manifests define behavior and provenance.
2. Current configuration files define intended operational settings.
3. Frozen data and knowledge-base manifests define the materialized inputs.
4. Tests provide executable architectural checks.
5. Documentation and README files explain context, but older prose is superseded where it conflicts with current code or manifests.

Status labels have the following meaning:

| Label                                | Meaning                                                                                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Current and authoritative            | Used by the current final matrix or current frozen input path.                                                                                    |
| Implemented and verified             | Implemented in code and covered by a completed smoke or unit/integration artifact. This is not equivalent to a completed final thesis experiment. |
| Implemented, not final-authoritative | Present and usable in code, but not selected by the current final matrix.                                                                         |
| Legacy or superseded                 | Historically used or documented, but inconsistent with the current final dataset, model matrix, or output layout.                                 |
| Planned or pending                   | Configured or described, but no completed artifact was found.                                                                                     |

The central reproducibility distinction is therefore between a method being implemented, a pipeline being smoke-verified, and a final multi-seed result being completed. The repository currently demonstrates the first two for all four method families, but it does not contain completed final-matrix coverage for all requested models, methods, and seeds.

## 5.2 Structured Output Schema

### 5.2.1 Contract and semantic purpose

The system maps a dashboard brief to a structured design recommendation. The output contract is represented by Pydantic models in src/core/schemas.py, summarized by the generated schema artifact at data/frozen/dashboard_v4/schema.json, and referenced by the common prompt in src/core/prompts.py.

The top-level recommendation object is DesignOutput. It contains six top-level fields:

| Field             | Type             | Intended content                                                            | Runtime behavior                                               |
| ----------------- | ---------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| context_summary   | object           | Users, goals, data context, and other contextual interpretation             | Default is an empty object; additional keys are allowed.       |
| kpi_chart_mapping | array of objects | One or more KPI-to-chart recommendations                                    | Each item is typed by KPIChartMapping; extra keys are allowed. |
| layout            | object           | Spatial organization, hierarchy, placement, and dashboard composition       | Default is an empty object; additional keys are allowed.       |
| styling           | object           | Visual style, scales, colors, typography, and accessibility-related choices | Default is an empty object; additional keys are allowed.       |
| interactions      | array            | Filters, cross-filtering, drill-down, tooltips, and related behavior        | Default is an empty array.                                     |
| rationales        | array of objects | Claims and design principles supporting recommendations                     | Each item is typed by Rationale; extra keys are allowed.       |

The required recommendation entry has the following fields:

| Field        | Type        | Constraint                                                                         |
| ------------ | ----------- | ---------------------------------------------------------------------------------- |
| kpi          | string      | KPI or analytical objective being addressed.                                       |
| task_type    | enum string | One of the nine controlled task types.                                             |
| chart_type   | enum string | One of the seventeen controlled chart types.                                       |
| alternatives | array       | Optional alternative chart recommendations; values are normalized when possible.   |
| encoding     | object      | Data-to-visual encoding such as dimensions, measures, axes, color, or aggregation. |

The nine task types are trend, comparison, composition, distribution, correlation, ranking, deviation, part_to_whole, and flow.

The seventeen chart types are line, bar, stacked_bar, grouped_bar, area, pie, donut, scatter, heatmap, histogram, box, kpi_card, table, gauge, sankey, treemap, and map.

The rationale object contains claim and principle string fields. The implementation preserves extra fields because design outputs may carry useful method-specific detail without invalidating the core contract. This is intentional extensibility, not permission to omit the core top-level structure during evaluation.

### 5.2.2 Dataset record and validation layers

The output contract is distinct from the dataset record contract. A GoldItem contains:

| Field          | Requiredness | Role                                                              |
| -------------- | ------------ | ----------------------------------------------------------------- |
| item_id        | Required     | Stable join key for training, inference, caching, and evaluation. |
| brief          | Required     | Structured dashboard request represented by DashboardBrief.       |
| recommendation | Required     | Reference DesignOutput.                                           |
| split          | Optional     | Dataset partition metadata.                                       |

DashboardBrief stores item_id, users, goals, kpis, columns, constraints, and optional extra. The common prompt currently renders users, goals, KPIs, columns, and constraints. It does not separately render item_id or the extra dictionary. The recommendation target remains the complete reference object.

The repository uses several validation layers with different purposes:

1. JSON extraction checks whether a model response contains an object that can be decoded.
2. Lenient post-processing normalizes common enum spellings and converts simple representation errors into the canonical internal form.
3. Pydantic parsing creates a DesignOutput or records a schema_error.
4. Strict evaluation checks the raw extracted JSON object for required top-level keys, full schema validity, non-empty required content, field coverage, and valid raw enum values.
5. Frozen-dataset validation checks the GoldItem wrapper, reference completeness, duplicate identifiers, duplicate content, and leakage constraints.

The distinction between parser acceptance and strict metric validity is necessary. DesignOutput fields have defaults, and its model configuration allows extra fields. Consequently, direct Pydantic parsing can produce an object with empty defaults when a raw response omits fields. The strict evaluator separately checks the raw JSON object and required-key presence so that a parser convenience does not become an inflated schema-compliance result.

data/frozen/dashboard_v4/schema.json records GoldItem as the dataset-level JSON Schema. It requires item_id, brief, and recommendation. Within recommendation, the mapping requires task_type and chart_type; the core objects otherwise retain the permissive additional-property behavior. The artifact also records schema_version: GoldItem, enrichment_spec_version: phase3-enrichment-v1, and the required enrichment fields users, context_summary, layout, styling, interactions, and rationales.

The parser in src/inference/postprocess.py accepts raw JSON, fenced JSON objects, and brace-delimited objects. It then applies explicit aliases such as comparision to comparison, kpi card to kpi_card, and line graph to line. It drops mapping entries whose task or chart type cannot be normalized. It converts simple strings into objects or rationale claims where that can be done without inventing semantic content. The original raw text and parse error remain available in GenerationResult.

This design separates recoverable serialization variance from substantive design validity. That separation is scientifically important because malformed JSON, an unknown chart label, an omitted field, and a semantically poor chart choice are different failure modes. The repository reports them through different fields and metrics rather than collapsing them into one score.

### 5.2.3 Scientific motivation and challenges

The structured contract makes the model's design decision inspectable. It exposes the analytical task, chart type, alternatives, encoding, layout, styling, interaction, and rationale rather than treating the recommendation as an unscored prose paragraph. This is aligned with the repository's structured-output reference represented by BibTeX key geng2025structured and with the visualization-design literature represented by keys such as mackinlay1986apt and munzner2014visualization in docs/thesis/references.bib. Those references motivate the need for explicit analytical and visual decisions; this record makes no additional empirical claim about their effectiveness.

The main implementation challenge is that a strict contract can fail for syntactic reasons before the design content can be assessed. The two-stage parser and explicit strict metric layer address that challenge. A second challenge is that a permissive schema is useful for extensibility but can hide omissions. The repository therefore retains both the normalized object and the raw response and computes strict completeness from the raw object.

The schema is not a guarantee of design quality. A response can be valid JSON, use a valid enum, and still select an unsuitable chart. The repository's synthetic top-k and macro-F1 diagnostics must therefore be interpreted together with the pending independent human-effectiveness layer described in src/evaluation/reporting.py.

## 5.3 Common Prompting and Generation Pipeline

### 5.3.1 Shared architecture

Methods A and C use HFMethod in src/methods/base.py. Methods B and D use RAGHFMethod, which adds retrieval and a retrieval-conditioned system prompt. All four methods are registered through src/core/registry.py and expose the same method interface:

| Stage               | Implemented behavior                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Configuration       | Hydra composes model, method, dataset, training, and evaluation settings.                                                |
| Model setup         | src/models/hf_causal.py loads the configured Hugging Face causal language model and tokenizer.                           |
| Prompt construction | src/core/prompts.py builds one system message and one user message.                                                      |
| Chat formatting     | The model tokenizer applies its chat template; a text fallback exists when no template is available.                     |
| Generation          | The method calls the common model wrapper with the method's generation configuration.                                    |
| Parsing             | Raw text is decoded, extracted, normalized, and parsed into DesignOutput when possible.                                  |
| Recording           | GenerationResult stores the raw response, parsed output, error, latency, model, method, seed, and configuration hash.    |
| Evaluation          | Predictions are joined to references by item_id, reparsed with the current parser, and scored by the configured metrics. |
| Persistence         | Append-only JSONL predictions, error logs, run manifests, cache identity, and reports are written to the run directory.  |

This common path is the principal control mechanism for comparing A, B, C, and D. Method-specific changes are retrieval context and adapter loading; prompt content, generation parameters, parser, evaluator, and artifact format are shared unless a configuration explicitly overrides them.

### 5.3.2 System and user messages

The current system instruction in src/core/prompts.py is:

“You are an expert dashboard design consultant. Given a dashboard brief, you generate structured, professional design recommendations. Always respond with a single valid JSON object following the exact schema provided. Do not wrap the JSON in markdown fences. Do not add commentary outside the JSON object.”

The user message contains:

1. Users.
2. Goals.
3. KPIs.
4. Data columns.
5. Constraints.
6. A direct instruction to return only a valid JSON object.
7. The six required top-level keys and their expected types.
8. The mapping-entry fields.
9. The allowed task-type and chart-type vocabularies.
10. The rationale fields.
11. A compact JSON-shaped example.

The prompt builder is shared by inference and training. src/data_pipeline/formatter.py imports the same prompt construction logic, so the supervised target is conditioned on the same instruction format used at inference time. This minimizes a train–test prompt-template mismatch. The formatter serializes the gold recommendation as indented JSON with UTF-8 characters preserved and appends the tokenizer end-of-sequence token when available.

### 5.3.3 Model-family and tokenizer behavior

The model wrapper loads the tokenizer from the base model or from the adapter directory when an adapter is present. If the tokenizer lacks a padding token, the end-of-sequence token is used. Padding is right-sided. Input length is bounded by the selected model's max_seq_length.

Qwen3 configurations pass enable_thinking: false through the chat-template keyword path. The repository does not append a literal /think suffix. Llama 3.1 uses the generic chat-template path and requires an HF token only for model loading; the token is not written into artifacts. All current model revisions are null, so the model identifiers are recorded but an immutable Hub revision is not pinned.

### 5.3.4 Generation settings

The current method configurations use the following generation values unless a smoke configuration overrides them:

| Parameter          | Current value |
| ------------------ | ------------: |
| max_new_tokens     |          1024 |
| temperature        |           0.1 |
| top_p              |           0.9 |
| do_sample          |          true |
| repetition_penalty |          1.15 |

The smoke runs use a shorter generation budget and disable sampling for bounded end-to-end verification. Those settings are smoke settings, not final thesis settings.

The optional constrained-decoding path uses Outlines and a Pydantic-derived JSON schema when method.generate.constrained is enabled. The authoritative A–D final configurations do not enable constrained decoding. Therefore, the final methods rely on instruction-following, parsing, and strict evaluation rather than claiming hard decoding-time schema enforcement.

### 5.3.5 Parser, cache, and error behavior

For each item, the runner:

1. Loads the brief and constructs the method-specific prompt.
2. Retrieves context only for B and D.
3. Generates text through the common wrapper.
4. Extracts and parses JSON.
5. Stores the result as one JSONL record keyed by item_id.

The inference cache is append-only and item-keyed. A run can resume after a crash. The cache guard rejects stale configuration hashes and compares dataset, model, method, seed, training, inference, and knowledge-base identity. This is a reproducibility safeguard against mixing outputs generated under incompatible conditions.

Generation exceptions are caught per item and written to an errors\*.jsonl file with the item identifier, exception type, message, and traceback. The runner continues with remaining items and prints the error. This prevents silent disappearance of failures. However, failed items are not inserted into the successful prediction list, and current run-level metrics use the available GenerationResult records as their denominator. The error is therefore visible but the denominator policy is not fully explicit; this remains a limitation recorded later.

### 5.3.6 Scientific reasoning and fairness

The shared pipeline makes the four methods differ primarily in information access and parameter state:

| Method | Additional information            | Parameter state                              |
| ------ | --------------------------------- | -------------------------------------------- |
| A      | No retrieved guidelines           | Base model only.                             |
| B      | Retrieved local design guidelines | Base model only.                             |
| C      | No retrieved guidelines           | Base model plus a trained LoRA adapter.      |
| D      | Retrieved local design guidelines | Base model plus the corresponding C adapter. |

This factorization supports an interpretable comparison. It does not by itself prove causal attribution because model revisions, GPU precision, sampling, and incomplete seed coverage can affect outputs. Those controls and limitations are documented in Section 5.9.

## 5.4 Method A — Prompt-Only

### 5.4.1 Definition and implementation

Method A is registered as prompt_only and implemented by PromptOnlyMethod in src/methods/base.py. It loads the selected base model and tokenizer, builds the common system and user messages, generates under the shared generation settings, and parses the response. It does not load an adapter and does not instantiate a retriever.

The current final matrix maps method A to experiment label E01. The small Qwen2.5 configuration remains the designated smoke model. The default src/config/config.yaml also selects Qwen2.5, prompt-only, and a legacy run profile for convenience. That default is not the authority for the final four-model matrix; src/config/matrix/final.yaml is the current final-matrix authority.

### 5.4.2 Current authoritative model configurations

The final matrix lists four final models and one separate smoke model:

| Model key    | Hugging Face identifier          | Family    | Nominal size | Maximum sequence length | Configured inference dtype | Trust remote code | HF token | Thinking |
| ------------ | -------------------------------- | --------- | -----------: | ----------------------: | -------------------------- | ----------------- | -------- | -------- |
| qwen3_1_7b   | Qwen/Qwen3-1.7B                  | Qwen3     |         1.7B |                    4096 | bfloat16                   | true              | no       | disabled |
| qwen3_8b     | Qwen/Qwen3-8B                    | Qwen3     |           8B |                    4096 | bfloat16                   | true              | no       | disabled |
| qwen3_14b    | Qwen/Qwen3-14B                   | Qwen3     |          14B |                    4096 | bfloat16                   | true              | no       | disabled |
| llama3_1_8b  | meta-llama/Llama-3.1-8B-Instruct | Llama 3.1 |           8B |                    4096 | bfloat16                   | false             | yes      | disabled |
| qwen2_5_0_5b | Qwen/Qwen2.5-0.5B-Instruct       | Qwen2.5   |         0.5B |                    2048 | float16                    | true              | no       | disabled |

The first four are final_models in src/config/matrix/final.yaml. Qwen2.5-0.5B is the smoke_model. All model revisions are currently unset. The loader records the identifier and configuration, but exact immutable Hub revision reproducibility is not yet available.

### 5.4.3 Expected behavior and scientific limits

Prompt-only is the cleanest base condition for measuring what the pretrained instruction model can infer from the brief and the explicit output contract. It isolates the effect of retrieval and fine-tuning in the method matrix.

The method has no access to the local guideline corpus beyond the common prompt. It may still produce design knowledge learned during pretraining, but that knowledge is not traceable to a repository document. A valid JSON response is therefore not evidence that the recommendation is grounded in an explicit design source. The absence of a retriever also makes the grounding metric not applicable.

Prompt-only has the same sampling default as the other methods. Because do_sample is true, the default path is not bitwise deterministic even when the run records seed 42, 43, or 44. The inference entry point records the seed in results but does not explicitly reseed the inference RNG before each run. Exact stochastic reproducibility therefore requires an additional controlled seeding policy.

## 5.5 Method B — Retrieval-Augmented Generation

### 5.5.1 Motivation and scope

Method B is registered as rag and implemented by RAGMethod through RAGHFMethod. The method retrieves repository-local design guidance from a frozen knowledge base and inserts the selected passages into the system message before the shared user brief.

Retrieval augmentation is motivated by the repository's RAG reference lewis2020rag: external context can make the source of a recommendation more explicit and can reduce reliance on untraceable parametric memory. This is a methodological motivation, not a measured claim that the current corpus improves accuracy or reduces hallucination. Such a claim would require completed comparative experiments and an appropriate independent evaluation.

### 5.5.2 Authoritative knowledge base

The current final RAG configuration points to data/knowledge_base/chunks.jsonl and uses the TF-IDF retriever with top_k: 3.

| KB property         | Current value                                                              |
| ------------------- | -------------------------------------------------------------------------- |
| Source documents    | 3 Markdown documents                                                       |
| Materialized chunks | 41                                                                         |
| KB version          | 46b8575d98d37312                                                           |
| Chunk file SHA-256  | cad3bb0c1606fab3062d3235740d7701e1fb8ac46702af8b09f615f66bfc3afb           |
| Chunking split      | Markdown headings                                                          |
| Minimum chunk size  | More than 5 words; chunks at or below the threshold are dropped            |
| Chunk identifier    | Eight-character MD5 prefix derived from source content and heading context |
| Source ordering     | Sorted source files                                                        |
| Manifest            | data/knowledge_base/kb_manifest.json                                       |

The source documents are:

1. data/knowledge_base/guidelines/accessibility_guidelines.md
2. data/knowledge_base/guidelines/chart_selection_guidelines.md
3. data/knowledge_base/guidelines/dashboard_design_guidelines.md

The first document covers color use, contrast, labels, keyboard access, cognitive load, and accessible data tables. The second covers task-to-chart mappings such as trends to line charts, category comparisons to bar charts, composition to stacked bars, part-to-whole relationships to pie or donut charts, correlations to scatter plots, and constraints on category counts. The third covers dashboard hierarchy, KPI placement, filters, cross-filtering, drill-down, labels and scales, annotations, consistency, and grid alignment.

The manifest stores source byte sizes and SHA-256 hashes. It does not encode external URLs, DOIs, or a formal provenance statement for the guideline text. The corpus is therefore authoritative as a repository input, but its status as an independently sourced public or scientific reference cannot be established from the KB manifest alone.

### 5.5.3 Chunk construction

src/data_pipeline/kb_builder.py builds the materialized corpus as follows:

1. Enumerate Markdown source files under the configured guideline directory.
2. Sort the source file names.
3. Split at Markdown heading lines.
4. Keep the heading with the following body text.
5. Discard chunks whose word count is at or below the configured minimum.
6. Create a deterministic chunk identifier from source and heading content.
7. Write one JSON record per chunk.
8. Hash the resulting chunk file and source files.
9. Derive the KB version from the sorted source hash manifest.

This design favors deterministic provenance and heading-level interpretability. It also creates a retrieval granularity trade-off: very short guidance may be discarded, while long heading sections may contain multiple rules. The current repository does not implement an additional semantic splitting or passage reranking stage.

### 5.5.4 Retrieval algorithm

The authoritative retriever is src/retrievers/tfidf.py. It fits TfidfVectorizer(stop_words="english") over the 41 chunk texts, transforms a query built from the dashboard users, goals, and KPIs, computes cosine similarity, retains positive-score matches, and returns the top three chunks with their metadata and score.

The query intentionally omits the full serialized recommendation and focuses on the brief's user and analytical intent. This keeps retrieval independent of the model's generated output and prevents target leakage.

The retriever is deterministic conditional on the fixed chunk file, vectorizer implementation, query text, and stable ordering. No retrieval model weights or remote service are required. The returned documents are stored in each RAG GenerationResult so that the context used for a recommendation can be inspected after generation.

src/retrievers/dense.py is implemented as an optional alternative using sentence-transformers, with default model BAAI/bge-small-en-v1.5 and configuration in src/config/retriever/bge.yaml. It is not selected by the current final matrix. No BM25 or hybrid retriever implementation was found. Therefore, the authoritative description is TF-IDF, not dense or hybrid retrieval.

### 5.5.5 Prompt integration

When positive retrieval matches exist, the RAG method extends the base system prompt with:

1. A Relevant Design Guidelines delimiter.
2. Each retrieved passage formatted as an indexed entry containing source, heading, and text.
3. An End of Guidelines delimiter.

The original user message remains the same. If no positive-score passages are found, the method uses the base system prompt without an empty context block.

The retrieved context is guidance, not a hard constraint. The model can ignore it, misinterpret it, or generate a rationale that is only weakly related to the passages. The grounding metric checks claim–passage overlap as a claim-based diagnostic. With GROUNDING_SEMANTIC=1 and an available sentence-transformers encoder, it reports a semantic mode using a cosine threshold of 0.5. Otherwise it reports a lexical proxy using content-word overlap with threshold 0.2. The fallback is explicitly labeled lexical_proxy; it is not a faithfulness judge.

RAG generation latency is measured after retrieval in the current RAGHFMethod.generate implementation. Consequently, the stored latency_ms represents model generation and parsing timing after retrieval, not complete end-to-end retrieval-plus-generation latency. This matters for a fair systems comparison and remains an implementation limitation.

### 5.5.6 Scientific reasoning and challenges

The method tests whether explicit, retrievable design rules change structured recommendations and rationale content. The local corpus includes rules that can be compared to the output, which supports traceability analysis. It does not provide a complete theory of visualization effectiveness, and it does not replace independent human judgment.

The main challenge is distinguishing retrieval availability from retrieval usefulness. A retrieved passage can be relevant, irrelevant, redundant, or too general. The repository records retrieved text and scores but does not yet provide a human-judged relevance label, passage-level attribution assessment, or independent causal estimate of retrieval benefit. The experiment must therefore report retrieval and grounding diagnostics as method characterization, not as proof of improved visual-design quality.

## 5.6 Method C — QLoRA Fine-Tuning

### 5.6.1 Definition and implementation

Method C is registered as ft and implemented by FineTunedMethod. It trains a parameter-efficient LoRA adapter over the configured causal language model and then uses the base model plus that adapter during inference. The current final matrix maps it to E03.

The implementation is a supervised fine-tuning pipeline with 4-bit base-model loading and LoRA updates. The methodological rationale is consistent with the repository references hu2021lora for low-rank adaptation and dettmers2023qlora for quantized low-rank fine-tuning. The repository does not claim that its particular hyperparameters are optimal; they are the current reproducible configuration.

### 5.6.2 Training data and partition policy

The operational data configuration is src/config/data/dashboard_v4.yaml. It names the dataset dashboard_v4, points to the frozen directory data/frozen/dashboard_v4, and records frozen_manifest_version: dashboard_v4_1.

The current frozen counts are:

| Materialized input           | Count | Role                                                                                   |
| ---------------------------- | ----: | -------------------------------------------------------------------------------------- |
| train.jsonl                  | 2,932 | Training source consumed by the current training entry point.                          |
| val.jsonl                    |   613 | Configured validation partition; not currently loaded by experiments/scripts/train.py. |
| test.jsonl                   |   274 | Held-out evaluation source.                                                            |
| human_eval_test_items_40.csv |    40 | Human-evaluation selection; not a training source.                                     |
| test_paraphrased.jsonl       |   274 | Robustness variant; not a training source.                                             |
| test_missing_info.jsonl      |   274 | Robustness variant; not a training source.                                             |

The frozen manifest describes 2,000 generated records: 1,651 assigned to training and 349 to validation. It preserves the v3 train and validation records, producing 2,932 train and 613 validation items. The test set remains the v3 test set, and the human-evaluation CSV remains separate. The manifest reports zero generated test overlap and zero generated human-evaluation overlap.

The current training script loads only cfg.data.train_file through load_and_format_train_dataset. It does not load cfg.data.val_file and passes no evaluation dataset to the trainer. The validation path is therefore a declared data partition and a reproducibility input, but it is not evidence of an executed validation loop in the current training entry point. This distinction must be preserved in a thesis methods chapter.

The data policy is reinforced by src/config/data/dashboard_v4.yaml, whose not_for_training list includes test, human-evaluation, robustness, L1, and real-brief paths. The frozen validation and leakage reports provide additional checks. The implementation does not train on the test or human-evaluation files.

### 5.6.3 Training-example formatting

experiments/scripts/train.py obtains a tokenizer for the base model, loads the GoldItem records from the configured training file, and creates a dataset with one text field. src/data_pipeline/formatter.py formats each record as:

1. The shared system instruction.
2. The shared user prompt containing the brief fields and schema instructions.
3. The serialized gold recommendation.
4. The tokenizer end-of-sequence token when available.

The recommendation is serialized as pretty-printed JSON with indentation of two spaces and ensure_ascii=false. The text field is named text. The trainer uses the model-specific maximum sequence length: 2,048 for the Qwen2.5 smoke model and 4,096 for each current final model. The current trainer configuration disables sequence packing.

The target is the full recommendation object, not a single chart label. This allows the adapter to learn structured mapping, encoding, layout, styling, interactions, and rationales jointly. It also creates a harder optimization problem than single-label chart prediction. The formatter itself emits the complete text example without truncation or padding; the tokenizer/trainer applies the configured maximum length during SFT preparation. The model wrapper uses right padding and substitutes the EOS token as the pad token when needed. Long briefs or recommendations can approach the maximum sequence length, but the current implementation does not log a separate per-example truncation audit.

### 5.6.4 Authoritative QLoRA configuration

The current configuration is src/config/training/qlora_default.yaml:

| Component                                                  | Setting                              |
| ---------------------------------------------------------- | ------------------------------------ |
| Trainer type                                               | qlora_sft                            |
| LoRA rank                                                  | 16                                   |
| LoRA alpha                                                 | 32                                   |
| LoRA dropout                                               | 0.05                                 |
| LoRA bias                                                  | none                                 |
| Target modules                                             | all-linear                           |
| Base loading                                               | 4-bit                                |
| Quantization type                                          | NF4                                  |
| Double quantization                                        | true                                 |
| Configured quantization compute dtype                      | float16                              |
| Epochs                                                     | 3                                    |
| Per-device batch size                                      | 2                                    |
| Gradient accumulation                                      | 4                                    |
| Effective batch size per device before distributed scaling | 8 examples                           |
| Learning rate                                              | 2e-4                                 |
| Scheduler                                                  | cosine                               |
| Optimizer                                                  | adamw_torch                          |
| Warmup ratio                                               | 0.1                                  |
| Weight decay                                               | 0.01                                 |
| Maximum gradient norm                                      | 1                                    |
| Precision                                                  | auto                                 |
| Gradient checkpointing                                     | true                                 |
| Sequence packing                                           | false                                |
| Maximum length                                             | Model configuration's max_seq_length |
| Evaluation strategy                                        | no                                   |
| Logging interval                                           | Every 10 steps                       |
| Save interval                                              | Every 50 steps                       |
| Maximum retained checkpoints                               | 4                                    |
| Reporting                                                  | none                                 |

The string target all-linear makes the LoRA target selection model-agnostic and is covered by the architecture tests. It is the current authority. Older artifacts may contain explicit query/value/output module lists; those are historical configurations and must not be substituted for qlora_default.yaml.

src/training/sft_trainer.py constructs a 4-bit BitsAndBytesConfig, resolves a safe effective compute dtype through the repository precision helper, prepares the quantized model for k-bit training, applies a PEFT LoraConfig, and creates an SFT trainer. CUDA execution uses automatic device mapping and the trainer contains a CPU fallback path. The effective precision can therefore be constrained by visible hardware even though the configuration records a preferred quantization dtype. The run metadata records the effective training precision and hardware.

### 5.6.5 Adapter training and persistence

The training flow is:

1. Compose the resolved Hydra configuration.
2. Determine the adapter output path and any resume checkpoint.
3. Load and format the train split.
4. Write initial run metadata.
5. Set training seeds.
6. Load the base model and apply 4-bit preparation and LoRA.
7. Train with the configured SFT settings.
8. Reject non-finite logs or invalid checkpoint states through finite-value callbacks and resume validation.
9. Save adapter weights, adapter configuration, tokenizer, and training metadata.
10. Update the resume manifest and finalize the run manifest.

The adapter directory contains PEFT configuration and weights plus training_metadata.json. Metadata records the base model, model key, revision, model configuration hash, trainer type, quantization mode, LoRA settings, dataset version, train-file path and SHA-256, training configuration hash, maximum steps, batch and accumulation settings, optimizer and scheduler, maximum sequence length, effective precision, parameter counts, duration, experiment identity, and train metrics.

The adapter is not a full copy of the base model. It is valid only with a compatible base model and compatible training provenance. src/utils/adapter.py checks base model, model key, revision when available, model configuration hash, seed, dataset version, and training configuration hash. Fresh final-layout adapters are resolved under the dataset/model/method/seed hierarchy. Legacy profiles use a flat experiment-oriented path.

### 5.6.6 Scientific reasoning and challenges

Parameter-efficient adaptation is appropriate for comparing task-specific behavior while limiting trainable parameters and storage. It also makes the trained intervention inspectable: the base model and adapter provenance can be separated in the manifest.

The major threats to interpretation are data contamination, target formatting mismatch, and under-specified validation. The current repository addresses the first two with frozen split manifests, test/human separation, shared prompt formatting, and adapter compatibility checks. It does not yet establish validation-based checkpoint selection because the current training entry point does not pass val.jsonl to the trainer. The selected epoch count and final adapter should therefore be described as configuration-driven rather than validation-optimized unless a later corrected training run is produced.

## 5.7 Method D — Fine-Tuning + RAG

### 5.7.1 Definition

Method D is registered as ft_rag and implemented by FineTunedRAGMethod. It composes the two interventions:

1. Load the compatible C adapter for the same model, dataset, and seed.
2. Retrieve the same local guideline corpus with the same TF-IDF configuration as B.
3. Insert retrieved passages into the system prompt.
4. Generate and parse with the shared inference path.

The current final matrix maps D to E04 and declares adapter_from: C. The method does not train a second adapter during the D inference run.

### 5.7.2 Adapter provenance and compatibility

In the final/smoke layout, an adapter is resolved from the corresponding C path:

experiments/outputs/<profile>/<dataset>/<model>/C/seed\_<seed>/adapter

The D resolver enforces same-model and same-seed provenance. It also checks dataset version, training configuration hash, base-model identity, model configuration hash, and revision when present. run_final_matrix.py adds dependency checks for the C manifest, adapter files, source method, profile, model, dataset, and seed. If a compatible C adapter is unavailable, D is not a valid completed method instance.

Explicit adapter paths remain supported and take precedence. This is useful for controlled reuse but increases the risk of accidental cross-condition contamination. The final matrix's automatic dependency resolution is therefore the preferred reproducibility path.

### 5.7.3 Retrieval and prompt equality

D uses the same KB version, chunk file, source documents, TF-IDF vectorizer, query construction, top-k value, formatting delimiters, and generation configuration as B. The only intended additional intervention is the C adapter. This is important for attributing any difference between B and D to parameter adaptation rather than a changed corpus or retriever.

The C and D comparison is still conditional on the adapter having been trained with the same data and base model. The compatibility guards are necessary but cannot compensate for an unpinned upstream model revision or a missing final run manifest.

### 5.7.4 Scientific reasoning and challenges

D tests whether explicit guideline context remains useful after the model has been adapted to the dashboard-design task. It is a compositional condition, not a simple “best method” assumption. RAG may add useful context, duplicate learned information, distract the model, or change rationale wording without changing the primary chart choice.

Because D depends on C, a missing or incompatible C adapter is a dependency failure, not a D model result. The current repository contains completed smoke D artifacts for Qwen2.5-0.5B, but no completed final-matrix D artifacts for the four larger models and requested seeds. The distinction is material and is preserved in the coverage audit.

## 5.8 Inference Pipeline

### 5.8.1 Input selection and run preparation

src/pipeline/runner.py loads the configured test file and optional robustness variants. The current dashboard_v4 evaluation inputs are:

| Input                                           | Purpose                                                    |
| ----------------------------------------------- | ---------------------------------------------------------- |
| data/frozen/dashboard_v4/test.jsonl             | Primary held-out test evaluation.                          |
| data/eval/robustness_v4/test_paraphrased.jsonl  | Paraphrase stability and paraphrase accuracy diagnostic.   |
| data/eval/robustness_v4/test_missing_info.jsonl | Clarification and under-specification behavior diagnostic. |

The runner can cap items through max_samples; this is used by smoke configurations and must not be mistaken for full-test evaluation. It constructs references from GoldItem records and joins predictions by item_id.

### 5.8.2 Per-item generation

For each item:

1. The method builds the common prompt, optionally with RAG context.
2. The model wrapper applies the tokenizer chat template with a generation prompt.
3. Inputs are truncated to a safe budget derived from maximum sequence length minus the requested new-token budget.
4. Generation uses the configured sampling and repetition parameters.
5. Only newly generated token IDs are decoded.
6. The raw text is passed to JSON extraction and normalization.
7. A GenerationResult is written with model, method, seed, configuration hash, parse status, latency, and optional retrieved documents.

For adapter methods, the loader creates a PeftModel over the selected base model and checks that adapter files and metadata are finite and compatible. For RAG methods, retrieved passages are stored with the result. For A and C, retrieved_docs is absent and grounding is not applicable.

### 5.8.3 Parsing and re-parsing

The runner reloads predictions and reparses raw text with the current parser before evaluation. This permits parser improvements to be applied consistently to stored outputs. The raw response remains the basis for strict schema metrics, so improved normalization does not retroactively turn missing raw keys into complete JSON.

GenerationResult stores:

| Field family | Examples                                        |
| ------------ | ----------------------------------------------- |
| Identity     | item_id, method_name, model_name, seed, variant |
| Provenance   | config_hash                                     |
| Content      | raw_text, parsed                                |
| Failure      | parse_error                                     |
| Retrieval    | retrieved_docs                                  |
| Timing       | latency_ms                                      |

### 5.8.4 Error and cache behavior

The inference runner resumes from an append-only prediction cache when the requested item identifiers and cache identity match. It rejects stale identity or configuration hashes. Per-item exceptions are written to an errors file with traceback and do not halt the remaining items.

The current behavior is transparent but not denominator-complete: an item that raises an exception is absent from predictions.jsonl, while metrics generally use the result list. The errors file prevents silent omission, but a complete failure-aware metric policy would need to score missing predictions explicitly or report requested-item and successful-item denominators side by side.

### 5.8.5 Automatic metrics

The automatic evaluation code reports several distinct diagnostics:

| Metric family     | Meaning and limitation                                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Schema compliance | JSON parse rate, required-key presence, strict raw schema validity, non-empty completeness, and field coverage.                             |
| Top-k accuracy    | Primary and alternative chart agreement with the synthetic reference. Top-3 is reported only when support and distinctness conditions pass. |
| Macro-F1          | Fixed chart-type label comparison, with parse/empty outputs represented as (none).                                                          |
| Latency           | Mean, p50, and p95 from stored per-item latency values. RAG latency excludes retrieval time in the current method implementation.           |
| Grounding         | Claim-based supported or unsupported rationale rate for RAG results; semantic mode is opt-in, lexical mode is a proxy.                      |
| Robustness        | Paraphrase consistency, paraphrase accuracy and delta, missing-information clarification rate, and missing-information schema rate.         |

The reporting layer explicitly labels synthetic chart accuracy as internal-circular because the synthetic reference follows the generator's rule-based chart choice. It does not treat that metric as independent design effectiveness. The independent L1 human-effectiveness scorer is pending. L3 realism and L4 human ratings are also pending, as recorded by src/evaluation/reporting.py.

### 5.8.6 Robustness interpretation

Paraphrase consistency measures whether the primary chart remains unchanged for shared items. It must be read with paraphrase accuracy because a model can be stable and consistently wrong. Missing-information behavior uses a regular-expression detector for clarification or uncertainty language and separately reports the rate at which a full schema is emitted despite missing information. These are useful behavioral diagnostics, not semantic judges.

The grounding metric similarly requires caution. A lexical overlap rate can be influenced by generic vocabulary and does not establish that a rationale is faithful to the retrieved rule. A semantic score depends on an optional sentence encoder and threshold. Neither metric substitutes for human assessment of whether the recommendation is justified by the guideline.

## 5.9 Reproducibility

### 5.9.1 Environment and dependency control

The project declares Python version range >=3.11,<3.14 and pins the main environment in pyproject.toml and poetry.lock. Relevant pinned packages include:

| Package family   | Repository pin or declaration |
| ---------------- | ----------------------------- |
| PyTorch          | 2.6.0                         |
| Transformers     | 5.7.0                         |
| Hugging Face Hub | 1.13.0                        |
| Pydantic         | 2.11.4                        |
| Hydra Core       | 1.3.2                         |
| OmegaConf        | 2.3.0                         |
| NumPy            | 2.2.6                         |
| SciPy            | 1.16.1                        |
| scikit-learn     | 1.8.0                         |
| pandas           | 2.2.3                         |
| PEFT             | 0.19.1 in the training extras |
| TRL              | 1.3.0 in the training extras  |
| bitsandbytes     | 0.49.2 in the training extras |
| Accelerate       | 1.13.0 in the training extras |
| datasets         | 4.8.5 in the training extras  |

Run manifests record Python, PyTorch, CUDA, GPU, and package versions. One legacy full-size artifact records Python 3.12.4, PyTorch 2.6.0 with CUDA 12.4, and an NVIDIA RTX A1000 Laptop GPU. The current smoke artifacts include a CPU execution environment. These are artifact-specific observations, not a claim about the environment available for future final runs.

### 5.9.2 Configuration control

Hydra composes the configuration from src/config/config.yaml and its model, method, data, training, evaluation, and matrix groups. The resolved configuration is written to each run directory. src/utils/config_hash.py computes a SHA-256-derived configuration hash, stored with run metadata and cache identity.

The current final matrix is src/config/matrix/final.yaml:

| Matrix setting   | Current value                                               |
| ---------------- | ----------------------------------------------------------- |
| Dataset          | dashboard_v4 operational package                            |
| Frozen revision  | dashboard_v4_1                                              |
| Final models     | qwen3_1_7b, qwen3_8b, qwen3_14b, llama3_1_8b                |
| Smoke model      | qwen2_5_0_5b                                                |
| Methods          | A prompt-only, B RAG, C fine-tuning, D fine-tuning plus RAG |
| Seeds            | 42, 43, 44                                                  |
| Output root      | experiments/outputs/final                                   |
| Final run layout | Dataset, model, method, seed hierarchy                      |
| D dependency     | C adapter for the same model, dataset, and seed             |

The configuration name dashboard_v4 is the operational package identity. The frozen manifest and hashes identify the materialized revision as dashboard_v4_1. This is not an accidental interchange: the repository currently uses the package name in configuration and the repair/freeze revision in the manifest. Both identifiers must be reported together.

### 5.9.3 Dataset provenance and hashes

The current data lineage is:

1. A local nvBench v1 archive is recorded under data/raw_external/nvbench with source archive SHA-256 2c95244aca93aaca689fc954f8ae228c6c17fd47c81e1d7b265c4191cb012e4c.
2. The repository builder extracts and normalizes source records into the v3 split.
3. v3 preserves 1,281 train, 264 validation, and 274 test records, with a separate 40-item human-evaluation selection.
4. v4 augmentation adds 2,000 generated train/validation records only: 1,651 train and 349 validation.
5. v4.1 applies semantic repair to generated fields and freezes the resulting package.

The v4.1 manifest records the generated fields as AI-generated and not_gold=true. Protected fields include goals, KPIs, columns, constraints, task types, chart types, and encodings. Test and human-evaluation records are preserved unchanged according to the manifest reports.

The current file hashes in data/frozen/dashboard_v4/hashes.json are:

| File                         | SHA-256                                                          |      Bytes |
| ---------------------------- | ---------------------------------------------------------------- | ---------: |
| train.jsonl                  | aa1e7ec912ea155fc335bc6d93477ab37e9e4f8983d7f80f95d3e77fefd9feb4 | 23,618,651 |
| val.jsonl                    | 91a98f29226ef93e31d2617ce888622fc05ad8667392c0d361d12b9166e2cdbc |  4,927,924 |
| test.jsonl                   | e2df055d0a75c25f53a88cb830a5b6d66411fa179413f352f45ca2f6873829d5 |  1,254,865 |
| human_eval_test_items_40.csv | 2c336ee26398e8e487991df7e1c6e31385787c555228962d77e811f9a920cb8a |    451,967 |

The hashes artifact also records the current schema, manifest, dataset card, and report hashes. The exact hash values are the reproducibility anchors; file names alone are not sufficient.

The dataset provenance document cites nvBench v1 and records the archive hash but does not pin an upstream Git commit. The repository therefore has a local archive identifier rather than a source-commit identifier. data/raw_external/nvbench2 exists as a separate pending source and is not the current authoritative training source.

The provenance discussion is compatible with repository bibliography keys luo2021nvbench and luo2025nvbench2. The latter describes a source that is present as a pending alternative, not the current training dataset.

### 5.9.4 Model and adapter identity

The model identifiers are pinned in configuration, but revisions are null for all current model entries. A future final run must preserve the exact downloaded revision or commit in its manifest to achieve immutable model provenance.

Adapter reproducibility is stronger than manual path naming. Adapter metadata contains base-model and configuration hashes, dataset version, training configuration hash, seed, and training settings. D must consume the compatible C adapter under the same model and seed. The cache identity includes adapter-related provenance when applicable.

The model loader handles Qwen3 thinking settings, Llama token requirements, padding, dtype, and maximum sequence length in one shared path. This reduces family-specific divergence, but the different model families still have different tokenizer and architecture behavior. The final matrix must therefore retain model identity in every comparison and must not aggregate across models without model-stratified reporting.

### 5.9.5 Seed policy and actual coverage

The intended final seeds are 42, 43, and 44. src/utils/seed.py seeds Python's random module, NumPy when available, and PyTorch. The training script sets seeds after dataset formatting and before trainer execution. The implementation intentionally does not call torch.cuda.manual_seed_all; it relies on torch.manual_seed for the default CUDA generator. The seed is also stored in GenerationResult and run manifests.

The default final generation configuration uses sampling. The current inference entry point records the seed but does not explicitly call the training seed helper before each inference run. Exact stochastic replication is therefore not guaranteed by the seed field alone. Smoke configurations disable sampling, which improves bounded smoke repeatability but does not make full final runs deterministic.

The exact completed coverage found in the workspace is:

| Model        | Dataset and artifact class                            | Method A          | Method B                  | Method C          | Method D          | Interpretation                                             |
| ------------ | ----------------------------------------------------- | ----------------- | ------------------------- | ----------------- | ----------------- | ---------------------------------------------------------- |
| Qwen3 1.7B   | Current final profile, dashboard_v4, seeds 42/43/44   | None found        | None found                | None found        | None found        | Planned final coverage; no completed artifacts found.      |
| Qwen3 8B     | Current final profile, dashboard_v4, seeds 42/43/44   | None found        | None found                | None found        | None found        | Planned final coverage; no completed artifacts found.      |
| Qwen3 14B    | Current final profile, dashboard_v4, seeds 42/43/44   | None found        | None found                | None found        | None found        | Planned final coverage; no completed artifacts found.      |
| Llama 3.1 8B | Current final profile, dashboard_v4, seeds 42/43/44   | None found        | None found                | None found        | None found        | Planned final coverage; no completed artifacts found.      |
| Qwen2.5 0.5B | Legacy full-size dashboard_v3 artifacts               | Seed 43 completed | Seeds 42 and 43 completed | No completed run  | None found        | Historical v3 evidence; not current final matrix evidence. |
| Qwen2.5 0.5B | Current-layout dashboard_v4 smoke, seed 42, two items | Seed 42 completed | Seed 42 completed         | Seed 42 completed | Seed 42 completed | End-to-end smoke verification only.                        |
| Qwen2.5 0.5B | Current-layout dashboard_v3 smoke, seed 42, two items | Seed 42 completed | Seed 42 completed         | Seed 42 completed | Seed 42 completed | Historical smoke verification only.                        |

The legacy Qwen2.5 A seed-42 folder has a failed run manifest despite containing prediction and metric files, so it is not counted as completed. The legacy C seed-42 folder is incomplete; C seed-43 has an adapter and 33 predictions but no completed metrics and is not counted as a completed full-size result. No full-size final D artifact was found. No model currently has completed final coverage for all three intended seeds and all four methods.

### 5.9.6 Run artifacts and provenance

src/utils/artifacts.py writes or records:

| Artifact                         | Reproducibility role                                                    |
| -------------------------------- | ----------------------------------------------------------------------- |
| Resolved configuration snapshot  | Exact method and environment settings used by the run.                  |
| Configuration hash               | Compact identity for cache and artifact comparison.                     |
| Dataset hashes                   | Input-file identity.                                                    |
| Knowledge-base hashes            | Retrieved-context identity for RAG.                                     |
| Model and chat-template metadata | Model-family and tokenizer behavior.                                    |
| Git hash and dirty flag          | Source revision provenance stored by the run code.                      |
| Hardware and package versions    | Execution environment.                                                  |
| Cache identity                   | Prevents reuse across incompatible condition identities.                |
| Adapter provenance               | Connects C and D to base model, seed, data, and training configuration. |
| Status and completion timestamps | Distinguishes completed, failed, and partial runs.                      |
| Error JSONL                      | Makes per-item failures auditable.                                      |

The final-layout path is:

experiments/outputs/<profile>/<dataset>/<model>/<method>/seed\_<seed>

with methods A, B, C, and D represented by method directories and adapters stored under C. Legacy flat paths such as experiments/outputs/final/E01_qwen0_5b_prompt_43 are retained evidence but do not satisfy the current final profile, model, dataset, and seed hierarchy.

### 5.9.7 Reproducibility checklist

| Requirement                      | Current state                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------- |
| Source code path                 | Present; current repository may contain dirty state recorded by old manifests.              |
| Dependency lock                  | Present in poetry.lock; training extras declared.                                           |
| Resolved configuration           | Written by run metadata code for executed runs.                                             |
| Data file hashes                 | Present for current dashboard_v4_1 files.                                                   |
| KB hashes and manifest           | Present and internally verifiable.                                                          |
| Base model identifiers           | Present.                                                                                    |
| Immutable model revision         | Missing; revisions are null.                                                                |
| Adapter provenance               | Implemented and stored for trained adapters.                                                |
| Intended seed set                | 42, 43, 44.                                                                                 |
| Completed final seed coverage    | Missing for all four final models.                                                          |
| Validation execution             | Not established; current training entry point does not load the configured validation file. |
| Inference RNG policy             | Seed recorded, but explicit inference reseeding is not established.                         |
| Per-item failure denominator     | Errors logged, but missing predictions are not fully represented in metrics denominators.   |
| Independent human effectiveness  | Pending.                                                                                    |
| Human ratings and realism layers | Pending.                                                                                    |

## Historical Evolution and Superseded Implementations

### Dataset evolution

| Stage | Evidence                                                                                             | Status and methodological meaning                                                                                                                   |
| ----- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| v2    | src/config/data/dashboard_v2.yaml, older frozen data, legacy docs                                    | Superseded synthetic small dataset with 18 train, 3 validation, and 3 internal-test items. Useful for early pipeline checks, not current evidence.  |
| v3    | src/config/data/dashboard_v3.yaml, data/frozen/dashboard_v3, nvBench builder and source manifest     | Source-backed normalized dataset with 1,281 train, 264 validation, and 274 test items. Historical full-size artifacts use this dataset.             |
| v4    | src/config/data/dashboard_v4.yaml, v4 generation scripts and manifests                               | Operational dashboard_v4 package combining preserved v3 records with generated train/validation records. Generated records are explicitly not gold. |
| v4.1  | data/frozen/dashboard_v4/manifest.json, hashes.json, dataset card, validation/leakage/repair reports | Current frozen materialized revision. Semantic repair is recorded; test and human-evaluation records are preserved.                                 |

The repository contains data/raw_external/nvbench2, but the source is pending inspection and licensing confirmation and is not part of the authoritative current training input. Quda-related paths are also pending and are not current evidence.

### Method and infrastructure evolution

The repository evolved from a small synthetic data path and legacy flat run directories to a four-model final matrix, dataset-specific output hierarchy, cached run identities, adapter compatibility checks, structured report layers, and explicit robustness inputs. The current architecture is implemented in code, but older documents and artifact folders remain in place.

The following items are superseded or must be treated carefully:

| Older item                                     | Conflict with current authority                                                 | Current interpretation                                                                   |
| ---------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| README.md primary-model description            | Describes Qwen2.5-0.5B and older paths as primary.                              | Use as historical orientation; use src/config/matrix/final.yaml for final models.        |
| docs/project/SUPERVISOR_FULL_GPU_RUNBOOK.md    | Describes dashboard_v3 and older Qwen2.5 execution paths.                       | Legacy runbook; not current final-matrix authority.                                      |
| Later sections of PROJECT_COMMANDS.md          | Mix dashboard_v3, Qwen2.5, and older output layouts with newer v4 instructions. | Mixed operational guide; resolve conflicts through current configs and manifests.        |
| src/config/config.yaml default                 | Selects Qwen2.5, prompt-only, and legacy profile.                               | Convenience default and smoke-compatible baseline; final matrix overrides it.            |
| E01–E04 legacy experiment files                | Encode old Qwen2.5 experiment examples.                                         | Historical examples; current final model and seed authority is the matrix file.          |
| src/core/interfaces.py BaseRetriever docstring | Says RAG retrievers are stubbed in v1.                                          | Stale wording; TF-IDF and optional dense retrievers are implemented.                     |
| src/methods/base.py RAG comment                | Describes retrieval as future work.                                             | Stale wording; RAGHFMethod performs retrieval.                                           |
| Older full-size adapters                       | May use old target-module lists, v3 data, old paths, or old save limits.        | Do not substitute for current qlora_default.yaml or current v4 final runs.               |
| frozen_validation.py module wording            | Retains historical v2 wording in places.                                        | The implementation validates current v3/v4 JSONL paths; inspect arguments and manifests. |

The old configurations remain useful for reconstructing the project's evolution and for understanding smoke artifacts. They must not be silently combined with current v4/final-matrix claims.

### Reproducibility safeguards added by the current architecture

The current codebase contains several safeguards that address common failure modes:

1. Configuration hashes and cache identity prevent stale predictions from being reused under a new method or dataset.
2. Dataset and KB hashes make input drift visible.
3. Adapter validation prevents accidental cross-model, cross-seed, or cross-dataset reuse.
4. The D dependency is resolved from C for the same condition.
5. JSONL caches resume after interruptions without replacing successful items.
6. Per-item exceptions are retained in error logs.
7. Stored raw text permits parser improvements and forensic review.
8. Reporting marks synthetic metrics as internal-circular and leaves independent layers pending.
9. Tests cover schema, retriever behavior, adapter resolution, model architecture, cache guards, manifest structure, metrics, splits, resume logic, and matrix planning.

These safeguards improve auditability. They do not remove the remaining gaps in model revision pinning, final coverage, validation execution, inference seeding, or independent effectiveness assessment.

## Repository Evidence Map

| Evidence path                                                               | Relevant evidence                                                                                                       |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| src/core/schemas.py                                                         | GoldItem, DashboardBrief, DesignOutput, mapping, rationale, result models, enum definitions.                            |
| src/core/constants.py                                                       | Required top-level keys, controlled vocabulary, report schema version.                                                  |
| src/core/prompts.py                                                         | Exact system prompt, user prompt, schema instructions, allowed values.                                                  |
| src/data_pipeline/formatter.py                                              | Shared training/inference prompt formatting and JSON target serialization.                                              |
| src/inference/postprocess.py                                                | JSON extraction, alias normalization, parser errors, reparsing behavior.                                                |
| src/inference/decoders.py                                                   | Optional Outlines constrained decoder.                                                                                  |
| src/models/hf_utils.py                                                      | Model-family chat-template keyword behavior, loading kwargs, token handling.                                            |
| src/models/hf_causal.py                                                     | Tokenizer/model loading, padding, dtype, chat generation, adapter loading, token budgets.                               |
| src/methods/base.py                                                         | Common HF method, RAG method, retrieval prompt insertion, result construction.                                          |
| src/methods/ft.py                                                           | Fine-tuned method registration and adapter resolution.                                                                  |
| src/methods/ft_rag.py                                                       | Fine-tuned RAG registration, adapter validation, and composition.                                                       |
| src/core/interfaces.py                                                      | Method, retriever, trainer, and metric interfaces.                                                                      |
| src/core/registry.py                                                        | Method, retriever, trainer, and metric registration.                                                                    |
| src/retrievers/tfidf.py                                                     | Authoritative TF-IDF retrieval implementation.                                                                          |
| src/retrievers/dense.py                                                     | Implemented optional dense retriever; not final-authoritative.                                                          |
| src/data_pipeline/kb_builder.py                                             | Deterministic Markdown chunking and KB manifest construction.                                                           |
| data/knowledge_base/chunks.jsonl                                            | Current 41-chunk materialized KB.                                                                                       |
| data/knowledge_base/kb_manifest.json                                        | KB version, source hashes, chunk hash, and builder settings.                                                            |
| data/knowledge_base/guidelines/accessibility_guidelines.md                  | Accessibility guidance source.                                                                                          |
| data/knowledge_base/guidelines/chart_selection_guidelines.md                | Chart-selection guidance source.                                                                                        |
| data/knowledge_base/guidelines/dashboard_design_guidelines.md               | Layout and interaction guidance source.                                                                                 |
| src/config/config.yaml                                                      | Convenience defaults and legacy profile.                                                                                |
| src/config/matrix/final.yaml                                                | Current final-model, method, dataset, seed, and dependency matrix.                                                      |
| src/config/data/dashboard_v4.yaml                                           | Operational dataset package and train/test/robustness paths.                                                            |
| src/config/data/dashboard_v3.yaml                                           | Historical v3 selectable dataset.                                                                                       |
| src/config/data/dashboard_v2.yaml                                           | Historical v2 selectable dataset.                                                                                       |
| src/config/model/qwen2_5_0_5b.yaml                                          | Qwen2.5 smoke model identity and limits.                                                                                |
| src/config/model/qwen3_1_7b.yaml                                            | Qwen3 1.7B final model identity and limits.                                                                             |
| src/config/model/qwen3_8b.yaml                                              | Qwen3 8B final model identity and limits.                                                                               |
| src/config/model/qwen3_14b.yaml                                             | Qwen3 14B final model identity and limits.                                                                              |
| src/config/model/llama3_1_8b.yaml                                           | Llama 3.1 8B final model identity, token requirement, and limits.                                                       |
| src/config/method/prompt_only.yaml                                          | Method A generation settings.                                                                                           |
| src/config/method/rag.yaml                                                  | Method B retriever, KB path, top-k, and generation settings.                                                            |
| src/config/method/ft.yaml                                                   | Method C adapter settings and generation settings.                                                                      |
| src/config/method/ft_rag.yaml                                               | Method D adapter source, retriever, KB, top-k, and generation settings.                                                 |
| src/config/training/qlora_default.yaml                                      | Current QLoRA/SFT hyperparameters.                                                                                      |
| src/config/training/lora_default.yaml                                       | Older/non-authoritative LoRA alternative.                                                                               |
| src/config/training/dora.yaml                                               | Implemented DoRA alternative, not final-authoritative.                                                                  |
| src/config/training/rslora.yaml                                             | Implemented RSLoRA alternative, not final-authoritative.                                                                |
| src/training/sft_trainer.py                                                 | 4-bit loading, PEFT application, SFT trainer, checkpoints, adapter metadata.                                            |
| experiments/scripts/train.py                                                | Current training entry point and train-file-only loading behavior.                                                      |
| src/utils/adapter.py                                                        | Adapter path resolution and compatibility checks.                                                                       |
| experiments/scripts/run_final_matrix.py                                     | Final layout, seed matrix, C-before-D dependency, stale-run checks.                                                     |
| src/pipeline/runner.py                                                      | Test/robustness loading, inference execution, reparsing, metrics, reports.                                              |
| src/inference/runner.py                                                     | Append-only cache, cache identity, per-item errors, prediction persistence.                                             |
| src/evaluation/metrics/schema_compliance.py                                 | Strict schema and completeness metrics.                                                                                 |
| src/evaluation/metrics/top_k_accuracy.py                                    | Top-k semantics and top-3 validity rule.                                                                                |
| src/evaluation/metrics/grounding.py                                         | Semantic opt-in and lexical-proxy grounding behavior.                                                                   |
| src/evaluation/metrics/robustness.py                                        | Paraphrase and missing-information diagnostics.                                                                         |
| src/evaluation/reporting.py                                                 | Layered metrics, internal-circular labeling, pending L1/L3/L4 layers, per-item reports, and legacy backfill safeguards. |
| src/utils/artifacts.py                                                      | Run manifest, config, environment, data, KB, adapter, hardware, and cache provenance.                                   |
| src/utils/seed.py                                                           | Seed initialization behavior.                                                                                           |
| src/utils/config_hash.py                                                    | Resolved configuration hash.                                                                                            |
| pyproject.toml                                                              | Project Python range, core dependencies, and optional training/retrieval extras.                                        |
| poetry.lock                                                                 | Locked dependency resolution used for environment reproducibility.                                                      |
| requirements-train.txt                                                      | Training dependency export generated from the project specification.                                                    |
| experiments/outputs/smoke/dashboard_v4/qwen2_5_0_5b/A/seed_42/env.txt       | Per-run environment capture for a completed smoke artifact.                                                             |
| experiments/outputs/smoke/dashboard_v4/qwen2_5_0_5b/A/seed_42/manifest.json | Example completed run manifest with model, method, seed, data, environment, and status provenance.                      |
| src/data_pipeline/dataset.py                                                | GoldItem loading and identifier handling.                                                                               |
| src/data_pipeline/splits.py                                                 | Deterministic item-ID split helper.                                                                                     |
| src/data_pipeline/frozen_validation.py                                      | Frozen JSONL schema, semantic, duplicate, distribution, leakage, and hash validation.                                   |
| data/frozen/dashboard_v4/manifest.json                                      | Current v4.1 freeze identity, lineage, counts, repair metadata, and checks.                                             |
| data/frozen/dashboard_v4/hashes.json                                        | Current data, schema, manifest, card, and report hashes.                                                                |
| data/frozen/dashboard_v4/dataset_card.md                                    | Current data-card description and generated-field status.                                                               |
| data/frozen/dashboard_v4/validation_report.json                             | Current frozen-data validation status.                                                                                  |
| data/frozen/dashboard_v4/leakage_report.json                                | Generated-record leakage checks.                                                                                        |
| docs/datasets/DATASET_CONSTRUCTION_HISTORY_AND_METHODOLOGY.md               | Source lineage, v2/v3/v4 evolution, generation and repair history.                                                      |
| docs/thesis/references.bib                                                  | Repository bibliography keys for NVBench, LoRA, QLoRA, RAG, structured output, and visualization theory.                |
| src/tests/test_schema.py                                                    | Schema and model behavior checks.                                                                                       |
| src/tests/test_retriever.py                                                 | KB chunking, TF-IDF ranking, and method registration checks.                                                            |
| src/tests/test_final_multi_model_architecture.py                            | Model matrix, thinking behavior, token gating, output paths, adapter identity, and dataset selection checks.            |
| src/tests/test_adapter_resolution.py                                        | Adapter path and compatibility behavior.                                                                                |
| src/tests/test_run_manifest.py                                              | Manifest and provenance structure checks.                                                                               |
| src/tests/test_inference_cache_guard.py                                     | Cache identity and stale-output protection.                                                                             |
| src/tests/test_postprocess.py                                               | Parser and normalization behavior.                                                                                      |
| src/tests/test_metrics.py and src/tests/test_metric_semantics.py            | Metric behavior and denominator semantics.                                                                              |
| src/tests/test_generator.py and src/tests/test_generator_v2.py              | Dataset generation and validation utilities.                                                                            |
| src/tests/test_frozen_v2.py                                                 | Frozen dataset validation behavior.                                                                                     |
| src/tests/test_multi_run_statistics.py                                      | Multi-seed aggregation and confidence-interval safeguards.                                                              |
| src/tests/test_run_final_matrix.py                                          | Final matrix planning and dependency behavior.                                                                          |
| src/tests/test_training_resume.py                                           | Resume and checkpoint behavior.                                                                                         |
| src/tests/test_gpu_precision.py                                             | Hardware-aware precision resolution.                                                                                    |
| src/tests/test_splits.py                                                    | Deterministic split behavior.                                                                                           |
| src/tests/test_scientific_validity.py                                       | Scientific-evaluation safeguards and validity checks.                                                                   |
| experiments/outputs/smoke/dashboard_v4/qwen2_5_0_5b/A/seed_42               | Completed two-item prompt-only smoke artifact.                                                                          |
| experiments/outputs/smoke/dashboard_v4/qwen2_5_0_5b/B/seed_42               | Completed two-item RAG smoke artifact.                                                                                  |
| experiments/outputs/smoke/dashboard_v4/qwen2_5_0_5b/C/seed_42               | Completed two-item QLoRA smoke artifact with adapter.                                                                   |
| experiments/outputs/smoke/dashboard_v4/qwen2_5_0_5b/D/seed_42               | Completed two-item fine-tuning plus RAG smoke artifact.                                                                 |
| experiments/outputs/final/E01_qwen0_5b_prompt_43                            | Completed legacy full-size v3 prompt-only artifact.                                                                     |
| experiments/outputs/final/E02_qwen0_5b_rag_42                               | Completed legacy full-size v3 RAG artifact.                                                                             |
| experiments/outputs/final/E02_qwen0_5b_rag_43                               | Completed legacy full-size v3 RAG artifact.                                                                             |

## Final Consistency Audit

The following statements are the current consistent interpretation of code, configurations, manifests, and artifacts:

| Audit item                | Consistent statement                                                                                                                                                                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Authoritative dataset     | The operational package is dashboard_v4; its frozen materialized revision is dashboard_v4_1 in data/frozen/dashboard_v4.                                                                                                     |
| Current training input    | data/frozen/dashboard_v4/train.jsonl, 2,932 records, as selected by src/config/data/dashboard_v4.yaml and consumed by the current training entry point.                                                                      |
| Validation status         | val.jsonl has 613 records and is configured, but the current training entry point does not pass it to the SFT trainer.                                                                                                       |
| Held-out evaluation       | test.jsonl has 274 records; robustness files are separate test variants; the human-evaluation CSV has 40 items.                                                                                                              |
| Current final models      | Qwen3 1.7B, Qwen3 8B, Qwen3 14B, and Llama 3.1 8B.                                                                                                                                                                           |
| Smoke model               | Qwen2.5-0.5B.                                                                                                                                                                                                                |
| Current methods           | A prompt-only, B TF-IDF RAG, C QLoRA fine-tuning, D QLoRA plus the same TF-IDF RAG.                                                                                                                                          |
| Current RAG authority     | 41 chunks from three local Markdown guideline files, KB version 46b8575d98d37312, top-k 3, TF-IDF retriever.                                                                                                                 |
| Current QLoRA authority   | qlora_default.yaml: 4-bit NF4 double quantization, LoRA rank 16, alpha 32, dropout 0.05, all-linear, 3 epochs, batch 2, accumulation 4, learning rate 2e-4, cosine schedule, gradient checkpointing, no validation strategy. |
| Intended seed coverage    | 42, 43, and 44 for each final model and method.                                                                                                                                                                              |
| Completed final coverage  | None found for any of the four final models.                                                                                                                                                                                 |
| Completed legacy evidence | Qwen2.5 v3 A seed 43 and B seeds 42/43; no completed full-size C or D.                                                                                                                                                       |
| Completed smoke evidence  | Qwen2.5 v4 current-layout A/B/C/D seed 42, two items each; smoke only.                                                                                                                                                       |
| Model revision            | All current model revisions are null; exact immutable Hub revisions are missing.                                                                                                                                             |
| Results interpretation    | Existing legacy and smoke reports are implementation evidence, not current final thesis results.                                                                                                                             |
| Independent validity      | Human effectiveness, realism, and human-rating layers are pending.                                                                                                                                                           |

The audit also identifies concrete documentation conflicts rather than hiding them. README and supervisor runbook descriptions of Qwen2.5/v3 are historical. The default config is legacy-profile convenience configuration. The operational v4 name and frozen v4.1 revision are both retained. Dense retrieval exists but is not the final retriever. Stale comments claiming that retrieval is stubbed are contradicted by the working TF-IDF implementation and retriever tests.

No numerical performance result is promoted here as a final scientific finding. The artifact inventory establishes completion state only. A final thesis results chapter must use completed final-layout runs with matching dataset, model, method, seed, configuration hash, KB identity, adapter provenance, and evaluation layer.

## Remaining Information Gaps

- No completed final-profile runs were found for Qwen3 1.7B, Qwen3 8B, Qwen3 14B, or Llama 3.1 8B under methods A, B, C, or D for seeds 42, 43, and 44.
- No completed full-size current-v4 C or D result was found; the existing C/D Qwen2.5 evidence is smoke-scale or incomplete legacy evidence.
- The current model configurations leave revision unset, so exact immutable Hugging Face model revisions are not recorded.
- The current training entry point loads only the training split. It does not execute validation on the configured 613-item val.jsonl; checkpoint selection and validation-based model selection are therefore not established.
- The inference entry point records seeds but does not establish explicit per-run inference RNG seeding while default generation uses sampling.
- Failed inference items are recorded in error files but are absent from successful prediction lists; run-level denominator handling does not yet represent requested, successful, and failed item counts separately.
- The knowledge-base manifest hashes local guideline files but does not record external URLs, DOIs, or independent source provenance for those texts.
- The grounding metric is claim-based and either semantic-threshold or lexical-proxy; it is not a human faithfulness judgment.
- Synthetic chart-selection metrics remain internal and circular because the generated gold follows the generator's chart-choice rule.
- The independent L1 human-effectiveness scorer is not implemented; L3 realism and L4 human-rating layers remain pending.
- The current repository does not establish a completed final multi-seed aggregation for the intended matrix, so confidence intervals and cross-seed thesis comparisons cannot yet be reported as final evidence.
