# Chapter 5 — System Design and Methods

This chapter describes the system and the experimental methods used to study structured dashboard-design recommendations. The chapter begins by defining the research object and the comparison logic of the four methods. It then formalizes the input and output of the task, describes the model and seed matrix, and explains the shared prompt and generation pipeline. The following sections present the prompt-only baseline, retrieval-augmented generation (RAG), QLoRA fine-tuning, and the combined fine-tuning-plus-RAG method. The chapter also discusses alternative parameter-efficient fine-tuning methods and explains why QLoRA is the main adaptation method in this thesis. It ends with the inference procedure, the evaluation protocol, and the measures used to preserve reproducibility and to limit over-interpretation of the results.

The design of this study separates decisions that are shared by all methods from decisions that define a particular method. The dashboard brief, the output schema, the test split, the generation settings, and the evaluation code are shared wherever the implementation allows it. Retrieval and parameter adaptation are introduced as separate interventions. This makes it possible to compare a prompt-only model with a model that receives design guidance, with a model that learns from the task data, and with a model that uses both sources of information. Chapter 4 described how the dataset was constructed and frozen. The present chapter describes how the frozen data and the implemented models are used; it does not present outcome values. Those values and their interpretation belong to the results and discussion chapters.

The implementation record and the repository configuration are the primary evidence for the project-specific details in this chapter. Scientific literature is used to justify the general method choices, including Transformer generation, low-rank adaptation, quantized fine-tuning, retrieval augmentation, structured outputs, and behavioral evaluation. Where the current implementation is incomplete or where a metric is only a diagnostic, this is stated explicitly. In particular, an internal synthetic chart label is not treated as independent evidence of visualization effectiveness, and a seed listed in a configuration file is not treated as proof that the corresponding run has already been completed.

## 5.1 Methodological Scope and Research Design

The system studied in this thesis maps a dashboard brief to a structured design recommendation. The brief describes the intended users, analytical goals, key performance indicators (KPIs), available data columns, and design constraints. The output is not a rendered dashboard and it is not a SQL query. It is an intermediate design object that can be inspected, validated, and used by a later rendering or application layer. The object contains a context summary, KPI-to-chart mappings, layout and styling decisions, interaction suggestions, and rationales. This scope is important because chart selection is only one part of a dashboard design. A method can select a plausible chart and still fail to provide a coherent layout, accessible styling, or a useful explanation.

The experimental design uses four method conditions. They differ in two binary factors: whether the prompt contains retrieved design guidance and whether the base model contains a task-specific adapter. Method A is the prompt-only reference condition. Method B adds retrieval but keeps the base model unchanged. Method C adds QLoRA adaptation but does not retrieve documents. Method D combines the adapter from Method C with the retrieval procedure from Method B. The names used by the repository are shown in Table 5.1.
Table 5.1. The four method conditions and the factors that distinguish them.


| Method | Repository name | Retrieval context | Task-specific adapter | Main purpose |
| --- | --- | --- | --- | --- |
| A | <code>prompt_only</code> | No | No | Reference condition for prompting alone |
| B | <code>rag</code> | Yes, TF-IDF over the frozen design-guideline corpus | No | Estimate the effect of retrieved design guidance |
| C | <code>ft</code> | No | Yes, QLoRA adapter trained on the task data | Estimate the effect of task-specific adaptation |
| D | <code>ft_rag</code> | Yes, TF-IDF over the same corpus | Yes, the compatible C adapter | Test the combination of adaptation and retrieval |

The comparison is designed as a two-by-two intervention study for a fixed base model and seed. The B–A comparison changes retrieval while keeping the model parameters unchanged. The C–A comparison changes the parameters through QLoRA while keeping the prompt closed-book. The D–C comparison asks whether retrieval still changes the output after task adaptation. The D–B comparison asks whether adaptation changes the output when the same retrieval context is available. The direct D–A comparison is also useful as an overall system comparison, but it does not isolate one intervention because both retrieval and adaptation change at the same time. This distinction prevents a combined system from being described as evidence for one component alone.

The unit of analysis is one dashboard brief and one generated recommendation for a specified model, method, seed, and input variant. The same item identifier is used to join a prediction with its reference and with any robustness variant. A model run is therefore not treated as an independent observation when its predictions are later compared item by item with another method. Paired comparisons are used where the evaluation protocol supports them. This is appropriate because the methods receive the same test items, while the model or context condition changes.

The implementation follows a staged experimental design. The primary dataset is <code>dashboard_v4</code> with the frozen revision <code>dashboard_v4_1</code>, and the available seed values are 42, 43, and 44. The executed model matrix comprises three Qwen3 profiles of increasing capacity — Qwen3 1.7B, Qwen3 8B, and Qwen3.8 27B — together with OLMo-2-0425-1B-Instruct as a second model family at a scale comparable to the smallest Qwen3 profile. The execution plan uses seed 42 first and reserves repeated seeds for the two smaller profiles, where the stability of the method contrast can be examined within the available budget. A condition that is executed only with seed 42 is reported as a single-seed condition. It is not combined with a multi-seed mean and is not used as evidence of seed stability. The small Qwen2.5-0.5B model is a local smoke-test profile and is not part of the reported comparison. Chapter 6 records the executed coverage cell by cell, including the cells that are missing and the runs that were rejected.

This staged policy is a resource decision, not a claim that larger models are deterministic or that one seed is sufficient in general. Training and generation for several models, four methods, and several seeds require substantial memory and time. QLoRA reduces the cost of adaptation, but it does not remove the cost of model loading, generation, validation, and repeated evaluation. Reporting the seed coverage together with the result is therefore part of the method. It makes clear which conclusions describe a repeated small-model stability check and which conclusions describe a single executed run.

## 5.2 Formal Task and Output Contract

The input to every method is represented as a dashboard brief

\[
x = (u, G, K, F, c),
\]

where \(u\) denotes the intended users, \(G\) is a list of analytical goals, \(K\) is a list of KPIs, \(F\) is the set of available data columns and their types, and \(c\) denotes optional constraints. The project also stores an <code>item_id</code>, but this identifier is used for provenance and joining records rather than as semantic input to the recommendation. A column can be represented as a pair

\[
F = \{(n_i, d_i)\}_{i=1}^{p},
\]

where \(n_i\) is a field name and \(d_i\) is its declared data type. The brief may also contain additional fields, but the shared prompt exposes the users, goals, KPIs, columns, and constraints as the common interface for all methods.

The output is a structured object

\[
y = (s, m, l, t, i, r),
\]

where \(s\) is <code>context_summary</code>, \(m\) is <code>kpi_chart_mapping</code>, \(l\) is <code>layout</code>, \(t\) is <code>styling</code>, \(i\) is <code>interactions</code>, and \(r\) is <code>rationales</code>. A mapping entry connects a KPI to an analytical task and a chart type. It also contains optional alternatives and an encoding object. The schema therefore separates the semantic task from the visual representation. This separation is consistent with the view of visualization design as a mapping between data, task, and graphical encoding \cite{mackinlay1986apt,munzner2014visualization}. It also makes it possible to evaluate a chart token without losing the context in which the token was recommended.
Table 5.2. Required top-level fields of the structured recommendation.


| Output field | Type in the project contract | Role in the recommendation |
| --- | --- | --- |
| <code>context_summary</code> | Object | Summarises the design situation and the main goal |
| <code>kpi_chart_mapping</code> | Array of objects | Connects each KPI to a task, chart type, alternatives, and encodings |
| <code>layout</code> | Object | Describes placement, grouping, order, or emphasis of dashboard elements |
| <code>styling</code> | Object | Describes visual styling and accessibility-related choices |
| <code>interactions</code> | Array | Describes possible filters, drill-downs, or other interactions |
| <code>rationales</code> | Array of objects | Gives a claim and the design principle supporting it |

The project defines controlled vocabularies for the two most important categorical fields. The available task types are <code>trend</code>, <code>comparison</code>, <code>composition</code>, <code>distribution</code>, <code>correlation</code>, <code>ranking</code>, <code>deviation</code>, <code>part_to_whole</code>, and <code>flow</code>. The available chart types include <code>line</code>, <code>bar</code>, <code>stacked_bar</code>, <code>grouped_bar</code>, <code>area</code>, <code>pie</code>, <code>donut</code>, <code>scatter</code>, <code>heatmap</code>, <code>histogram</code>, <code>box</code>, <code>kpi_card</code>, <code>table</code>, <code>gauge</code>, <code>sankey</code>, <code>treemap</code>, and <code>map</code>. These values are a project interface and not a universal taxonomy of all visualization tasks or chart forms. Their purpose is to provide a stable vocabulary for the dataset, the prompt, the parser, and the metrics.

The recommendation is generated as a token sequence whose final text should be one JSON object. For a model with parameters \(\theta\), the autoregressive generation process can be written as

\[
p_{\theta}(y \mid x) =
\prod_{q=1}^{T}
p_{\theta}(y_q \mid y_{<q}, x),
\]

where \(y_q\) is the next output token and \(y_{<q}\) is the generated prefix. This is the standard decoder formulation used by Transformer language models \cite{vaswani2017attention}. In the prompt-only and RAG conditions, the model parameters remain those of the selected base model. In the fine-tuned conditions, the base model is combined with the learned adapter described in Section 5.7.

For the supervised adaptation condition, the target is the serialized reference object. If \(D_{\mathrm{train}}=\{(x_j,y_j)\}_{j=1}^{n}\) is the frozen training set and \(y_j=(y_{j,1},\ldots,y_{j,T_j})\) is the target token sequence, the training objective is

\[
\mathcal{L}(\theta) =
-\sum_{j=1}^{n}\sum_{q=1}^{T_j}
\log p_{\theta}(y_{j,q}\mid y_{j,<q},x_j).
\]

The target is therefore a complete structured recommendation rather than only a chart label. The loss does not prove that every target design is the only correct design. It only defines how the selected model is adapted to reproduce the frozen reference representation. This distinction becomes important in the evaluation: exact agreement with the synthetic reference is an internal diagnostic, while design effectiveness requires independent evidence.

The schema is implemented in <code>src/core/schemas.py</code> with Pydantic models. The regular runtime models allow additional keys so that a slightly extended model output can be retained for inspection. The strict generation metric uses the separate strict response schema and validates the raw extracted JSON object before lenient parser repairs are applied. This creates two clearly separated purposes. The permissive model helps preserve information during analysis, while the strict metric measures whether the model produced the required contract without relying on a correction made after generation. JSON Schema provides a general formal language for describing JSON structure and validation constraints; the project uses a project-specific Pydantic contract and a generated schema artifact for the same practical purpose \cite{jsonschema2022draft}.

## 5.3 Model Matrix and Experimental Factors

### 5.3.1 Primary model profiles

The study uses four base models. Three of them are Qwen3-family profiles that span an order of magnitude in capacity, which makes it possible to ask whether a method effect changes with model size inside one family \cite{yang2025qwen3,qwen2026qwen38}. The fourth is an OLMo 2 instruction-tuned checkpoint of comparable scale to the smallest Qwen3 profile, which makes it possible to ask whether an effect observed in Qwen3 reappears in an independently developed family \cite{olmo2025furious}. Model identifiers, context lengths, preferred data types, revisions, and access requirements are read from the model configuration files and from the run manifests rather than inferred from a result file. The repository also contains a Qwen2.5-0.5B smoke-test profile, which is used only to verify that the pipeline runs end to end and is not part of the reported comparison.
Table 5.3. Base-model profiles used in the study.


| Model key | Hugging Face identifier | Nominal size | Pinned revision | Sequence length in this study | Runtime profile | Role |
| --- | --- | ---: | --- | ---: | --- | --- |
| <code>qwen3_1_7b</code> | <code>Qwen/Qwen3-1.7B</code> | 1.7B | not pinned | 4096 | bfloat16 preferred; thinking disabled | Smallest scale; three-seed stability condition |
| <code>qwen3_8b</code> | <code>Qwen/Qwen3-8B</code> | 8B | not pinned | 4096 | bfloat16 preferred; thinking disabled | Intermediate scale |
| <code>qwen3_8_27b</code> | <code>Qwen/Qwen3.8-27B</code> | 27B | <code>1d4bf0f2ff60…</code> | 4096 | bfloat16 preferred; thinking disabled | Largest scale |
| <code>olmo2_1_49b</code> | <code>allenai/OLMo-2-0425-1B-Instruct</code> | 1.49B | <code>48d788eca847…</code> | 4096 | bfloat16 preferred | Cross-family check at the small scale |
| <code>qwen2_5_0_5b</code> | <code>Qwen/Qwen2.5-0.5B-Instruct</code> | 0.5B | not pinned | 2048 | float16 on GPU; float32 CPU fallback | Smoke test only |

The `qwen3_8_27b` profile is used through its text-only causal-language-model path. Its published model card describes a hybrid attention stack and a native context window far larger than the 4,096-token window configured here \cite{qwen2026qwen3827bcard}; the project deliberately holds the effective sequence length constant across profiles so that the context budget is not an uncontrolled factor in the comparison.

The OLMo profile is not an arbitrary fourth model. It is selected because it provides a second model family whose construction is documented in unusual detail: the OLMo 2 release publishes training data, code, recipes and intermediate checkpoints alongside the weights, so the ways in which it differs from Qwen3 can be described rather than assumed \cite{olmo2025furious}. Its instruction-tuned variant is post-trained with supervised fine-tuning on an OLMo-specific variant of the Tülu 3 mixture, followed by direct preference optimisation and reinforcement learning with verifiable rewards \cite{allenai2025olmo2instruct1b,lambert2025tulu3}. Its parameter count, measured when the model is loaded for adapter training, is 1.50 billion against 1.74 billion for the smallest Qwen3 profile, so the cross-family comparison is made at a comparable scale rather than across a capacity gap.

Because architecture, tokenizer, pretraining corpus, post-training recipe and native context window all differ simultaneously between the two families, the OLMo profile supports a model-family comparison and nothing narrower. The published configurations record that the OLMo checkpoint uses multi-head attention with 16 query and 16 key-value heads while the small Qwen3 profile uses grouped-query attention with 16 query and 8 key-value heads; this is a documented difference and not an explanation of any observed behaviour. A result from this comparison can support a statement about whether a pattern generalises across families. It cannot isolate any single architectural component as the cause. The OLMo profile is compared conceptually with the smallest Qwen3 profile only, and is not treated as size-matched to the 8B or 27B profiles.

All models are used as decoder-only language models through the Hugging Face model interface, and the model-specific chat template is applied before generation. For the Qwen3 profiles the configuration disables the optional thinking mode, because the target is a compact JSON object and the comparison concerns the requested design output rather than unobserved reasoning traces. The same output contract is used for the OLMo profile, while its tokenizer and chat template remain model-specific.

### 5.3.2 Methods, seeds, and controlled factors

For each selected model and seed, the method factor is A, B, C, or D. The dataset identity, prompt contract, output parser, evaluation code, and generation settings are held fixed unless a method explicitly adds retrieval or an adapter. The main generation settings are <code>max_new_tokens=512</code>, <code>temperature=0.1</code>, <code>top_p=0.9</code>, <code>do_sample=true</code>, and <code>repetition_penalty=1.15</code>. The low temperature reduces unnecessary variation in structured output, but sampling remains enabled; the setting is therefore not deterministic generation.

The available seeds in the matrix are 42, 43, and 44. The staged execution uses 42 as the first seed. Repeated seeds are used to check whether the smaller Qwen3-1.7B condition gives stable conclusions across training and generation. Larger models can be run at seed 42 to extend the capacity comparison under the available budget. The resulting evidence is labelled accordingly. A mean across seeds is calculated only when all required seed runs for the stated comparison exist. Missing runs are not silently replaced by a value from another seed.

The seed is an experimental factor, not a row-level label that increases the number of independent dashboard briefs. For a fixed test item, the predictions from different seeds remain repeated observations of the same item. This is why seed variation is reported as spread or as a stability diagnostic, while method comparisons use the same items whenever possible. The project records the seed in every <code>GenerationResult</code> and in training metadata. Exact bitwise reproduction is not promised merely because the same integer is recorded: sampling, hardware kernels, and CUDA execution can introduce variation. The seed is therefore necessary provenance, but it is not by itself a guarantee of identical output.

## 5.4 Shared Prompt and Structured Generation Pipeline

### 5.4.1 Single prompt source

All four methods use the prompt builder in <code>src/core/prompts.py</code>. The system message defines the model as a dashboard design consultant and requires one valid JSON object without Markdown fences or commentary outside the object. The user message renders the current brief and explicitly names the six required top-level fields. It also specifies the types of the fields, the required mapping entries, the allowed task and chart vocabularies, and the rule that KPI and encoding names must come from the current brief. This explicit contract gives the prompt-only baseline the same structural information that the fine-tuned methods receive.

The prompt does not ask the model to invent data values or to generate a rendered image. It asks for a recommendation based on the users, goals, KPIs, columns, and constraints. The prompt also distinguishes the rationale from the design decision. A rationale must contain a claim and a principle, so a response can state both what it recommends and why the recommendation fits the brief. This separation is useful for later grounding analysis, although a generated rationale is not automatically a faithful explanation.

The formatter in <code>src/data_pipeline/formatter.py</code> imports the same prompt builder for supervised training. It serializes the reference recommendation as indented JSON and appends the tokenizer end-of-sequence token when one is available. Using one prompt construction path for training and inference reduces an avoidable difference between the input seen during adaptation and the input seen during evaluation. The output still depends on model-specific tokenization and chat templates, so the implementation records the model profile and tokenizer provenance for each run.

### 5.4.2 Pipeline stages

The common execution path is shown in Table 5.4. A method-specific difference appears only at the retrieval and adapter stages. The separation between preparation, generation, parsing, and scoring is deliberate. It allows a prompt-length failure or parser failure to be recorded as a failure of the relevant stage instead of being hidden inside one aggregate score.
Table 5.4. Shared execution path; only stages 3 and the adapter load differ between methods.


| Stage | Operation | Main artifact or check |
| --- | --- | --- |
| 1. Load | Read the resolved model, method, dataset, seed, and evaluation configuration | Configuration and dataset identity |
| 2. Prepare | Build the system and user messages from the dashboard brief | Shared prompt text |
| 3. Retrieve | For B and D, retrieve up to three positive-scoring guideline passages | Retrieved passage IDs, scores, and context |
| 4. Fit | Count prompt tokens and reserve the output budget | Input-token budget and truncation flag |
| 5. Generate | Apply the model chat template and generate up to 512 new tokens | Raw model text |
| 6. Parse | Extract a JSON object, normalize recoverable aliases, and validate the result | Parsed object and parse error |
| 7. Evaluate | Reparse consistently and compute the configured metrics | Per-item records and aggregate reports |

For a model with maximum sequence length \(L\) and an output limit \(M\), the available input budget is

\[
B_{\mathrm{input}} = L - M.
\]

For the final model profiles, the nominal values are \(4096-512=3584\) input tokens. For the smoke profile, the corresponding budget is \(2048-512=1536\) tokens. The model wrapper checks this budget before generation and raises an error instead of silently dropping part of the brief. In the RAG conditions, the retrieved passages are fitted to the same budget. If the full passages do not fit, the implementation clips each passage to an equal token limit and records <code>rag_context_truncated=true</code>. If even the passage headings do not fit, the item is treated as a preparation failure.

The standard A–D configurations use sequential generation with batch size one. This choice makes the item-level execution order and failure attribution simple. The code supports larger batches only when batching is explicitly acknowledged in the configuration. A batched generation call cannot provide a separate model duration for every item, so the implementation amortizes the fused duration and records the corresponding semantics. Batch size one is therefore the default for the main methods.

### 5.4.3 Parsing and strict validation

The parser in <code>src/inference/postprocess.py</code> first searches for a JSON object. It can handle a fenced JSON object, a brace-delimited object surrounded by short prose, or a complete JSON response. It then applies limited normalization before Pydantic parsing. Examples include mapping <code>comparision</code> to <code>comparison</code>, mapping <code>column chart</code> to <code>bar</code>, and removing alternatives that are not valid chart tokens. Object-like fields returned as strings can be wrapped so that the content remains available for inspection.

These repairs are useful for diagnosing near-miss outputs, but they must not be confused with a strict success. The schema-compliance metric extracts the raw JSON object again and validates the raw values against the strict response contract. An output that becomes valid only after an alias repair is therefore not counted as strict raw-schema validity. The raw text, the parsed representation, and the parse error are stored together in <code>GenerationResult</code>. This permits later audits without asking the model to generate the item again.

The repository also contains an optional constrained decoder based on an output schema. Constrained decoding can reduce invalid structures, and structured-output benchmarks show that the behavior of constrained generation depends on both the constraint method and the model \cite{geng2025structured}. The constrained decoder is not enabled in the authoritative A–D final configurations. It is treated as a separate engineering or ablation path so that an improvement from hard decoding is not incorrectly attributed to RAG or QLoRA.

## 5.5 Method A — Prompt-Only Baseline

Method A is the reference condition for the study. It loads the selected base model and tokenizer, applies the shared system and user messages, generates the JSON response under the common generation settings, and passes the raw text to the common parser. It does not load a task-specific adapter and it does not add a retrieval context. The method therefore measures how much structured dashboard recommendation the base instruction-tuned model can produce from the explicit prompt alone.

The baseline is not intended to represent an unprompted language model. It is a controlled prompt-only condition. The model receives the same output contract, field names, and allowed vocabularies as Methods B–D. Without this information, a comparison would mix the effect of the method with the effect of giving one condition a clearer task definition. The baseline also does not use the target recommendation at inference time. It has access to the brief only.

Method A provides two comparison points. First, it shows whether retrieval adds useful information beyond the prompt. Second, it shows whether task-specific adaptation improves the structured output beyond the capability of the base model. Because the model is still stochastic at <code>temperature=0.1</code> with sampling enabled, repeated seeds or repeated executions must be labelled when they are used. A single successful JSON response does not establish general reliability.

## 5.6 Method B — Retrieval-Augmented Generation

Method B adds a non-parametric information source to the prompt-only condition. Retrieval-augmented generation combines the language model's parametric knowledge with passages selected from an external or local knowledge source \cite{lewis2020rag}. In this thesis, the retrieved information consists of dashboard and visualization design guidance. The method tests whether an explicit, auditable design context helps the model produce recommendations that are better aligned with the project principles or that provide more supported rationales.

### 5.6.1 Frozen knowledge base

The authoritative knowledge base contains three Markdown documents: <code>accessibility_guidelines.md</code>, <code>chart_selection_guidelines.md</code>, and <code>dashboard_design_guidelines.md</code>. The builder splits the documents at Markdown headings, keeps a heading with its body text, removes chunks below the configured minimum word count, assigns deterministic identifiers, and writes the resulting records to <code>data/knowledge_base/chunks.jsonl</code>. The frozen manifest reports three source documents and 41 chunks. It records the source hashes, byte sizes, the chunk-file hash, and the builder settings. The knowledge-base version recorded in the repository manifest is <code>46b8575d98d37312</code>; the version recorded by every executed retrieval run is <code>3d1409d8347fc165</code>. The two identities describe the same content and differ only in line terminators, as Section 6.3.1 documents; the value recorded by the runs is authoritative for the reported results.

Heading-based chunking was selected because the corpus is small and the headings already provide meaningful topical boundaries. The resulting chunks can be inspected by a researcher and cited in a retrieval trace. A smaller chunk may improve lexical matching for a short rule, while a larger chunk may preserve the context needed to interpret that rule. The project accepts this trade-off and records the exact chunk file so that a later change in chunking is a new knowledge-base version rather than an invisible modification.

The knowledge base is project guidance, not an independent human gold set for chart effectiveness. Its role is to provide design principles to the model and to support a retrieval trace. It is not used to claim that a retrieved sentence proves a chart is effective. This distinction is necessary because a rule document can be incomplete, simplified, or written for the project rather than validated as a complete theory of visualization.

### 5.6.2 Query and retrieval algorithm

For each brief, the retriever builds a query from the users, the joined goals, and the joined KPIs. It intentionally omits the target recommendation and the reference chart mapping. This exclusion prevents the retrieval step from receiving information that would not be available at inference time. The current retriever is TF-IDF with English stop-word removal. It vectorizes the 41 chunks and the query, computes cosine similarity, orders the chunks by score, and returns at most the top three chunks with a positive score.

For a term \(w\) in document \(d\), TF-IDF can be understood as weighting a term by its frequency in the document and reducing the weight of terms that occur in many documents. The retriever then compares the query vector \(q\) with a chunk vector \(v_d\) using

\[
\operatorname{cos}(q,v_d)
=\frac{q\cdot v_d}{\lVert q\rVert\lVert v_d\rVert}.
\]

The use of sparse TF-IDF is a deliberate scope choice. Term weighting by inverse document frequency is a long-established retrieval principle: a term that occurs in few documents carries more information about which document is relevant than a term that occurs in many \cite{sparckjones1972idf}. For a corpus of 41 short, topically distinct guideline chunks, this simple weighting is sufficient to separate an accessibility rule from a chart-selection rule, and it has four properties that matter for an experiment rather than for a production system. It is deterministic, so the same brief retrieves the same passages in every run. Its scores can be inspected directly, so a retrieval trace can be explained. It runs on the CPU and adds no model to the experiment. And it is model-independent, so the same retrieval mechanism can be applied unchanged across two model families without introducing a component that was itself trained on one of them.

Two better-known alternatives were not used. BM25 refines the same sparse principle with document-length normalisation and term-frequency saturation \cite{robertson2009bm25}; those refinements matter most for long and heterogeneous documents, which this corpus does not contain. Dense retrieval encodes queries and passages with a learned encoder and matches them in embedding space, which can retrieve a relevant passage that shares no vocabulary with the query \cite{karpukhin2020dpr}; a dense profile using BGE embeddings exists in the repository as an optional configuration. Neither was used for the reported conditions, and neither is presented as an evaluated method. The thesis does not compare retrievers, and the claim made here is that a simple sparse retriever is defensible for this corpus and this experimental purpose — not that it is the best retriever for the task.

The parameter <code>top_k=3</code> limits the amount of external context. A larger value could provide more coverage, but it would also consume input tokens and increase the chance that unrelated rules distract the model. A smaller value could omit a relevant rule. The selected value is kept fixed across B and D so that the comparison between the two RAG conditions changes the adapter, not the retrieval depth.

### 5.6.3 Prompt integration and grounding records

When positive matches exist, the retrieved passages are appended to the system prompt between the delimiters <code>--- Relevant Design Guidelines ---</code> and <code>--- End of Guidelines ---</code>. Each passage includes its identifier, source, heading, text, and retrieval score. The user message remains the same as in Method A. If no passage has a positive score, the method falls back to the base system prompt rather than adding an empty context block.

The retrieved context is guidance and not a hard constraint. The model can ignore it, misunderstand it, or generate a recommendation that is not supported by it. The pipeline therefore stores the retrieved passages in every RAG <code>GenerationResult</code>. It also calculates a claim-based grounding diagnostic for the generated rationales. The default mode is a lexical-overlap proxy. An optional semantic mode uses a sentence-transformer encoder when explicitly enabled. The value is labelled with its mode because word overlap is not a faithfulness judge. It can show that a claim shares words with a passage, but it cannot prove that the passage entails the claim.

The RAG method also records whether the context had to be truncated to fit the input budget. This field is important for interpretation. A model that receives three full passages is not under the same context condition as a model that receives only short excerpts. The context-truncation flag allows the results chapter to report this difference instead of treating all RAG calls as identical.

## 5.7 Method C — QLoRA Fine-Tuning

Method C adapts the selected base model to the structured dashboard task with Quantized Low-Rank Adaptation (QLoRA). LoRA freezes the original model weights and learns a low-rank update for selected linear transformations \cite{hu2022lora}. QLoRA combines this idea with 4-bit quantization of the frozen base model and trains the adapter while the quantized base remains fixed \cite{dettmers2023qlora}. In the project, Method C is the main task-adaptation condition and Method D reuses its compatible adapter.

### 5.7.1 Why QLoRA is the primary adaptation method

The first reason for choosing QLoRA is the resource profile of the study. The model matrix spans roughly 1.5B to 27B parameters. Full-parameter fine-tuning would require updating and storing the full parameter set and its optimizer state for every selected model and seed. QLoRA keeps the base weights frozen and limits the learned update to a small adapter while reducing the memory needed to load the base model. This makes a multi-condition master-thesis experiment more feasible. The reason is practical and methodological: a method that cannot be executed consistently across the planned profiles cannot provide a useful comparison.

The second reason is experimental control. The adapter is a separate artifact that can be stored with its base-model identity, seed, dataset version, training configuration hash, and training metadata. Method D can then use the adapter produced by Method C for the same model, dataset, and seed. The difference between C and D is consequently the addition of retrieval at inference time, rather than a newly trained and potentially different adapter. This provenance relation is harder to maintain when every combined condition is fine-tuned independently.

The third reason is alignment with the research question. The thesis asks how prompting, retrieval, and task-specific adaptation affect structured dashboard recommendations. It does not aim to establish a universal ranking of all fine-tuning algorithms. Selecting one well-defined PEFT method for the primary matrix keeps the adaptation factor interpretable. Alternative methods are discussed in Section 5.9 and can be treated as separate ablations, but they are not mixed into the main A–D comparison.

These reasons do not imply that QLoRA is always better than other adaptation methods. QLoRA can introduce quantization error, depends on hardware and software support, and may behave differently across model families. The scientific claim is narrower: QLoRA is an appropriate and reproducible primary method for this resource-constrained structured-generation study, provided that its configuration, adapter provenance, and limitations are reported.

### 5.7.2 Training data and target formatting

The training examples are taken from the frozen <code>dashboard_v4_1</code> package described in Chapter 4. The package contains 2,932 training records, 613 validation records, and 274 test records. The 40-item human-evaluation selection is held apart from training and is not used as an additional training split. The test and human-evaluation records remain byte-identical to their parent records according to the manifest checks. The fact that the training package contains generated enrichment fields is carried into the interpretation of the fine-tuned model: adaptation teaches the model to reproduce the frozen structured representation, but it does not turn AI-generated enrichment into independent expert gold.

For each training record, the formatter creates the system message and user message from the brief and appends the full recommendation as pretty-printed JSON. The target includes the context summary, KPI-to-chart mapping, layout, styling, interactions, and rationales. The primary configuration uses full-sequence loss with <code>completion_only_loss=false</code>. This setting is retained to preserve compatibility with the existing C and D training path. A response-only prompt-completion mode exists in the formatter and trainer, but enabling it creates a separately identified training condition and is not silently substituted into the primary run.

Using the same prompt builder in training and inference is important for this task because small wording changes can alter the model's expected output structure. It also means that the prompt-only and fine-tuned conditions see the same field names and allowed values. The difference between them is the learned adapter, not a hidden change in the task definition. Instruction tuning research shows the value of aligning language-model behavior with explicit instructions; the present training procedure applies that principle to a project-specific structured output contract \cite{wei2022flan}.

### 5.7.3 QLoRA configuration

The authoritative training configuration is <code>src/config/training/qlora_default.yaml</code>. Table 5.5 records the parameters that define the primary adaptation condition. The values are included because the rank, target modules, quantization mode, optimizer, and schedule can all affect the result. Reporting only the word “QLoRA” would not be sufficient to reproduce the experiment.
Table 5.5. Primary QLoRA configuration as requested by `src/config/training/qlora_default.yaml`.


| Configuration group | Parameter | Primary value |
| --- | --- | --- |
| Adapter | Rank \(r\) | 16 |
| Adapter | LoRA scaling \(\alpha\) | 32 |
| Adapter | Dropout | 0.05 |
| Adapter | Bias | <code>none</code> |
| Adapter | Target modules | <code>all-linear</code> |
| Quantization | Weight loading | 4-bit |
| Quantization | Quantization type | NF4 |
| Quantization | Double quantization | Enabled |
| Quantization | Compute dtype request | float16 |
| SFT | Epochs | 3 |
| SFT | Per-device train batch size | 2 |
| SFT | Per-device evaluation batch size | 1 |
| SFT | Gradient accumulation | 4 |
| SFT | Learning rate | \(2.0\times10^{-4}\) |
| SFT | Scheduler | Cosine |
| SFT | Optimizer | <code>adamw_torch</code> |
| SFT | Warm-up ratio | 0.1 |
| SFT | Weight decay | 0.01 |
| SFT | Gradient checkpointing | Enabled |
| SFT | Validation | Every epoch |
| SFT | Best checkpoint criterion | Lowest <code>eval_loss</code> |
| SFT | Packing | Disabled |

The adapter update can be written conceptually as

\[
W' = W + \Delta W,
\qquad
\Delta W = BA,
\]

where \(W\) is a frozen base weight matrix, \(B\) and \(A\) are trainable low-rank matrices, and the rank \(r\) is much smaller than the dimensions of \(W\). The actual implementation uses PEFT to attach the adapter to the supported linear layers. Quantization affects how the frozen base is loaded and computed; it does not mean that the adapter itself is treated as a substitute for the complete base model.

The YAML requests NF4 4-bit loading and double quantization, while the training precision helper resolves the effective hardware-safe precision from <code>precision: auto</code>. The effective precision and the selected device are written to training metadata. This distinction matters because a configuration request and the effective runtime setting are not always identical across different accelerators. A run is therefore interpreted from its saved metadata rather than from the YAML file alone.

The same principle applies to three further parameters that the executed runs resolved differently from the defaults in Table 5.5: the per-device batch size and gradient-accumulation count, the maximum training sequence length, and the numerical precision. All eight trained adapters share the same effective optimisation batch of eight sequences, but they reach it through different combinations depending on the memory of the assigned device, they trained at either 1,024 or 4,096 tokens, and two of them trained in fp16 because the assigned device does not support bfloat16. Section 6.4 records the effective value for every adapter.

One consequence of the shared rank deserves explicit statement, because it is easy to state incorrectly. A fixed LoRA rank does not produce a fixed number of trainable parameters. The number and width of the adapted linear projections depend on the architecture, so rank 16 with <code>all-linear</code> targeting yields a different adapter size for every profile in the matrix. The actual counts recorded by the trainer are reported in Section 6.4; the share of the model that is trainable falls as the base model grows.

### 5.7.4 Validation and adapter provenance

The trainer loads the validation file when the evaluation strategy is not <code>no</code>. For the primary configuration, evaluation and checkpoint saving occur at epoch boundaries, and the trainer records the best validation loss and best checkpoint. The validation split is used for training control and checkpoint selection; the held-out test split remains reserved for evaluation. This separation prevents a test score from being used to select the adapter.

The trainer also checks for non-finite training metrics and trainable weights before continuing or saving an adapter. On completion, it stores the adapter weights, tokenizer, configuration, and <code>training_metadata.json</code>. The metadata includes the base model identifier, model key and revision, model configuration hash, training configuration hash, dataset version, training-file hash, seed, effective precision, parameter counts, optimizer and schedule, training duration, and validation history.

Method D can reuse an adapter only when the adapter resolver and validator confirm the expected identity. The checks cover the base model, model key, model revision where available, model configuration hash, seed, dataset version, and training configuration. If a mismatch is detected, the run fails rather than silently attributing an adapter from another model or seed to Method D. This is a central provenance safeguard because the D–C comparison depends on the two methods sharing the same learned adapter.

## 5.8 Method D — QLoRA Fine-Tuning with Retrieval

Method D combines the task adaptation of Method C with the retrieval process of Method B. It does not train a second independent adapter for the RAG condition in the primary matrix. Instead, it resolves the adapter produced by the corresponding C run and loads it into the same base-model profile. It then builds the RAG query from the brief, retrieves up to three passages from the same frozen knowledge base, fits the passages to the input budget, and adds them to the system prompt before generation.

The dependency is defined in <code>src/config/method/ft_rag.yaml</code> and in the final matrix through <code>adapter_from: C</code>. The resolver maps the dependency to the same model, dataset, and seed. Thus, C seed 43, when executed, is the adapter source for D seed 43. The method would be invalid if D used a C adapter from another seed or from another model size, because the comparison would then contain an uncontrolled training difference.

Method D tests a specific interaction between the two interventions. Retrieval may supply design principles that are absent from the learned task representation, but it may also add irrelevant or conflicting text. Fine-tuning may make the model better at converting the brief into the required JSON structure, but it may also make the model less responsive to new context. The method is therefore not based on the assumption that combining two interventions must improve the result. Its purpose is to measure the combined behavior and to examine whether the retrieved rationales and chart decisions remain compatible with the adapted output style.

The interpretation of D requires special care. A difference between D and A reflects both the adapter and the retrieved context. A difference between D and C is the cleaner test of adding retrieval after adaptation. A difference between D and B is the cleaner test of adding adaptation in the presence of retrieval. The results chapter must keep these comparisons separate rather than presenting one combined score as proof that both components independently caused the change.

## 5.9 Alternative Fine-Tuning Methods and Selection Rationale

QLoRA is the primary method, but it is not the only possible way to adapt a language model. The repository contains or documents several alternatives. They are useful for understanding the design space and for possible ablations, but they are not included in the primary A–D matrix. Their status and the reason for excluding them from the main comparison are given in Table 5.6.
Table 5.6. Alternative adaptation methods and why each is not a primary A–D condition.


| Method | Scientific idea | Project status | Reason it is not a primary A–D condition |
| --- | --- | --- | --- |
| Plain LoRA | Learns low-rank updates while loading the base model without 4-bit quantization \cite{hu2022lora} | Functional optional configuration in <code>lora_default.yaml</code> | Changes the quantization factor as well as the adaptation method and uses more memory |
| AdaLoRA | Allocates a parameter budget adaptively according to the estimated importance of updates \cite{zhang2023adalora} | Not part of the authoritative final matrix | Adds dynamic rank allocation and a second budget policy that would require a separate matched experiment |
| DoRA | Decomposes weight magnitude and direction and applies a low-rank update to the directional component \cite{liu2024dora} | Supported by the trainer and an optional <code>dora.yaml</code> configuration | Tests a different adapter parameterization and should be reported as an ablation, not pooled with QLoRA |
| RSLoRA | Uses rank-stabilized scaling to reduce sensitivity to the adapter rank \cite{kalajdzievski2023rslora} | Supported by the trainer and an optional <code>rslora.yaml</code> configuration | Uses a different rank and scaling rule, so it is not a like-for-like continuation of the primary adapter |
| GaLore | Projects full-parameter gradients into a low-rank subspace while retaining full-parameter training \cite{zhao2024galore} | Optional trainer and configuration | Produces a full-model training condition with different memory, optimizer, and artifact semantics |
| Full fine-tuning | Updates the complete model parameter set | Placeholder and not used in the final matrix | Exceeds the intended resource and provenance scope of the primary experiment |

Plain LoRA is the closest alternative to QLoRA because it uses the same low-rank idea without quantizing the frozen base. It could answer whether quantization changes the adaptation result, but that is a separate question from whether adaptation helps dashboard recommendation. Including it in the main A–D matrix would add a new factor and make the interpretation less direct. It is therefore better treated as a matched quantization ablation with the same data, seed, prompt, evaluation, and training budget wherever such an ablation is actually executed.

AdaLoRA changes the allocation of the parameter budget during training. This can be useful when different layers or updates have different importance, but it introduces additional choices about the budget schedule and importance estimates. DoRA changes the parameterization by separating weight magnitude and direction, while RSLoRA changes the scaling behavior of the low-rank update. Both are scientifically relevant alternatives, and the project includes optional configurations for them. Their outputs should not be combined with QLoRA outputs as if they were repetitions of the same method.

GaLore follows a different resource strategy. It performs full-parameter learning while using a low-rank gradient projection to reduce optimizer memory. This can be valuable when full-model adaptation is a research goal, but it produces a full model rather than the same small adapter artifact used by C and D. The resulting training cost, checkpoint behavior, and provenance checks are different. It is therefore outside the primary controlled comparison.

The selection of QLoRA is consequently based on scope, resource compatibility, and experimental control. It is not based on a claim that QLoRA dominates DoRA, AdaLoRA, RSLoRA, GaLore, or plain LoRA for every task. A future algorithmic comparison would need to match the base model, data, prompt, seed, number of update steps, evaluation split, and output processing before drawing a conclusion about algorithm quality. The current thesis keeps those alternatives separate so that its main question remains answerable.

## 5.10 Inference and Output Processing

### 5.10.1 Loading and preparing a method

The inference runner reads the frozen test file and the selected method configuration. Method A and Method B load the base model. Method C loads the base model and its own adapter. Method D validates and loads the adapter produced by Method C before setting up the retriever. For C and D, the method configuration requests 4-bit inference loading with NF4 and double quantization. A and B use the base-model runtime profile. This difference is part of the actual implementation and is considered when latency and memory are interpreted.

For each item, the method constructs the system prompt and user prompt, retrieves context if required, and calls the model's prompt preparation function. The preparation function counts input tokens and returns the input budget. It does not silently truncate the dashboard brief. In a RAG condition, context fitting occurs before final tokenization, and the resulting system prompt is recorded through the retrieval fields and truncation flag.

The generation call uses the shared settings from the method YAML files: a maximum of 512 new tokens, temperature 0.1, top-p 0.9, sampling enabled, and a repetition penalty of 1.15. The maximum is a safety boundary for the structured response; it is not a statement that every response requires 512 tokens. If the model reaches the boundary, the raw output and any parse failure are retained so that truncation is visible in the error analysis.

### 5.10.2 Parsing and re-parsing

The raw response is passed to <code>parse_json_safe</code>. The function extracts a JSON object, normalizes a limited set of spelling and alias variants, and attempts to construct the runtime <code>DesignOutput</code> object. The <code>GenerationResult</code> stores the raw text, the parsed result or <code>null</code>, and a parse-error code such as <code>no_json_found</code> or <code>schema_error</code>. For RAG methods it also stores the retrieved documents and their scores.

Before evaluation, cached predictions can be reparsed with the current parser. This permits an improvement to the parser to be applied consistently to existing raw outputs. Strict schema metrics do not use the lenient parsed object as their sole evidence; they re-extract the raw JSON object and apply the strict validation path. The two paths therefore answer different questions: the lenient path asks whether a usable structured object can be recovered, while the strict path asks whether the model emitted a valid object without repair.

### 5.10.3 Failure handling and artifacts

The runner joins predictions and references by <code>item_id</code>, writes predictions as JSONL, records item-level errors, and writes automatic and layered metrics. A failed item is not silently removed from the denominator. If a run becomes incomplete, the run status and error artifact make that fact visible. The output layout includes the experiment identifier, model, method, seed, configuration hash, and relevant result files.

The cache identity includes a short SHA-256 prefix of the resolved configuration. This prevents a prediction generated under one method setting from being reused without notice after the setting changes. The hash is not a replacement for storing the full resolved configuration; it is an index used together with the configuration and run metadata.

Latency is measured by the method wrapper around the generation call after item preparation. Retrieval and prompt preparation occur before this timer in the shared implementation. Consequently, the recorded latency is primarily model-generation latency and should not be described as complete end-to-end user latency unless retrieval and preparation are measured separately. This boundary is important for a fair interpretation of the RAG condition.

## 5.11 Evaluation Protocol

The evaluation is layered because no single automatic metric can establish that a generated dashboard recommendation is useful. A valid JSON object is necessary for downstream use, but it is not sufficient for a good visualization. A chart token that matches a synthetic reference is also not sufficient evidence that users would perform a task effectively with that chart. The protocol therefore separates technical format checks, internal structured diagnostics, robustness, retrieval and grounding diagnostics, independent chart-effectiveness evidence, realism, and human usefulness.

### 5.11.1 Technical and structured-output metrics

The implemented schema metrics report JSON extraction, required-key presence, strict raw-schema validity, encoding-object validity, completeness, and field coverage. Strict schema validity checks the raw extracted object against the strict contract. Empty objects, empty arrays, missing fields, invalid enum values, and invalid mapping structures are not treated as complete success. Parse failures count in the denominator, so a method cannot improve its schema rate by omitting difficult items.

The chart-type metrics compare the primary predicted chart with the primary chart in the frozen synthetic reference. Top-1 counts a parse failure or missing chart as incorrect, so it does not reward a method for omitting the items it finds difficult.

The synthetic chart and macro-F1 scores are labelled <code>internal-circular</code> in the reporting layer. The reason is that the chart labels and some structural targets originate from the project data-generation and transformation pipeline. Agreement with those labels is useful for checking whether the implementation is working and whether methods learn the frozen representation. It is not independent evidence of human-effective chart choice. This limitation follows the distinction between a formal output contract and the broader effectiveness of a visualization \cite{mackinlay1986apt,munzner2014visualization}.

### 5.11.2 Robustness and behavioral checks

The runner supports a paraphrased test variant and a missing-information variant. Paraphrase consistency measures whether the output remains stable after the wording changes while the intended task remains similar. Paraphrase accuracy compares the variant with the synthetic reference, but it inherits the internal-circular limitation of that reference. Consistency and correctness are therefore reported separately. A model can be consistently wrong.

The missing-information condition checks whether the model asks for clarification or produces a structured response when important information is absent. The current clarification measure uses a regular-expression heuristic over the generated text. It is an implementation diagnostic, not a complete measure of whether the model recognized the ambiguity. A high missing-information schema rate is not automatically good: a model that confidently produces a complete-looking recommendation from an under-specified brief may be less trustworthy than a model that requests the missing information.

These perturbation tests follow the broader principle that accuracy on one fixed test form is not enough to assess model behavior. Behavioral testing can reveal failures under controlled changes to the input \cite{ribeiro2020checklist}. In the present study, the perturbations are limited to the variants implemented in the repository and should not be generalized to all possible paraphrases or incomplete briefs.

### 5.11.3 Retrieval and grounding

For RAG methods, the pipeline records the ranked chunk identifiers, scores, and text. Retrieval relevance metrics such as Recall@3, MRR@3, and nDCG@3 are available only when explicit human or expert qrels are configured. The default evaluation configuration does not contain such qrels, so the reporting layer marks retrieval relevance as unavailable rather than inferring relevance from lexical overlap.

The grounding metric checks the rationale claims against the retrieved passages. In semantic mode, a claim is considered supported when its maximum cosine similarity to a retrieved passage reaches the configured threshold. In the default lexical-proxy mode, support is based on content-word overlap. Both modes have limitations. Similarity is not entailment, and a retrieved passage can be relevant even when the generated explanation contains an unsupported conclusion. The mode, claim coverage, number of scored claims, and number of items with retrieved documents are therefore reported together.

The RAG metrics answer different questions. Retrieval relevance asks whether the selected chunks match an independent relevance judgement. Grounding asks whether generated rationale text resembles the retrieved context. Neither metric directly measures whether the final chart and layout are effective. The results chapter must keep these quantities separate from human usefulness.

### 5.11.4 Independent effectiveness and human evaluation

The evaluation protocol defines an independent chart-effectiveness layer based on human-effectiveness evidence from Saket et al. \cite{saket2019taskbased} and Kim and Heer \cite{kim2018taskdata}. The intended score is set-valued membership: a predicted primary chart is correct when it belongs to the human-effective set for the relevant task and data shape. Items without coverage in the independent table are excluded from that accuracy denominator and reported as uncovered. This is a stronger validity anchor than comparing only with the project's synthetic single-label rule, but the scorer is documented as designed and not yet implemented in the current evaluation code.

The planned usefulness layer uses human ratings on six dimensions: chart appropriateness, layout quality, styling and accessibility, interaction design, rationale quality, and overall usefulness. Each dimension uses a 1–5 rubric. The infrastructure for assignment, storage, inter-rater reliability, and the rating interface exists, but no human ratings have been collected in the current project state. Consequently, this thesis must not claim human usefulness or overall dashboard quality from automatic metrics alone.

Human evaluation should follow explicit reporting practices: the rating task, rater instructions, sampling, number of raters, aggregation rule, and agreement measure must be reported \cite{vanderlee2019human}. The planned protocol uses paired outputs for the same briefs and treats the six dimensions as separate outcomes. If ratings are collected, the analysis uses the documented ordinal statistics and reports uncertainty and inter-rater reliability. The absence of ratings is reported as a limitation rather than filled with an LLM-judge score.

### 5.11.5 Statistical comparison

The evaluation code and protocol define paired comparisons because the same items are evaluated under several methods. Binary outcomes such as strict schema validity or covered chart membership can be compared with an omnibus Cochran's \(Q\) test and exact pairwise McNemar tests with Holm correction. Continuous or ordinal paired measures such as completeness and human rubric scores can be analysed with Friedman and Wilcoxon procedures where their assumptions and sample sizes permit. Bootstrap confidence intervals are reported for rates and paired differences when the required per-item vector is available.

These procedures are part of the analysis plan, not evidence that a statistical test has already been completed for every model and method. A confidence interval cannot repair a biased gold set, and a significant difference on an internal-circular metric cannot establish a better dashboard design. The statistical result must therefore be reported together with the evidence layer, the sample size, the seed coverage, and the status of the metric implementation.

## 5.12 Reproducibility and Provenance

Reproducibility is implemented through configuration, data hashes, model and adapter metadata, raw-output retention, and explicit run identities. The environment is represented by <code>pyproject.toml</code> and <code>poetry.lock</code>. The resolved configuration is written with the run, and a SHA-256 configuration hash is used in the cache identity and result metadata. The goal is not only to allow a later rerun; it is also to make it possible to determine which data, method, model, and settings produced a prediction.

### 5.12.1 Dataset and knowledge-base anchors

The dataset and knowledge-base identifiers used by the final methods are shown in Table 5.7. The full manifest contains additional source and report hashes. The exact values are included here for the files that directly define the test and retrieval inputs.
Table 5.7. Dataset and knowledge-base anchors for the executed runs.


| Artifact | Version or file | SHA-256 or identifier |
| --- | --- | --- |
| Dataset | <code>dashboard_v4_1</code> manifest | <code>291f57a50b0f7fa7c1c515031468d0b9458d9a11fb353f3cdf00c66168a3729e</code> |
| Dataset test split | <code>test.jsonl</code> | <code>e2df055d0a75c25f53a88cb830a5b6d66411fa179413f352f45ca2f6873829d5</code> |
| Human-evaluation selection | <code>human_eval_test_items_40.csv</code> | <code>2c336ee26398e8e487991df7e1c6e31385787c555228962d77e811f9a920cb8a</code> |
| Dataset schema | <code>schema.json</code> | <code>50e264be71f5d052f489e9a2753c4d878ec6b1ea17b8b38e3b19cf8cd0129183</code> |
| Knowledge base, as executed | <code>kb_version</code> | <code>3d1409d8347fc165</code> |
| Knowledge-base chunks, as executed | <code>chunks.jsonl</code> | <code>19fbbfa4ad333dec7298f53c98b25bf5a8bf5059236e8fa4885febede0e54159</code> |
| Knowledge base, repository manifest | <code>kb_version</code> | <code>46b8575d98d37312</code> |
| Knowledge-base chunks, repository copy | <code>chunks.jsonl</code> | <code>cad3bb0c1606fab3062d3235740d7701e1fb8ac46702af8b09f615f66bfc3afb</code> |

The dataset manifest identifies 2,932 training records, 613 validation records, 274 test records, and 40 human-evaluation items. It also records that the generated enrichment fields are not gold, that protected structural fields were kept unchanged, and that duplicate and leakage checks passed at freeze time. Chapter 4 provides the full transformation and provenance discussion. Repeating the anchors here connects the model methods to the exact frozen inputs without redefining the dataset construction chapter.

The knowledge-base manifest records the three source hashes and the deterministic builder settings. A change to a guideline document, a heading boundary, a minimum chunk size, or the chunk identifier function changes the knowledge-base artifact. Such a change must receive a new version and must not be described as the same RAG condition.

### 5.12.2 Model and adapter provenance

The model configuration records the Hugging Face identifier, nominal size, family, maximum sequence length, preferred dtype, chat-template options, token requirement, and model revision field. The current revision fields are <code>null</code>. This means that the identifier is known but an immutable upstream commit is not pinned in the final configuration. Exact model revisions should be recorded for the final published runs if the upstream repository supports it. Until then, model identity is less reproducible than the frozen dataset and knowledge base.

Every adapter stores its own base model identifier, model key, model revision field, model configuration hash, training configuration hash, dataset version, seed, training-file hash, and validation history. Method D validates these fields before loading the adapter. This prevents a technically loadable but scientifically incompatible adapter from being used in a final comparison.

The run manifest also records Python, PyTorch, CUDA, device, and package information for executed runs. The project uses a precision helper because the safe effective dtype can depend on the available hardware. The saved effective precision is therefore more authoritative than a generic statement such as “the model was trained in mixed precision.”

### 5.12.3 Reproducibility boundaries

The project deliberately keeps raw model text. This is necessary because parser versions can change, and because a parsed object alone cannot show whether a response contained extra prose, invalid enum values, or a truncated JSON object. The results can be reparsed from the raw text, while strict metrics can still use the raw extracted object.

Seed recording, configuration hashes, and fixed input files improve reproducibility but do not guarantee exact output equality. The inference entry point records the seed in the result object, but stochastic generation and hardware-specific operations can still differ. The training seed is passed to the trainer, while the project seed helper avoids a CUDA seeding call that caused a Windows DLL load-order problem. This implementation choice is documented because it affects the strength of any claim about bitwise reproducibility.

The final methods are therefore reproducible in the sense of documented inputs, code paths, configuration, and provenance checks, subject to the unpinned upstream model revisions and hardware-dependent numerical behavior. The results chapter must state these boundaries whenever it reports repeated runs or compares models from different families.

## 5.13 Methodological Limitations and Threats to Validity

### 5.13.1 Construct validity

The first limitation concerns what the automatic metrics actually measure. JSON parsing, required-key coverage, strict schema validity, and completeness are meaningful for a system that must produce a machine-readable recommendation. They do not measure whether a dashboard is visually clear, whether a user can complete a task quickly, or whether the layout supports a real decision. The structured output contract is an operational definition of the system interface, not a complete definition of dashboard quality.

The chart-type comparison has a related limitation. The frozen synthetic reference provides one target representation, but visualization tasks can admit several valid charts. The project therefore plans an independent set-valued effectiveness layer and a human usefulness layer. Until these layers are implemented and rated, automatic chart agreement must be described as an internal diagnostic or as format-level evidence. This boundary must be stated directly in the reported tables.

### 5.13.2 Dataset and supervision validity

The dataset is based mainly on nvBench-derived analytical examples and project transformations. Chapter 4 records which fields originate from the source and which enrichment fields were generated during the project. The generated fields are useful for constructing a complete training target, but they are not human expert annotations. Fine-tuning on them can teach the model the conventions and errors of the transformation pipeline. It cannot establish that the generated layout, styling, interaction, or rationale is the best design for a user.

The test split is protected from the later semantic repair and is byte-identical to the parent according to the frozen manifest. This protects the split from the enrichment operation, but it does not remove the broader source limitation. The coverage is still concentrated in the source task and domain distribution, and complex multi-KPI dashboards may be underrepresented. Results should therefore be generalized to this structured recommendation setting, not to all business dashboards.

### 5.13.3 Experimental confounds

Model-family comparisons include more than parameter count. The Qwen3 profiles and the OLMo 2 profile differ in tokenizer, pretraining data, post-training recipe, architecture, native context window, and model-specific chat template. A result difference between the two families cannot be attributed to size or to any one of these properties alone. The thesis treats model family as an experimental factor and avoids language such as “the larger model is better because it is larger” unless the evidence supports that narrower comparison.

Seed coverage is also not symmetric by design. Repeated seeds are concentrated on the small-model stability condition, while larger profiles may be run first with seed 42. This gives the study a feasible staged execution but limits claims about variance for larger models. A single-seed result can show the behavior of that executed condition; it cannot show the expected distribution across seeds.

The primary QLoRA configuration also introduces a quantization choice, a specific adapter rank, a target-module policy, a schedule, and a full-sequence loss. The result is evidence about this QLoRA configuration, not about every possible QLoRA setting. Similarly, the A–D comparison does not establish that QLoRA is the best adaptation algorithm. Alternative methods must be evaluated in a separately matched experiment.

### 5.13.4 Retrieval and grounding limitations

The RAG corpus contains only three project guideline documents and 41 chunks. It may not cover all visualization principles, domain terminology, or user constraints. TF-IDF is transparent and suitable for this small corpus, but lexical similarity can miss a useful paraphrase and can retrieve a passage because of shared words without understanding the design task. The fixed <code>top_k=3</code> policy also creates a trade-off between coverage and context length.

The default grounding score is a lexical proxy and the optional semantic score is a similarity test. Neither one establishes logical entailment or faithful causal use of the retrieved passage. Retrieval relevance is unavailable without independent qrels. A RAG result can therefore be discussed as showing retrieved-context behavior and a grounding diagnostic, but not as proof that the model followed a design rule.

Method D depends on Method C. If an adapter is damaged, incompatible, or trained with a different dataset, D cannot be interpreted as a clean combined condition. The adapter validator reduces this risk, but it does not prove that the adapter learned only the intended signal. It also means that failures in the C training pipeline can propagate to D.

### 5.13.5 Measurement and implementation limitations

The parser can repair some spelling and format errors. This is useful for a recoverable-output analysis but can make a lenient metric look better than the raw model output. The strict metric and raw-text retention reduce this risk, but the results must state which path was used. Similarly, the missing-information clarification rate is based on a regular-expression heuristic and should not be interpreted as a complete test of uncertainty recognition.

The recorded latency begins after item preparation in the shared method wrapper. RAG retrieval and context fitting are therefore not necessarily included in the same way as model generation. Memory use also differs between the base and 4-bit adapter conditions. These measurements are useful for the implemented pipeline, but they are not a complete deployment benchmark.

The model revisions in the final configurations are not yet immutable commit identifiers. The repository can preserve the code, dataset, and configuration used by a run, but an upstream model repository can change its default revision. Final published runs should record the resolved model commit or revision, together with the environment metadata. Until this is done, the model part of reproducibility is weaker than the dataset and knowledge-base part.

### 5.13.6 External validity and current evidence status

The system produces recommendations for a structured brief and does not render or test a dashboard with real users in the current automatic pipeline. It therefore cannot support claims about adoption, decision quality, time-on-task, or practical business value without the planned human evaluation and, where relevant, a rendered-dashboard study. The current experiment can establish how the methods behave on the frozen structured task and how reliably they satisfy the output contract.

The current evidence status is consequently part of the method: L2 format and robustness checks are implemented; synthetic chart agreement is internal-circular; semantic grounding is opt-in; independent human-effectiveness scoring is designed but not implemented; the realism layer is pending; and human usefulness ratings have not yet been collected. The next chapter reports only the completed conditions and metrics, separates internal diagnostics from independent evidence, and labels incomplete seed or model coverage instead of filling the gaps with assumptions.

## References Used in Chapter 5

Generated from `docs/thesis/references.bib` by `experiments/scripts/build_chapter_reference_lists.py`. The BibTeX key of each entry is given in brackets.

Allen Institute for AI (2025). *OLMo-2-0425-1B-Instruct Model Card*. Hugging Face model repository. Revision 48d788eca847d4d7548f375ad03d3c9312f6139e; Apache-2.0; accessed 2026-09-10. https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct [`allenai2025olmo2instruct1b`]

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. Advances in Neural Information Processing Systems, 36, 10088–10115. https://doi.org/10.52202/075280-0441 [`dettmers2023qlora`]

Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N., et al. (2025). *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models*. https://doi.org/10.48550/arXiv.2501.10868 [`geng2025structured`]

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference on Learning Representations. https://openreview.net/forum?id=nZeVKeeFYf9 [`hu2022lora`]

JSON Schema (2022). *JSON Schema Draft 2020-12*. JSON Schema specification. https://json-schema.org/draft/2020-12 [`jsonschema2022draft`]

Kalajdzievski, D. (2023). *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*. arXiv:2312.03732. https://arxiv.org/abs/2312.03732 [`kalajdzievski2023rslora`]

Kim, Y., & Heer, J. (2018). *Assessing Effects of Task and Data Distribution on the Effectiveness of Visual Encodings*. Computer Graphics Forum, 37(3), 157–167. https://doi.org/10.1111/cgf.13409 [`kim2018taskdata`]

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., et al. (2025). *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*. https://doi.org/10.48550/arXiv.2411.15124 [`lambert2025tulu3`]

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems, 33, 9459–9474. https://arxiv.org/abs/2005.11401 [`lewis2020rag`]

Liu, S., Wang, C., Yin, H., Molchanov, P., Wang, Y. F., Cheng, K., et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. Proceedings of the 41st International Conference on Machine Learning, 235, 32100–32121. https://proceedings.mlr.press/v235/liu24bn.html [`liu2024dora`]

Mackinlay, J. (1986). *Automating the Design of Graphical Presentations of Relational Information*. ACM Transactions on Graphics, 5(2), 110–141. https://doi.org/10.1145/22949.22950 [`mackinlay1986apt`]

Munzner, T. (2014). *Visualization Analysis and Design*. CRC Press. https://doi.org/10.1201/b17511 [`munzner2014visualization`]

OLMo Team, Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., et al. (2025). *2 OLMo 2 Furious*. https://doi.org/10.48550/arXiv.2501.00656 [`olmo2025furious`]

Qwen Team (2026). *Qwen3.8-Max: A New Bar for Coding and Cowork*. Qwen blog. Citation given on the Qwen3.8-27B model card; accessed 2026-09-10. https://qwen.ai/blog?id=qwen3.8 [`qwen2026qwen38`]

Qwen Team (2026). *Qwen3.8-27B Model Card*. Hugging Face model repository. Revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; Apache-2.0; accessed 2026-09-10. https://huggingface.co/Qwen/Qwen3.8-27B [`qwen2026qwen3827bcard`]

Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 4902–4912. https://doi.org/10.18653/v1/2020.acl-main.442 [`ribeiro2020checklist`]

Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333–389. https://doi.org/10.1561/1500000019 [`robertson2009bm25`]

Saket, B., Endert, A., & Demiralp, C. (2019). *Task-Based Effectiveness of Basic Visualizations*. IEEE Transactions on Visualization and Computer Graphics, 25(7), 2505–2512. https://doi.org/10.1109/TVCG.2018.2829750 [`saket2019taskbased`]

Sparck Jones, K. (1972). *A Statistical Interpretation of Term Specificity and Its Application in Retrieval*. Journal of Documentation, 28(1), 11–21. https://doi.org/10.1108/eb026526 [`sparckjones1972idf`]

Unknown author (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550 [`karpukhin2020dpr`]

van der Lee, C., Gatt, A., van Miltenburg, E., Wubben, S., & Krahmer, E. (2019). *Best Practices for the Human Evaluation of Automatically Generated Text*. Proceedings of the 12th International Conference on Natural Language Generation, 355–368. https://doi.org/10.18653/v1/W19-8643 [`vanderlee2019human`]

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems, 30, 5998–6008. https://arxiv.org/abs/1706.03762 [`vaswani2017attention`]

Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., et al. (2022). *Finetuned Language Models are Zero-Shot Learners*. International Conference on Learning Representations. https://arxiv.org/abs/2109.01652 [`wei2022flan`]

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Gao, G., et al. (2025). *Qwen3 Technical Report*. arXiv preprint arXiv:2505.09388. https://doi.org/10.48550/arXiv.2505.09388 [`yang2025qwen3`]

Zhang, Q., Chen, M., Bukharin, A., Karampatziakis, N., He, P., Cheng, Y., et al. (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*. International Conference on Learning Representations. https://arxiv.org/abs/2303.10512 [`zhang2023adalora`]

Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., & Tian, Y. (2024). *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection*. Proceedings of the 41st International Conference on Machine Learning, 235, 61121–61143. https://proceedings.mlr.press/v235/zhao24s.html [`zhao2024galore`]
