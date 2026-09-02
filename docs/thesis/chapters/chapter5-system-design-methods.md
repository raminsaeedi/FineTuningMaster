# 5 System Design and Methods

In this chapter we formalize the dashboard design recommendation task, describe the structured-output contract, and present the four method families A–D together with the common prompting and generation pipeline, inference procedure, and reproducibility safeguards implemented in the thesis repository. The exposition is aligned with the implementation record in `METHODS_IMPLEMENTATION_AND_REPRODUCIBILITY` and focuses on scientific motivation, formalization, and methodological choices rather than repeating low-level configuration details.\\cite{MethodsRecord2026}

## 5.1 Formal Problem Formulation

We view dashboard recommendation as a structured prediction problem. An input \(x\\in\\mathcal{X}\) is a dashboard brief encoded as natural language text plus structured fields for users, goals, KPIs, columns, and constraints.\\cite{MethodsRecord2026} An output \(y\\in\\mathcal{Y}\) is a structured design recommendation following the `DesignOutput` schema implemented in `src/core/schemas.py`.\\cite{MethodsRecord2026}

At the semantic level, we can decompose
\\[
  y = (c, m, l, s, i, r),
\\]
where \(c\) is a context summary, \(m\) is an array of KPI-to-chart mappings, \(l\) captures layout decisions, \(s\) encodes styling choices, \(i\) describes interactions, and \(r\) stores rationales. Each component is realized as a JSON-serializable Pydantic object with its own sub-fields and constraints.\\cite{MethodsRecord2026} The system therefore learns a conditional distribution
\\[
  p_\\theta(y\\mid x) = p_\\theta(c, m, l, s, i, r \\mid x),
\\]
where \(\\theta\) denotes the parameter state of a particular model family and method (e.g., base Qwen3-1.7B or OLMo-2-1B, with or without a QLoRA adapter, with or without RAG context).

Given a frozen training set
\\[
  D_{\\text{train}} = \\{(x_j, y_j)\\}_{j=1}^n,
\\]
constructed from `GoldItem` records in the `dashboard_v4_1` package, the fine-tuned methods minimize a sequence-to-sequence cross-entropy loss over the serialized JSON representation of \(y_j\) subject to the schema contract.\\cite{MethodsRecord2026} This formulation follows the general paradigm of structured prediction in NLP, where models learn to emit complex objects—such as sequences, trees, or multi-field records—rather than single labels.\\cite{AdversarialStructured2020,FlexibleStructured2020} Recent work on structured information inference shows that transformer-based language models can infer multi-field device-level information directly from textual input and update structured scientific datasets, supporting our choice of a single decoder-only model per experimental condition.\\cite{StructuredInference2024}

**Definition 5.1 (Method family mapping).** Let \(f_\\theta : \\mathcal{X} \\to \\mathcal{Y}\) denote the mapping implemented by the common prompting and generation pipeline together with a parameter state \(\\theta\) for a given base model (Qwen3 or OLMo 2) and method (A–D). Methods are distinguished by whether retrieval-derived context \(r(x)\) is concatenated to the prompt and whether \(\\theta\) includes a task-specific LoRA adapter or corresponds to the base model alone.\\cite{MethodsRecord2026}

**Proposition 5.2 (Controlled variation across methods).** Under the implementation described in the repository, all four methods share the same high-level mapping
\\[
  x \\mapsto (\\text{prompt}(x, r(x)), \\text{model}, \\theta) \\mapsto y,
\\]
where the prompt builder, tokenizer chat template, schema-based parser, and evaluation metrics are held fixed.\\cite{MethodsRecord2026} Method A sets \(r(x)=\\emptyset\) and uses base parameters \(\\theta_0\). Method B uses \(r(x)\\neq\\emptyset\) (RAG context) with \(\\theta_0\). Method C sets \(r(x)=\\emptyset\) and uses adapted parameters \(\\theta_0+\\Delta\\theta_{\\text{LoRA}}\). Method D uses both \(r(x)\\neq\\emptyset\) and \(\\theta_0+\\Delta\\theta_{\\text{LoRA}}\).\\cite{MethodsRecord2026} Consequently, differences between A–D for a fixed model family and seed can be attributed to retrieval and parameter adaptation, up to confounders such as model-family architecture and incomplete seed coverage documented in the implementation record.\\cite{MethodsRecord2026,AllenAI_OLMo2,An2024Qwen2.5}

### 5.1.1 Experimental Model and Seed Matrix

While the implementation record defines a symmetric final matrix over multiple models and seeds, the experiments actually executed in this thesis follow a compute-constrained design.\\cite{MethodsRecord2026} We therefore restrict the model and seed coverage as follows.

**Model families and sizes.** We focus on two dense decoder-only Transformer families, Qwen3 and OLMo 2, and a subset of sizes:

- Qwen3-1.7B, with 28 layers, hidden size 2048, 16/8 grouped-query attention heads, RoPE, SwiGLU, RMSNorm, and QK-Norm.
- Qwen3-8B and Qwen3-14B, which increase depth and feed-forward width while retaining the same core architectural components.
- OLMo-2-0425-1B-Instruct, an approximately 1.49B-parameter model with 16 layers, hidden size 2048, 16/16 full multi-head attention heads, RoPE, SwiGLU, QK-Norm, and reordered/output RMSNorm.\\cite{AllenAI_OLMo2,AllenAI_OLMo2_1B,An2024Qwen2.5}

This selection provides one size-matched pair at the smallest scale—Qwen3-1.7B versus OLMo-2-1B—where overall capacity is comparable but attention and normalization designs differ, and two larger Qwen3 models that extend capacity without introducing additional architectural confounders.\\cite{AllenAI_OLMo2}

**Seed coverage.** Let \(M_{\\text{family},\\text{method}}(s)\) denote an evaluation metric (e.g., schema-compliance rate or macro-F1) computed for a given model family, method (A–D), and random seed \(s\). For the Qwen3-1.7B and OLMo-2-1B pair we train and evaluate Method C (QLoRA fine-tuning) and the derived Method D (FT+RAG) under three distinct seeds (42, 43, 44), and report
\\[
  \\hat{M}_{\\text{family},\\text{method}} = \\frac{1}{3} \\sum_{s \\in \\{42,43,44\\}} M_{\\text{family},\\text{method}}(s),
\\]
with an empirical seed variance that quantifies stochastic variability in adapter training.\\cite{MethodsRecord2026,Zhou2025Seeds}

For larger models Qwen3-8B und Qwen3-14B we restrict QLoRA fine-tuning and evaluation to a single seed (42) because of limited compute and time; these runs provide additional evidence on behavior at higher capacities but do not support full seed-variance analysis. No OLMo-2-7B oder OLMo-2-13B models are fine-tuned or evaluated in this thesis, even though the PDF analysis identifies them as conceptually interesting counterparts to Qwen3-8B and Qwen3-14B.\\cite{AllenAI_OLMo2}

**Resource-aware justification and QLoRA choice.** Training and evaluating multiple seeds and large models simultaneously is expensive even with parameter-efficient methods. QLoRA substantially reduces memory and compute requirements by quantizing base weights and updating low-rank adapters, but the total budget still scales with model size, number of seeds, und number of methods.\\cite{dettmers2023qlora,FinLoRA2025,GreenCodeSummarization2025,VizQLoRA2023} Recent work on Green AI emphasizes explicitly trading off model size, number of variants, und evaluation depth under fixed resource constraints, rather than implicitly over-committing to symmetric experimental designs that cannot be fully executed.\\cite{GreenAIReview2024,GreenAIEnsembles2024}

Given the limited GPU time and energy budget of a Master thesis, we therefore (i) select QLoRA as the sole fine-tuning algorithm, (ii) allocate three seeds to the smallest, most interpretable architecture pair (Qwen3-1.7B vs. OLMo-2-1B), and (iii) limit larger Qwen3 models to a single seed. The results chapter explicitly separates claims supported by multi-seed evidence from those based on single-seed runs and discusses the limitations that arise from this compute-bounded design.\\cite{MethodsRecord2026}

## 5.2 Structured Output Schema

The output contract is represented by Pydantic models for `DesignOutput` und its subcomponents und summarized by the generated JSON Schema artifact at `data/frozen/dashboard_v4/schema.json`.\\cite{MethodsRecord2026} At the top level, `DesignOutput` exposes six fields: `context_summary`, `kpi_chart_mapping`, `layout`, `styling`, `interactions`, und `rationales`. Each KPI-to-chart mapping in `kpi_chart_mapping` contains a `kpi` string, `task_type` und `chart_type` enum fields, optional `alternatives`, und an `encoding` object that records data-to-visual mappings.\\cite{MethodsRecord2026}

From a visualization-science perspective, the schema encodes the decomposition advocated by Mackinlay’s APT framework, which formalizes the mapping from data und task to graphical encodings, und Munzner’s nested model, which distinguishes analytical tasks, encoding choices, und layout decisions as separate but interconnected design levels.\\cite{mackinlay1986apt,munzner2014visualization} The controlled vocabularies for \(\\texttt{task\\_type}\) (e.g., trend, comparison, correlation) und \(\\texttt{chart\\_type}\) (e.g., line, bar, scatter, treemap) instantiate this theory in a machine-readable form that can be enforced by the parser und strict evaluator.\\cite{MethodsRecord2026}

The implementation record describes multiple validation layers: JSON extraction, lenient post-processing that normalizes enum spellings, Pydantic parsing into `DesignOutput`, strict evaluation of raw JSON objects for required keys und non-empty content, und frozen-dataset validation of `GoldItem` wrappers.\\cite{MethodsRecord2026} These layers separate recoverable serialization variance (e.g., spelling errors) from substantive design validity und prevent defaults in the Pydantic model from inflating schema-compliance metrics.

## 5.3 Common Prompting and Generation Pipeline

All four methods use a shared prompting und generation pipeline implemented in `HFMethod` und `RAGHFMethod`.\\cite{MethodsRecord2026} The system message instructs the model to act as an expert dashboard design consultant und to respond with a single valid JSON object that follows the schema exactly, without Markdown fences oder extra commentary. The user message renders the dashboard brief und enumerates the required top-level keys, mapping-entry fields, allowed task types und chart types, und rationale fields, including a compact JSON-shaped example.\\cite{MethodsRecord2026}

The same prompt construction logic is used for both training und inference: `src/data_pipeline/formatter.py` imports the prompt builder und serializes gold recommendations as indented JSON, appending the tokenizer end-of-sequence token when available.\\cite{MethodsRecord2026} This minimizes train–test prompt-template mismatch, a known source of instability in instruction-tuned LLMs.\\cite{Pauk2026} Generation settings such as `max_new_tokens`, `temperature`, `top_p`, und `repetition_penalty` are configured via Hydra und kept identical across methods A–D for final runs, with shorter, deterministic settings reserved for smoke tests.\\cite{MethodsRecord2026}

At runtime, the model wrapper loads the tokenizer (from the base model oder adapter directory), applies the chat template, bounds input length by the model’s maximum sequence length, und generates new tokens under the configured sampling regime.\\cite{MethodsRecord2026} The parser then extracts JSON (including fenced oder brace-delimited objects), normalizes known aliases, drops unsupported mapping entries, und parses into `DesignOutput` when possible, retaining raw text und parse errors in `GenerationResult` for forensic review.

## 5.4 Method A — Prompt-Only (Baseline)

Method A, registered as `prompt_only`, serves as the baseline. It loads the selected base model und tokenizer, builds the common system und user messages, generates under the shared sampling configuration, und parses the response into `DesignOutput` without any adapter oder retriever.\\cite{MethodsRecord2026}

For the final experimental design, the Qwen3 family provides three dense decoder-only Transformers—Qwen3-1.7B, Qwen3-8B, und Qwen3-14B—that share architectural patterns such as grouped-query attention, RoPE positional encoding, SwiGLU activation functions, RMSNorm und QK-Norm, but differ in depth, width, und feed-forward dimensionality.\\cite{An2024Qwen2.5} The OLMo 2 family provides size-matched dense decoder-only models with full multi-head attention und reordered/output RMSNorm, plus fully documented training data und recipes sowie Apache 2.0 licensing.\\cite{AllenAI_OLMo2,AllenAI_OLMo2_1B}

Prompt-only corresponds to a closed-book condition: the model can only draw on parametric knowledge acquired during pretraining und post-training. Large decoder-only LLMs have demonstrated strong zero-shot und few-shot capabilities in structured prediction und information extraction when guided by well-designed prompts und schemas.\\cite{StructuredInference2024,Pauk2026} In this thesis, Method A thus provides a baseline for how much structure und task alignment can be achieved without retrieval oder fine-tuning, und it defines the reference against which Methods B–D are compared for each model family und seed.

## 5.5 Method B — Retrieval-Augmented Generation

Method B, registered as `rag`, augments the baseline with a non-parametric knowledge base und a retrieval step, following Retrieval-Augmented Generation (RAG).\\cite{MethodsRecord2026} The authoritative configuration uses a local corpus of three Markdown guideline documents (accessibility, chart selection, dashboard design) that is chunked at heading boundaries into 41 passages, hashed, und versioned as a frozen knowledge base in `data/knowledge_base`.\\cite{MethodsRecord2026}

### 5.5.1 Knowledge Base Construction

The knowledge-base builder enumerates Markdown source files, sorts their names, splits at heading lines, keeps each heading together with its body text, discards chunks below a configured word-count threshold, assigns deterministic identifiers derived from source content und heading context, und writes one JSON record per chunk.\\cite{MethodsRecord2026} A manifest stores source byte sizes und SHA-256 hashes, und the KB version is derived from the sorted source-hash manifest, making guideline-provenance auditable.

This design mirrors contemporary RAG practice, where domain corpora are materialized as chunked, hashed collections that support provenance tracking und integrity checks.\\cite{Pinecone2023Chunking,FedRAG2026} Chunking at headings preserves semantic coherence und interpretability, at the cost of sometimes mixing multiple rules within longer sections.

### 5.5.2 Retrieval Algorithm

The authoritative retriever fits a `TfidfVectorizer` with English stop-word removal over the chunk texts und, für each brief, builds a query from users, goals, und KPIs. It then computes cosine similarity scores und returns the top-\(k\) (currently \(k=3\)) positive-scoring chunks with metadata und scores.\\cite{MethodsRecord2026} Queries intentionally omit full serialized recommendations to avoid target leakage.

Sparse TF-IDF retrieval remains competitive for relatively small, well-structured corpora und has the advantage of determinism und transparency compared with dense retrieval.\\cite{lewis2020rag} Recent work on hybrid und dense RAG systems shows that retrieval quality heavily influences grounding und hallucination rates, motivating explicit retrieval diagnostics in our pipeline.\\cite{ControlTokenDPR2024,SMELLM2025}

### 5.5.3 Prompt Integration and Grounding

When positive matches exist, Method B extends the base system prompt with a "Relevant Design Guidelines" block that lists each retrieved passage as an indexed entry containing source, heading, und text, followed by an "End of Guidelines" delimiter.\\cite{MethodsRecord2026} The user message remains unchanged. If retrieval finds no positive-score passages, the method falls back to the base prompt without an empty context block.

The retrieved context is guidance rather than a hard constraint: the model can ignore, misinterpret, oder partially use it. To support analysis, the pipeline records retrieved texts in each `GenerationResult` und computes claim-based grounding metrics that estimate the proportion of rationale claims supported by retrieved passages, using lexical overlap oder an optional sentence-transformers encoder.\\cite{MethodsRecord2026} This follows recent work on Finetune-RAG und RAG evaluation, which emphasizes claim-level support rather than document-level relevance alone.\\cite{Lee2025FinetuneRAG,FedRAG2026}

## 5.6 Method C — QLoRA Fine-Tuning

Method C, registered as `ft`, introduces task-specific parameter-efficient adaptation via Quantized Low-Rank Adaptation (QLoRA).\\cite{MethodsRecord2026} The training pipeline loads the base model in 4-bit quantized form, attaches low-rank LoRA adapters to linear layers across the network, und trains only these adapters while keeping the quantized base weights frozen. The authoritative configuration (`qlora_default.yaml`) uses rank 16, \(\\alpha=32\), dropout 0.05, NF4 quantization with double quantization, a cosine learning-rate schedule, three epochs, per-device batch size 2, gradient accumulation 4, und gradient checkpointing.\\cite{MethodsRecord2026}

QLoRA has been shown to enable fine-tuning of models up to 65B parameters on a single 48GB GPU while achieving 99.3% of ChatGPT’s performance on the Vicuna benchmark, by combining NF4 quantization, double quantization, und carefully configured LoRA adapters.\\cite{dettmers2023qlora} Follow-up work has applied QLoRA to financial LLMs, code summarization, und copyright-aware marketplaces, confirming that quantized low-rank adapters offer an attractive trade-off between resource efficiency und downstream performance.\\cite{FinLoRA2025,GreenCodeSummarization2025,VizQLoRA2023} In this thesis, Method C leverages these advances to adapt Qwen3 und OLMo 2 models specifically to structured dashboard design.

### 5.6.1 Training Data Formatting

Training data are stored as `GoldItem` records in the frozen `dashboard_v4_1` package und are formatted into instruction-style examples by `src/data_pipeline/formatter.py`. For each item, the formatter constructs the same system und user messages used at inference time und appends the pretty-printed JSON recommendation as the target, followed by an end-of-sequence token when available.\\cite{MethodsRecord2026} The resulting text field contains the entire structured recommendation—task type, chart type, encodings, layout, styling, interactions, und rationales—rather than a single label.

This setup aligns with instruction-tuning und structured information inference practice, where models are trained to emit complete multi-field objects from prompts instead of predicting isolated tags.\\cite{StructuredInference2024,Pauk2026} By using identical prompts und schemas für training und inference, the pipeline avoids train–test template mismatch und supports interpretable comparisons of prompt-only versus fine-tuned behavior.

### 5.6.2 QLoRA Configuration and Adapter Training

`src/training/sft_trainer.py` constructs a 4-bit `BitsAndBytesConfig`, resolves an effective compute dtype through a precision helper, prepares the quantized model für k-bit training, applies a `LoraConfig` to the targeted linear modules, und instantiates a supervised fine-tuning trainer.\\cite{MethodsRecord2026} The training flow composes the resolved Hydra configuration, determines adapter output paths und any resume checkpoints, loads und formats the train split, sets seeds, runs training under finite-value callbacks, und saves adapter weights, configuration, tokenizer, und `training_metadata.json` upon completion.\\cite{MethodsRecord2026}

Adapter metadata captures the base model identifier, model configuration hash, dataset version, training configuration hash, seed, effective precision, parameter counts, und training duration, enabling provenance-aware reuse only under compatible conditions.\\cite{MethodsRecord2026} This is consistent with best practices für reproducible LLM fine-tuning, which call für explicit reporting of model version, configuration, customizations (such as adapters), und training settings.\\cite{LLMGuidelines2026,Zhou2025Seeds}

## 5.7 Method D — Fine-Tuning + RAG

Method D, registered as `ft_rag`, composes Methods C und B by applying retrieval augmentation to an adapter-enhanced base model.\\cite{MethodsRecord2026} At inference time, the system resolves the compatible C adapter für the same dataset, model family, und seed, attaches it to the base model, runs the TF-IDF retriever over the dashboard-design knowledge base, und integrates retrieved passages into the system prompt before generating und parsing outputs. The final matrix declares D as depending on C, enforcing same-model und same-seed provenance.\\cite{MethodsRecord2026}

Conceptually, Method D is related to Finetune-RAG, which fine-tunes language models to resist hallucination und better exploit retrieved evidence during generation.\\cite{Lee2025FinetuneRAG} In our setting, combining structured-output adapters with RAG tests whether retrieval remains beneficial after task-specific adaptation und how fine-tuned models change their rationale content und grounding behavior. The architecture is explicitly compositional: retrieval, adapters, und evaluation layers are modular but tightly coupled through configuration hashes und dependency checks.\\cite{MethodsRecord2026,FedRAG2026}

## 5.8 Inference Pipeline

The inference pipeline für all methods is implemented in `src/pipeline/runner.py` und associated components.\\cite{MethodsRecord2026} It loads the frozen test file und optional robustness variants (`test_paraphrased.jsonl`, `test_missing_info.jsonl`), constructs references from `GoldItem` records, und iterates over items by building method-specific prompts, applying tokenizer chat templates, enforcing safe token budgets, generating text, extracting und normalizing JSON, und writing `GenerationResult` records with identity, provenance, content, failure, retrieval, und latency fields.\\cite{MethodsRecord2026}

Predictions are reparsed with the current parser before evaluation, so parser improvements can be applied consistently without altering the raw JSON basis für strict schema metrics.\\cite{MethodsRecord2026} Automatic metrics report schema-compliance rates (parse success, required-key presence, raw-schema validity, completeness), synthetic chart-type macro-F1, synthetic top-\(k\) accuracy, latency statistics, claim-based grounding diagnostics für RAG methods, und robustness measures such as paraphrase consistency und missing-information behavior.\\cite{MethodsRecord2026}

The reporting layer explicitly marks synthetic chart accuracy as internal-circular because chart references are generated by deterministic rule-based mappings rather than independent human ratings; it treats these metrics as implementation diagnostics rather than direct measures of visualization effectiveness.\\cite{MethodsRecord2026,mackinlay1986apt,munzner2014visualization} Independent human-effectiveness scoring und realism layers (L1–L4) are pending und are described in the implementation record as future work.\\cite{MethodsRecord2026}

## 5.9 Reproducibility

Reproducibility is a primary design goal und is supported at several levels. The environment is pinned via `pyproject.toml` und `poetry.lock`, with declared Python und key package versions (e.g., PyTorch, Transformers, Hydra, PEFT, TRL, bitsandbytes, datasets).\\cite{MethodsRecord2026} Run manifests record Python, PyTorch, CUDA, GPU, und package versions für executed runs. Configuration composition is handled by Hydra, und resolved configurations are written to each run directory. A SHA-256 configuration hash is computed und stored with run metadata und cache identity, ensuring that stale predictions are not reused under incompatible conditions.\\cite{MethodsRecord2026}

Dataset provenance is documented through frozen manifests und hashes für `train.jsonl`, `val.jsonl`, `test.jsonl`, und the 40-item human-evaluation selection, including SHA-256 values und byte sizes that serve as reproducibility anchors.\\cite{MethodsRecord2026} The lineage from the nvBench v1 archive, through v3 splits, to the augmented und repaired `dashboard_v4_1` package is explicitly recorded, with generated fields marked as AI-generated und protected fields (goals, KPIs, columns, constraints, tasks, charts, encodings) preserved für test und human-evaluation records.\\cite{MethodsRecord2026}

Model und adapter identity are maintained via configuration files und adapter metadata, though immutable upstream model revisions are not yet pinned; future final runs must record exact hub revisions für full model provenance.\\cite{MethodsRecord2026} The final matrix (`src/config/matrix/final.yaml`) specifies intended seeds (42, 43, 44), models (Qwen3 family und legacy Llama 3.1-8B für historical comparison), methods (A–D), dataset identity (`dashboard_v4` with frozen revision `dashboard_v4_1`), und output layout.\\cite{MethodsRecord2026} Seed policies und actual coverage are documented, revealing that current artifacts provide smoke verification und historical Qwen2.5 results but not complete multi-seed coverage für all final models und methods—a limitation that is explicitly acknowledged.

These practices are consistent with emerging guidelines für LLM experiments, which recommend reporting model versions, configurations, customizations, dataset hashes, seeds, und environment details to support auditability und scientific validity.\\cite{LLMGuidelines2026,Zhou2025Seeds} In this thesis, the results chapter refers back to this methods und reproducibility chapter whenever interpreting experimental outcomes, ensuring that claims are grounded in the documented implementation und coverage state rather than inferred from configuration intent alone.\\cite{MethodsRecord2026}
