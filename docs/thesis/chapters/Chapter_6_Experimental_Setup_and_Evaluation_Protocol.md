# Chapter 6 — Experimental Setup and Evaluation Protocol

Chapter 5 described the system: the output contract, the four methods, the training procedure, and the provenance machinery. This chapter describes the experiment that was actually executed with that system and the protocol used to measure it. It states which model, method and seed combinations were run and which were excluded, records the hyperparameters and hardware that each run actually used rather than the values requested in the configuration files, defines every metric together with what it can and cannot establish, specifies the two perturbation conditions, and fixes the statistical analysis plan. It also specifies the human-evaluation study that the protocol requires but that has not been carried out; that specification is included here so that the gap is explicit and so that the study can be executed later without redesigning it.

Two conventions apply throughout the chapter and the two that follow. First, a value is reported only if it can be read from a generated artifact in the repository; configuration files are treated as a statement of intent and run manifests as a statement of fact, and where the two disagree the manifest wins. Second, a metric is always reported together with the evidence class it belongs to. Agreement with the frozen reference recommendation is agreement with the project's own construction, not independent evidence about visualization effectiveness, and it is labelled as such every time it appears.

## 6.1 Executed experiment matrix

The experiment is a three-factor design: model (4 levels), method (4 levels: A, B, C, D), and seed (up to 3 levels: 42, 43, 44). A complete crossing would give 48 runs. Thirty-two runs were executed and passed the completeness check. Table 6.1 records the executed coverage.

Table 6.1. Executed run coverage. Each cell lists the seeds for which a complete run exists.

| Model profile | Hugging Face identifier | Nominal size | A (prompt-only) | B (RAG) | C (QLoRA) | D (QLoRA + RAG) |
| --- | --- | ---: | --- | --- | --- | --- |
| `qwen3_1_7b` | `Qwen/Qwen3-1.7B` | 1.7B | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 |
| `qwen3_8b` | `Qwen/Qwen3-8B` | 8B | 42 | 42 | 42 | 42 |
| `qwen3_8_27b` | `Qwen/Qwen3.8-27B` | 27B | 42 | 42 | 42 | 42 |
| `olmo2_1_49b` | `allenai/OLMo-2-0425-1B-Instruct` | 1.49B | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 |

A run counts as complete only if it satisfies six conditions, all of which were verified programmatically: `predictions.jsonl`, `predictions_paraphrased.jsonl` and `predictions_missing_info.jsonl` each contain exactly 274 records; every line is valid JSON; item identifiers are unique within each file; and an automatic metrics file exists. All thirty-two runs listed in Table 6.1 pass all six conditions with no file-level defects.

**Every model has a complete A–D matrix at the primary seed.** The four-way method contrast is therefore available for all four models, which is the coverage the research questions require. What the matrix does not have is three-seed replication at the two larger scales.

### 6.1.1 Missing cells

No runs exist for `qwen3_8b` seeds 43 and 44 or for `qwen3_8_27b` seeds 43 and 44. These were not executed. Repeating the full A–D matrix for the two larger profiles at three seeds would have required roughly three times the GPU time already spent on them, which the available allocation did not permit within the schedule of the thesis. The consequence is stated directly rather than argued away: the two smaller profiles support statements about seed-to-seed variability, and the two larger profiles do not. A single-seed run is one realisation of a stochastic process; it is reported as such and is never combined with a multi-seed mean or used to estimate a standard deviation.

One earlier defect is worth recording because it shaped the schedule. A first attempt at the prompt-only and RAG conditions for `qwen3_8b` produced files with duplicated item identifiers carrying different generated responses, because two processes were started against the same output paths and wrote into the same files. Metrics over such a file mix two executions of the same condition, so those exports were set aside in a separate `incomplete` directory and both conditions were inferred again from a single process. The runs reported in this thesis are the clean re-executions; the defective exports contribute nothing to any number. The detection mechanism — checking that item identifiers are unique within each prediction file — is now part of the completeness check quoted above and is applied to every run.

### 6.1.2 Priorities behind the allocation

The compute budget was spent according to a fixed order of priorities. The first priority was that every model that appears in the study should have a complete A–D matrix for at least the primary seed, so that the four-way method contrast is available for every model. The second was that the method contrast should be replicated across three seeds for at least one model, so that the stability of the contrast itself can be examined. The third was that the replication should be available in two different model families rather than only in Qwen3, so that a family-specific artifact can be distinguished from a general pattern. The fourth was that the scale sweep within Qwen3 should reach a model an order of magnitude larger than the smallest profile. Table 6.1 is the result of applying these priorities in order until the allocation was exhausted: the first three are satisfied in full, and the fourth is satisfied for absolute performance but not for seed variability. Across the thirty-two runs, roughly 162 hours of GPU wall-clock time were spent on inference and roughly 23 hours on the eight adapter trainings, giving about 186 GPU-hours in total for the reported experiment.

## 6.2 Models

Four base models were used. Two are small enough for three-seed replication, and two extend the capacity range within the Qwen3 family. Table 6.2 records the architectural facts that are relevant for interpreting the comparison; all values are read from the published model configurations.

Table 6.2. Base models. Attention notation is heads (query) / heads (key-value).

| Profile | Identifier | Total parameters (measured at load) | Layers | Hidden size | Attention | Vocabulary | Native context | Family |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `qwen3_1_7b` | `Qwen/Qwen3-1.7B` | 1,738,007,552 | 28 | 2,048 | 16 / 8 (grouped-query) | 151,936 | 40,960 | Qwen3 \cite{yang2025qwen3} |
| `qwen3_8b` | `Qwen/Qwen3-8B` | 8,234,382,336 | 36 | 4,096 | 32 / 8 (grouped-query) | 151,936 | 40,960 | Qwen3 \cite{yang2025qwen3} |
| `qwen3_8_27b` | `Qwen/Qwen3.8-27B` | 27,012,726,272 | 64 | 5,120 | hybrid: gated linear attention interleaved with gated full attention | 248,320 | 262,144 | Qwen3.8 \cite{qwen2026qwen38,qwen2026qwen3827bcard} |
| `olmo2_1_49b` | `allenai/OLMo-2-0425-1B-Instruct` | 1,496,975,360 | 16 | 2,048 | 16 / 16 (multi-head) | 100,352 | 4,096 | OLMo 2 \cite{olmo2025furious,allenai2025olmo2instruct1b} |

The parameter counts in Table 6.2 are not taken from marketing names; they are the totals recorded by the trainer when the model was loaded for adapter training, which is why the OLMo profile is described as roughly 1.49B rather than 1B.

### 6.2.1 Why a second model family, and why this one

Three Qwen3 profiles alone would answer a question about capacity inside one family. They cannot distinguish a general property of the task from a property of how Qwen3 models are built and post-trained. A second family is therefore part of the design rather than an extra data point.

OLMo 2 was selected for four reasons that can be verified from primary sources. It is developed independently of Qwen by a different organisation, so its tokenizer, pretraining corpus, architecture and post-training recipe differ simultaneously from the Qwen3 profiles \cite{olmo2025furious}. Its development is unusually transparent: the training data, code, intermediate checkpoints and recipes are released alongside the weights, which makes the differences from Qwen3 documentable rather than a matter of speculation \cite{olmo2025furious}. The selected instruction-tuned checkpoint has a parameter count close to the smallest Qwen3 profile — 1.50B against 1.74B — so the cross-family comparison is made at a comparable scale rather than across a capacity gap. And the post-training pipeline of the checkpoint is documented: supervised fine-tuning on an OLMo-specific variant of the Tülu 3 mixture, followed by direct preference optimisation and reinforcement learning with verifiable rewards \cite{allenai2025olmo2instruct1b,lambert2025tulu3}.

The interpretive limit of this design must be stated at the same time. Because architecture, tokenizer, pretraining corpus and post-training recipe all change together between the families, a difference between the OLMo profile and the Qwen3 profiles is a *model-family* difference. It cannot be attributed to any single component. In particular, the fact that the OLMo checkpoint uses multi-head attention while the small Qwen3 profile uses grouped-query attention is a recorded architectural difference and not an explanation of any observed behaviour. The cross-family comparison can support a statement of the form "the pattern observed in Qwen3 does or does not reappear in an independently developed family at a comparable scale". It cannot support a statement of the form "the pattern is caused by attention type".

OLMo is compared conceptually with `qwen3_1_7b`. It is not size-matched to the 8B or 27B profiles, and no conclusion in this thesis treats it as such.

## 6.3 Data and inputs

All runs consume the frozen `dashboard_v4` package described in Chapter 4, whose exact frozen revision is `dashboard_v4_1`. Table 6.3 lists the input artifacts and the digests recorded by the runs.

Table 6.3. Input artifacts, with SHA-256 digests as recorded in the run manifests.

| Artifact | Records | SHA-256 |
| --- | ---: | --- |
| `data/frozen/dashboard_v4/train.jsonl` | 2,932 | `c618488d92016e6c67925c2cbf021fb860a9e37ad6322107d535413f0bca84b1` |
| `data/frozen/dashboard_v4/val.jsonl` | 613 | `78847c3fe42a0a7221ead9b8970cfa17e51b4d9e79e103e4c4f15961e6462356` |
| `data/frozen/dashboard_v4/test.jsonl` | 274 | `e2df055d0a75c25f53a88cb830a5b6d66411fa179413f352f45ca2f6873829d5` |
| `data/eval/robustness_v4/test_paraphrased.jsonl` | 274 | `46de83228fa8aaef24c0aca459502345b62ade634afa475ee6a472e5135cf4ac` |
| `data/eval/robustness_v4/test_missing_info.jsonl` | 274 | `f0c4e695d9c71f369247ca76bd707f2223d85880b59e3e5eb079d6fc3946f088` |
| `data/frozen/dashboard_v4/human_eval_test_items_40.csv` | 40 | `2c336ee26398e8e487991df7e1c6e31385787c555228962d77e811f9a920cb8a` |

Every one of the thirty-two runs records the same train, validation and test digests, so all reported comparisons are made on identical inputs. The train, validation and test digests were additionally recomputed from the local files during the preparation of this chapter and matched the values in the frozen `hashes.json`.

### 6.3.1 Knowledge base

Conditions B and D retrieve from a frozen corpus of three Markdown guideline documents — `accessibility_guidelines.md`, `chart_selection_guidelines.md` and `dashboard_design_guidelines.md` — split at Markdown headings into 41 chunks with a minimum chunk length of five words and deterministic chunk identifiers. All sixteen executed RAG runs record the same knowledge-base identity: version `3d1409d8347fc165`, chunk file digest `19fbbfa4ad333dec7298f53c98b25bf5a8bf5059236e8fa4885febede0e54159`.

The repository manifest for the same corpus records version `46b8575d98d37312` and chunk digest `cad3bb0c1606fab3062d3235740d7701e1fb8ac46702af8b09f615f66bfc3afb`. The two digests differ, and the cause was determined rather than assumed: converting the line endings of the local chunk file from CRLF to LF reproduces the digest recorded by the runs exactly. The working copy on the Windows development machine carries CRLF line endings, while the copy transferred to the compute cluster carried LF. The retrieved *content* is therefore identical; only the byte encoding of the line terminator differs. The version recorded by the runs is the authoritative identity for the reported results, and the difference is noted here because a reader who recomputes the hash on a Windows checkout will obtain the other value.

A separate distinction applies to what the corpus is. The knowledge base is project-authored guidance whose statements are traceable to the visualization literature summarised in Chapter 2. A file digest establishes that the same bytes were retrieved in every run; it establishes nothing about the scientific authority of the guidance itself. Retrieval of a passage is therefore never treated as evidence that the recommendation it supports is correct.

### 6.3.2 Perturbation conditions

Two perturbation sets were derived from the held-out test split by a deterministic, dependency-free transformation with no language model, no network access and no randomness, and both preserve the original `item_id` so that every perturbed prediction pairs with its original.

The **paraphrase** condition rewrites only the free-text fields of the brief — `users`, `constraints`, and the string entries of `goals` and `kpis` — using a fixed table of meaning-preserving synonym substitutions applied to framing and verb words (for example *show* ↔ *display*, *track* ↔ *monitor*, *compare* → *contrast*, *goal* → *objective*, *metric* → *measure*). Domain nouns such as KPI expressions and column names are left untouched, so the analytical content of the brief is unchanged and a system that recommends a different primary chart has changed its answer for a reason that is not in the data.

The **missing-information** condition produces an under-specified brief by removing the constraints field entirely and dropping the last KPI and the last data column when more than one is present. Here the correct behaviour is not obvious by construction, and this is the point of the condition: a system may reasonably ask for the missing information instead of producing a confident complete recommendation.

Both conditions are narrow. They test one paraphrase family and one information-removal family. Results from them support statements about robustness *under these perturbations* and nothing broader. The general principle — that accuracy on one fixed test form is insufficient to characterise behaviour, and that controlled input changes reveal failures a single score hides — is the standard argument for behavioural testing of language systems \cite{ribeiro2020checklist}.

## 6.4 Executed training configuration

Chapter 5 documents the training configuration file. The values that the runs actually used differ from that file in three places, and the manifests rather than the YAML are reported here. Table 6.4 records the effective configuration of each of the eight trained adapters.

Table 6.4. Effective QLoRA training configuration per adapter, read from `training_metadata.json`. All adapters use rank 16, scaling α = 32, three epochs, learning rate 2 × 10⁻⁴, a cosine schedule with 10 % warm-up, the `adamw_torch` optimiser, and 4-bit NF4 quantization of the frozen base with double quantization.

| Model | Seeds | Training-config hash | Batch × accumulation (effective) | Max sequence length | Precision | Trainable parameters | Share of total | Training GPU | Hours per adapter |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| `olmo2_1_49b` | 42, 43, 44 | `94bd7af87e2d` | 2 × 4 (8) | 4,096 | bf16 | 12,058,624 | 0.81 % | H100 NVL | 0.54 |
| `qwen3_1_7b` | 42 | `823472d2903a` | 1 × 8 (8) | 1,024 | bf16 | 17,432,576 | 1.00 % | A30 | 1.00 |
| `qwen3_1_7b` | 43, 44 | `94bd7af87e2d` | 2 × 4 (8) | 1,024 | fp16 | 17,432,576 | 1.00 % | Tesla T4 | 3.35 |
| `qwen3_8b` | 42 | `94bd7af87e2d` | 2 × 4 (8) | 4,096 | bf16 | 43,646,976 | 0.53 % | GB10 | 4.87 |
| `qwen3_8_27b` | 42 | `8998747cdc16` | 1 × 8 (8) | 1,024 | bf16 | 116,727,808 | 0.43 % | GB10 | 8.82 |

Four observations follow from this table and are carried into the interpretation of the results.

The intended common recipe held for the parameters that define the adaptation itself. Rank, scaling factor, dropout, target-module policy, learning rate, schedule, optimiser, epoch count and quantization scheme are identical across all eight adapters. The effective optimisation batch is also identical at eight sequences, although it is reached by different combinations of per-device batch size and gradient accumulation depending on the memory available on the assigned device.

The trainable parameter count is not constant. The same rank produces 12.1 M trainable parameters on the OLMo profile, 17.4 M on Qwen3-1.7B, 43.6 M on Qwen3-8B and 116.7 M on Qwen3.8-27B, because the number and width of the adapted linear projections depend on the architecture. The share of total parameters that is trainable consequently *falls* as the model grows, from 1.00 % to 0.43 %. A statement such as "the same adapter capacity was used for every model" would be false and is not made anywhere in this thesis.

The maximum training sequence length is not constant. Two profiles trained at 4,096 tokens and two at 1,024. A shorter training window truncates longer training targets, and the structured recommendation is a long target. This is a real difference between conditions that share a training-config hash label, and it is treated as a potential confound in Chapter 8 rather than as a detail.

The training precision is hardware-determined. The precision helper resolves `precision: auto` against the weakest available device, so the two Qwen3-1.7B adapters for seeds 43 and 44 were trained in fp16 on a Tesla T4, which does not support bfloat16, while every other adapter was trained in bf16. Seed 42 for the same model also used a different per-device batch size and a different code state. The three Qwen3-1.7B seeds are therefore *not* three draws from one identical procedure, and Chapter 7 reports them individually rather than as a pooled mean.

## 6.5 Inference configuration and execution environment

Every one of the thirty-two runs used identical generation settings, recorded in each manifest: at most 512 new tokens, temperature 0.1, top-p 0.9, sampling enabled, and a repetition penalty of 1.15. Generation is sequential with batch size one. Conditions C and D additionally load the base model in 4-bit NF4 with double quantization; conditions A and B use the base runtime profile. Conditions B and D retrieve at most three positively scoring chunks (`top_k = 3`).

The 512-token output limit deserves emphasis here because it becomes important in Chapter 7. It was chosen to bound key-value cache growth during 27B inference while leaving room for a structured response. It is a hard ceiling: a response that has not closed its outermost JSON object by token 512 is cut off, and the stored raw text ends mid-structure. The pipeline records the raw text in every case, so truncation is recoverable after the fact, but the metric that reads the parsed object cannot score a truncated response.

The runs were executed on heterogeneous hardware across several compute environments: NVIDIA GB10, H100 NVL, A30, L4 and Tesla T4 devices, with Python 3.11.7 or 3.12.x, PyTorch 2.6.0+cu124 or 2.13.0+cu130. This heterogeneity has one direct methodological consequence: **latency is not comparable across conditions that ran on different devices.** Section 6.6.10 restricts the interpretation of latency accordingly.

### 6.5.1 Code-state comparability

Each run manifest records the git commit of the code that produced it. These commits are not uniform across the matrix, and the differences are material enough that the analysis records them per comparison rather than assuming a single protocol. Three levels are distinguished:

- **homogeneous** — every run in the comparison recorded the same git commit, and the trained conditions share one training-config hash;
- **mixed code state** — the runs recorded more than one known commit;
- **unknown code state** — at least one run did not record its commit.

The label is computed at two granularities, because they answer different questions. The **block** label covers all four methods entering the omnibus test for one model and one seed. The **pair** label covers only the two runs actually being contrasted, and is the one that matters when reading an individual contrast: a block can be mixed because a third condition came from a different commit while the two runs in a given contrast did not.

Of the eight model-by-seed blocks, none is homogeneous: three are mixed and five contain at least one run whose commit was not recorded. At the level of individual contrasts the picture is better. Of the forty distinct contrasts, **seven are homogeneous**, fourteen are mixed and nineteen contain an unrecorded commit. The homogeneous seven are the B − A contrasts for `qwen3_8b` and `qwen3_8_27b` at seed 42 and for `olmo2_1_49b` at seed 42, and the D − C contrasts for `qwen3_8b`, `qwen3_8_27b` and `olmo2_1_49b` at seed 42 and for `olmo2_1_49b` at seed 43. Both contrasts that carry the main retrieval finding of this study — B − A at 8B and at 27B — are therefore code-state homogeneous.

Contrasts drawn from non-homogeneous pairs are reported as descriptive: the method changed, but so did something else that the manifests cannot rule out. This remains the largest threat to the internal validity of those contrasts, and it is a consequence of scheduling the runs across four compute environments over several weeks. It is discussed again in Chapter 8. Every contrast in Chapter 7 carries its pair-level label.

### 6.5.2 Adapter provenance in condition D

Condition D must load the adapter produced by condition C for the same model, dataset, training configuration and seed. Six of the eight D runs record the identifier of their source C run explicitly — for example `dashboard_v4_olmo2_1_49b_C_seed_43` for OLMo seed 43 — together with a digest of the adapter manifest. The remaining two D runs (`qwen3_1_7b` seed 42 and `qwen3_8_27b` seed 42) were produced by an earlier version of the runner that did not write the source-run field; for these the evidence is the recorded adapter path, which in both cases points at the condition-C adapter directory for the same model, method and seed. All eight D runs record a training-config hash identical to that of the corresponding C run. The pairing is therefore verified for all eight, by manifest field for six and by path plus configuration hash for two.

## 6.6 Metrics

The evaluation is layered because no single automatic number can establish that a dashboard recommendation is good. Each metric below is defined, motivated, and bounded. For every metric the chapter states whether it is *independent evidence* about the task or an *internal diagnostic* of the pipeline and its reference construction.

### 6.6.1 JSON parse rate

**What it measures.** The percentage of the 274 test responses from which a JSON object could be extracted from the raw model text. The extractor accepts a fenced JSON block, a brace-delimited object surrounded by short prose, or a bare object.

**Why it matters.** The whole point of a structured recommendation is that a downstream system can read it. A response that cannot be decoded at all is unusable regardless of the design it describes.

**How to read it.** A high value means the model reliably emits something machine-readable. It says nothing about whether the fields are correct, complete, or sensible. A parse failure has two very different causes — the model produced prose instead of JSON, or the model produced JSON that the token budget cut off — and Chapter 7 separates them.

**Evidence class.** Internal diagnostic of output form. Independent of the reference labels.

### 6.6.2 Runtime-contract parse rate

**What it measures.** The percentage of responses whose extracted object also validated into the runtime `DesignOutput` model after a limited set of alias repairs (for example `comparision` → `comparison`, `column chart` → `bar`).

**Why it matters.** This is the level at which the downstream metrics operate. If a response does not reach it, the pipeline has no typed object from which to read a chart type, and the item is scored as incorrect by the chart metric.

**How to read it.** The gap between the JSON parse rate and this rate is exactly the set of responses that were decodable but did not match the expected types. In this study the dominant cause is an `encoding` field emitted as a string where the contract requires an object.

**Evidence class.** Internal diagnostic of contract conformance. This is a property of the model output *relative to this project's contract*, not a general capability measure.

### 6.6.3 Strict schema validity

**What it measures.** The percentage of responses whose *raw* extracted object validates against the strict response contract, with no alias repair, no extra keys anywhere, non-empty strings for encoding channels, and an aggregate drawn from a closed set.

**Why it was included.** It was intended as the honest counterpart to the lenient parse rate: a measure of whether the model emitted a valid record without the pipeline fixing it afterwards.

**Why it must be reported with a ceiling statement.** During the preparation of this chapter the strict contract was applied to the frozen reference recommendations themselves. **None of the 274 reference records satisfies it**: the strict encoding model forbids extra keys, while every reference encoding carries the additional source-grounded keys `x_aggregate`, `y_aggregate`, `grouped`, `classify`, `group_field`, `filters`, `sort`, `limit`, `having`, `time_grain`, `visual_grouping`, `source_x` and `source_y`. A system that reproduces the reference format perfectly therefore scores 0 % on this metric by construction. The check is reproducible via `experiments/scripts/analyze_final_results.py`, which writes `strict_schema_ceiling.json`.

**How to read it.** Strict schema validity in this study measures *distance from a narrow contract*, and that contract is narrower than the training target. A high value indicates output closer to the minimal contract, which for fine-tuned systems means output further from the reference. It is reported for completeness and is explicitly **not** used as a quality ranking in Chapter 7.

**Evidence class.** Internal diagnostic with a known ceiling artifact.

### 6.6.4 Completeness and field coverage

**What it measures.** Completeness is the mean fraction of the six required top-level fields that are present *and non-empty*; an empty list, empty object or empty string does not count. Field coverage reports the presence rate of each of the six fields separately.

**Why it matters.** A record can decode successfully and still omit the layout, the interactions or the rationales. Completeness detects that failure mode without requiring type-level validity.

**How to read it.** High completeness means the response has the expected shape. It says nothing about whether the content of any field is appropriate. A complete recommendation can be a wrong recommendation.

**Evidence class.** Internal diagnostic of output form.

### 6.6.5 Chart agreement (top-1), reported at three levels

**What it measures.** Whether the primary recommended chart type matches the primary chart type in the frozen reference recommendation for the same item.

**Why it matters.** Chart selection is the single decision in the output that the source corpus can support with evidence, and it is the decision most directly connected to the visualization literature.

**Why three levels.** The metric as implemented in the pipeline reads the chart from the parsed runtime object. That couples two distinct failures: a wrong chart, and a record that never reached the runtime contract. Because the raw text of every response is stored, the same comparison can be repeated under weaker preconditions, and the difference between the levels isolates the cause. The three levels are:

1. **Pipeline top-1** — the pre-specified metric. Requires a valid runtime object. A parse failure counts as incorrect.
2. **Format-tolerant chart agreement** — reads `kpi_chart_mapping[0].chart_type` directly from the raw extracted JSON object, ignoring whether the record satisfies the runtime contract. Still requires a complete, decodable object.
3. **Text-level chart agreement** — reads the first `"chart_type": "<token>"` occurrence in the raw text. Survives a response that the token budget cut off before the object could be closed.

The difference between levels 1 and 2 is the cost of the runtime contract; the difference between levels 2 and 3 is the cost of the generation-length budget. All three are computed by the same script and normalise the chart token identically. Levels 2 and 3 are **post-hoc diagnostics** introduced after inspecting the level-1 results, and they are labelled as such wherever they appear. Level 1 remains the headline metric.

**How to read it.** Agreement is agreement with the project's frozen reference. Because a visualization task frequently admits more than one defensible chart, a mismatch is not automatically an error and a match is not automatically a good design \cite{saket2019taskbased,kim2018taskdata,mackinlay1986apt}.

**Evidence class.** Internal diagnostic relative to the project's own reference construction. Not independent evidence about visualization effectiveness.

### 6.6.6 Macro-F1 over chart classes and the confusion matrix

**What it measures.** The unweighted mean of the per-class F1 scores over the chart-type vocabulary, plus the full predicted-versus-reference confusion matrix.

**Why it matters.** The held-out split is strongly imbalanced towards bar charts. A system that predicts `bar` for everything would score well on plain agreement. Macro-F1 weights every chart class equally and exposes that behaviour.

**How to read it.** A large gap between top-1 agreement and macro-F1 indicates that performance is concentrated in the frequent classes. Classes with very small support — the split contains only two scatter records — produce unstable per-class F1 values, so per-class numbers are read as descriptive.

**Evidence class.** Internal diagnostic, same reference as top-1.

### 6.6.7 Robustness under the two perturbations

Four quantities are computed by pairing each perturbed prediction with its original by `item_id`.

**Paraphrase consistency** is the percentage of items whose primary predicted chart is unchanged after the paraphrase. It measures stability of the answer, not its correctness — a system can be perfectly consistent and consistently wrong.

**Paraphrase accuracy** is chart agreement computed on the paraphrased briefs, over the items present in both runs. **Paraphrase accuracy delta** is the difference between paraphrased and original accuracy on the same items; a negative value means accuracy was lost under a change that should not have mattered.

**Missing-information clarification rate** is the percentage of under-specified responses that ask for clarification or signal uncertainty. It is computed by a regular-expression detector over the generated text, matching phrases such as *clarify*, *more information*, *ambiguous*, *insufficient*, *cannot determine*, *not enough*, *please specify* and *underspecified*. This is a lexical heuristic, not a semantic judge: it will miss an implicit request and can fire on an unrelated use of a matched word. It is reported as an implementation diagnostic.

**Missing-information schema rate** is the percentage that instead produce a complete valid record despite the missing information. It is reported for contrast, and a high value is not automatically a good outcome: confidently completing an under-specified brief may be less trustworthy than asking for what is missing.

All four apply only to the two implemented perturbation families and are never generalised beyond them.

### 6.6.8 Retrieval grounding

**What it measures.** For conditions B and D, the percentage of generated rationale claims that are supported by at least one retrieved passage. Support is decided in the default lexical-proxy mode by content-word overlap; an optional semantic mode using a sentence encoder exists but was not enabled for the reported runs. The metric is reported together with the number of scored claims, the claim coverage rate (the share of items that contributed any scorable claim), and the number of items that received retrieved documents.

**How to read it.** Lexical overlap is not entailment. A high value means the rationale text reuses vocabulary from the retrieved passage; it does not establish that the passage justifies the claim, and it does not establish that the chart the rationale defends is appropriate. Claim coverage must be read first: a grounding rate computed over a handful of claims from a handful of items describes those claims only.

**Evidence class.** Weak proxy diagnostic.

### 6.6.9 Retrieval relevance

Recall@3, MRR@3 and nDCG@3 require independent relevance judgements linking briefs to guideline chunks. No such judgements exist in the project, the evaluation configuration leaves the qrels path empty, and the reporting layer marks the metric unavailable rather than inferring relevance from the retriever's own scores. Retrieval relevance is therefore **not measured**.

### 6.6.10 Latency

**What it measures.** Wall-clock duration of the generation call for one item, averaged over the run, together with the median and the 95th percentile.

**Interpretive limits.** Two restrictions apply. The timer wraps the generation call after item preparation, so retrieval and prompt construction are not included in the same way as generation; the value is generation latency, not end-to-end user latency. More importantly, the runs were executed on five different GPU types, so a latency difference between two conditions that ran on different devices is partly a hardware difference. Latency is therefore reported as a descriptive cost indicator for the executed configuration and is not used to compare methods.

### 6.6.11 Metrics that are specified but not executed

Three layers of the protocol are defined and remain unmeasured. They are listed here so that their absence is visible in the setup rather than discovered in the results.

The **independent chart-effectiveness layer** would score a predicted chart as correct if it belongs to the set of charts shown to be effective for the relevant task and data shape in published human-subject studies \cite{saket2019taskbased,kim2018taskdata}, reporting covered accuracy and coverage rate separately so that uncovered items are visible. This would be a genuinely independent criterion rather than agreement with the project's own reference. The gold table exists in the repository; the scorer is not implemented, and every per-item record accordingly carries `l1_covered = not_applicable`.

The **realism layer**, which would compare the distribution of generated dashboard structures against a corpus of real-world dashboards, requires a corpus that was not acquired. It is not measured.

The **human-rating layer** is specified in Section 6.8 and has not been executed.

## 6.7 Statistical analysis plan

The design is paired: every method sees the same 274 held-out items for a given model and seed, and the per-item outcome records are stored, so comparisons are made item by item rather than between run-level averages. The tests were chosen from the structure of the outcome and the design, not from what happened to be available in the code.

**Binary outcomes.** Chart agreement, JSON extraction, runtime-contract validity, strict schema validity and encoding-object validity are dichotomous per item. For a block with three or more methods, Cochran's Q provides the omnibus test for matched binary outcomes across k conditions \cite{cochran1950}. Pairwise comparisons then use the exact McNemar test, which conditions on the discordant pairs — items where exactly one of the two methods succeeded — and evaluates them against a binomial null \cite{mcnemar1947}. The exact binomial form is used rather than the chi-square approximation because several contrasts have few discordant pairs.

**Multiple comparisons.** Within each block of pairwise tests, p-values are adjusted with the Holm step-down procedure, which controls the family-wise error rate without the conservatism of a single Bonferroni threshold \cite{holm1979}. A contrast is called significant when its Holm-adjusted p-value falls below 0.05. The correction family is one block: one model, one seed, one outcome.

**Effect size and uncertainty.** A p-value alone does not describe how large a difference is. Every contrast is therefore also reported as the mean paired difference in percentage points with a 95 % percentile bootstrap confidence interval obtained by resampling items — not runs — 10,000 times with pairing preserved \cite{efron1979bootstrap}. Per-condition rates are reported with their own bootstrap intervals. Statistical significance is never described as practical importance; a difference of one or two percentage points on 274 items can be significant and still be irrelevant for a design tool.

**Ordinal and continuous outcomes.** Completeness and, once collected, the human Likert dimensions are ordinal or bounded-continuous. The plan for these is the Friedman omnibus test across methods \cite{friedman1937} followed by pairwise Wilcoxon signed-rank tests with Holm correction \cite{wilcoxon1945,holm1979}, reported with paired rank-biserial correlation and bootstrap intervals. These procedures are part of the plan; they are executed in Chapter 7 only where the corresponding data exists.

**Seeds.** Seeds are repeated realisations of the same 274 items and are not additional observations. Where three seeds exist, the per-seed values are shown individually and summarised descriptively by mean and standard deviation. No confidence interval is computed across three seeds, and no variance estimate of any kind is computed for a single-seed condition. The reporting layer enforces this: it labels seed summaries with "not reported for fewer than 5 independent seeds" rather than emitting a spurious interval. That random seeds can produce non-trivial variation in fine-tuned language models — at the level of aggregate metrics and at the level of individual predictions — is documented in the literature and is a reason to report the spread rather than a single run \cite{bui2025seeds}.

**Comparability gating.** Every reported test carries the code-state label from Section 6.5.1. A significant result from a `mixed_code_state` or `unknown_code_state` block establishes that the two runs differ; it does not establish that the method is the only reason they differ.

**What the statistics cannot do.** A confidence interval cannot repair a reference set, and a significant difference on an internal diagnostic cannot establish a better dashboard design. Every test in Chapter 7 is reported together with the evidence class of the outcome it was applied to.

## 6.8 Human evaluation: specification of a study that has not been run

Automatic metrics in this study measure output form and agreement with a constructed reference. They cannot judge whether a layout is sensible, whether a styling choice is accessible, whether an interaction is useful, or whether a rationale is a real explanation. Those dimensions require human judgement, and reporting practice for human evaluation of generated text requires that the task, the instructions, the sampling, the number of raters, the aggregation rule and the agreement measure all be stated \cite{vanderlee2019human}.

The study is fully specified in the repository and its infrastructure is implemented. **No ratings have been collected.** The directory for the study contains an item file and a rater assignment and no `ratings/` content. Nothing in this thesis reports a human-rated result, and no automatic metric is presented as a substitute for one.

The specification, recorded here before data collection, is as follows. The instrument covers six dimensions, each on a 1–5 scale with anchors defined at 1, 3 and 5: chart appropriateness, layout quality, styling and accessibility, interaction design, rationale quality, and overall usefulness. The sampling frame is the fixed 40-row selection drawn from the held-out test split and frozen with the dataset. To keep the burden within a planned 30-minute session, the final collection uses a fixed task-stratified subset of eight items: three comparison tasks, two trend tasks, and one each for part-to-whole, composition and correlation. The design is 8 items × 4 methods × 3 independent ratings per output, giving 32 rating units and 96 ratings distributed as 16 ratings per rater across six raters. The workload calculation assumes five instruction minutes and 1.5 minutes per rating, or 29 minutes; this is a planning estimate, not a guarantee of individual speed. Raters see the dashboard brief, one anonymised recommendation, the six scales and an optional comment field; method identity, model identity, seed, automatic metrics and the reference recommendation are hidden. Each study is isolated by dataset, model, seed and protocol directory, and its manifest records the source prediction paths and digests, run identifiers, configuration hashes, dataset and knowledge-base digests, the selected item identifiers, the rubric hash, assignment settings, expected counts and time budget, so that an analysis refuses to run if the underlying predictions changed after the study was created. Because only eight items and a non-expert convenience sample are used, resulting method comparisons are exploratory estimates of perceived quality for Qwen3.8-27B at seed 42; they are not universal or expert-validity claims.

The planned analysis reports Krippendorff's ordinal α separately for each of the six dimensions as the inter-rater reliability measure \cite{krippendorff2018content}; per-method means and standard deviations for each dimension and for the composite mean across dimensions; paired Friedman tests across the four methods \cite{friedman1937}; pairwise Wilcoxon signed-rank tests with Holm correction \cite{wilcoxon1945,holm1979}; paired rank-biserial correlations, Cohen's *d_z*, and bootstrap intervals for the paired differences \cite{efron1979bootstrap}; and a derived human chart-acceptability outcome — chart appropriateness ≥ 4, aggregated by majority across raters — analysed with Cochran's Q and exact McNemar tests with Holm correction \cite{cochran1950,mcnemar1947,holm1979}. Human chart acceptability would be a supportive human measure of whether raters accept a chart, not an objective statement that the chart is correct.

Until this study is executed, the thesis makes no claim about the usefulness, quality or design adequacy of any generated recommendation.

## 6.9 Reproducibility of the reported analysis

Every number in Chapter 7 is produced by code in the repository from artifacts in the repository. The aggregation of run-level metrics is produced by `experiments/scripts/aggregate_results.py` and the figures by `experiments/scripts/make_figures.py`. The paired statistics, the provenance table, the three-level chart re-scoring, the failure-mode counts and the strict-schema ceiling check are produced by `experiments/scripts/analyze_final_results.py`, which reads only `manifest.json`, `metrics_auto.json`, `eval_per_item.jsonl` and `predictions.jsonl` from the thirty-two completed runs plus the frozen test file, and writes its outputs to `experiments/results/final/dashboard_v4/statistics/`. None of these scripts requires a GPU, loads a model, or contacts a network service, so the entire analysis can be re-executed from the stored artifacts:

```bash
python experiments/scripts/analyze_final_results.py --dataset dashboard_v4
```

The reproducibility of the analysis is not the same as the reproducibility of the runs. The runs depend on upstream model repositories, and the model revision field is recorded for two of the four profiles (`allenai/OLMo-2-0425-1B-Instruct` at revision `48d788eca847d4d7548f375ad03d3c9312f6139e` and `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`) and left null for the two other Qwen3 profiles. For those two, the identifier is known but no immutable upstream commit is pinned. Sampling is enabled at temperature 0.1, and hardware kernels differ across the five device types used, so recording a seed documents the intended configuration without guaranteeing bitwise identical regeneration. These boundaries are stated wherever a reproduction claim is made.

## References Used in Chapter 6

Generated from `docs/thesis/references.bib` by `experiments/scripts/build_chapter_reference_lists.py`. The BibTeX key of each entry is given in brackets.

Allen Institute for AI (2025). *OLMo-2-0425-1B-Instruct Model Card*. Hugging Face model repository. Revision 48d788eca847d4d7548f375ad03d3c9312f6139e; Apache-2.0; accessed 2026-09-10. https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct [`allenai2025olmo2instruct1b`]

Bui, N., Savova, G., & Wang, L. (2025). *Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning Large Language Models*. arXiv:2503.07329. Accepted at IJCNLP 2025. https://arxiv.org/abs/2503.07329 [`bui2025seeds`]

Cochran, W. G. (1950). *The Comparison of Percentages in Matched Samples*. Biometrika, 37(3-4), 256–266. https://doi.org/10.1093/biomet/37.3-4.256 [`cochran1950`]

Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife*. The Annals of Statistics, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552 [`efron1979bootstrap`]

Friedman, M. (1937). *The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance*. Journal of the American Statistical Association, 32(200), 675–701. https://doi.org/10.1080/01621459.1937.10503522 [`friedman1937`]

Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 6(2), 65–70. https://www.jstor.org/stable/4615733 [`holm1979`]

Kim, Y., & Heer, J. (2018). *Assessing Effects of Task and Data Distribution on the Effectiveness of Visual Encodings*. Computer Graphics Forum, 37(3), 157–167. https://doi.org/10.1111/cgf.13409 [`kim2018taskdata`]

Krippendorff, K. (2018). *Content Analysis: An Introduction to Its Methodology*. SAGE Publications. 4th ed. [`krippendorff2018content`]

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., et al. (2025). *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*. https://doi.org/10.48550/arXiv.2411.15124 [`lambert2025tulu3`]

Mackinlay, J. (1986). *Automating the Design of Graphical Presentations of Relational Information*. ACM Transactions on Graphics, 5(2), 110–141. https://doi.org/10.1145/22949.22950 [`mackinlay1986apt`]

McNemar, Q. (1947). *Note on the Sampling Error of the Difference between Correlated Proportions or Percentages*. Psychometrika, 12(2), 153–157. https://doi.org/10.1007/BF02295996 [`mcnemar1947`]

OLMo Team, Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., et al. (2025). *2 OLMo 2 Furious*. https://doi.org/10.48550/arXiv.2501.00656 [`olmo2025furious`]

Qwen Team (2026). *Qwen3.8-Max: A New Bar for Coding and Cowork*. Qwen blog. Citation given on the Qwen3.8-27B model card; accessed 2026-09-10. https://qwen.ai/blog?id=qwen3.8 [`qwen2026qwen38`]

Qwen Team (2026). *Qwen3.8-27B Model Card*. Hugging Face model repository. Revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; Apache-2.0; accessed 2026-09-10. https://huggingface.co/Qwen/Qwen3.8-27B [`qwen2026qwen3827bcard`]

Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 4902–4912. https://doi.org/10.18653/v1/2020.acl-main.442 [`ribeiro2020checklist`]

Saket, B., Endert, A., & Demiralp, C. (2019). *Task-Based Effectiveness of Basic Visualizations*. IEEE Transactions on Visualization and Computer Graphics, 25(7), 2505–2512. https://doi.org/10.1109/TVCG.2018.2829750 [`saket2019taskbased`]

van der Lee, C., Gatt, A., van Miltenburg, E., Wubben, S., & Krahmer, E. (2019). *Best Practices for the Human Evaluation of Automatically Generated Text*. Proceedings of the 12th International Conference on Natural Language Generation, 355–368. https://doi.org/10.18653/v1/W19-8643 [`vanderlee2019human`]

Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods*. Biometrics Bulletin, 1(6), 80–83. https://doi.org/10.2307/3001968 [`wilcoxon1945`]

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Gao, G., et al. (2025). *Qwen3 Technical Report*. arXiv preprint arXiv:2505.09388. https://doi.org/10.48550/arXiv.2505.09388 [`yang2025qwen3`]
