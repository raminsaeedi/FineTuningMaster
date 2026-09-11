# Chapter 8 — Discussion, Error Analysis and Threats to Validity

Chapter 7 reported what the runs produced. This chapter explains what those observations mean, what caused them, and how far they can be trusted. It first answers the four research questions that the evidence supports. It then examines the measurement decomposition that changed the size and in several cases the sign of the estimated effects, and analyses the two failure modes that dominate the experiment. The following sections discuss why retrieval behaves differently after fine-tuning than before it, what the scale and cross-family observations do and do not license, and which methodological problems arose during the project and how they were handled. The chapter closes with a systematic account of the threats to validity and with the practical implications that the evidence actually supports.

The chapter deliberately keeps two things apart. An explanation of a mechanism inside this experiment — for example, that a specific guideline sentence produced a specific substitution — is supported by the artifacts. A general claim about how retrieval or fine-tuning behaves on structured design tasks is not, and is not made.

## 8.1 Answers to the research questions

**RQ1 — Structured-output reliability.** Reliability is a property of the base model rather than of the method, and it is acquired somewhere between 1.7B and 8B parameters. Qwen3-8B and Qwen3.8-27B both satisfy the output contract on all 274 items without any adaptation. Qwen3-1.7B produces a decodable JSON object almost always but satisfies the runtime contract on under 1.5 % of items in the untrained conditions, because it emits the `encoding` field as a string. OLMo-2-1B-Instruct recovers a complete object on only 30–33 % of items in the prompt-only condition. The interventions do not simply add reliability: fine-tuning raises contract compliance decisively at 1.7B under the current code state (from about 1 % to 84–99 %) but *lowers* it by 13.87 pp at 8B and by 27–30 pp at OLMo, and retrieval on top of an adapter lowers it by a further 12.04 pp at 8B. Retrieval alone improves recoverability only where it is missing: +15 to +17 pp for OLMo, replicated across three seeds, and exactly zero effect at both larger Qwen3 scales.

**RQ2 — Effect of the two interventions on the chart decision.** The answer depends on which of the three measurement levels is used, and that dependence is the main methodological result of this thesis. Measured by the pre-specified pipeline metric, the effect of fine-tuning ranges from −7.66 pp at 8B to +85.04 pp at 1.7B. Measured on the raw text, with the runtime-contract requirement and the generation-length penalty removed, the same contrast collapses into a narrow band of +0.36 to +10.58 pp, positive in all eight blocks and significant in seven. Retrieval alone is small and positive at 1.7B (+2.19 to +3.65 pp), neutral at OLMo, and **significantly negative at both larger Qwen3 scales** (−9.49 pp at 8B and −8.76 pp at 27B, both from code-state homogeneous pairs and both identical at all three measurement levels). Adding retrieval on top of an adapter is negative in eleven of twelve blocks and significantly negative in eight; the only exception is the 27B model, where D − C is +0.73 pp and not significant.

**RQ3 — Model scale and model family.** Within Qwen3, absolute performance rises with capacity while the marginal value of the interventions falls, and one of them reverses sign. Fine-tuning supplies both the output contract and about ten points of chart agreement at 1.7B, and about four points of chart agreement and nothing on the contract at 8B and 27B. Retrieval supplies a small gain at 1.7B and a significant loss at both larger scales. The two larger profiles agree closely on every base-model quantity and diverge on what fine-tuning costs: at 27B the adapter improves agreement and macro-F1 with no loss of validity, while at 8B the same recipe improves text-level agreement but costs 13.87 pp of contract satisfaction and lowers macro-F1. Across families at the small scale, the two base models are indistinguishable on the chart decision itself (86.9–87.2 % for Qwen3-1.7B, 86.1–87.6 % for OLMo at the text level) and very different on whether they can deliver it inside the output contract.

**RQ4 — Stability.** Seed variability is small for the untrained conditions (standard deviation 0.2–1.1 pp across three seeds) and larger for the trained conditions in the weaker family (up to 4.7 pp). Under the tested synonym paraphrase, systems that produce usable output are stable — 100 % consistency for both Qwen3-8B base-model conditions, 96.7–100 % for the 27B model, 81–99 % for the current-state Qwen3-1.7B fine-tuned conditions — and no condition loses more than 1.5 pp of agreement. Under information removal, no condition demonstrates a reliable clarification behaviour: the highest observed clarification rate is 3.65 %, and every fine-tuned condition is at exactly 0.0 %.

**RQ5 — Human-rated design quality.** Not answered. No ratings were collected.

## 8.2 What the measurement decomposition shows

The single most consequential finding of this study is not about retrieval or fine-tuning. It is that a metric which requires a valid record before it will score a decision measures both, and reports the result as if it measured only the decision.

The pre-specified top-1 metric reads the primary chart from the parsed runtime object. If the object does not parse, the item is counted as an incorrect chart. This is a defensible design for an end-to-end system evaluation: a recommendation the pipeline cannot read is useless in practice. It is not a defensible basis for the claim "fine-tuning improves chart selection", because it cannot distinguish a model that chose the wrong chart from a model that chose the right chart inside a record with one type error.

The size of the difference is not marginal. Qwen3-1.7B in the prompt-only condition scores 0.0–0.4 % on the pipeline metric and 86.5–87.2 % once the chart is read from the raw JSON object. A reader of the pipeline number alone would conclude that a 1.7B instruction-tuned model cannot perform this task at all. The correct conclusion is that it performs the analytical part of the task about as well as the OLMo model of comparable size, and fails a single structural convention: it writes `"encoding": "payment_method_code"` where the contract expects `"encoding": {"x": ..., "y": ..., "aggregate": ...}`.

The decomposition also changes a sign. At Qwen3-8B, the pipeline metric reports C − A as −7.66 pp, which reads as "fine-tuning made chart selection worse". At the text level the same contrast is +4.38 pp and significant in the opposite direction. Both numbers are correct about different things: the adapter did improve the chart decision, and it did cost the system 36 responses that no longer fit inside the output budget. A single figure cannot carry both statements, and reporting only the first would have been wrong in a way that no amount of statistical care would have caught.

Two lessons follow, and they generalise beyond this project.

The first concerns the design of structured-generation evaluations. Format conformance and content correctness must be measured on separate axes, because interventions affect them differently and because a single combined number can move by eighty percentage points, or change sign, for reasons that have nothing to do with the quality of the decision. The literature already reports that format restrictions themselves change what models produce \cite{tam2024formatrestrictions} and that language models are sensitive to superficial features of prompt format in ways that shift measured performance substantially \cite{sclar2024formatspread}. The present study adds an evaluation-side observation to that picture: the *scoring* pipeline can introduce a comparable artifact, independently of the model.

The second concerns the value of retaining raw outputs. The decomposition in Chapter 7 was possible only because every run stored the complete raw model text alongside the parsed object. Had the pipeline stored only the parsed representation, both artifacts would have been undetectable after the fact and would have propagated into every conclusion in this thesis. Raw-output retention should be treated as a requirement of a structured-generation experiment rather than as a debugging convenience.

The three-level decomposition is itself a post-hoc analysis, introduced after inspecting the level-1 results, and Chapter 7 labels it as such throughout. It is not a replacement for the pre-specified metric; it is a diagnostic that explains it. The honest summary is that the pre-specified metric answered a different question from the one the research question asked, and that the raw outputs allowed the intended question to be answered as well.

## 8.3 Error analysis

### 8.3.1 The encoding type error

For Qwen3-1.7B in conditions A and B across all three seeds, and in conditions C and D at seed 42, essentially every response fails the runtime contract with the same Pydantic error: `kpi_chart_mapping.0.encoding — Input should be a valid dictionary`. The model produces a complete, well-formed JSON object with the six required fields, a plausible task type, a correct chart type in about 87 % of cases, and an `encoding` value that is a single column name rather than a channel mapping.

The failure is systematic rather than random: it occurs on 270–274 of 274 items in the affected conditions and on essentially none in the unaffected ones. It disappeared in the seed-43 and seed-44 adapters trained under the later code state, which raise contract compliance to 84–99 %. And no other model in the study shows it at all — Qwen3-8B, Qwen3.8-27B and OLMo never emit a string-valued encoding. This makes it a property of the interaction between the smallest model and the prompt's field specification rather than an intermittent generation error or a defect in the prompt as such. A 1.7B model appears to read "encoding" as the name of a thing to be filled in with a field, and the prompt's type description was not sufficient to override that reading.

The methodological consequence for this thesis is that every chart number for the affected conditions must be read at level 2 or 3. The engineering consequence — that the prompt should specify the encoding object explicitly by example, or that constrained decoding should enforce the type \cite{willard2023guided,geng2025structured} — is noted as an implication, not as an executed experiment.

### 8.3.2 Truncation at the output budget

The second failure mode is quantitatively larger and is not a modelling failure at all. Of the OLMo prompt-only responses at seed 42 that failed to parse, 189 of 191 do not end with a closing brace. Across all conditions, OLMo truncates on 254–274 of 274 items, Qwen3-8B on 0 (A and B), 36 (C) and 69 (D) items, and the current-state Qwen3-1.7B fine-tuned conditions on 0–38 items. The 27B model never truncates.

Inspection of the stored prefixes shows why this matters. A truncated OLMo fine-tuned response opens with a valid `context_summary`, gives the correct `chart_type`, and populates the encoding channels correctly before stopping mid-array inside `having`. The analytical content is present and correct; the response simply did not fit. Text-level chart agreement recovers 87–95 % from these prefixes in the OLMo A, B and C conditions, against pipeline values of 2.6–24.1 %.

The Qwen3-8B row isolates the mechanism cleanly, and it is the clearest evidence in the study that the ceiling is an experimental-design artifact rather than a model property. The same base model produces **zero** truncations in the prompt-only and RAG conditions and **thirty-six** after fine-tuning, rising to **sixty-nine** when retrieval is added on top. Nothing about the model changed except which target format it had learned and how much input budget the prompt consumed.

The cause is the interaction between three fixed design choices. The generation budget was set to 512 new tokens to bound key-value cache growth during 27B inference. The reference format is long: it carries thirteen encoding keys per mapping in addition to the six top-level fields, and mean raw responses run to roughly 1,500–2,300 characters. And fine-tuning teaches the model to reproduce exactly that long format. The three choices are individually reasonable and jointly produce a systematic ceiling for any model whose tokenizer is less efficient or whose generation is more verbose.

Reporting "fine-tuning destroys OLMo's output validity" would be a true statement about the measured pipeline and a misleading statement about the model. The accurate statement is that under a 512-token budget, fine-tuning on this verbose target moves the typical response past the budget, and the pipeline records the consequence as a parse failure.

### 8.3.3 Chart confusions

Where the chart decision can be read, the errors are concentrated in a small number of interpretable substitutions rather than spread over the vocabulary.

For both larger Qwen3 models the dominant substitution under retrieval is `pie` → `donut`. At 27B it rises from 5 items under prompt-only to 28 of the 36 pie items under retrieval; at 8B it rises from **zero** to 28, and appears again at 29 items in the fine-tuning-plus-retrieval condition. The second family is `stacked_bar` → `bar`, at 4–6 items per condition, which is also why the fine-tuned 27B macro-F1 of 0.762 sits well below its 96.7 % top-1: recall on the eleven `stacked_bar` references is only 0.45. The third is `bar` → `histogram`, at 5–9 items in the base-model conditions of both larger models.

For Qwen3-1.7B at seeds 43 and 44, the fine-tuned condition confuses `stacked_bar` → `bar` (10 items) and `pie` → `bar` (4). The retrieval-plus-adapter condition adds a different and larger error family: `bar` → `stacked_bar` (17), `bar` → `line` (10), `bar` → `scatter` (7) and `bar` → `pie` (5). Retrieval moves this model *away* from the dominant reference class in several directions at once, which is consistent with the D − C degradation reported in Chapter 7.

At 8B, fine-tuning produces a different pattern again. Text-level agreement rises by 4.38 pp, but macro-F1 falls from 0.558 to 0.530, and the per-class values show why: F1 on `line` drops from 1.00 to 0.67 and on `stacked_bar` from 0.59 to 0.15, while `bar` and `pie` hold. The adapter moved the model toward the majority class. An aggregate agreement metric rewards that move on a split where three quarters of the references are bars; a class-balanced metric does not. Neither metric settles whether the resulting recommendations are better.

Two of these substitution families deserve separate comment because the reference is not obviously right. A donut and a pie encode the same part-to-whole relationship; a stacked bar and a grouped set of bars can both express a composition. Whether a substitution of this kind is an error depends on the analytical task and on the audience, and the task-based visualization literature is explicit that effectiveness is task- and distribution-dependent rather than fixed per chart type \cite{saket2019taskbased,kim2018taskdata}. The reference-agreement metric cannot make that judgement. This is precisely the gap that the unexecuted human-evaluation layer was designed to fill.

### 8.3.4 Alternatives collapse to empty after adaptation

Every fine-tuned condition of every model produced exactly zero items with more than one distinct recommendation. The contrast with the base-model conditions is total: Qwen3-8B offers alternatives on all 274 items under both A and B, and Qwen3.8-27B on 266 and 267 items.

The adapters learned to emit an empty `alternatives` list because that is what the frozen training targets predominantly contain. This is a faithful reproduction of the supervision, and it is an instance of a general property of supervised adaptation that matters for design tasks: the model learns the *distribution* of the targets, including the places where the targets are impoverished. It is also the clearest example in the study of a behavioural change that the aggregate metrics do not capture. A system that proposes one chart and a system that proposes a ranked set of defensible charts are different products, and for a visualization task where more than one encoding is frequently defensible, the second is arguably the more useful one. Fine-tuning on this dataset converts the second into the first, and every agreement metric in Chapter 7 is blind to that conversion. If the intended behaviour is to offer alternatives, the training targets have to contain them.

## 8.4 Why retrieval behaves differently before and after adaptation

Retrieval alone and retrieval after adaptation are two different interventions in this experiment, and their effects differ both in sign and in mechanism.

Before adaptation, the effect depends entirely on what the base model is missing. For OLMo, retrieval supplies structure the model does not have: JSON recovery rises by 15–17 pp across all three seeds while text-level chart agreement stays essentially unchanged (−4.01 to +0.36 pp). The retrieved passages help the model produce more of a well-formed record before the budget runs out, without changing which chart it names. For Qwen3-1.7B the same intervention gives a small positive effect on the chart decision (+2.19 to +3.65 pp) and no effect on format. For the two larger Qwen3 models, which have nothing structural to gain, the only measurable effect is on content — and it is negative.

After adaptation, retrieval competes with what the adapter has learned. The adapter encodes a specific output format and a specific mapping policy derived from the frozen targets; the retrieved passages introduce text that was never present during training and that sometimes recommends a different choice. The D − C contrast at the text level is negative in eleven of twelve blocks, with effect sizes of −3.7 to −8.4 pp for Qwen3-1.7B, −18.98 pp for Qwen3-8B and −27.4 to −36.5 pp for OLMo. The magnitude tracks how fragile the model's output behaviour already is: the more marginal the model's capacity for the task, the more damage the additional context does.

Three mechanisms are consistent with the observations, and the study can separate them only partially.

**Guideline conflict.** This mechanism is directly traceable, and it now replicates. The retrieved chart-selection document states *"Prefer donut over pie: Donut charts are easier to read because the center can show a total value."* At 27B, retrieval raised pie→donut substitutions from 5 to 28 of 36 pie items, which accounts for almost the entire −8.76 pp B − A difference. At 8B, the same retrieval raised them from 0 to 28, which more than accounts for the −9.49 pp difference. Two independently sized models, two code-state homogeneous contrasts, one sentence in the corpus. The models followed the guidance they were given; the metric penalised them for doing so. This is not a model failure and it is not a retrieval failure. It is a disagreement between the knowledge base and the reference, exposed by a metric that treats the reference as ground truth.

**Input-budget competition.** Retrieved passages consume part of the 3,584-token input budget. For a model that is already near the output ceiling, less room and a longer prompt make truncation more likely. Qwen3-8B shows this cleanly: truncations rise from 36 under C to 69 under D, and JSON recovery falls by 12.04 pp on a code-state homogeneous pair. OLMo shows the same direction at a higher baseline.

**Distraction.** For Qwen3-1.7B under D, the error profile changes shape rather than merely degrading: `bar` references are answered with `stacked_bar`, `line`, `scatter` and `pie`. A small model given several partly relevant design rules appears to be pulled toward whichever chart the retrieved text mentions.

The practical reading is narrow and specific to this setting. Retrieval was investigated here as a mechanism for supplying explicit domain guidance at inference time. On this task it helped a weak model produce more usable structure, it did not improve the chart decision at any scale and significantly worsened it at the two larger ones, and it degraded the chart decision further when combined with an adapter trained on the same task. Whether the degradation reflects worse recommendations or merely different ones cannot be settled by reference agreement, and for the donut cases there is concrete reason to think the metric is at least partly at fault.

## 8.5 Scale, family, and what the comparison licenses

The Qwen3 sweep now covers three scales with a complete A–D matrix at each, and it supports one clear statement about the relationship between capacity and the value of an intervention: **the interventions in this study substitute for capacity, and the substitution is not uniform across what they supply.**

Output-contract compliance is acquired by the base model between 1.7B and 8B. Below that threshold, fine-tuning is the only thing in this study that supplies it. Above it, fine-tuning cannot add compliance and at 8B actively costs it. Chart agreement behaves differently: fine-tuning improves it at every scale, by about ten points at 1.7B and about four at 8B and 27B, so its value diminishes but does not vanish. Retrieval is the intervention that reverses: mildly helpful at the small scale, significantly harmful at both larger ones.

This pattern is worth stating carefully because it can be misread in two ways. It does not say that fine-tuning is unnecessary at larger scales: +4.38 pp on 274 items is a real and statistically detectable improvement at both 8B (Holm-adjusted p = 0.034) and 27B (p = 9.8 × 10⁻⁴), and the fine-tuned 27B condition also produces the highest macro-F1 in the study (0.762 for C, 0.789 for D against 0.445 for A). It also does not say that a bigger model is always the better choice: the 27B fine-tuned condition took 8.8 GPU-hours to train and 76–81 seconds per item to run, against 4.9 hours and 22.5 seconds for the 8B profile on the same class of hardware.

The 8B and 27B profiles are also a useful internal control on each other. They agree on the base-model quantities to within one percentage point — 91.61 % against 92.34 % prompt-only agreement, −9.49 against −8.76 pp for B − A, the same donut mechanism at the same 28 items — and they disagree on what the common fine-tuning recipe costs. At 27B the adapter adds agreement and macro-F1 with no loss of validity; at 8B it adds agreement but loses 13.87 pp of contract satisfaction and lowers macro-F1. The two adapters differ in training window (1,024 tokens for 27B against 4,096 for 8B) and the two base models differ in context architecture, so the study cannot say which of these explains the divergence. It can say that "fine-tuning with this recipe is safe at large scale" is not supported: it was safe at one of the two large scales tested.

The cross-family comparison licenses less. OLMo-2-1B-Instruct and Qwen3-1.7B agree on the chart decision at almost identical rates in the prompt-only condition, so the analytical difficulty of the task is apparently similar for both. They diverge sharply on delivering that decision inside the required contract and within the output budget. Because architecture, tokenizer, pretraining corpus, post-training recipe and native context length all differ simultaneously between the two families — the OLMo checkpoint uses multi-head attention with 16 query and 16 key-value heads and a 4,096-token position limit, the Qwen3 profile grouped-query attention with 16 query and 8 key-value heads and a 40,960-token limit — no single factor can be identified as the cause. The correct formulation is that this is a model-family comparison in which several properties change together.

What the cross-family check does establish is worth having. A conclusion drawn from the Qwen3 sweep alone — "QLoRA fine-tuning on this task improves the chart decision at every scale" — happens to survive in OLMo at the text level (+0.36 to +8.76 pp). A conclusion about usable output does not: fine-tuning takes OLMo from 30 % JSON recovery to 2.9 %. The check therefore did what a cross-family control is for: it separated a finding that generalises from one that does not. OLMo is compared conceptually with the smallest Qwen3 profile only; it is not size-matched to the 8B or 27B profiles and no conclusion treats it as such.

## 8.6 Methodological problems encountered and how they were handled

Several problems arose during the project that affected methodological validity rather than only implementation. Each is reported with the reason it mattered, the response, and what remains unresolved. Debugging history that had no methodological consequence is not included.

**Source-task mismatch between candidate datasets and the target task.** Several public visualization resources were examined during dataset construction. Datasets whose primary task depends on a rendered chart image do not expose the query, aggregation, field roles or constraint scope that the target task requires, so they could not supply the analytical core of a brief. nvBench was selected because it provides source-grounded natural-language, schema, query and visualization information aligned with the analytical component of the task \cite{luo2021nvbenchsigmod}. **Residual limitation:** nvBench is not a dashboard-design corpus, so the dashboard-level fields had to come from somewhere else, which is the origin of the next problem.

**Generated supervision for the dashboard-level fields.** Six fields — users, context summary, layout, styling, interactions and rationales — are not present in the source corpus and were produced by an LLM under a constrained enrichment procedure. This matters because a model fine-tuned on generated targets learns the conventions of the generator, and agreement with those targets is not independent evidence of quality. The response was to keep the source-grounded analytical fields immutable during enrichment, to record field-level provenance, to validate deterministically, to mark generated records `not_gold`, and to keep the held-out test byte-identical across the freeze steps. **Residual limitation:** generated annotation cannot substitute for expert gold. The evaluation design compensates through an independent effectiveness layer and a human layer, and neither has been executed, so the compensation is currently theoretical.

**A strict validator narrower than the training target.** The strict response schema forbids extra keys in the encoding object, while the reference encodings carry thirteen additional source-grounded keys. The metric therefore has a construction ceiling of 0 % for any faithful system. This was detected by applying the validator to the 274 reference records during the preparation of Chapter 6 rather than by inspecting the code, and it invalidates any interpretation of strict schema validity as a quality ranking. The response was to verify the ceiling programmatically, publish the check as an artifact, and exclude the metric from every comparative statement. **Residual limitation:** the metric remains in the reported tables for completeness and is a standing invitation to misread; it should be either aligned with the reference contract or removed in future work.

**A fixed output budget below the length of the target format.** Described in Section 8.3.2. The response was to detect truncation explicitly, quantify it per condition, and re-score the chart decision at a level that survives it. **Residual limitation:** the affected runs were not repeated with a larger budget, so the study cannot report what OLMo or the fine-tuned Qwen3-8B condition would achieve without the ceiling. Every OLMo conclusion and the Qwen3-8B C and D conclusions are conditional on the 512-token budget.

**Duplicate concurrent writes to the same output files.** A first attempt at the prompt-only and RAG conditions for Qwen3-8B was started twice, and two processes wrote into the same prediction files, producing 548 lines for 274 unique identifiers with conflicting generations. Metrics over such a file mix two executions. The response had three parts: the defect was detected by an identifier-uniqueness check rather than by inspection of the numbers; the affected exports were quarantined and excluded; and both conditions were re-inferred from a single process, which is what Chapter 7 reports. The uniqueness check is now part of the standing completeness criterion for every run. **Residual limitation:** none for the reported results — the Qwen3-8B row is now a complete and clean A–D matrix — but the episode is the reason the 8B A and B runs carry a later git commit than the 8B C and D runs, which makes the C − A contrast at that scale code-state mixed.

**Heterogeneous code states across runs.** The thirty-two runs were executed over several weeks across four compute environments, and their manifests record different git commits; five of the eight model-by-seed blocks contain at least one run whose commit was not recorded at all. The response was to make code-state comparability an explicit, machine-computed label attached to every reported contrast, at both block and pair level, rather than an unstated assumption. This is what makes it possible to say that the two B − A contrasts carrying the main retrieval finding, and the D − C contrasts at 8B, 27B and OLMo seeds 42 and 43, are code-state homogeneous. **Residual limitation:** nineteen of forty contrasts still involve a run with an unrecorded commit, and this cannot be repaired by analysis. Only re-running the affected conditions under one pinned commit would repair it.

**Hardware-determined training precision.** The precision helper resolves `precision: auto` against the weakest available device, so two of the three Qwen3-1.7B adapters were trained in fp16 on a Tesla T4 while the rest were trained in bf16. The response was to record the effective precision in the training metadata and to report the three seeds individually rather than as a pooled mean. **Residual limitation:** the three Qwen3-1.7B seeds are not three draws from one procedure, which weakens them as a variance estimate.

**Knowledge-base identity across platforms.** The knowledge-base digest recorded by the runs differs from the digest of the local repository copy. The cause was determined rather than assumed: converting the local file from CRLF to LF reproduces the recorded digest exactly, so the retrieved content is identical and only the line terminator differs. **Residual limitation:** a digest that depends on checkout platform is a weak identity. Normalising line endings before hashing would make the artifact identity platform-independent.

## 8.7 Threats to validity

### 8.7.1 Internal validity

**Code-state confounding.** Seven of the forty contrasts are code-state homogeneous, fourteen are mixed and nineteen involve a run with an unrecorded commit. A significant contrast from a non-homogeneous pair establishes that the runs differ; it does not establish that the method is the only reason. The contrasts most central to the conclusions are in better shape than the average: B − A at 8B and 27B, and D − C at 8B, 27B and OLMo seeds 42 and 43, are all homogeneous. The C − A contrasts, on which the fine-tuning conclusion rests, are not homogeneous in any block.

**Differing training sequence length.** Two adapters were trained with a maximum sequence length of 4,096 tokens and two with 1,024, despite sharing every other hyperparameter. Truncating longer training targets changes what the adapter learns. Qwen3-8B and OLMo trained at 4,096, Qwen3-1.7B and Qwen3.8-27B at 1,024, so the 8B-versus-27B comparison of what fine-tuning costs is partly confounded with training window.

**Differing inference hardware.** Runs executed on five GPU types. This does not affect correctness metrics but makes latency non-comparable across conditions, as Chapter 7 states. The Qwen3-8B block is the one place where all four conditions ran on the same device.

**Adapter provenance in two runs.** Six of the eight D runs record their source C run identifier explicitly. Two record only the adapter path, which points at the correct condition-C directory for the same model and seed, together with a matching training-configuration hash. This is strong but indirect evidence for those two runs.

### 8.7.2 Construct validity

**The reference is not ground truth.** Every chart number in this thesis is agreement with the project's frozen reference. The analytical fields of that reference are source-grounded, but the dashboard-level fields are LLM-generated, and even for the chart type the reference records one defensible choice where several may exist. The donut-versus-pie case, now observed at two model scales, and the stacked-bar-versus-bar case are concrete instances where the reference decides a question that the visualization literature treats as task-dependent \cite{saket2019taskbased,kim2018taskdata}.

**No independent effectiveness criterion was applied.** The layer that would score a predicted chart against the set of charts shown effective for the corresponding task in published human-subject studies is specified but not implemented, and every per-item record carries `l1_covered = not_applicable`.

**No human judgement of design quality exists.** Layout, styling, accessibility, interaction design, rationale quality and overall usefulness are unmeasured. The automatic metrics do not approximate them, and no claim about them is made.

**Aggregate agreement hides a behavioural change.** The collapse of the alternatives list after adaptation is invisible to every metric reported in Chapter 7 and was detected only by counting distinct recommendations per item. Other behavioural changes of the same kind may be present and unmeasured.

**Grounding is a lexical proxy.** The grounding numbers measure content-word overlap between rationale claims and retrieved passages. Overlap is not entailment. Additionally, only six of the sixteen RAG conditions have claim coverage above 70 %, so ten of the reported grounding rates describe a small and non-representative subset of items.

**The clarification measure is a regular expression.** The missing-information clarification rate is detected by matching phrases such as *clarify*, *insufficient* and *please specify*. It will miss an implicit request and can fire on an unrelated use of a matched word. The near-zero values are consistent across every condition, which makes a purely detector-side explanation unlikely, but the measure is a diagnostic and not a semantic judgement.

### 8.7.3 External validity

**One task, one output contract, one source corpus.** The held-out split contains 274 items drawn from one source corpus, with a chart distribution dominated by `bar` (208 of 274) and only two `scatter` items. Generalisation beyond this distribution is not supported. Task families that were introduced only in the generated training material — distribution, ranking, deviation, flow — do not appear in the test at all and are therefore untested.

**Structured recommendations, not rendered dashboards.** The system predicts a design object. It does not render it, and no user completed a task with it. Claims about decision quality, time on task, adoption or business value are outside the evidence.

**One adaptation configuration.** The evidence concerns one QLoRA setting: rank 16, α = 32, dropout 0.05, all linear target modules, NF4 4-bit quantization, three epochs, learning rate 2 × 10⁻⁴, cosine schedule. It is not evidence about QLoRA in general, and it is not a comparison between adaptation algorithms.

**One retriever and one small corpus.** TF-IDF over 41 chunks from three project-authored documents. A different retriever or a larger corpus could produce different retrieval effects, and the donut mechanism shows how directly a single sentence in the corpus can move a metric by nine percentage points.

**Four models, asymmetric seed coverage.** Two models have three seeds and two have one. The scale trend within Qwen3 rests on single runs at 8B and 27B.

### 8.7.4 Statistical validity

**Pairing is respected; independence across seeds is not assumed.** Comparisons are item-level and paired, Holm correction is applied within each block of pairwise tests, and bootstrap intervals resample items with pairing preserved. Seeds are treated as repeated realisations and are never pooled into a sample of independent observations.

**No cross-block correction.** Holm is applied within a block — one model, one seed, one outcome. Across the whole thesis many blocks are tested, and no global correction is applied. Individual contrasts should therefore be read with their effect sizes and intervals rather than as isolated significance decisions.

**Small discordant counts.** Several contrasts rest on very few discordant pairs (for example two for the 27B D − C comparison). The exact binomial form of McNemar is used precisely for this reason, but a non-significant result from two discordant pairs carries almost no information and is not evidence of equivalence.

**Statistical significance is not practical importance.** Several significant contrasts are four or five percentage points on 274 items. Whether such a difference matters for a design assistant is a question the statistics cannot answer.

**Variance is not estimated where it cannot be.** No standard deviation, confidence interval or stability claim is reported for a single-seed condition, and the reporting layer refuses to compute seed-level intervals from three values.

### 8.7.5 Reproducibility

**The analysis is fully reproducible; the runs are not fully reproducible.** Every number in Chapter 7 can be regenerated from the stored artifacts by one script with no GPU and no network access. The runs themselves depend on upstream model repositories, and the model revision is pinned for two of the four profiles and null for the other two. Sampling is enabled at temperature 0.1 and the runs used five GPU types, so recording a seed documents the configuration without guaranteeing bitwise regeneration.

**Dataset and knowledge-base inputs are anchored.** The train, validation and test digests are identical in all thirty-two run manifests and were recomputed from the local files. The knowledge-base content is identical across all sixteen RAG runs, subject to the line-ending caveat above.

**Engineering correctness is not scientific validity.** The repository's tests establish that the implementation behaves as intended; the frozen-data validation establishes data integrity; the automatic metrics establish task performance against a constructed reference; the multi-seed analysis establishes stochastic stability where three seeds exist; the statistical analysis quantifies uncertainty about observed differences. None of these establishes that a generated dashboard recommendation is a good design. That evidence layer is missing and is named as missing.

## 8.8 Practical implications

The implications below follow from the evidence and are stated at the strength it supports.

**Below about 8B parameters, fix the output contract before anything else; above it, do not pay for what you already have.** The largest single gain observed anywhere in this study was contract compliance at 1.7B, and it was worth eighty percentage points of measured accuracy while the underlying chart decision changed by ten. At 8B and 27B the base model already satisfies the contract, and fine-tuning buys about four points of agreement while risking output validity. Where the contract is the problem, constrained decoding is the obvious alternative to training and is available in the repository but was not enabled for the reported runs \cite{willard2023guided,geng2025structured}.

**Set the generation budget from the target length, not from memory convenience.** A 512-token ceiling against a target that routinely needs more produced the most severe apparent failures in the study, and the Qwen3-8B block shows that they were not model failures: the same base model truncated zero responses without an adapter and sixty-nine with an adapter and retrieval.

**Check that the knowledge base and the reference agree before interpreting a retrieval result.** The clearest negative retrieval effect in this study was produced by two different models correctly following a single guideline sentence that the reference contradicts. A retrieval corpus is an opinion about design, and if it disagrees with the evaluation reference the experiment measures the disagreement, not the method.

**Do not assume that retrieval and fine-tuning compose.** In eleven of twelve blocks, adding retrieval to a fine-tuned adapter lowered chart agreement, and at 8B it also cost twelve points of output validity. If both are used, the combination needs its own evaluation rather than an assumption of additivity.

**Watch what adaptation removes, not only what it adds.** Fine-tuning eliminated the alternatives list in every condition of every model, and no metric in the pre-specified set would have shown it.

**Report structured-generation results at more than one level of strictness, and keep the raw outputs.** Both recommendations follow directly from Section 8.2.

Nothing in this study supports a recommendation about which dashboard design a practitioner should adopt. That would require the human evidence that has not been collected.

## References Used in Chapter 8

Generated from `docs/thesis/references.bib` by `experiments/scripts/build_chapter_reference_lists.py`. The BibTeX key of each entry is given in brackets.

Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N., et al. (2025). *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models*. https://doi.org/10.48550/arXiv.2501.10868 [`geng2025structured`]

Kim, Y., & Heer, J. (2018). *Assessing Effects of Task and Data Distribution on the Effectiveness of Visual Encodings*. Computer Graphics Forum, 37(3), 157–167. https://doi.org/10.1111/cgf.13409 [`kim2018taskdata`]

Luo, Y., Tang, N., Li, G., Chai, C., Li, W., & Qin, X. (2021). *Synthesizing Natural Language to Visualization (NL2VIS) Benchmarks from NL2SQL Benchmarks*. Proceedings of the 2021 International Conference on Management of Data, 1235–1247. https://doi.org/10.1145/3448016.3457261 [`luo2021nvbenchsigmod`]

Saket, B., Endert, A., & Demiralp, C. (2019). *Task-Based Effectiveness of Basic Visualizations*. IEEE Transactions on Visualization and Computer Graphics, 25(7), 2505–2512. https://doi.org/10.1109/TVCG.2018.2829750 [`saket2019taskbased`]

Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I Learned to Start Worrying about Prompt Formatting*. The Twelfth International Conference on Learning Representations (ICLR). arXiv:2310.11324. https://arxiv.org/abs/2310.11324 [`sclar2024formatspread`]

Tam, Z. R., Wu, C., Tsai, Y., Lin, C., Lee, H., & Chen, Y. (2024). *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Large Language Model Performance*. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, 1218–1236. https://doi.org/10.18653/v1/2024.emnlp-industry.91 [`tam2024formatrestrictions`]

Willard, B. T., & Louf, R. (2023). *Efficient Guided Generation for Large Language Models*. https://doi.org/10.48550/arXiv.2307.09702 [`willard2023guided`]
