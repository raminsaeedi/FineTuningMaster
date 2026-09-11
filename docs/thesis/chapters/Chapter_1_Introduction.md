# Chapter 1 — Introduction

This chapter introduces the research problem, defines the task that the thesis studies, states the research gap and the research questions, and explains how the remaining chapters answer them. It first describes why the design of a dashboard is a decision problem rather than a rendering problem, and why that decision problem is a plausible target for a language model. It then narrows the general idea into the concrete task used throughout this work: mapping a short natural-language dashboard brief to a machine-readable design recommendation. The chapter closes with the contributions, the scope of the study, and an overview of the thesis structure. Readers who finish this chapter should know what is being predicted, which four system configurations are compared, which evidence the study can produce, and which questions it deliberately leaves open.

## 1.1 Motivation

Dashboards are one of the most common ways in which organisations present quantitative information. They combine several coordinated views so that an audience can monitor a situation, compare alternatives, or decide on an action. The visualization literature has repeatedly pointed out that a dashboard is defined less by the number of charts it shows than by the fit between the displayed information and the decision context of its audience. A review of dashboards in performance management describes them as systems that combine performance information, visual display, and interaction in support of management decisions \cite{yigitbasioglu2012dashboard}. A systematic analysis of how the term is actually used in research and practice finds a broad and partly inconsistent design space, ranging from static reporting displays to interactive analytic environments \cite{sarikaya2019dashboards}. More recent work argues that recurring solutions in this space can be described as design patterns, for example patterns for overview, comparison, filtering, and detail \cite{bach2023dashboardpatterns}.

Producing a good dashboard therefore involves a sequence of connected decisions. Which measures matter for this audience and this goal? What analytical task does each measure imply — a comparison, a trend, a distribution, a part-to-whole relation? Which chart type expresses that task in a way that people can decode accurately? How should the fields be mapped onto visual channels? How should the views be arranged so that the most important information is found first? Which styling and accessibility conventions apply? Which interactions are needed, and what justifies each choice? Each of these questions is informed by an established body of knowledge. Controlled studies of graphical perception show that some encodings support more accurate quantitative judgements than others \cite{cleveland1984graphical}. Expressiveness and effectiveness criteria distinguish a representation that states the intended information from one that also communicates it efficiently \cite{mackinlay1986apt}. Task-based studies show that the effectiveness of a basic visualization depends on the analytical task and on the data distribution rather than on a universal ranking of chart types \cite{saket2019taskbased,kim2018taskdata}.

The practical problem is that this knowledge is not evenly distributed. The people who know the data and the decision are frequently not the people who know the visualization literature. Existing recommendation systems address part of the gap. Constraint-based approaches encode design knowledge as explicit rules that can rank or reject encodings \cite{moritz2019draco}, and mixed-initiative tools suggest encodings during exploratory analysis \cite{wongsuphasawat2016voyager,wongsuphasawat2017voyager2}. Natural-language interfaces let an analyst request a chart in ordinary words and translate that request into a visualization specification \cite{narechania2021nl4dv,luo2022nmt}. These systems are valuable, but almost all of them stop at the level of one chart for one query. They do not produce a coordinated dashboard design that also fixes layout, styling, interaction, and the reasons behind each decision.

Large language models change what is feasible here, because they can read an unstructured description of a situation and emit a structured object in one step. Two established mechanisms make it possible to adapt such a model to a specialised task without training it from scratch. Retrieval-augmented generation attaches relevant external passages to the prompt so that a frozen model can use knowledge that is not reliably present in its parameters \cite{lewis2020rag}. Parameter-efficient fine-tuning freezes the pretrained weights and learns a small low-rank update instead \cite{hu2022lora}, and its quantized variant makes this affordable on a single accelerator by keeping the frozen base model in 4-bit precision \cite{dettmers2023qlora}. Both mechanisms are widely used, but they intervene at different points: one changes what the model sees, the other changes what the model is. Whether either of them — or their combination — actually helps on a structured dashboard-design task is an empirical question, and it is the question this thesis addresses.

## 1.2 The task studied in this thesis

The thesis studies one narrowly defined prediction problem. The input is a *dashboard brief*: a short specification containing the intended users, the analytical goals, the key performance indicators (KPIs), the available data columns with their types, and any constraints. The output is a *structured dashboard-design recommendation*: a single JSON object with six required top-level fields — `context_summary`, `kpi_chart_mapping`, `layout`, `styling`, `interactions`, and `rationales`. Each entry of `kpi_chart_mapping` links one KPI to an analytical task type, a primary chart type, optional alternatives, and an encoding that names the fields placed on the visual channels. Chapter 4 documents how the data for this task was built and frozen; Chapter 5 gives the formal contract and the implementation.

The output is deliberately a structured object rather than free prose or a rendered image. A structured object can be parsed deterministically, validated against a schema, compared field by field, and consumed by a later rendering layer. It also makes a controlled comparison between systems possible, because every condition is scored against the same field names and the same controlled vocabularies. This choice has a cost that the thesis takes seriously throughout: format compliance and design quality are different properties. A response can be syntactically valid JSON and still recommend the wrong chart, and a response can recommend a defensible chart inside a record that fails the format contract. Structured-generation research shows both that schema conformance is itself a non-trivial capability \cite{geng2025structured} and that imposing format restrictions can measurably change what a model produces \cite{tam2024formatrestrictions}. The evaluation design in Chapter 6 therefore keeps parse success, contract validity, completeness, and chart agreement as separate quantities.

## 1.3 Research gap

Three gaps motivate the study.

The first gap is in task scope. Public resources for natural-language-to-visualization research supply a query, a database schema, and a target chart specification. nvBench, for example, was synthesised from a text-to-SQL benchmark and provides cross-domain natural-language-to-visualization pairs at large scale \cite{luo2021nvbenchsigmod,yu2018spider}, and its successor addresses ambiguity in that mapping \cite{luo2025nvbench2}. These resources cover the analytical core of the present task well. They do not cover the dashboard-level fields — users, layout, styling, interactions, rationales — that distinguish a coordinated dashboard from a single chart. No public benchmark maps a dashboard brief of the kind defined above onto the full structured recommendation used here. A dataset for this task therefore has to be constructed, and its construction has to be transparent about which fields are source-grounded and which were generated.

The second gap is in controlled comparison. Retrieval augmentation and parameter-efficient fine-tuning are usually studied separately, on different tasks, with different base models and different prompts. Reports that compare them on one task, with one frozen dataset, one prompt contract, one output parser, one evaluation code path, and matched seeds are rare — and rarer still across several model sizes and more than one model family. Without those controls it is difficult to attribute an observed difference to the intervention rather than to an incidental change in prompt, data, or decoding settings.

The third gap is in evaluation. Many reported results for structured generation compress several distinct properties into a single accuracy number. If a metric requires a fully valid record before it will score the chart decision, then a formatting failure and a design failure become indistinguishable. The present study needs an evaluation protocol that separates these layers explicitly, that states which reference labels are independent evidence and which are the project's own construction, and that marks what has not been measured.

## 1.4 Research questions

The study is organised around four answerable research questions and one question that is defined but deliberately left open.

**RQ1 — Structured-output reliability.** How reliably does each of the four conditions produce a machine-usable dashboard recommendation, measured as the rate at which a JSON object can be recovered from the response, the rate at which the recovered object satisfies the output contract, and the completeness of the required fields?

**RQ2 — Effect of the two interventions on the chart decision.** How does adding retrieved design guidance, adding QLoRA-based task adaptation, and combining the two change agreement with the reference primary chart? Specifically, how much of any observed change is attributable to the chart decision itself and how much to whether the surrounding record satisfied the output contract and the generation-length budget?

**RQ3 — Model scale and model family.** Do the effects observed for RQ1 and RQ2 persist across increasing capacity within one model family (Qwen3 at roughly 1.7B, 8B, and 27B parameters), and do they persist in a second, independently developed model family at a comparable small scale (OLMo 2 at roughly 1.49B parameters)?

**RQ4 — Stability.** How stable are the observed effects across repeated training and generation seeds where repeated seeds were executed, and how do the conditions behave under two controlled input perturbations: a meaning-preserving paraphrase of the brief, and a brief from which information has been removed?

**RQ5 — Human-rated design quality (defined, not answered).** How do human raters judge chart appropriateness, layout quality, styling and accessibility, interaction design, rationale quality, and overall usefulness across the four conditions? The instrument, the rubric, the assignment design, and the analysis plan for this question are specified in Chapter 6, and the study infrastructure exists in the repository. No ratings have been collected in the current project state. RQ5 is therefore reported as an open question and not as a result. Guidance on human evaluation of generated text requires that the rating task, rater instructions, sampling, number of raters, aggregation rule, and agreement measure all be reported \cite{vanderlee2019human}; none of these can be reported for data that does not exist.

The corresponding expectations were formulated before the results were inspected. Adding explicit design guidance at inference time was expected to help most where the base model has little task-specific structure available, and adding task-specific adaptation was expected to help most with the output contract. No directional expectation was formulated for the combination, because retrieval can also displace input budget and introduce text that conflicts with a learned output style.

## 1.5 Approach

The study compares exactly four system conditions for a fixed base model and a fixed seed:

- **A — Prompt-only.** The base instruction-tuned model receives the shared system and user prompt and nothing else. This is the reference condition.
- **B — Retrieval-augmented generation.** The same base model additionally receives up to three passages retrieved from a small, frozen corpus of visualization and dashboard design guidelines.
- **C — QLoRA fine-tuning.** The base model is adapted with a quantized low-rank adapter trained on the frozen training split. No retrieval is used.
- **D — Fine-tuning with retrieval.** The adapter produced by condition C for the *same* base model, dataset, training configuration, and seed is loaded, and the retrieval procedure of condition B is applied on top of it.

Condition D reuses the C adapter rather than training a second one. Training D independently would introduce a second stochastic training trajectory and would make the incremental effect of retrieval impossible to isolate. The contrasts that the design supports are therefore B − A for retrieval alone, C − A for adaptation alone, D − C for retrieval added after adaptation, and D − B for adaptation added under retrieval. The direct D − A contrast compares the complete systems but changes two factors at once and cannot attribute the difference to either one.

Everything not named by a condition is held fixed: the frozen dataset release and its hashes, the prompt builder, the controlled task and chart vocabularies, the output parser, the generation settings, the held-out test split, and the evaluation code. The experiment was executed across four base models — three Qwen3 profiles spanning roughly 1.7B to 27B parameters and one OLMo 2 profile at roughly 1.49B parameters \cite{yang2025qwen3,olmo2025furious} — with three seeds where the compute budget allowed and a single seed for the more expensive profiles. Chapter 6 states the executed coverage precisely and explains why the coverage is asymmetric.

## 1.6 Contributions

The thesis makes the following contributions, each stated at the strength the available evidence supports.

1. **A frozen, lineage-annotated dataset for the dashboard-brief-to-recommendation task.** The release combines nvBench-derived, source-grounded analytical records with LLM-generated dashboard-level enrichment and additional generated records. Every field carries an explicit evidence class, the generated material is labelled as generated rather than as gold, the held-out test is byte-identical across the freeze steps, and the package is published with a manifest and file hashes. Chapter 4 documents the construction; the limitations of generated supervision are stated there and revisited in Chapter 8.

2. **A configuration-driven implementation of the four conditions with enforced adapter provenance.** One prompt source, one parser, one evaluation path, and a validator that refuses to load an adapter into condition D unless the base model, model revision, model-configuration hash, training-configuration hash, dataset version, and seed match the producing condition-C run.

3. **A layered evaluation protocol that keeps evidence types apart.** Parse success, contract validity, completeness, chart agreement, robustness under two perturbation families, and retrieval grounding are reported as separate quantities, each with an explicit statement of what it does and does not establish. Reference-based chart agreement is labelled as agreement with the project's own reference, not as evidence of visualization effectiveness.

4. **An executed A/B/C/D comparison across three Qwen3 scales and one cross-family OLMo 2 model**, with paired per-item statistics — Cochran's Q, exact McNemar tests with Holm correction, and percentile bootstrap confidence intervals for paired differences — computed from the stored per-item records \cite{cochran1950,mcnemar1947,holm1979,efron1979bootstrap}.

5. **A diagnostic decomposition of the headline chart metric.** The stored raw model outputs are re-scored at three levels of strictness so that a failure of the output contract, a failure of the generation-length budget, and a failure of the chart decision can be separated. This decomposition is reported in Chapter 7 and is the basis for several of the conclusions in Chapter 8.

6. **A transparent record of what was not measured.** The independent chart-effectiveness scorer, the realism layer, and the human-rating layer are specified but not executed. They are reported as open, not silently omitted.

## 1.7 Scope and delimitation

The system predicts a structured design recommendation. It does not render a dashboard, does not execute queries against a database, and does not measure how quickly a human completes a task with the resulting display. Claims about decision quality, adoption, or business value are therefore outside what the current evidence can support.

The adaptation method is fixed to one QLoRA configuration. The thesis compares prompting, retrieval, and adaptation; it does not compare adaptation algorithms against each other. Alternative parameter-efficient methods are discussed in Chapters 2, 3, and 5 to justify the selection, and they are explicitly not part of the experiment matrix.

The retriever is a sparse TF-IDF retriever over a small frozen guideline corpus. Dense retrieval is available in the repository as an optional configuration and was not used for the reported conditions; it is discussed only where it helps explain the design choice.

Model coverage is limited to four base models and to the seed coverage that the compute budget allowed. Seed replication is complete for the two smaller models and partial for the larger ones. The consequence — that variability statements are weaker for the larger models — is stated wherever it applies rather than being generalised away.

Finally, the reference recommendations against which the automatic metrics score contain LLM-generated fields. Agreement with them measures how well a system reproduces the project's frozen representation. It is not, by itself, evidence that a recommendation is a good dashboard design. This boundary is maintained in every results statement.

## 1.8 Structure of the thesis

**Chapter 2** introduces the background: dashboard design, visualization principles, analytical tasks and chart types, language models and instruction tuning, parameter-efficient fine-tuning including LoRA and QLoRA, retrieval-augmented generation, and structured outputs.

**Chapter 3** reviews related work on visualization recommendation, natural-language interfaces, datasets and benchmarks for visualization generation, LLM-assisted visualization, parameter-efficient fine-tuning for structured generation, retrieval augmentation, and the evaluation of structured outputs, and positions the present study against them.

**Chapter 4** documents the dataset: source selection, the target schema and its evidence classes, the source-faithful transformation of nvBench, quality filtering, deduplication and split isolation, the constrained LLM enrichment, the controlled augmentation and semantic repair, the freeze, and the resulting limitations.

**Chapter 5** describes the system: the formal task contract, the model matrix and the controlled factors, the shared prompt and generation pipeline, the four methods, the reasons for selecting QLoRA over the alternatives, inference and output processing, reproducibility and provenance, and methodological threats to validity.

**Chapter 6** defines the experimental setup and the evaluation protocol: the executed run matrix and its seed coverage, every metric with its interpretation and its limits, the perturbation conditions, the statistical analysis plan, and the specification of the human evaluation that has not yet been carried out.

**Chapter 7** reports the results from the generated artifacts only: structured-output reliability, chart agreement at three levels of strictness, the paired method contrasts with confidence intervals and corrected p-values, seed variability, robustness, retrieval grounding, and cost.

**Chapter 8** interprets the results, analyses the failure modes item by item, and states the threats to validity that bound each conclusion.

**Chapter 9** summarises what the study established, what it did not, and which next steps follow directly from the open items.

## References Used in Chapter 1

Generated from `docs/thesis/references.bib` by `experiments/scripts/build_chapter_reference_lists.py`. The BibTeX key of each entry is given in brackets.

Bach, B., Freeman, E., Abdul-Rahman, A., Turkay, C., Khan, S., Fan, Y., et al. (2023). *Dashboard Design Patterns*. IEEE Transactions on Visualization and Computer Graphics, 29(1), 342–352. https://doi.org/10.1109/TVCG.2022.3209448 [`bach2023dashboardpatterns`]

Cleveland, W. S., & McGill, R. (1984). *Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods*. Journal of the American Statistical Association, 79(387), 531–554. https://doi.org/10.1080/01621459.1984.10478080 [`cleveland1984graphical`]

Cochran, W. G. (1950). *The Comparison of Percentages in Matched Samples*. Biometrika, 37(3-4), 256–266. https://doi.org/10.1093/biomet/37.3-4.256 [`cochran1950`]

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. Advances in Neural Information Processing Systems, 36, 10088–10115. https://doi.org/10.52202/075280-0441 [`dettmers2023qlora`]

Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife*. The Annals of Statistics, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552 [`efron1979bootstrap`]

Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N., et al. (2025). *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models*. https://doi.org/10.48550/arXiv.2501.10868 [`geng2025structured`]

Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 6(2), 65–70. https://www.jstor.org/stable/4615733 [`holm1979`]

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference on Learning Representations. https://openreview.net/forum?id=nZeVKeeFYf9 [`hu2022lora`]

Kim, Y., & Heer, J. (2018). *Assessing Effects of Task and Data Distribution on the Effectiveness of Visual Encodings*. Computer Graphics Forum, 37(3), 157–167. https://doi.org/10.1111/cgf.13409 [`kim2018taskdata`]

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems, 33, 9459–9474. https://arxiv.org/abs/2005.11401 [`lewis2020rag`]

Luo, T., Huang, C., Shen, L., Li, B., Shen, S., Zeng, W., et al. (2025). *nvBench 2.0: Resolving Ambiguity in Text-to-Visualization through Stepwise Reasoning*. Advances in Neural Information Processing Systems, 38, 138749–138786. https://doi.org/10.52202/085713-4172 [`luo2025nvbench2`]

Luo, Y., Tang, N., Li, G., Chai, C., Li, W., & Qin, X. (2021). *Synthesizing Natural Language to Visualization (NL2VIS) Benchmarks from NL2SQL Benchmarks*. Proceedings of the 2021 International Conference on Management of Data, 1235–1247. https://doi.org/10.1145/3448016.3457261 [`luo2021nvbenchsigmod`]

Luo, Y., Tang, N., Li, G., Tang, J., Chai, C., & Qin, X. (2022). *Natural Language to Visualization by Neural Machine Translation*. IEEE Transactions on Visualization and Computer Graphics, 28(1), 217–226. https://doi.org/10.1109/TVCG.2021.3114848 [`luo2022nmt`]

Mackinlay, J. (1986). *Automating the Design of Graphical Presentations of Relational Information*. ACM Transactions on Graphics, 5(2), 110–141. https://doi.org/10.1145/22949.22950 [`mackinlay1986apt`]

McNemar, Q. (1947). *Note on the Sampling Error of the Difference between Correlated Proportions or Percentages*. Psychometrika, 12(2), 153–157. https://doi.org/10.1007/BF02295996 [`mcnemar1947`]

Moritz, D., Wang, C., Nelson, G. L., Lin, H., Smith, A. M., Howe, B., et al. (2019). *Formalizing Visualization Design Knowledge as Constraints: Actionable and Extensible Models in Draco*. IEEE Transactions on Visualization and Computer Graphics, 25(1), 438–448. https://doi.org/10.1109/TVCG.2018.2865240 [`moritz2019draco`]

Narechania, A., Srinivasan, A., & Stasko, J. (2021). *NL4DV: A Toolkit for Generating Analytic Specifications for Data Visualization from Natural Language Queries*. IEEE Transactions on Visualization and Computer Graphics, 27(2), 369–379. https://doi.org/10.1109/TVCG.2020.3030378 [`narechania2021nl4dv`]

OLMo Team, Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., et al. (2025). *2 OLMo 2 Furious*. https://doi.org/10.48550/arXiv.2501.00656 [`olmo2025furious`]

Saket, B., Endert, A., & Demiralp, C. (2019). *Task-Based Effectiveness of Basic Visualizations*. IEEE Transactions on Visualization and Computer Graphics, 25(7), 2505–2512. https://doi.org/10.1109/TVCG.2018.2829750 [`saket2019taskbased`]

Sarikaya, A., Correll, M., Bartram, L., Tory, M., & Fisher, D. (2019). *What Do We Talk About When We Talk About Dashboards?*. IEEE Transactions on Visualization and Computer Graphics, 25(1), 682–692. https://doi.org/10.1109/TVCG.2018.2864903 [`sarikaya2019dashboards`]

Tam, Z. R., Wu, C., Tsai, Y., Lin, C., Lee, H., & Chen, Y. (2024). *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Large Language Model Performance*. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, 1218–1236. https://doi.org/10.18653/v1/2024.emnlp-industry.91 [`tam2024formatrestrictions`]

van der Lee, C., Gatt, A., van Miltenburg, E., Wubben, S., & Krahmer, E. (2019). *Best Practices for the Human Evaluation of Automatically Generated Text*. Proceedings of the 12th International Conference on Natural Language Generation, 355–368. https://doi.org/10.18653/v1/W19-8643 [`vanderlee2019human`]

Wongsuphasawat, K., Moritz, D., Anand, A., Mackinlay, J., Howe, B., & Heer, J. (2016). *Voyager: Exploratory Analysis via Faceted Browsing of Visualization Recommendations*. IEEE Transactions on Visualization and Computer Graphics, 22(1), 649–658. https://doi.org/10.1109/TVCG.2015.2467191 [`wongsuphasawat2016voyager`]

Wongsuphasawat, K., Qu, Z., Moritz, D., Chang, R., Ouk, F., Anand, A., et al. (2017). *Voyager 2: Augmenting Visual Analysis with Partial View Specifications*. Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems, 2648–2659. https://doi.org/10.1145/3025453.3025768 [`wongsuphasawat2017voyager2`]

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Gao, G., et al. (2025). *Qwen3 Technical Report*. arXiv preprint arXiv:2505.09388. https://doi.org/10.48550/arXiv.2505.09388 [`yang2025qwen3`]

Yigitbasioglu, O. M., & Velcu, O. (2012). *A Review of Dashboards in Performance Management: Implications for Design and Research*. International Journal of Accounting Information Systems, 13(1), 41–59. https://doi.org/10.1016/j.accinf.2011.08.002 [`yigitbasioglu2012dashboard`]

Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., et al. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 3911–3921. https://doi.org/10.18653/v1/D18-1425 [`yu2018spider`]
