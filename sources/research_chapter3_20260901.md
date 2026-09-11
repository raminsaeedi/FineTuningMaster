# Literature lookup record for Chapter 3

Date: 2026-09-01

Scope: visualization recommendation, natural-language visualization interfaces, visualization datasets and benchmarks, LLM-based visualization assistance, parameter-efficient fine-tuning, RAG, and evaluation of structured outputs.

The sources below were checked against publisher, proceedings, ACL Anthology, PMLR, NeurIPS, W3C, JSON Schema, or arXiv landing pages. DOI links are retained when a publisher page is access-protected. Project-specific statements were checked against repository files and are not treated as findings from the external literature.

## Visualization recommendation and natural-language interfaces

Mackinlay (1986), “Automating the Design of Graphical Presentations of Relational Information,” *ACM Transactions on Graphics*, 5(2), 110–141. DOI: https://doi.org/10.1145/22949.22950. Used for the APT formulation and the distinction between expressiveness and effectiveness.

Mackinlay, Hanrahan, and Stolte (2007), “Show Me: Automatic Presentation for Visual Analysis,” *IEEE Transactions on Visualization and Computer Graphics*, 13(6), 1137–1144. DOI: https://doi.org/10.1109/TVCG.2007.70594. The Tableau publication page is also available at https://www.tableau.com/research/publications/show-me-automatic-presentation-visual-analysis. Used for automatic mark selection, presentation commands, and the analyst-controlled interaction model.

Wongsuphasawat et al. (2016), “Voyager: Exploratory Analysis via Faceted Browsing of Visualization Recommendations,” *IEEE Transactions on Visualization and Computer Graphics*, 22(1), 649–658. DOI: https://doi.org/10.1109/TVCG.2015.2467191. The authors’ project page is https://idl.uw.edu/papers/voyager. Used for mixed-initiative exploration and a gallery of recommended views.

Wongsuphasawat et al. (2017), “Voyager 2: Augmenting Visual Analysis with Partial View Specifications,” *Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems*, 2648–2659. DOI: https://doi.org/10.1145/3025453.3025768. Used for recommendations that complete partially specified views.

Moritz et al. (2019), “Formalizing Visualization Design Knowledge as Constraints: Actionable and Extensible Models in Draco,” *IEEE Transactions on Visualization and Computer Graphics*, 25(1), 438–448. DOI: https://doi.org/10.1109/TVCG.2018.2865240. Used for hard and soft visualization constraints and inspectable design knowledge.

Narechania, Srinivasan, and Stasko (2021), “NL4DV: A Toolkit for Generating Analytic Specifications for Data Visualization from Natural Language Queries,” *IEEE Transactions on Visualization and Computer Graphics*, 27(2), 369–379. DOI: https://doi.org/10.1109/TVCG.2020.3030378. Used for analytic specifications, attribute/task extraction, and JSON-based intermediate output.

Mitra et al. (2022), “Facilitating Conversational Interaction in Natural Language Interfaces for Visualization,” *2022 IEEE Visualization and Visual Analytics (VIS)*, 6–10. DOI: https://doi.org/10.1109/VIS54862.2022.00010. An open author version and metadata are available at https://arxiv.org/abs/2207.00189. Used for conversational follow-up, multiple conversations, and ambiguity resolution.

Dibia and Demiralp (2019), “Data2Vis: Automatic Generation of Data Visualizations Using Sequence-to-Sequence Recurrent Neural Networks,” *IEEE Computer Graphics and Applications*, 39(5), 33–46. DOI: https://doi.org/10.1109/MCG.2019.2924636. Used for neural generation of Vega-Lite visualization specifications.

Luo et al. (2022), “Natural Language to Visualization by Neural Machine Translation,” *IEEE Transactions on Visualization and Computer Graphics*, 28(1), 217–226. DOI: https://doi.org/10.1109/TVCG.2021.3114848. Used for the learned natural-language-to-visualization mapping.

Wu et al. (2022), “NL2Viz: Natural Language to Visualization via Constrained Syntax-Guided Synthesis,” *Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering*, 972–983. DOI: https://doi.org/10.1145/3540250.3549140. Crossref confirms the ten authors, proceedings title, and page range. Used for constrained syntax-guided visualization synthesis.

## Datasets and benchmarks

Fu et al. (2020), “Quda: Natural Language Queries for Visual Data Analytics,” arXiv:2005.03257. https://arxiv.org/abs/2005.03257. The abstract reports 14,035 queries with one or more analytic-task annotations and describes analyst seed queries followed by crowdsourced paraphrase generation and validation.

Luo et al. (2021), “Synthesizing Natural Language to Visualization (NL2VIS) Benchmarks from NL2SQL Benchmarks,” *Proceedings of the 2021 International Conference on Management of Data*, 1235–1247. DOI: https://doi.org/10.1145/3448016.3457261. Crossref confirms the author order Yuyu Luo, Nan Tang, Guoliang Li, Chengliang Chai, Wenbo Li, and Xuedi Qin. The benchmark counts used in Chapter 3 are 25,750 pairs, 750 tables, and 105 domains.

Luo et al. (2025), “nvBench 2.0: Resolving Ambiguity in Text-to-Visualization through Stepwise Reasoning,” *Advances in Neural Information Processing Systems 38*, 138749–138786, Datasets and Benchmarks Track. Official NeurIPS page: https://papers.nips.cc/paper_files/paper/2025/hash/b5f985cdc1796defe83a6b2f84fe2ab6-Abstract-Datasets_and_Benchmarks_Track.html. DOI: https://doi.org/10.52202/085713-4172. The published title replaces the older arXiv wording used in the previous draft. The official abstract reports 7,878 queries, 24,076 visualizations, 780 tables, and 153 domains, with controlled ambiguity injection and reasoning paths. The page range was checked against the DOI metadata and copied to the final BibTeX entry.

## LLM-based visualization assistance

Sah et al. (2024), “Generating Analytic Specifications for Data Visualization from Natural Language Queries using Large Language Models,” arXiv:2408.13391. https://arxiv.org/abs/2408.13391. Used for LLM-generated analytic specifications, phrase-to-entity mappings, design-principle explanations, conversational support, and ambiguity detection.

Tian et al. (2025), “ChartGPT: Leveraging LLMs to Generate Charts from Abstract Natural Language,” *IEEE Transactions on Visualization and Computer Graphics*, 31(3), 1731–1745. DOI: https://doi.org/10.1109/TVCG.2024.3368621. The arXiv record is https://arxiv.org/abs/2311.01920. Used for stepwise chart generation, task-specific visualization data, fine-tuning, and editable intermediate outputs.

Pesaran Zadeh et al. (2024), “Text2Chart31: Instruction Tuning for Chart Generation with Automatic Feedback,” *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 11459–11480. ACL Anthology: https://aclanthology.org/2024.emnlp-main.640/. DOI: https://doi.org/10.18653/v1/2024.emnlp-main.640. The author family name is “Pesaran Zadeh,” not “Zadeh.” The published record confirms 31 plot types and the page range used in the revised reference list.

Shin, Hong, and Elmqvist (2025), “Visualizationary: Automating Design Feedback for Visualization Designers Using Large Language Models,” *IEEE Transactions on Visualization and Computer Graphics*, 31(10), 8796–8813. DOI: https://doi.org/10.1109/TVCG.2025.3579700. The arXiv record is https://arxiv.org/abs/2409.13109. Crossref confirms the journal volume, issue, page range, title, and three authors. The arXiv abstract describes guideline prompts, perceptual filters, and a longitudinal study with 13 designers.

Snyder, Wang, and Drucker (2025), “Challenges & Opportunities with LLM-Assisted Visualization Retargeting,” *2025 IEEE Visualization and Visual Analytics (VIS)*, 141–145. DOI: https://doi.org/10.1109/VIS60296.2025.00034. The arXiv record is https://arxiv.org/abs/2507.01436. Crossref confirms the author order, title, venue, and page range. The ampersand in the published title is escaped in the BibTeX record. Used for the contrast between direct code generation and constrained structural guidance.

## PEFT, QLoRA, RAG, and evaluation

Hu et al. (2022), “LoRA: Low-Rank Adaptation of Large Language Models,” ICLR 2022. https://arxiv.org/abs/2106.09685. Used for low-rank updates with frozen pretrained weights.

Dettmers et al. (2023), “QLoRA: Efficient Finetuning of Quantized LLMs,” *Advances in Neural Information Processing Systems 36*, 10088–10115. Official NeurIPS page: https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html. DOI: https://doi.org/10.52202/075280-0441. Used for 4-bit base quantization, NF4, double quantization, and paged optimizers.

Houlsby et al. (2019), “Parameter-Efficient Transfer Learning for NLP,” *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, 2790–2799. https://proceedings.mlr.press/v97/houlsby19a.html. Used for adapter tuning.

Li and Liang (2021), “Prefix-Tuning: Optimizing Continuous Prompts for Generation,” ACL-IJCNLP 2021, 4582–4597. DOI: https://doi.org/10.18653/v1/2021.acl-long.353. Used for prefix tuning.

Lester, Al-Rfou, and Constant (2021), “The Power of Scale for Parameter-Efficient Prompt Tuning,” EMNLP 2021, 3045–3059. DOI: https://doi.org/10.18653/v1/2021.emnlp-main.243. Used for prompt tuning.

Ben Zaken, Goldberg, and Ravfogel (2022), “BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models,” ACL 2022, 1–9. DOI: https://doi.org/10.18653/v1/2022.acl-short.1. Used for bias-only adaptation.

Liu et al. (2022), “Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning,” *Advances in Neural Information Processing Systems 35*, 1950–1965. Official NeurIPS page: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html. Used for IA3 and activation scaling.

Zhang et al. (2023), “AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning,” ICLR 2023. https://arxiv.org/abs/2303.10512. Used for adaptive rank allocation.

Kalajdzievski (2023), “A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA,” arXiv:2312.03732. https://arxiv.org/abs/2312.03732. Used for RSLoRA scaling.

Liu et al. (2024), “DoRA: Weight-Decomposed Low-Rank Adaptation,” *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 32100–32121. https://proceedings.mlr.press/v235/liu24bn.html. Used for magnitude–direction decomposition.

Zhao et al. (2024), “GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection,” *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 61121–61143. https://proceedings.mlr.press/v235/zhao24s.html. Used to distinguish low-rank gradient projection with full-parameter learning from strict PEFT.

Lewis et al. (2020), “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” *Advances in Neural Information Processing Systems 33*, 9459–9474. https://arxiv.org/abs/2005.11401. Used for the separation of parametric generation and retrieved non-parametric memory.

JSON Schema (2022), *JSON Schema Draft 2020-12*. https://json-schema.org/draft/2020-12. Used for schema-based structural validation.

Geng et al. (2025), “JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models,” arXiv:2501.10868. https://doi.org/10.48550/arXiv.2501.10868. Used for the distinction between schema validity, coverage, efficiency, and output quality.

Saket, Endert, and Demiralp (2019), “Task-Based Effectiveness of Basic Visualizations,” *IEEE Transactions on Visualization and Computer Graphics*, 25(7), 2505–2512. DOI: https://doi.org/10.1109/TVCG.2018.2829750. Used for task-dependent visual encoding effectiveness.

Kim and Heer (2018), “Assessing Effects of Task and Data Distribution on the Effectiveness of Visual Encodings,” *Computer Graphics Forum*, 37(3), 157–167. DOI: https://doi.org/10.1111/cgf.13409. Used for the role of task and data distribution in evaluation.

Ribeiro et al. (2020), “Beyond Accuracy: Behavioral Testing of NLP Models with CheckList,” ACL 2020, 4902–4912. DOI: https://doi.org/10.18653/v1/2020.acl-main.442. Used for systematic robustness and failure-mode testing.

van der Lee et al. (2019), “Best Practices for the Human Evaluation of Automatically Generated Text,” INLG 2019, 355–368. https://aclanthology.org/W19-8643/. Used for rubrics, independent ratings, and agreement reporting.

## Project evidence checked for the seed and method statements

The current model profile identifies the smallest final model as Qwen/Qwen3-1.7B: `src/config/model/qwen3_1_7b.yaml`. The final matrix lists Qwen3-1.7B, Qwen3-8B, Qwen3-14B, and Llama-3.1-8B: `src/config/matrix/final.yaml`.

The repository contains QLoRA, DoRA, RSLoRA, and optional GaLore training configurations. The main A/B/C/D comparison and adapter hand-over are documented in `src/config/training/qlora_default.yaml`, `src/config/training/dora.yaml`, `src/config/training/rslora.yaml`, `src/config/training/galore.yaml`, and `docs/project/PIPELINE_IMPLEMENTATION_AND_SCIENTIFIC_RATIONALE_V4.md`. The Chapter 3 text treats QLoRA as the main intervention and the other algorithms as optional ablations.

The seed wording in Chapter 3 follows the user-approved thesis plan rather than the older three-seed wording still present in some repository runbooks: three seeds are used only for the Qwen3-1.7B stability check; if the predefined stability criterion is met, remaining planned configurations use Seed 42. No stability result is claimed in Chapter 3 before those observations exist.
