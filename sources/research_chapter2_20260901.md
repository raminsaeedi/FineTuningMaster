# Literature lookup record for Chapter 2

Date: 2026-09-01

Scope: dashboard design, visualization principles, analytical tasks, Transformer language models, instruction tuning, PEFT, LoRA variants, QLoRA, RAG, accessibility, and structured outputs.

The sources below were checked through publisher, proceedings, ACL Anthology, W3C, JSON Schema, Google Research, or arXiv landing pages. The URLs are retained so that the metadata and the claims used in Chapter 2 can be checked again before submission. Project-specific implementation claims were checked against the repository and are not treated as external literature findings.

The 27 URLs in the Chapter 2 reference list were checked on 2026-09-01. They either returned successfully or resolved through a DOI redirect. Several publisher pages returned HTTP 403 because of access protection; this was treated as an access response, not as evidence of a broken DOI.

## Dashboard and visualization

Yigitbasioglu and Velcu (2012), “A review of dashboards in performance management: Implications for design and research,” _International Journal of Accounting Information Systems_, 13(1), 41–59. DOI: https://doi.org/10.1016/j.accinf.2011.08.002. Used for the role of dashboards in performance management and design research.

Bach et al. (2023), “Dashboard design patterns,” _IEEE Transactions on Visualization and Computer Graphics_, 29(1), 342–352. DOI: https://doi.org/10.1109/TVCG.2022.3209448. Used for recurring dashboard design problems and patterns.

Cleveland and McGill (1984), “Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods,” _Journal of the American Statistical Association_, 79(387), 531–554. DOI: https://doi.org/10.1080/01621459.1984.10478080. Used for the relative accuracy of elementary graphical encodings.

Mackinlay (1986), “Automating the Design of Graphical Presentations of Relational Information,” _ACM Transactions on Graphics_, 5(2), 110–141. DOI: https://doi.org/10.1145/22949.22950. Used for expressiveness, effectiveness, and automated graphical presentation.

Munzner (2014), _Visualization Analysis and Design_. CRC Press. DOI: https://doi.org/10.1201/b17511. Used for the relation between data, tasks, and visualization design.

Brehmer and Munzner (2013), “A Multi-Level Typology of Abstract Visualization Tasks,” _IEEE Transactions on Visualization and Computer Graphics_, 19(12), 2376–2385. DOI: https://doi.org/10.1109/TVCG.2013.124. Used for the distinction between task purpose, means, and data.

Saket et al. (2019), “Task-Based Effectiveness of Basic Visualizations,” _IEEE Transactions on Visualization and Computer Graphics_, 25(7), 2505–2512. DOI: https://doi.org/10.1109/TVCG.2018.2829750. Used for task-dependent visualization effectiveness. The publication year is 2019.

Kim and Heer (2018), “Assessing Effects of Task and Data Distribution on the Effectiveness of Visual Encodings,” _Computer Graphics Forum_, 37(3), 157–167. DOI: https://doi.org/10.1111/cgf.13409. Used for the effect of task and data distribution on encoding effectiveness.

Moritz et al. (2019), “Formalizing Visualization Design Knowledge as Constraints: Actionable and Extensible Models in Draco,” _IEEE Transactions on Visualization and Computer Graphics_, 25(1), 438–448. DOI: https://doi.org/10.1109/TVCG.2018.2865240. Used for constraint-based visualization recommendation.

## Language models and generation

Vaswani et al. (2017), “Attention Is All You Need,” _Advances in Neural Information Processing Systems_, 30, 5998–6008. arXiv: https://arxiv.org/abs/1706.03762. Used for the Transformer and self-attention overview.

Wei et al. (2022), “Finetuned Language Models are Zero-Shot Learners,” ICLR 2022. Google Research record: https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/. arXiv: https://arxiv.org/abs/2109.01652. Used for the definition and motivation of instruction tuning.

Ji et al. (2023), “Survey of Hallucination in Natural Language Generation,” _ACM Computing Surveys_, 55(12), Article 248. DOI: https://doi.org/10.1145/3571730. Used for the limitation that fluent generated text can be unsupported.

## PEFT and memory-efficient fine-tuning

Houlsby et al. (2019), “Parameter-Efficient Transfer Learning for NLP,” _Proceedings of the 36th International Conference on Machine Learning_, PMLR 97, 2790–2799. https://proceedings.mlr.press/v97/houlsby19a.html. Used for adapter-based parameter-efficient transfer.

Hu et al. (2022), “LoRA: Low-Rank Adaptation of Large Language Models,” ICLR 2022. arXiv: https://arxiv.org/abs/2106.09685. Used for low-rank updates with frozen pretrained weights.

Dettmers et al. (2023), “QLoRA: Efficient Finetuning of Quantized LLMs,” _Advances in Neural Information Processing Systems_, 36, 10088–10115. DOI: https://doi.org/10.52202/075280-0441. Used for 4-bit base quantization, NF4, double quantization, and paged optimizers.

Li and Liang (2021), “Prefix-Tuning: Optimizing Continuous Prompts for Generation,” ACL-IJCNLP 2021, 4582–4597. DOI: https://doi.org/10.18653/v1/2021.acl-long.353. Used for prefix tuning.

Lester et al. (2021), “The Power of Scale for Parameter-Efficient Prompt Tuning,” EMNLP 2021, 3045–3059. DOI: https://doi.org/10.18653/v1/2021.emnlp-main.243. Used for prompt tuning.

Liu et al. (2022), “Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning,” _Advances in Neural Information Processing Systems_, 35, 1950–1965. DOI: https://doi.org/10.52202/068431-0142. Used for IA3 and activation-scaling PEFT.

Ben Zaken et al. (2022), “BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models,” ACL 2022, 1–9. DOI: https://doi.org/10.18653/v1/2022.acl-short.1. Used for bias-only fine-tuning.

Zhang et al. (2023), “Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning,” ICLR 2023. arXiv: https://arxiv.org/abs/2303.10512. Used for AdaLoRA and adaptive rank allocation.

Kalajdzievski (2023), “A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA,” arXiv:2312.03732. https://arxiv.org/abs/2312.03732. Used for the RSLoRA scaling rule.

Liu et al. (2024), “DoRA: Weight-Decomposed Low-Rank Adaptation,” _Proceedings of the 41st International Conference on Machine Learning_, PMLR 235, 32100–32121. https://proceedings.mlr.press/v235/liu24bn.html. Used for magnitude–direction decomposition.

Zhao et al. (2024), “GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection,” _Proceedings of the 41st International Conference on Machine Learning_, PMLR 235, 61121–61143. https://proceedings.mlr.press/v235/zhao24s.html. Used to distinguish low-rank gradient projection with full-parameter learning from strict PEFT.

## RAG, accessibility, and structured outputs

Lewis et al. (2020), “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” _Advances in Neural Information Processing Systems_, 33, 9459–9474. arXiv: https://arxiv.org/abs/2005.11401. Used for the RAG framework and the separation of parametric and retrieved knowledge.

World Wide Web Consortium (W3C) (2024), _Web Content Accessibility Guidelines (WCAG) 2.2_, W3C Recommendation. https://www.w3.org/TR/WCAG22/. Used for the accessibility principle that information should not depend on color alone and for contrast-related guidance.

JSON Schema (2022), _JSON Schema Draft 2020-12_. https://json-schema.org/draft/2020-12. Used for schema-based description and validation of JSON instances.

Geng et al. (2025), “JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models,” arXiv preprint arXiv:2501.10868. DOI: https://doi.org/10.48550/arXiv.2501.10868. Used for the distinction between structured-output validity, coverage, efficiency, and quality. The record follows the current arXiv version available on 2026-09-01.

## Claim boundary

The chapter uses the sources above for general concepts and published method descriptions. It does not transfer their reported benchmark results to this thesis. Statements about the project’s six-field output, A/B/C/D methods, QLoRA configuration, same-seed adapter reuse, local RAG corpus, and evaluation layers are repository-specific implementation statements and must be checked against the corresponding project files before final submission.

This audit covers the references used by Chapter 2. The global bibliography also contains older records used by other chapters, including entries that still require a separate metadata and source check before the complete thesis is submitted. They were not used as evidence for the Chapter 2 text.
