# Chapter 3 — Related Work

## 3.1 Visualization Recommendation Systems

Automated visualization recommendation predates modern large language models by several decades. Early systems attempted to encode visualization design knowledge explicitly. Mackinlay’s APT system formalized graphical presentation as a search problem constrained by expressiveness and effectiveness criteria (Mackinlay, 1986). This work established an important idea that remains relevant today: visualization recommendation can be treated as a structured reasoning problem rather than a purely aesthetic one.

Commercial and research systems later moved toward mixed-initiative interfaces. Show Me integrated automatic chart recommendation into Tableau so that users could select data fields and receive suitable views without manually specifying every encoding (Mackinlay et al., 2007). Voyager extended recommendation toward exploratory analysis by systematically suggesting views over combinations of data attributes (Wongsuphasawat et al., 2016). Voyager 2 combined manual specification with recommendations, allowing analysts to progressively constrain the space of suggested charts (Wongsuphasawat et al., 2017).

These systems are important for this thesis because they show two complementary design philosophies. One approach represents visualization knowledge as explicit rules or constraints; another uses recommendations to support exploration while leaving final control to the user.

Draco provides one of the clearest formalizations of the constraint-based approach. Moritz et al. (2019) represent visualization design knowledge as hard and soft constraints. The framework can incorporate design principles and empirical findings, and can learn weights for soft constraints from experimental evidence. This is closely related to the motivation of the present work: structured dashboard recommendations should respect data and task constraints instead of being free-form suggestions.

However, traditional visualization recommender systems generally operate on structured data specifications rather than rich natural-language dashboard briefs. They also do not usually generate full textual rationales, styling recommendations, interactions, and layout suggestions in a single structured response. These differences motivate the use of LLMs as a more flexible generation layer.

## 3.2 Natural Language Interfaces and NL-to-Visualization

Natural-language interfaces for visualization aim to reduce the effort required to specify charts and analyses. Instead of manually selecting fields, transformations, and encodings, users describe their intent in natural language.

### 3.2.1 Rule- and Parser-Based Systems

NL4DV is a prominent example. It takes a tabular dataset and a natural-language query and produces a JSON analytic specification that includes data attributes, analytic tasks, and recommended Vega-Lite visualizations (Narechania et al., 2021). The system is especially relevant because it separates query understanding from the final visualization and makes intermediate analytic structure explicit.

Subsequent work extended NL4DV toward conversational interaction, allowing follow-up questions and ambiguity resolution across multiple turns (Mitra et al., 2022). This demonstrates that visualization intent is often not fully specified in a single utterance.

The strength of these systems is interpretability: inferred attributes and tasks can be inspected. Their limitation is that rule-based and semantic-parser approaches require substantial engineering and can be difficult to extend to broader language variation or complex reasoning.

### 3.2.2 Neural NL-to-Visualization

Data2Vis reframed visualization generation as neural sequence-to-sequence translation. Dibia and Demiralp (2019) trained an encoder–decoder model to map data descriptions to Vega-Lite specifications. Their work showed that models can learn syntax, transformations, and common visualization structures from examples.

Luo et al. (2022) proposed a neural machine translation approach for natural language to visualization. Their work addresses limitations of heuristic systems and supports more complex transformations. This line of research made large training corpora increasingly important and directly motivated benchmark development.

NL2VIZ introduced constrained syntax-guided synthesis and combined natural-language input with data and program context (2022). The method uses hard and soft constraints to handle uncertainty and underspecification. This is conceptually close to the present thesis because both approaches distinguish what is strongly supported by the input from what remains uncertain.

## 3.3 Datasets and Benchmarks for Visualization Generation

### 3.3.1 Quda

Quda was introduced by Fu et al. (2020) as a dataset for recognizing analytic tasks from free-form natural-language queries. It contains 14,035 queries annotated with one or more analytic tasks. The dataset was created using analyst seed queries followed by crowdsourced paraphrase generation and validation.

Quda is valuable for task recognition and linguistic diversity, but it does not provide complete dashboard designs. It therefore addresses one component of the present task rather than the full output schema.

### 3.3.2 nvBench

nvBench is a large-scale cross-domain benchmark for NL-to-visualization (Luo et al., 2021). It contains 25,750 natural-language/visualization pairs derived from 750 tables across 105 domains. The benchmark was synthesized from existing natural-language-to-SQL resources and includes visualization information suitable for training and evaluating NL2VIS models.

nvBench is especially relevant to this thesis because it provides natural-language analytical requests, database context, chart labels, and visualization structure. These components can ground goals, KPIs, fields, aggregations, and chart encodings.

However, nvBench should not be interpreted as a dataset of full dashboard designs. It primarily represents query-to-visualization examples, not audience-specific layout, styling, interaction design, or multi-view dashboard rationale. This limitation is important in the present thesis because these missing fields must be modeled separately and their provenance must be explicit.

### 3.3.3 nvBench 2.0

nvBench 2.0 extends the benchmark to ambiguous natural-language requests (Luo et al., 2025). It contains 7,878 queries and 24,076 corresponding visualizations over 780 tables and 153 domains. Ambiguity is introduced in a controlled manner so that multiple valid interpretations can be associated with one query.

The benchmark is important because it challenges the assumption that every natural-language request has exactly one correct chart. This is a useful observation for dashboard recommendation as well. In real applications, several visualizations can be defensible, and uncertainty should sometimes be preserved instead of collapsed into one arbitrary label.

### 3.3.4 Scope of Existing Benchmarks

Taken together, Quda, nvBench, and nvBench 2.0 cover analytical-task recognition, query understanding, chart selection, encoding, ambiguity, and cross-domain generalization. They do not directly cover the complete target of this thesis: a dashboard brief containing user context, goals, KPIs, fields, and constraints mapped to a structured recommendation containing chart mappings, layout, styling, interactions, and rationales.

This does not mean that no related benchmarks exist. Rather, no directly matching public benchmark was identified that covers the complete end-to-end task and output schema used here.

## 3.4 LLMs for Visualization and Dashboard Generation

The rise of instruction-tuned LLMs has shifted natural-language visualization research from task-specific parsers toward general-purpose models that can reason over textual requests and generate code or structured specifications.

### 3.4.1 LLM-Based Analytic Specification

Sah et al. (2024) investigated the use of LLMs to generate analytic specifications from natural-language queries. Their approach produces explicit attributes, analytic tasks, and visualization recommendations, and emphasizes explainability and debuggability. This work is closely related to NL4DV but uses LLM reasoning instead of relying only on traditional semantic parsing.

This line of work supports the idea that LLMs can act as flexible front ends for visualization systems. At the same time, it highlights that intermediate structure remains useful even when a powerful language model is used.

### 3.4.2 ChartGPT

ChartGPT uses LLMs to generate charts from abstract natural-language inputs (Tian et al., 2025). The system decomposes the task into a step-by-step reasoning pipeline because abstract user requests may be ambiguous or underspecified. The authors also create a task-specific dataset and use fine-tuning to provide visualization knowledge.

ChartGPT is highly relevant to the present thesis for two reasons. First, it demonstrates that general-purpose LLMs benefit from domain-specific adaptation for visualization tasks. Second, it supports decomposing a complex generation task into explicit intermediate decisions.

However, ChartGPT focuses primarily on chart generation, while the present thesis targets a broader structured dashboard recommendation including context, layout, styling, interactions, and design rationales.

### 3.4.3 LLM-Assisted Visualization Design

More recent work studies LLMs not only as code generators but as visualization assistants. Visualizationary, for example, explores the use of LLMs to provide design feedback based on visualization guidelines and perceptual measurements (Shin et al., 2025). This is conceptually close to the goal of providing actionable design guidance rather than simply producing plotting code.

Other work studies LLM-assisted visualization retargeting, where models adapt existing visualization code or structure to new data. Results show that models can help but remain sensitive to data transformation and structural mismatches (Snyder et al., 2025).

These studies reinforce a central point of this thesis: language models can assist visualization design, but reliable systems need structure, grounding, validation, and careful evaluation.

## 3.5 Fine-Tuning LLMs for Structured Generation

Fine-tuning has long been used to adapt general models to domain-specific tasks. For LLMs, full fine-tuning can be expensive, which has made PEFT methods especially important.

LoRA freezes the original model and learns low-rank updates to selected weight matrices (Hu et al., 2021). QLoRA further reduces memory requirements by quantizing the base model to 4-bit precision while training LoRA adapters (Dettmers et al., 2023).

For visualization generation, ChartGPT provides direct evidence that fine-tuning can improve task-specific chart generation (Tian et al., 2025). Text2Chart31 similarly studies instruction tuning for chart generation and combines a diverse chart dataset with automatic feedback (Zadeh et al., 2024).

The broader relevance is that structured recommendation tasks require both domain knowledge and output discipline. Fine-tuning can teach a model recurrent output patterns, field relationships, and vocabulary. Yet it also creates the risk of learning dataset-specific conventions. This is why the present thesis separates training data from independent evaluation and reports multi-seed experiments rather than relying on one trained adapter.

## 3.6 RAG for Domain-Grounded Generation

RAG was originally introduced as a way to combine parametric language-model knowledge with retrieved external evidence (Lewis et al., 2020). In visualization design, RAG is attractive because design guidance is distributed across books, empirical studies, guidelines, accessibility recommendations, and domain-specific documentation.

A RAG-based dashboard assistant can retrieve passages relevant to a brief and include them in the generation context. In principle, this allows the model to justify recommendations using explicit guidance rather than relying only on knowledge encoded during pretraining.

However, RAG introduces additional failure modes. Retrieval quality depends on the corpus, chunking, indexing, and query representation. Even relevant evidence can be ignored or misused by the generator. Therefore, a RAG system should be evaluated both for output quality and for grounding.

In this thesis, the value of RAG is tested experimentally rather than assumed. Method B compares RAG with prompt-only generation, while Method D tests whether retrieval adds value after fine-tuning.

## 3.7 Evaluation of LLM-Generated Structured and Visualization Outputs

Evaluation is particularly difficult because a recommendation can be correct in one dimension and wrong in another.

### 3.7.1 Structural Validity

For machine-readable outputs, the first level is technical validity:

- parse success,
- JSON validity,
- schema compliance,
- required-field completeness.

Structured-output research shows why this distinction matters. JSONSchemaBench, for example, evaluates constrained generation across schema coverage, efficiency, and output quality rather than treating valid syntax as the only criterion (Geng et al., 2025).

### 3.7.2 Semantic and Visualization Correctness

Visualization systems also need semantic evaluation. This can include:

- task identification,
- chart-type correctness,
- field and encoding correctness,
- aggregation correctness,
- constraint preservation,
- per-class performance.

nvBench and related NL2VIS benchmarks provide structured references for some of these dimensions. Yet exact-match evaluation has limitations because multiple charts can be reasonable for the same intent. nvBench 2.0 makes this explicit by representing ambiguity and multiple valid visualizations.

### 3.7.3 Robustness

A useful system should behave consistently when a request is paraphrased and should respond safely when information is missing. Behavioral testing approaches such as CheckList argue for testing model behavior systematically rather than relying on aggregate accuracy alone (Ribeiro et al., 2020).

For this thesis, robustness is therefore evaluated with paraphrased and partially specified briefs. A robust model should preserve correct recommendations under harmless wording changes and avoid inventing unsupported information when required context is missing.

### 3.7.4 Human Evaluation

Automatic metrics cannot fully measure actionability, coherence, layout quality, or the usefulness of a rationale. Human evaluation is therefore needed for the higher-level questions in the thesis.

Best-practice work in NLG recommends clear rubrics, controlled rater instructions, sufficient sample sizes, and reporting of agreement (van der Lee et al., 2019). Inter-rater reliability is important because subjective ratings may vary even when evaluation criteria are well defined.

### 3.7.5 LLM-as-Judge

Recent work often uses strong LLMs as evaluators. This can reduce cost, but LLM judgments should not automatically be treated as ground truth. Model-based judges can inherit biases, be sensitive to prompt formulation, and disagree with human preferences. For this reason, the present thesis treats human evaluation as the main evidence for subjective quality dimensions.

## 3.8 Research Gap and Positioning of This Thesis

The literature provides strong building blocks for the problem studied here:

- visualization recommendation systems formalize design constraints;
- NL4DV and related systems map natural language to analytic specifications;
- Quda provides task-labeled natural-language queries;
- nvBench provides large-scale cross-domain NL-to-visualization supervision;
- nvBench 2.0 introduces ambiguity and multiple valid outputs;
- Data2Vis and ncNet show neural generation of visualization specifications;
- ChartGPT demonstrates LLM-based chart generation with fine-tuning;
- RAG provides a mechanism for external grounding;
- LoRA and QLoRA provide efficient adaptation;
- structured-output methods support machine-readable generation.

However, these works mainly address components of the complete task. In the literature reviewed for this thesis, no directly matching public benchmark was identified for the end-to-end mapping:

\[
\text{dashboard brief}
\rightarrow
\text{structured multi-component dashboard design recommendation}
\]

where the input contains user goals, KPIs, data fields, and constraints, and the output jointly includes chart mappings, layout, styling, interactions, and rationales.

A second gap concerns experimental comparison. Prior work typically evaluates one system family or one generation strategy. The present thesis instead evaluates four controlled variants:

- A: prompt-only,
- B: RAG,
- C: QLoRA fine-tuning,
- D: QLoRA fine-tuning + RAG.

This design separates the effect of parameter adaptation from the effect of external knowledge retrieval.

A third gap concerns evaluation breadth. The thesis does not treat schema validity or chart accuracy as sufficient evidence by themselves. It combines structured-output metrics, chart/task correctness, robustness tests, grounding checks, and human evaluation.

The contribution is therefore not the claim that LLMs are the first method for visualization recommendation. Rather, it is a controlled study of how prompting, retrieval, fine-tuning, and their combination behave for a structured dashboard-design recommendation task built from source-grounded data and evaluated with explicit provenance and separation between training and final evaluation.

## 3.9 Chapter Summary

This chapter reviewed the main research areas that support the thesis. Visualization recommendation systems show how design knowledge can be formalized and operationalized. Natural-language interfaces demonstrate how user intent can be translated into analytic tasks and chart specifications. Datasets such as Quda, nvBench, and nvBench 2.0 provide important supervision but do not directly cover complete dashboard-level recommendations. Recent LLM systems such as ChartGPT show that language models and fine-tuning can improve chart generation, while RAG provides a mechanism for grounding recommendations in external design knowledge.

The review also highlights the need for careful evaluation. A system may produce valid JSON while remaining semantically wrong, or select a defensible chart while providing weak layout or interaction guidance. This motivates the multi-dimensional evaluation framework used in the following chapters.

---

## References Used in Chapter 3

Dibia, V., & Demiralp, Ç. (2019). Data2Vis: Automatic generation of data visualizations using sequence-to-sequence recurrent neural networks. *IEEE Computer Graphics and Applications, 39*(5), 33–46. https://doi.org/10.1109/MCG.2019.2924636

Fu, S., Xiong, K., Ge, X., Tang, S., Chen, W., & Wu, Y. (2020). Quda: Natural language queries for visual data analytics. arXiv:2005.03257. https://arxiv.org/abs/2005.03257

Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N., West, R., Horvitz, E., & Nori, H. (2025). Generating structured outputs from language models: Benchmark and studies. arXiv:2501.10868. https://arxiv.org/abs/2501.10868

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.

Luo, T., Huang, C., Shen, L., Li, B., Shen, S., Zeng, W., Tang, N., & Luo, Y. (2025). nvBench 2.0: A benchmark for natural language to visualization under ambiguity. arXiv:2503.12880. https://arxiv.org/abs/2503.12880

Luo, Y., Tang, N., Li, G., Tang, J., Chai, C., & Qin, X. (2022). Natural language to visualization by neural machine translation. *IEEE Transactions on Visualization and Computer Graphics, 28*(1), 217–226. https://doi.org/10.1109/TVCG.2021.3114848

Luo, Y., Tang, J., & Li, G. (2021). nvBench: A large-scale synthesized dataset for cross-domain natural language to visualization task. arXiv:2112.12926. https://arxiv.org/abs/2112.12926

Mackinlay, J. (1986). Automating the design of graphical presentations of relational information. *ACM Transactions on Graphics, 5*(2), 110–141. https://doi.org/10.1145/22949.22950

Mitra, R., Narechania, A., Endert, A., & Stasko, J. (2022). Facilitating conversational interaction in natural language interfaces for visualization. *IEEE VIS 2022*. https://doi.org/10.1109/VIS54862.2022.00010

Moritz, D., Wang, C., Nelson, G. L., Lin, H., Smith, A. M., Howe, B., & Heer, J. (2019). Formalizing visualization design knowledge as constraints: Actionable and extensible models in Draco. *IEEE Transactions on Visualization and Computer Graphics, 25*(1), 438–448. https://doi.org/10.1109/TVCG.2018.2865240

Narechania, A., Srinivasan, A., & Stasko, J. (2021). NL4DV: A toolkit for generating analytic specifications for data visualization from natural language queries. *IEEE Transactions on Visualization and Computer Graphics*. https://doi.org/10.1109/TVCG.2020.3030378

Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond accuracy: Behavioral testing of NLP models with CheckList. *ACL 2020*, 4902–4912. https://doi.org/10.18653/v1/2020.acl-main.442

Sah, S., Mitra, R., Narechania, A., Endert, A., Stasko, J., & Dou, W. (2024). Generating analytic specifications for data visualization from natural language queries using large language models. arXiv:2408.13391. https://arxiv.org/abs/2408.13391

Shin, S., Hong, S., & Elmqvist, N. (2025). Visualizationary: Automating design feedback for visualization designers using LLMs. *IEEE Transactions on Visualization and Computer Graphics*. https://doi.org/10.1109/TVCG.2025.3579700

Snyder, L. S., Wang, C., & Drucker, S. (2025). Challenges & opportunities with LLM-assisted visualization retargeting. *IEEE VIS 2025*. https://doi.org/10.1109/VIS60296.2025.00034

Tian, Y., Cui, W., Deng, D., Yi, X., Yang, Y., Zhang, H., & Wu, Y. (2025). ChartGPT: Leveraging LLMs to generate charts from abstract natural language. *IEEE Transactions on Visualization and Computer Graphics, 31*(3), 1731–1745. https://doi.org/10.1109/TVCG.2024.3368621

van der Lee, C., Gatt, A., van Miltenburg, E., Wubben, S., & Krahmer, E. (2019). Best practices for the human evaluation of automatically generated text. *INLG 2019*. https://aclanthology.org/W19-8643/

Wongsuphasawat, K., Qu, Z., Moritz, D., Chang, R., Ouk, F., Anand, A., Mackinlay, J., Howe, B., & Heer, J. (2017). Voyager 2: Augmenting visual analysis with partial view specifications. *CHI 2017*. https://doi.org/10.1145/3025453.3025768

Zadeh, F. P., Kim, J., Kim, J.-H., & Kim, G. (2024). Text2Chart31: Instruction tuning for chart generation with automatic feedback. arXiv:2410.04064. https://arxiv.org/abs/2410.04064
