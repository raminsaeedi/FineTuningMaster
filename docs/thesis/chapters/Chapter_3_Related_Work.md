# Chapter 3 — Related Work

This chapter reviews the research that is most closely related to the dashboard-design recommendation task studied in this thesis. It first discusses visualization recommendation systems that encode design knowledge as rules, constraints, or interactive recommendations. It then covers natural-language interfaces and neural approaches for translating language into visualization specifications. The next section examines the datasets and benchmarks that support this research area. After that, the chapter reviews recent uses of large language models (LLMs) for analytic specifications, chart generation, and visualization assistance. The final technical sections discuss parameter-efficient fine-tuning (PEFT), retrieval-augmented generation (RAG), and the evaluation of structured visualization outputs. The chapter ends by identifying the remaining gap and positioning the present study within this literature.

The review makes a distinction between a visualization recommendation and a complete dashboard-design recommendation. A visualization recommendation usually selects one view or one visualization specification. The task in this thesis is broader: a dashboard brief is mapped to several coordinated design decisions, including KPI-to-chart mappings, layout, styling, interactions, and rationales. This distinction is important because a method can perform well on chart selection without solving the other parts of dashboard design. The discussion below therefore compares prior systems according to their input, output, design knowledge, and evaluation scope rather than treating all of them as equivalent.

## 3.1 Visualization Recommendation Systems

Visualization recommendation systems are an established research area. Their central goal is to reduce the manual effort required to choose data transformations and visual encodings. The main approaches differ in how they represent design knowledge and how much control remains with the analyst. Some systems formulate recommendation as a search problem over possible graphical presentations. Others use explicit constraints to filter or rank views. A third group integrates recommendations into an interactive interface so that users can explore and refine the results.

### 3.1.1 Explicit design knowledge and constraint-based recommendation

Mackinlay’s APT system is an early and influential example of automated visualization design. It treats the construction of a graphical presentation as a systematic search over possible encodings and evaluates candidates using the ideas of expressiveness and effectiveness (Mackinlay, 1986). Expressiveness concerns whether a presentation represents the intended data without adding unsupported information. Effectiveness concerns how well the chosen graphical properties support the required perceptual task. This formulation is relevant to the present thesis because it shows that visualization choice can be described as a structured design problem rather than as an arbitrary stylistic decision.

The same line of work was later integrated into a practical visual-analysis environment. Show Me added automatic presentation commands and defaults to Tableau, including automatic mark-type selection and the generation of views for one or more selected fields (Mackinlay, Hanrahan, & Stolte, 2007). The system did not remove the analyst from the process. Instead, it provided a fast starting point while allowing the analyst to continue the analysis and change the result. This human-in-the-loop character is important: a recommender can support design without claiming that one automatically generated view is the only valid answer.

Draco provides a more explicit formal representation of visualization design knowledge. Moritz et al. (2019) express design principles as constraints over visualization specifications. Hard constraints can exclude invalid or unsuitable specifications, while soft constraints can be used to rank alternatives. The framework also supports the use of empirical evidence when assigning weights to design preferences. Draco therefore connects a visualization grammar with a knowledge model that can be inspected and extended. This is close to the motivation for a structured dashboard recommendation: design knowledge should be available as a reason for a recommendation, not only as an unexplained model output.

These systems also illustrate a limitation of rule-based recommendation. The quality of a result depends on the coverage and correctness of the encoded design knowledge. A rule can express that a temporal variable is often useful for a trend view, but it cannot by itself understand every way in which a user may describe a business goal, nor can it reliably infer missing context. A rule system must also decide how to behave when several views satisfy the same constraints. These issues motivate the later development of natural-language interfaces and learned generation methods.

### 3.1.2 Mixed-initiative exploration and visualization galleries

Voyager addresses the problem from the perspective of exploratory analysis. Instead of requiring the analyst to specify every view manually, Voyager presents a gallery of automatically generated visualizations that can be browsed and refined (Wongsuphasawat et al., 2016). The system combines visualization recommendation with faceted browsing and uses statistical and perceptual measures to organize the recommended views. Its user study showed that this interaction model can support the exploration of unfamiliar data and increase the number of variables considered by users.

Voyager 2 extends the same general idea to partial view specifications (Wongsuphasawat et al., 2017). An analyst can specify some parts of a view and leave other parts open. The system then recommends compatible completions. This makes the interaction more flexible than a fully manual specification process and more controlled than a system that generates a complete view without user guidance. The work demonstrates that recommendation can be useful at different stages of analysis: it can support initial exploration, but it can also complete a partially formed design.

The mixed-initiative perspective provides two lessons for the thesis. First, recommendation should be understood as support for a user task, not only as an optimization over chart types. Second, several candidate views can be useful when the user's goal is exploratory or when the input does not determine one unique encoding. These lessons are consistent with later benchmark work on ambiguity and with the project decision to record chart and rationale information separately. They also show why a dashboard recommendation should be evaluated in relation to users, goals, and constraints rather than by chart-type accuracy alone.

### 3.1.3 Relation to the present dashboard task

The systems described above mainly operate on structured data tables, view specifications, or a limited set of user selections. Their outputs are usually individual views, view galleries, or completions of a visualization specification. They do not normally generate one coordinated object that combines KPI mappings, multiple views, dashboard layout, styling, interaction behaviour, and textual rationales.

This difference does not make the earlier systems less relevant. They provide the design concepts that a dashboard recommendation must respect. Their constraints can describe valid encodings, their perceptual criteria can inform chart choice, and their mixed-initiative interfaces show how recommendations can remain understandable to users. The contribution of the present thesis is positioned at a different level: it studies whether an LLM can produce a structured, multi-component recommendation from a dashboard brief while preserving the design and evaluation distinctions established by this earlier work.

## 3.2 Natural-Language Interfaces and NL-to-Visualization

Natural-language interfaces for visualization reduce the need for users to know the syntax of a visualization grammar or the details of a data-processing language. A user can state an analytical intention in ordinary language, and the system translates the request into data attributes, analytic tasks, transformations, and visual encodings. This translation is difficult because natural-language requests are often incomplete, ambiguous, or dependent on the structure of the data.

### 3.2.1 Semantic parsing and conversational interfaces

NL4DV is a prominent toolkit for natural-language visualization interfaces. Given a tabular dataset and a natural-language query, it identifies data attributes and analytic tasks and returns a structured analytic specification with relevant visualizations (Narechania, Srinivasan, & Stasko, 2021). The output makes intermediate decisions visible. A developer or user can inspect which phrases were linked to which fields, which task was inferred, and which visualization was recommended. This explicit intermediate representation improves debuggability compared with a system that only returns a final chart.

The toolkit also shows the value of separating language understanding from visualization rendering. A query may mention a measure, a grouping field, a filter, and an operation such as a comparison or trend. These elements can be represented before a chart is selected. The separation makes it possible to evaluate task recognition and chart recommendation as related but distinct steps. It also provides a useful conceptual basis for the structured output used in this thesis, where chart mappings and rationales are stored alongside the broader dashboard fields.

Mitra et al. (2022) extend NL4DV toward conversational interaction. Their system maintains multiple conversations, incorporates follow-up information into the analytic specification, and supports ambiguity resolution over several turns. This work is important because a single utterance does not always contain enough information for a reliable visualization. A follow-up question can specify a missing field, clarify a comparison, or change the intended view without requiring the user to restate the entire request.

Parser-based systems have a clear advantage in transparency. Their intermediate structures can be inspected and their errors can often be connected to a particular parsing rule or mapping. Their limitation is that extending the rule set to cover new language patterns, complex transformations, and domain-specific terminology requires substantial manual work. This limitation is one reason why later systems use neural generation and, more recently, instruction-tuned LLMs.

### 3.2.2 Neural generation of visualization specifications

Data2Vis treats visualization generation as a sequence-to-sequence problem. Dibia and Demiralp (2019) use a recurrent neural architecture to generate Vega-Lite specifications from data descriptions. The work shows that a model can learn recurring relationships between data characteristics, transformations, and visualization syntax. It also demonstrates the attraction of generating a formal specification rather than an image: the result can be rendered, inspected, and evaluated programmatically.

Luo et al. (2022) study natural language to visualization through neural machine translation. Their approach treats the translation from a natural-language request to a visualization specification as a learned mapping. This direction reduces the need to write a separate rule for every linguistic variation. At the same time, it makes the training data more important because the model can learn regularities, errors, and biases from the examples provided to it.

NL2Viz introduces constrained syntax-guided synthesis for natural-language visualization (Wu et al., 2022). The system combines the language input with data and program context and uses synthesis constraints to produce visualization programs that satisfy structural requirements. This is relevant to the present work because it treats generation as a combination of language interpretation and constrained program construction. It also shows that constraints remain useful even when a learned model is responsible for a large part of the translation process.

### 3.2.3 Common limitations of NL-to-visualization systems

The systems in this section generally focus on one query and one visualization, or on a conversation that incrementally refines one analytic specification. This scope is valuable, but it does not cover the complete dashboard problem. A dashboard recommendation must coordinate several KPIs and views and must also make decisions about spatial arrangement, visual hierarchy, colour use, interaction behaviour, and explanation. These decisions can depend on the audience and on the purpose of the dashboard, even when the underlying chart specifications are valid.

Natural-language input also creates a second problem: under-specification. A phrase such as “show performance” does not determine a measure, time grain, comparison group, or suitable level of aggregation. A system may infer a plausible choice, ask for clarification, or return several alternatives. The correct behaviour depends on the application and on the risk of making an unsupported assumption. This issue becomes central in the benchmark literature discussed next and motivates the use of explicit constraints and validation in the present thesis.

## 3.3 Datasets and Benchmarks for Visualization Generation

Datasets determine which parts of the visualization task a model can learn and which claims an evaluation can support. A dataset may contain natural-language queries, analytic-task labels, table schemas, visualization specifications, code, rendered images, or human judgements. These components are not interchangeable. A dataset that is suitable for chart-type prediction may not contain the information needed to train a model to recommend layout or interactions. The following benchmarks are therefore discussed according to their task and output coverage.

### 3.3.1 Quda and analytic-task recognition

Quda was introduced by Fu et al. (2020) for recognizing analytic tasks from free-form natural-language queries. It contains 14,035 queries, and each query is annotated with one or more analytic tasks. The authors first collected seed queries with data analysts and then used crowdsourced paraphrase generation and validation to increase linguistic diversity. The resulting data is useful for studying how different phrasings express tasks such as comparison, trend, or distribution.

Quda addresses an important subproblem, but it does not describe a complete visualization or dashboard design. It provides evidence about the language of analytical intent, not about the final selection of fields, encodings, layout, styling, or interaction. It can therefore support task-recognition research or data augmentation, but it cannot by itself serve as gold data for the full structured output required by this thesis.

### 3.3.2 nvBench and cross-domain NL-to-visualization

The nvBench benchmark was constructed to provide larger-scale supervision for natural-language to visualization research. Luo et al. (2021) report 25,750 natural-language/visualization pairs derived from 750 tables across 105 domains. The benchmark is synthesized from natural-language-to-SQL resources, which provides broad coverage of domains and table structures. Its construction was also evaluated by experts and crowd workers, which is important because a synthetic benchmark still needs quality checks.

nvBench is useful because it links language with data context and a visualization target. It can support the study of field selection, aggregation, and basic chart specifications across domains. However, its synthetic construction also defines its scope. The benchmark is designed for a query-to-visualization task, not for the design of a complete dashboard for a named audience. It does not provide the full set of contextual and presentational decisions required by the target object in this thesis.

### 3.3.3 nvBench 2.0 and ambiguity

nvBench 2.0 addresses a limitation of single-target benchmarks: one query can have more than one reasonable interpretation. The published version contains 7,878 natural-language queries and 24,076 corresponding visualizations derived from 780 tables across 153 domains (Luo et al., 2025). Its controlled ambiguity-injection pipeline starts from seed visualizations and creates alternative interpretations together with step-wise reasoning paths.

This one-to-many structure is important for visualization evaluation. Ambiguity can occur at the data level, for example when a word may refer to more than one column or when an aggregation is not stated. It can also occur at the visualization level, for example when “trend” could be represented by more than one chart type depending on the data and the purpose. A benchmark that stores only one target may incorrectly mark a defensible alternative as wrong. nvBench 2.0 does not solve the complete dashboard problem, but it provides a stronger basis for discussing uncertainty and multiple valid outputs.

### 3.3.4 What existing benchmarks cover

Together, Quda, nvBench, and nvBench 2.0 cover several important components: analytic-task recognition, natural-language variation, data and schema context, chart specifications, cross-domain supervision, and ambiguity. They do not cover the full mapping from a dashboard brief containing users, goals, KPIs, data fields, and constraints to a coordinated, multi-component dashboard-design recommendation.

This conclusion is a scope statement about the sources reviewed for this thesis, not a claim that no other visualization dataset exists. Other resources may cover chart code, chart images, visual question answering, or conversational changes to a visualization. The relevant point is that those resources do not automatically provide labels for every field in the target dashboard schema. The project dataset and its provenance are therefore described separately in Chapter 4 rather than being treated as a direct copy of one public benchmark.

## 3.4 LLMs for Visualization and Dashboard Assistance

Instruction-tuned LLMs have changed the design space for natural-language visualization systems. Instead of building a separate parser for every intent and language pattern, a general model can be prompted to interpret a brief and generate an analytic or visualization specification. This flexibility is useful for complex inputs, but it also makes it easier for unsupported assumptions and invalid structures to pass unnoticed. Recent work therefore combines LLMs with explicit prompts, intermediate representations, constraints, fine-tuning, or visual feedback.

### 3.4.1 LLMs for analytic specifications

Sah et al. (2024) study the generation of analytic specifications from natural-language queries with LLMs. Their prompt produces detected data attributes, inferred analytic tasks, and recommended visualizations. It also records mappings between input phrases and detected entities and states the design principles used for recommendations. The authors include conversational interaction and ambiguity detection in the prompt design and evaluate the approach with GPT-4.

The work is relevant to this thesis because it does not treat the LLM as an opaque chart generator. Instead, it asks the model to expose intermediate decisions that can be inspected and debugged. This approach is compatible with structured outputs and rationales. It also shows a practical limitation: a detailed prompt can improve the form of the result, but the model still depends on the data context and may fail when the request is ambiguous or when a requested field is not present.

### 3.4.2 Chart generation with LLMs

ChartGPT focuses on generating charts from abstract natural-language input (Tian et al., 2025). The authors decompose chart generation into several steps so that each step addresses a more specific decision. They also create task-specific training data and fine-tune a model to provide visualization knowledge that is not guaranteed to be present in a general language model. An interactive interface allows users to inspect and modify intermediate results.

ChartGPT demonstrates why decomposition and intermediate representations are useful when natural-language requests do not specify every visual parameter. It is closely related to the chart-selection part of this thesis, but its main target remains chart generation. The dashboard problem adds coordinated multi-view design and requires the system to express decisions about layout, styling, interactions, and rationales in the same structured response.

Text2Chart31 extends chart-generation research to a wider range of plot types and richer training records. Pesaran Zadeh et al. (2024) introduce a dataset with 31 Matplotlib plot types and approximately 11.1 thousand tuples that combine descriptions, code, data tables, and plots. They also propose instruction tuning with automatic feedback, including reinforcement-learning components for chart-generation tasks. This work shows that domain-specific adaptation can be used not only to learn a chart label, but also to connect language, data, code, and the resulting plot.

Text2Chart31 is nevertheless different from the present target. Its central output is a chart or chart-generating program, whereas this thesis studies a structured dashboard-design recommendation. The distinction matters for both training and evaluation. A model can generate executable chart code and still provide an unsuitable dashboard hierarchy, inaccessible styling, or an interaction that is not supported by the data.

### 3.4.3 LLM-assisted visualization design

Visualizationary studies LLMs as design-feedback assistants rather than only as generators. Shin, Hong, and Elmqvist (2025) combine a preamble of visualization guidelines with perceptual filters that extract salient measures from a visualization image. Their longitudinal study with 13 visualization designers suggests that natural-language feedback can help designers refine visualizations. The work is relevant because it connects generated advice with measurable properties of a visual result and with an iterative design process.

LLM-assisted visualization retargeting addresses a related but different problem: adapting existing visualization implementations to new data. Snyder, Wang, and Drucker (2025) compare direct code generation with a more constrained program-synthesis approach that provides structural information such as visual encodings. Their analysis reports that both approaches can struggle when the new data has not been transformed appropriately. This finding is a useful warning for the present thesis. Correct language generation cannot compensate for a mismatch between the requested design and the data representation on which that design depends.

### 3.4.4 General lessons from the LLM literature

The recent literature supports the use of LLMs as flexible interfaces for visualization work, but it does not support treating fluent output as reliable design evidence. Successful systems add structure in different ways: they expose analytic specifications, decompose chart generation, use task-specific data, apply constraints, or provide feedback based on the rendered result. These approaches address different failure modes and are not substitutes for one another.

For the present thesis, this leads to a layered system design. The LLM is used as a generation component, while the prompt specifies the output contract, retrieval supplies optional external guidance, fine-tuning provides task adaptation, and validation checks the generated structure. The evaluation must then distinguish syntactic validity from chart and dashboard quality. This separation is the basis for the comparison developed in the following sections.

## 3.5 Parameter-Efficient Fine-Tuning for Structured Generation

The background chapter introduced PEFT and its main variants. In related work, the important question is how these adaptation methods change the comparison between a general model and a task-specific model. Fine-tuning can help a model learn the vocabulary, field relations, and output patterns of a visualization task. It can also make the model reproduce regularities of a particular dataset without learning a general design principle. The choice of adaptation method therefore affects both resource use and scientific interpretation.

### 3.5.1 LoRA, QLoRA, and alternative adaptation methods

LoRA learns low-rank updates to selected weight matrices while keeping the pretrained model weights frozen (Hu et al., 2022). The adapter can be stored separately from the base model, which makes it possible to reuse one base model across several task-specific conditions. This property is useful for controlled experiments because the intervention can be described as a learned update rather than as a completely different model.

QLoRA combines the LoRA update with low-bit storage of the frozen base model (Dettmers et al., 2023). The method uses 4-bit quantization, NormalFloat4 (NF4), double quantization, and paged optimizers to reduce memory use during fine-tuning. The base model remains frozen while the LoRA parameters are trained. QLoRA therefore changes the memory representation of the base model without changing the basic idea of a separate low-rank task update. The reported results in the original paper motivate its use as a resource-aware method, but they do not guarantee the same outcome on a different model, dataset, or hardware setup.

Other adaptation methods make different trade-offs. Adapter tuning inserts small trainable bottleneck modules into the network (Houlsby et al., 2019). Prefix tuning and prompt tuning learn continuous task-specific vectors that condition a frozen model through virtual tokens or soft prompt embeddings (Li & Liang, 2021; Lester, Al-Rfou, & Constant, 2021). BitFit updates only bias parameters, while IA3 learns vectors that rescale internal activations (Ben Zaken, Goldberg, & Ravfogel, 2022; Liu et al., 2022). These methods can require fewer trainable parameters, but their update structure is different from a LoRA adapter.

Several methods modify the allocation or scaling of low-rank updates. AdaLoRA allocates different ranks to different matrices according to their estimated importance (Zhang et al., 2023). DoRA decomposes the weight update into magnitude and direction and applies low-rank adaptation to the directional component (Liu et al., 2024). Rank-Stabilized LoRA changes the scaling rule for the low-rank update and is intended to improve behaviour at higher ranks (Kalajdzievski, 2023). These methods address specific limitations of standard LoRA, but they also introduce additional choices that must be controlled in an ablation.

GaLore is related to memory-efficient training but is not PEFT in the strict sense used here. It projects gradients into a low-rank subspace while still allowing full-parameter training (Zhao et al., 2024). A GaLore comparison therefore changes not only the adapter algorithm but also the set of parameters that can be updated. Treating GaLore as an interchangeable name for QLoRA would make the experimental interpretation incorrect.

### 3.5.2 Fine-tuning in visualization-generation research

ChartGPT and Text2Chart31 provide visualization-specific evidence that task adaptation can be useful for chart generation. In both cases, the training data contains information about visualization tasks that is not represented by a generic language-instruction corpus alone. The results support the idea that a model may need examples of data fields, chart parameters, chart code, or intermediate decisions before it can reliably generate domain-specific visualization outputs.

The present thesis extends this idea from one chart to a structured dashboard recommendation. The required output contains several types of information, and the model must learn their names, relations, and allowed structure. Fine-tuning is therefore not evaluated only by whether a model produces fluent text. It is evaluated through parsing, schema validation, field completeness, chart and encoding behaviour, robustness, grounding, and human usefulness. This broader evaluation is necessary because a model can learn the output format without learning appropriate dashboard design.

### 3.5.3 Scientific reasons for selecting QLoRA in this thesis

QLoRA is selected as the main fine-tuning method for this thesis for four connected reasons. The first reason is resource feasibility. The project compares several controlled conditions and may use more than one model size. Storing the frozen base model in a 4-bit representation reduces the memory required during training, while the adapter keeps the task-specific parameters small. This makes the planned experiments more realistic under a limited GPU budget than full-parameter fine-tuning with the same base models. This is a feasibility argument, not a claim that QLoRA is the most accurate adaptation method for every task.

The second reason is experimental control. A QLoRA run leaves the base model identifiable and stores the learned task intervention as an adapter. In the project design, the prompt-only condition does not use an adapter, the RAG condition adds retrieved context without changing the model parameters, the QLoRA condition adds the adapter, and the combined condition uses the compatible adapter together with the same retrieval procedure. This factorization makes the intended comparison clearer than a design in which the retrieval mechanism and the training algorithm are changed at the same time.

The third reason is the structure of the research task. The goal is to adapt an existing instruction-tuned causal language model to produce a project-specific structured object. The task is not to train a new language model or to learn a new optimizer. Low-rank adaptation is a suitable intervention because it can change the model's behaviour for the target output while preserving the reusable base model. The adapter can also be checked for compatibility with the model, dataset, training configuration, and seed before it is used by the combined method.

The fourth reason is comparability with the alternatives that were considered. DoRA changes the parameter decomposition, AdaLoRA changes the rank allocation, RSLoRA changes the scaling behaviour, and GaLore changes the memory strategy while allowing full-parameter learning. Each of these methods could support a useful ablation, but choosing one as the main method would answer a different question or introduce a different control problem. The project therefore keeps QLoRA as the main fine-tuning intervention and treats the other algorithms as optional ablation paths rather than presenting them as universally worse or better.

This choice has clear limitations. QLoRA can be sensitive to the quantization configuration, adapter rank, target modules, learning rate, sequence length, and supervision quality. A small adapter can learn recurring formatting patterns without learning robust visualization principles. Quantization also introduces an approximation that may affect the result differently across model families. These limitations are addressed by recording configuration and adapter provenance and by evaluating the output at several levels instead of relying on training loss alone.

The seed design is part of the same control principle. In this thesis, three seeds are reserved for the smallest final model, Qwen3-1.7B, to check whether the main result is stable under different random conditions. These runs are a stability check; three seeds do not by themselves prove generalization. If the predefined stability criterion is met, the remaining planned model and method configurations will be run with Seed 42. If it is not met, the result must be reported as unstable and the plan must be reconsidered rather than silently treating Seed 42 as representative. The thesis will not describe the remaining single-seed runs as multi-seed evidence.

## 3.6 Retrieval-Augmented Generation for Domain-Grounded Generation

RAG provides a second way to add task-relevant information to a language model. Fine-tuning changes model parameters and stores knowledge in a learned task component. RAG leaves the model parameters unchanged at inference time and retrieves passages from an external knowledge base for each input. The two interventions can therefore be compared separately and can also be combined.

### 3.6.1 Retrieval and generation

Lewis et al. (2020) define RAG as a combination of a parametric generator and a non-parametric memory. A document collection is indexed, a retriever selects passages for a new input, and the generator produces an answer conditioned on the input and the retrieved context. The approach is useful when relevant information is too specific, too recent, or too large to depend only on the model parameters.

For visualization design, an external knowledge base can contain design guidelines, perceptual findings, accessibility recommendations, and domain-specific documentation. Retrieved passages can provide a source for a rationale or can remind the model of a design constraint. This is especially useful when a recommendation must be explained to a user. The retrieved text is not itself a guarantee of a correct design; it is an additional condition under which the model generates its output.

### 3.6.2 RAG and fine-tuning as different interventions

RAG and fine-tuning solve different parts of the adaptation problem. Fine-tuning can teach stable output conventions, field names, and recurring relations across the training examples. RAG can provide explicit information that can be changed without retraining the model. Fine-tuning is therefore a parameter intervention, while RAG is a context intervention. Combining them may be useful, but it also makes it necessary to verify which component is responsible for an observed change.

This distinction motivates the four conditions used in the project comparison. Method A is prompt-only generation and serves as the base condition. Method B adds retrieval while keeping the base model unchanged. Method C adds QLoRA fine-tuning without retrieval. Method D combines the QLoRA adapter with retrieval. The design is meaningful only if the prompt, tokenizer, model, generation settings, retrieval procedure, dataset, and adapter provenance are controlled. A difference between methods cannot be attributed to retrieval if the context length or post-processing rules also change.

### 3.6.3 RAG limitations in the present task

RAG introduces its own sources of error. A retriever may miss a relevant passage, return text that is only loosely related to the brief, or return several passages that do not agree. Chunk boundaries and query wording affect which evidence is available. The context may also exceed the model's input budget, or the generator may ignore a relevant passage even when it is present. These are system-level issues rather than simple model-quality issues.

The project therefore records retrieval results and knowledge-base identity separately from the generated recommendation. Grounding is evaluated as a separate dimension and must be interpreted together with its measurement mode. A lexical word-overlap score can indicate that terms from a retrieved passage appear in a rationale, but it is not a complete faithfulness assessment. The presence of retrieved text also does not prove that the chart, layout, or interaction recommendation is appropriate for the user's task.

## 3.7 Evaluation of Structured and Visualization Outputs

Evaluation is difficult because a dashboard recommendation can be correct in one dimension and weak in another. A response may be valid JSON but select the wrong field. It may select a defensible chart but propose an unusable layout. It may provide a plausible rationale that is not supported by the input or by the retrieved evidence. A meaningful evaluation must therefore separate structural validity, semantic correctness, robustness, grounding, and human usefulness.

### 3.7.1 Structural validity

The first level of evaluation asks whether the output can be used by software. The evaluation should distinguish successful extraction of a JSON object from validity against the required JSON Schema. It should also distinguish schema validity from completeness: all required keys may be present even when some values are empty or unusable. These checks answer different questions and should not be merged into one score.

JSON Schema Draft 2020-12 defines a formal language for describing JSON instances, including properties, types, required fields, arrays, and allowed values (JSON Schema, 2022). Structured-output research such as JSONSchemaBench evaluates more than parse success and separates schema coverage, generation efficiency, and output quality (Geng et al., 2025). This distinction is directly relevant to the thesis. A model can satisfy the formal structure while producing an unsupported KPI, an invalid chart choice, or an empty rationale.

### 3.7.2 Semantic and visualization correctness

The second level concerns the meaning of the recommendation. Relevant checks include whether the model identifies the intended analytical task, uses available fields, preserves constraints, chooses an appropriate chart type, and assigns fields to the expected visual channels. Aggregation and temporal granularity may also be important. These checks should be reported separately when a single exact-match score would hide the source of an error.

Task-based visualization studies show that the effectiveness of an encoding depends on the analytical task and on the distribution of the data, not only on the name of the chart (Saket, Endert, & Demiralp, 2019; Kim & Heer, 2018). This supports an evaluation design that considers task and data context. It also explains why an automatically generated chart should not be judged only by whether its label matches one synthetic target.

Exact-match evaluation has a further limitation when more than one visualization is defensible. nvBench 2.0 makes this problem explicit by associating ambiguous queries with multiple valid visualizations (Luo et al., 2025). The present thesis therefore treats independent human-effectiveness evidence and human evaluation as important complements to internal synthetic-gold metrics. A synthetic target can be useful for checking pipeline behaviour, but it should not automatically be interpreted as the only correct design.

### 3.7.3 Robustness and grounding

A useful system should not change its main recommendation because a brief was harmlessly paraphrased. It should also react appropriately when essential information is missing. Behavioural testing with CheckList argues for systematic tests of capabilities and failure modes instead of relying only on one aggregate accuracy value (Ribeiro et al., 2020). For the dashboard task, this means testing paraphrases, incomplete briefs, unsupported fields, and other controlled changes.

Robustness is not the same as correctness. A model can consistently repeat the same wrong chart under several paraphrases. Similarly, a model can mention words that overlap with a retrieved source without using that source correctly. The project therefore keeps paraphrase consistency, missing-information behaviour, and grounding support separate from chart and dashboard quality. This separation makes it possible to identify whether an intervention improves reliability, content, or only the surface form of the answer.

### 3.7.4 Human evaluation and LLM-based judges

Human evaluation is needed for dimensions that are difficult to reduce to a formal exact match. These dimensions include the appropriateness of the chart for the stated goal, the quality of the dashboard layout, styling and accessibility, interaction design, rationale quality, and overall usefulness. Best-practice guidance for evaluating generated language recommends a clear rubric, controlled instructions, independent ratings, and agreement analysis (van der Lee et al., 2019). The same principles apply when people evaluate structured dashboard recommendations.

An LLM can also be used as an auxiliary judge, but its score should not automatically replace human judgement. A model-based judge may be sensitive to prompt wording, may prefer fluent explanations, and may not detect an error in a chart-field mapping. In this thesis, an LLM judge is therefore treated as supportive evidence only. Claims about usefulness and actionability depend on the human-evaluation protocol and its reliability, not on an unvalidated automatic judge score.

### 3.7.5 Evaluation layers for this thesis

The project evaluation protocol assembles evidence across four layers. The first layer evaluates chart choice against an independent human-effectiveness reference where coverage exists. The second layer evaluates parsing, schema validity, completeness, robustness, and grounding on held-out and perturbed inputs. The third layer, if retained in the final claim scope, compares structural properties of generated dashboards with a real-dashboard corpus and is interpreted descriptively rather than as evidence of optimality. The fourth layer uses blind human ratings for quality and usefulness.

This layered design also protects against circular conclusions. The project's generated chart labels and internal synthetic test targets can be useful for diagnosing the pipeline, but they share part of the same task construction and are not sufficient as the primary evidence for chart quality. Independent references, real briefs, and human ratings are kept separate from the training and augmentation lineage. The detailed dataset and evaluation procedures are described in the following chapters.

## 3.8 Research Gap and Positioning of This Thesis

### 3.8.1 Gap in task scope

Prior work provides strong solutions for individual parts of visualization design. Constraint-based systems formalize visual encoding knowledge. Natural-language interfaces identify analytic tasks and relevant fields. Neural and LLM-based systems generate chart specifications or chart code. RAG can provide external guidance, and PEFT can adapt a model to a task. The remaining problem is how to combine these capabilities for a dashboard brief that contains user context, goals, KPIs, data fields, and constraints and that requires a coordinated multi-component output.

The target of this thesis is not a claim that LLMs replace visualization recommendation research. It is a narrower study of whether an LLM-based pipeline can produce a structured dashboard-design recommendation that is useful beyond a single chart label. The output includes KPI-to-chart mappings together with layout, styling, interactions, and rationales. This makes the task broader than the public query-to-visualization benchmarks reviewed above and creates a need for separate checks of each output component.

### 3.8.2 Gap in controlled intervention comparisons

The literature often studies one system, one adaptation method, or one generation pipeline. Such studies are valuable for introducing a method, but they do not always isolate the contribution of retrieval from the contribution of task-specific fine-tuning. The project addresses this issue with four controlled conditions: prompt-only generation, retrieval-augmented generation, QLoRA fine-tuning, and QLoRA fine-tuning combined with retrieval.

The comparison is designed as a factorization of two interventions. Retrieval changes the context given to the generator, whereas QLoRA changes the task-specific parameter state. This does not remove every possible confounder, and the model family and seed coverage must still be reported. It does, however, provide a clearer basis for asking whether retrieval remains useful after task adaptation and whether fine-tuning improves the structured output without retrieval.

### 3.8.3 Gap in evaluation and provenance

A second gap concerns evaluation breadth and provenance. A valid JSON response is not necessarily a good dashboard recommendation, and a high internal chart score is not necessarily independent evidence. The present study therefore records dataset versions, source lineage, knowledge-base identity, model configuration, seed, adapter provenance, raw outputs, parsing results, and evaluation status. Training and augmentation data are kept separate from the independent evidence used for final quality claims.

The seed policy follows the same principle of bounded interpretation. Three seeds are used only for the Qwen3-1.7B stability check. If the predefined stability criterion is met, the remaining planned configurations use Seed 42. Results from those runs will be reported as single-seed results unless additional seeds are actually executed. This wording prevents a stability check on one model from being presented as variance evidence for every model and method.

### 3.8.4 Position of the thesis

The thesis is positioned at the intersection of visualization recommendation, natural-language visualization, structured LLM generation, PEFT, RAG, and human-centred evaluation. Its contribution is an experimentally controlled study of these components for a dashboard-level output contract. QLoRA is used as the main fine-tuning method because it offers a resource-aware, adapter-based intervention that is compatible with the project hardware and comparison design. The alternative methods remain relevant as possible ablations, while the main scientific conclusions are restricted to the models, data, seeds, and evaluation layers that are actually run and verified.

This positioning leads to the next chapters. Chapter 4 describes the construction and provenance of the data used for training and evaluation. Chapter 5 formalizes the output schema and the four method conditions. The later evaluation chapters then test structure, chart behaviour, robustness, grounding, and human usefulness separately.

---

## References Used in Chapter 3

Ben Zaken, E., Goldberg, Y., & Ravfogel, S. (2022). BitFit: Simple parameter-efficient fine-tuning for Transformer-based masked language-models. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, 1–9. https://doi.org/10.18653/v1/2022.acl-short.1

Dibia, V., & Demiralp, Ç. (2019). Data2Vis: Automatic generation of data visualizations using sequence-to-sequence recurrent neural networks. *IEEE Computer Graphics and Applications, 39*(5), 33–46. https://doi.org/10.1109/MCG.2019.2924636

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems, 36*, 10088–10115. https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html

Fu, S., Xiong, K., Ge, X., Tang, S., Chen, W., & Wu, Y. (2020). Quda: Natural language queries for visual data analytics. arXiv:2005.03257. https://arxiv.org/abs/2005.03257

Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N., West, R., Horvitz, E., & Nori, H. (2025). JSONSchemaBench: A rigorous benchmark of structured outputs for language models. arXiv:2501.10868. https://doi.org/10.48550/arXiv.2501.10868

JSON Schema. (2022). *JSON Schema Draft 2020-12*. https://json-schema.org/draft/2020-12

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. In *Proceedings of the 36th International Conference on Machine Learning* (Vol. 97, pp. 2790–2799). PMLR. https://proceedings.mlr.press/v97/houlsby19a.html

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*. https://arxiv.org/abs/2106.09685

Kalajdzievski, D. (2023). A rank stabilization scaling factor for fine-tuning with LoRA. arXiv:2312.03732. https://arxiv.org/abs/2312.03732

Kim, Y., & Heer, J. (2018). Assessing effects of task and data distribution on the effectiveness of visual encodings. *Computer Graphics Forum, 37*(3), 157–167. https://doi.org/10.1111/cgf.13409

Lester, B., Al-Rfou, R., & Constant, N. (2021). The power of scale for parameter-efficient prompt tuning. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 3045–3059. https://doi.org/10.18653/v1/2021.emnlp-main.243

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-T., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474. https://arxiv.org/abs/2005.11401

Li, X. L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, 4582–4597. https://doi.org/10.18653/v1/2021.acl-long.353

Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., & Raffel, C. A. (2022). Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. *Advances in Neural Information Processing Systems, 35*, 1950–1965. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html

Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024). DoRA: Weight-decomposed low-rank adaptation. In *Proceedings of the 41st International Conference on Machine Learning* (Vol. 235, pp. 32100–32121). PMLR. https://proceedings.mlr.press/v235/liu24bn.html

Luo, T., Huang, C., Shen, L., Li, B., Shen, S., Zeng, W., Tang, N., & Luo, Y. (2025). nvBench 2.0: Resolving ambiguity in text-to-visualization through stepwise reasoning. *Advances in Neural Information Processing Systems, 38*, 138749–138786. https://doi.org/10.52202/085713-4172

Luo, Y., Tang, N., Li, G., Chai, C., Li, W., & Qin, X. (2021). Synthesizing natural language to visualization (NL2VIS) benchmarks from NL2SQL benchmarks. In *Proceedings of the 2021 International Conference on Management of Data*, 1235–1247. https://doi.org/10.1145/3448016.3457261

Luo, Y., Tang, N., Li, G., Tang, J., Chai, C., & Qin, X. (2022). Natural language to visualization by neural machine translation. *IEEE Transactions on Visualization and Computer Graphics, 28*(1), 217–226. https://doi.org/10.1109/TVCG.2021.3114848

Mackinlay, J. (1986). Automating the design of graphical presentations of relational information. *ACM Transactions on Graphics, 5*(2), 110–141. https://doi.org/10.1145/22949.22950

Mackinlay, J. D., Hanrahan, P., & Stolte, C. (2007). Show Me: Automatic presentation for visual analysis. *IEEE Transactions on Visualization and Computer Graphics, 13*(6), 1137–1144. https://doi.org/10.1109/TVCG.2007.70594

Mitra, R., Narechania, A., Endert, A., & Stasko, J. (2022). Facilitating conversational interaction in natural language interfaces for visualization. In *2022 IEEE Visualization and Visual Analytics (VIS)*, 6–10. https://doi.org/10.1109/VIS54862.2022.00010

Moritz, D., Wang, C., Nelson, G. L., Lin, H., Smith, A. M., Howe, B., & Heer, J. (2019). Formalizing visualization design knowledge as constraints: Actionable and extensible models in Draco. *IEEE Transactions on Visualization and Computer Graphics, 25*(1), 438–448. https://doi.org/10.1109/TVCG.2018.2865240

Narechania, A., Srinivasan, A., & Stasko, J. (2021). NL4DV: A toolkit for generating analytic specifications for data visualization from natural language queries. *IEEE Transactions on Visualization and Computer Graphics, 27*(2), 369–379. https://doi.org/10.1109/TVCG.2020.3030378

Pesaran Zadeh, F., Kim, J., Kim, J.-H., & Kim, G. (2024). Text2Chart31: Instruction tuning for chart generation with automatic feedback. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 11459–11480. Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.emnlp-main.640

Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond accuracy: Behavioral testing of NLP models with CheckList. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 4902–4912. https://doi.org/10.18653/v1/2020.acl-main.442

Sah, S., Mitra, R., Narechania, A., Endert, A., Stasko, J., & Dou, W. (2024). Generating analytic specifications for data visualization from natural language queries using large language models. arXiv:2408.13391. https://arxiv.org/abs/2408.13391

Saket, B., Endert, A., & Demiralp, Ç. (2019). Task-based effectiveness of basic visualizations. *IEEE Transactions on Visualization and Computer Graphics, 25*(7), 2505–2512. https://doi.org/10.1109/TVCG.2018.2829750

Shin, S., Hong, S., & Elmqvist, N. (2025). Visualizationary: Automating design feedback for visualization designers using large language models. *IEEE Transactions on Visualization and Computer Graphics, 31*(10), 8796–8813. https://doi.org/10.1109/TVCG.2025.3579700

Snyder, L. S., Wang, C., & Drucker, S. M. (2025). Challenges & opportunities with LLM-assisted visualization retargeting. In *2025 IEEE Visualization and Visual Analytics (VIS)*, 141–145. https://doi.org/10.1109/VIS60296.2025.00034

Tian, Y., Cui, W., Deng, D., Yi, X., Yang, Y., Zhang, H., & Wu, Y. (2025). ChartGPT: Leveraging LLMs to generate charts from abstract natural language. *IEEE Transactions on Visualization and Computer Graphics, 31*(3), 1731–1745. https://doi.org/10.1109/TVCG.2024.3368621

van der Lee, C., Gatt, A., van Miltenburg, E., Wubben, S., & Krahmer, E. (2019). Best practices for the human evaluation of automatically generated text. In *Proceedings of the 12th International Conference on Natural Language Generation*, 355–368. https://aclanthology.org/W19-8643/

Wongsuphasawat, K., Moritz, D., Anand, A., Mackinlay, J., Howe, B., & Heer, J. (2016). Voyager: Exploratory analysis via faceted browsing of visualization recommendations. *IEEE Transactions on Visualization and Computer Graphics, 22*(1), 649–658. https://doi.org/10.1109/TVCG.2015.2467191

Wongsuphasawat, K., Qu, Z., Moritz, D., Chang, R., Ouk, F., Anand, A., Mackinlay, J., Howe, B., & Heer, J. (2017). Voyager 2: Augmenting visual analysis with partial view specifications. In *Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems*, 2648–2659. https://doi.org/10.1145/3025453.3025768

Wu, Z., Le, V., Tiwari, A., Gulwani, S., Radhakrishna, A., Radiček, I., Soares, G., Wang, X., Li, Z., & Xie, T. (2022). NL2Viz: Natural language to visualization via constrained syntax-guided synthesis. In *Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering*, 972–983. https://doi.org/10.1145/3540250.3549140

Zhang, Q., Chen, M., Bukharin, A., Karampatziakis, N., He, P., Cheng, Y., Chen, W., & Zhao, T. (2023). AdaLoRA: Adaptive budget allocation for parameter-efficient fine-tuning. In *International Conference on Learning Representations*. https://arxiv.org/abs/2303.10512

Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., & Tian, Y. (2024). GaLore: Memory-efficient LLM training by gradient low-rank projection. In *Proceedings of the 41st International Conference on Machine Learning* (Vol. 235, pp. 61121–61143). PMLR. https://proceedings.mlr.press/v235/zhao24s.html
