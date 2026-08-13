# Chapter 2 — Background and Foundations

## 2.1 Dashboard Design

Dashboards are compact visual interfaces that bring together indicators, charts, and controls so that users can monitor a situation, compare performance, detect changes, and support decisions. In this thesis, a dashboard is treated as more than a collection of independent charts. A useful dashboard connects user goals, key performance indicators (KPIs), data attributes, analytical tasks, visual encodings, layout, styling, and interaction choices into one coherent design.

A KPI is a measurable quantity used to represent progress or performance in relation to a goal. A KPI therefore becomes meaningful only in context: the same measure can be relevant or irrelevant depending on the user, the decision, the time horizon, and the available data. This is one reason why dashboard design cannot be reduced to a fixed chart lookup table. A good design must first understand what the user wants to know and then select appropriate visual representations.

Visualization research has long treated visualization design as a mapping from data and analytical tasks to graphical representations. Mackinlay’s APT system formalized early ideas about automatic graphical presentation by encoding design criteria that can be used to construct effective graphical presentations (Mackinlay, 1986). Later systems such as Show Me, Voyager, Voyager 2, and Draco extended this direction by combining user intent, data types, perceptual principles, and constraint-based recommendation (Mackinlay et al., 2007; Wongsuphasawat et al., 2016, 2017; Moritz et al., 2019).

For dashboard design, the intended user and decision context are important because they affect what information should be emphasized. An executive dashboard may emphasize a small number of high-level KPIs and deviations, while an operational dashboard may require more detailed trends, filters, and drill-down interactions. Therefore, the target of this thesis is not simply “choose a chart.” The task is to generate a structured design recommendation that explains how a dashboard brief should be translated into a coherent presentation.

This view is consistent with the project’s structured output schema. The generated recommendation contains a context summary, KPI-to-chart mappings, layout, styling, interactions, and short rationales. The system therefore combines elements of visualization recommendation, natural-language understanding, and structured generation.

### 2.1.1 Dashboards versus Individual Visualizations

A single visualization usually answers a relatively focused analytical question, such as how a measure changes over time or how two variables relate. A dashboard combines several such views and must also manage relationships between them. This includes information hierarchy, visual consistency, shared filters, interaction, screen space, and the ordering of attention.

This distinction is important for the present work. Many public NL-to-visualization datasets focus on mapping one natural-language query to one visualization. They are valuable sources for chart type, encoding, aggregation, and analytical-task supervision, but they do not directly provide complete dashboard designs. The project therefore uses source-grounded visualization data for analytical semantics while separately representing dashboard-level fields such as layout, styling, interaction, and rationale.

### 2.1.2 Users, Goals, KPIs, and Data Context

A dashboard brief can be viewed as a compact specification of the design problem. Four parts are especially important:

1. **Users** describe the intended audience.
2. **Goals** describe what users want to understand or decide.
3. **KPIs** describe measurable quantities relevant to these goals.
4. **Data context** describes the available columns, data types, filters, groupings, and other constraints.

These elements reduce ambiguity. For example, “show sales” is underspecified, while “show monthly total sales by product category for a regional manager” already provides a temporal grain, aggregation, grouping dimension, and audience context. Structured generation benefits from this explicitness because the model has fewer opportunities to invent unsupported analytical assumptions.

## 2.2 Visualization Design Principles

Visualization design is guided by both empirical findings and practical heuristics. These should not be confused. Some principles are grounded in controlled perception studies, while others are conventions derived from long-term design practice.

### 2.2.1 Graphical Perception

Cleveland and McGill (1984) studied how accurately people decode elementary graphical encodings. Their work showed that position along a common scale is generally judged more accurately than encodings such as area or angle. This provides an empirical basis for preferring position-based encodings when precise quantitative comparison is important.

The broader lesson is not that one chart type is universally “best,” but that visual channels have different perceptual properties. A chart recommendation should therefore consider what comparison the user needs to make. If precise comparison is central, position and length are often strong channels. If the task is primarily composition, other encodings may be acceptable even if they are less precise.

### 2.2.2 Expressiveness and Effectiveness

Mackinlay (1986) distinguished between expressiveness and effectiveness. A visual representation is expressive when it represents the intended information without misleading implication, and effective when it uses graphical encodings that support accurate and efficient interpretation.

This distinction remains useful for automated recommendation. A chart may be technically valid but still be a poor design choice. For example, a pie chart may encode categories and values correctly, but it may be ineffective when there are many categories or when precise differences matter.

### 2.2.3 Task–Chart Fit

Visualization recommendations should be driven by analytical tasks. Saket et al. (2018) provide evidence that the effectiveness of common chart types varies by task. This supports task-aware recommendation rather than a universal ranking of charts.

Typical analytical intentions include:

- comparison,
- ranking,
- trend,
- distribution,
- correlation,
- composition or part-to-whole,
- deviation,
- lookup or retrieval,
- flow or relationship.

Mappings between these tasks and chart types are useful, but they are context-dependent. A line chart is often appropriate for ordered temporal trends, a bar chart for categorical comparisons, and a scatter plot for relationships between two quantitative variables. However, valid exceptions exist. For this reason, this thesis treats task–chart mappings as design guidance and source-grounded evidence, not as absolute universal laws.

### 2.2.4 Visual Hierarchy and Information Density

A dashboard must guide attention. Important information should be visually prominent, while secondary detail should be available without competing with primary signals. Practical techniques include grouping related views, using consistent alignment, limiting unnecessary decoration, and emphasizing the most decision-relevant KPIs.

Information density also requires balance. Too little information can make a dashboard uninformative, while too much can create clutter and increase cognitive effort. In this thesis, layout recommendations are therefore part of the structured output rather than an afterthought.

### 2.2.5 Readability, Labels, and Scales

Readable charts use clear titles, meaningful axis labels, suitable number formats, and scales that do not distort interpretation. Scale choices are especially important because visual changes can exaggerate or hide differences. Labels should use terminology understandable to the target user rather than internal database names whenever possible.

### 2.2.6 Color and Accessibility

Color can encode categories, magnitude, status, or emphasis. However, excessive or inconsistent color reduces readability. Accessibility also matters: designs should not rely on color alone for critical distinctions, and text/background contrast should be sufficient for legibility. These principles are widely accepted in visualization and interface design, even though exact palette recommendations depend on the application context.

## 2.3 Analytical Tasks and Chart Types

A useful visualization system needs a vocabulary of analytical tasks. Munzner’s visualization framework emphasizes that visualization design depends on what users need to do with data, not only on the data itself (Munzner, 2014). Natural-language visualization systems similarly attempt to infer analytical tasks before recommending charts (Narechania et al., 2021; Fu et al., 2020).

For this thesis, the most relevant task families are the following.

### 2.3.1 Comparison and Ranking

Comparison asks how values differ between categories, while ranking focuses on ordering. Bar charts are common because aligned positions and lengths make differences easy to judge. Sorting can further support ranking tasks.

### 2.3.2 Trend

Trend tasks examine change over an ordered dimension, most commonly time. Line charts are often effective because connected positions emphasize continuity and direction. Time grain matters: daily, monthly, and yearly aggregation can lead to very different interpretations.

### 2.3.3 Composition and Part-to-Whole

Composition asks how components contribute to a total. Stacked bars, normalized stacked bars, treemaps, and in limited cases pie or donut charts may support this task. Pie charts become difficult to interpret when categories are numerous or values are close, so their use should be conservative.

### 2.3.4 Distribution

Distribution tasks focus on the shape, spread, frequency, and outliers of a variable. Histograms, box plots, density plots, and related charts can support such questions. A simple bar chart may also be sufficient when the data are already aggregated into categorical frequencies.

### 2.3.5 Correlation and Relationship

Scatter plots are a standard representation when the task is to inspect the relationship between two quantitative variables. This assumes that both axes have meaningful quantitative semantics. Identifier-like fields should not be treated as measurements simply because they are stored as numbers.

### 2.3.6 Deviation

Deviation focuses on difference from a reference, target, baseline, or zero. Diverging bars, variance indicators, and annotated line charts may be useful depending on the context.

### 2.3.7 Flow and Network Relationships

Flow tasks concern movement between categories, stages, or entities. Sankey diagrams, node-link diagrams, and flow maps can be appropriate, although these structures are outside the main chart families represented in the project’s current source dataset.

Overall, task-to-chart mapping should be interpreted as a constrained design decision rather than a deterministic rule. This distinction is important because the thesis evaluates whether LLM-based methods can produce context-aware recommendations rather than memorize a fixed lookup table.

## 2.4 Large Language Models

Large language models are neural language models trained on large text corpora to predict and generate sequences of tokens. Modern LLMs are largely based on the Transformer architecture introduced by Vaswani et al. (2017). Transformers replace recurrence with self-attention, allowing each token representation to incorporate information from other positions in the sequence.

### 2.4.1 Transformer Architecture

The central mechanism is attention. In simplified form, a token representation is transformed into query, key, and value vectors. Attention weights are computed from query–key similarity, and these weights determine how value vectors are combined. Multi-head attention allows a model to learn several interaction patterns in parallel.

Transformer blocks also contain feed-forward layers, residual connections, and normalization. Decoder-only LLMs generate text autoregressively: each next token is predicted from the previously generated context.

For this thesis, the most important consequence is that an LLM can condition on a structured dashboard brief and generate a multi-field recommendation. However, the model does not inherently guarantee that all fields are correct, grounded, or valid JSON.

### 2.4.2 Instruction Following

Instruction-tuned models are adapted to respond to natural-language instructions rather than only continue text. This makes them suitable for tasks such as structured extraction, recommendation, and JSON generation. Prompt-only systems rely primarily on this general instruction-following capability.

### 2.4.3 Limitations and Hallucination

LLMs can produce plausible but unsupported content. In this thesis, hallucination includes inventing a field, KPI, filter, chart change, or design claim that is not supported by the dashboard brief or source evidence. This is particularly important because a syntactically valid output can still be semantically wrong.

The project therefore separates structural validity from semantic correctness. JSON parsing and schema compliance are necessary, but they do not by themselves prove that a recommendation is useful or source-faithful.

## 2.5 Parameter-Efficient Fine-Tuning

Full fine-tuning updates all model parameters. For modern LLMs, this can require large GPU memory and storage. Parameter-efficient fine-tuning (PEFT) instead adapts a relatively small subset of parameters while keeping most pretrained weights fixed.

### 2.5.1 LoRA

Low-Rank Adaptation (LoRA) was introduced by Hu et al. (2021). Instead of directly updating a full weight matrix, LoRA learns a low-rank update. If a pretrained weight matrix is \(W\), the adapted transformation can be written conceptually as:

\[
W' = W + BA
\]

where \(A\) and \(B\) are small trainable matrices whose rank is much lower than the original matrix dimensions.

The pretrained weights remain frozen. This reduces the number of trainable parameters and makes it practical to store separate adapters for different tasks. Hu et al. report that LoRA can achieve performance comparable to full fine-tuning while using far fewer trainable parameters.

### 2.5.2 QLoRA

QLoRA extends this idea by combining LoRA with low-bit quantization (Dettmers et al., 2023). The base model is stored in 4-bit precision while gradients are propagated to LoRA adapters. The method introduced techniques such as NormalFloat4 (NF4), double quantization, and paged optimizers to reduce memory use.

For this thesis, QLoRA is important because it enables experiments with larger open models on limited GPU resources. It also makes multi-seed experiments more feasible than full-parameter fine-tuning.

### 2.5.3 Relevance to This Thesis

Fine-tuning is expected to help the model learn:

- the project’s output schema,
- recurring dashboard-design structures,
- task and chart vocabulary,
- relations between goals, KPIs, and encodings,
- style and interaction conventions represented in the training data.

However, fine-tuning does not automatically guarantee better factual grounding. This motivates comparing fine-tuning both with and without retrieval.

## 2.6 Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a generative model with an external knowledge source. Lewis et al. (2020) introduced a general RAG architecture in which a model retrieves relevant documents from non-parametric memory and conditions generation on them.

A typical RAG pipeline contains:

1. a document or guideline corpus,
2. preprocessing and chunking,
3. an index,
4. a retrieval method,
5. selection of the top-\(k\) passages,
6. insertion of the retrieved passages into the generation context.

### 2.6.1 Sparse and Dense Retrieval

Sparse retrieval methods, such as TF-IDF or BM25, rely mainly on lexical overlap. They are transparent and inexpensive but can miss semantic matches that use different wording.

Dense retrieval uses learned vector embeddings and similarity in embedding space. It can capture semantic similarity beyond exact terms, but it introduces additional models and implementation complexity.

The retrieval method is only one part of RAG quality. Poorly chosen or poorly chunked source documents can produce irrelevant context even with a strong retriever.

### 2.6.2 RAG for Dashboard Design Guidance

In this thesis, RAG is used to provide external visualization-design knowledge during inference. Relevant passages may describe chart-selection principles, labeling, color use, hierarchy, or accessibility.

The motivation is complementary to fine-tuning:

- fine-tuning changes model behavior through learned parameters;
- RAG provides explicit information at inference time.

The A/B/C/D experimental design therefore isolates these two mechanisms.

### 2.6.3 RAG Limitations

Retrieval can fail in several ways:

- the relevant guideline is missing from the corpus;
- retrieval selects irrelevant passages;
- context is relevant but the LLM ignores it;
- the model combines retrieved evidence incorrectly;
- the answer appears grounded but contains unsupported claims.

Therefore, RAG should not be evaluated only by output fluency. Retrieval quality and grounding must also be considered.

## 2.7 Structured LLM Outputs

Many software applications require machine-readable output rather than free-form text. JSON is common because it can be parsed, validated, and integrated with downstream systems.

### 2.7.1 Schema-Based Generation

A schema specifies the required fields, field types, and sometimes enumerated values. For this thesis, the output schema allows automatic checks such as:

- whether valid JSON was generated,
- whether required sections are present,
- whether chart and task types use accepted values,
- whether nested fields are structurally valid.

Structured-output research has shown that constrained decoding can strongly improve syntactic validity, but syntactic validity should be separated from semantic correctness. Recent benchmarks such as JSONSchemaBench explicitly evaluate both coverage of schema constraints and output quality (Geng et al., 2025).

### 2.7.2 Parsing and Validation

A robust system should distinguish at least three cases:

1. output cannot be parsed,
2. output parses but violates the schema,
3. output is schema-valid but semantically incorrect.

This separation is essential in the thesis evaluation. A model that produces 100% valid JSON may still recommend the wrong chart or invent unsupported interactions.

### 2.7.3 Why Structured Output Matters Here

The thesis does not aim to produce open-ended design commentary. The target is a structured dashboard recommendation that can be evaluated consistently and potentially consumed by software. This design choice enables reproducible automatic evaluation and makes the comparison between prompting, RAG, fine-tuning, and fine-tuning+RAG clearer.

## 2.8 Chapter Summary

This chapter introduced the concepts needed for the remainder of the thesis. Dashboard design was framed as the translation of user goals, KPIs, data context, and analytical tasks into coordinated visual and interactive decisions. Visualization research provides empirical and heuristic guidance for graphical perception, task–chart fit, hierarchy, readability, and accessibility. Large language models offer a flexible mechanism for generating such recommendations, but they require controls for structure and grounding. LoRA and QLoRA enable efficient adaptation, while RAG provides access to explicit external knowledge. Finally, structured output and schema validation provide the technical basis for reproducible evaluation.

The next chapter reviews prior work that addresses these components and explains how the present thesis is positioned relative to visualization recommendation, NL-to-visualization systems, modern LLM-based chart generation, fine-tuning, RAG, and structured-output evaluation.

---

## References Used in Chapter 2

Cleveland, W. S., & McGill, R. (1984). Graphical perception: Theory, experimentation, and application to the development of graphical methods. _Journal of the American Statistical Association, 79_(387), 531–554. https://doi.org/10.1080/01621459.1984.10478080

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. _Advances in Neural Information Processing Systems, 36_. https://arxiv.org/abs/2305.14314

Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N., West, R., Horvitz, E., & Nori, H. (2025). Generating structured outputs from language models: Benchmark and studies. arXiv:2501.10868. https://arxiv.org/abs/2501.10868

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. _Advances in Neural Information Processing Systems, 33_. https://arxiv.org/abs/2005.11401

Mackinlay, J. (1986). Automating the design of graphical presentations of relational information. _ACM Transactions on Graphics, 5_(2), 110–141. https://doi.org/10.1145/22949.22950

Moritz, D., Wang, C., Nelson, G. L., Lin, H., Smith, A. M., Howe, B., & Heer, J. (2019). Formalizing visualization design knowledge as constraints: Actionable and extensible models in Draco. _IEEE Transactions on Visualization and Computer Graphics, 25_(1), 438–448. https://doi.org/10.1109/TVCG.2018.2865240

Munzner, T. (2014). _Visualization Analysis and Design_. CRC Press. https://doi.org/10.1201/b17511

Narechania, A., Srinivasan, A., & Stasko, J. (2021). NL4DV: A toolkit for generating analytic specifications for data visualization from natural language queries. _IEEE Transactions on Visualization and Computer Graphics_. https://doi.org/10.1109/TVCG.2020.3030378

Saket, B., Endert, A., & Stasko, J. (2018). Task-based effectiveness of basic visualizations. _IEEE Transactions on Visualization and Computer Graphics_. https://doi.org/10.1109/TVCG.2018.2865020

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. _Advances in Neural Information Processing Systems, 30_. https://arxiv.org/abs/1706.03762

Wongsuphasawat, K., Moritz, D., Anand, A., Mackinlay, J., Howe, B., & Heer, J. (2016). Voyager: Exploratory analysis via faceted browsing of visualization recommendations. _IEEE Transactions on Visualization and Computer Graphics, 22_(1), 649–658.

Wongsuphasawat, K., Qu, Z., Moritz, D., Chang, R., Ouk, F., Anand, A., Mackinlay, J., Howe, B., & Heer, J. (2017). Voyager 2: Augmenting visual analysis with partial view specifications. _CHI 2017_. https://doi.org/10.1145/3025453.3025768
