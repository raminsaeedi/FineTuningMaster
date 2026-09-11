# Chapter 2 — Background and Foundations

This chapter introduces the concepts needed to understand the thesis. It first defines dashboards, key performance indicators (KPIs), and the design problem addressed in this project. It then presents the visualization principles that guide chart selection and dashboard composition. The chapter next explains analytical tasks and common chart types, followed by a short description of large language models (LLMs), Transformer-based generation, and instruction tuning. The main part of the chapter describes parameter-efficient fine-tuning (PEFT), including LoRA, QLoRA, and related methods such as AdaLoRA, DoRA, RSLoRA, and GaLore. It also explains why QLoRA is used as the main fine-tuning approach in this thesis. The final sections introduce retrieval-augmented generation (RAG) and structured LLM outputs. The chapter therefore establishes the design, modelling, and evaluation concepts used in the following chapters.

## 2.1 Dashboard Design

A dashboard is an interactive view that brings together several related measures and visual elements for monitoring, analysis, or decision support. Dashboards are not defined only by the number of charts that they contain. Their value depends on whether the displayed information supports the needs of a particular audience and decision context. Reviews of dashboard research describe dashboards as systems that combine performance information, visual displays, and interaction in order to support monitoring and management decisions \cite{yigitbasioglu2012dashboard}. Practitioner-oriented work makes the same point from the design side, arguing that a dashboard should present the information needed to achieve one or more objectives on a single screen and that most dashboard failures are failures of visual communication rather than of data \cite{few2006dashboard}. More recent work also shows that recurring dashboard design problems can be described through design patterns, for example patterns for overview, comparison, filtering, and detail \cite{bach2023dashboardpatterns}, and a survey of how the term is used across research and practice documents a design space that is considerably broader than a single canonical dashboard type \cite{sarikaya2019dashboards}.

In this thesis, a dashboard is understood as a coordinated design rather than as a random collection of individual charts. A coherent dashboard connects the intended users, their goals, the relevant KPIs, the available data, the analytical tasks, the visual encodings, the layout, the styling, and the interactions. These elements should support the same decision context. A chart can be correct on its own and still be unsuitable when it is placed in the wrong dashboard, given too much visual emphasis, or separated from the information needed to interpret it.

A KPI is a measurable value used to represent progress or performance in relation to a goal. The value of a KPI depends on its context. The same measure may be useful for one user and irrelevant for another user because the decision, time period, comparison group, or available level of detail is different. A dashboard design should therefore make clear what is measured, for whom it is measured, over which period, and why the measure matters. This view also prevents a common mistake in automated design systems: treating a KPI name as enough information to choose a chart.

The design problem in this thesis can be described as a translation problem. A dashboard brief contains natural-language information about users and goals, KPI descriptions, data columns, and constraints. The system must translate this brief into a structured design recommendation. The recommendation does not only contain a chart type. It also contains the mapping from KPIs to analytical tasks and encodings, a dashboard layout, styling, interactions, and rationales. This formulation is related to the general visualization-design problem of mapping data and user tasks to graphical representations \cite{mackinlay1986apt,munzner2014visualization}. Constraint-based systems such as Draco make a similar distinction by representing design knowledge as explicit constraints that can be used to rank or reject visual encodings \cite{moritz2019draco}.

The project implements this design problem as structured generation. Its input is a dashboard brief with users, goals, KPIs, data columns, and constraints. Its target is a typed recommendation with the fields `context_summary`, `kpi_chart_mapping`, `layout`, `styling`, `interactions`, and `rationales`. This structure is a project-specific design contract. It is not presented as a universal definition of a dashboard. It makes the target explicit and allows the methods in the thesis to be compared on the same output interface.

### 2.1.1 Dashboards versus Individual Visualizations

A single visualization normally addresses a focused analytical question, such as whether a value changes over time or whether two variables are related. A dashboard combines several views and must also manage the relationships between them. These relationships include visual hierarchy, consistent scales and labels, shared filters, spatial grouping, interaction, and the amount of information shown at one time.

This distinction matters for the present research. Many visualization datasets and systems focus on one query and one chart specification. They can provide useful examples for analytical tasks, data fields, aggregations, and encodings. They do not necessarily provide a complete dashboard with a coordinated layout, styling, interaction design, and rationale. The project therefore treats chart selection as one part of a larger recommendation. The generated object is evaluated as a structured dashboard design, not only as a single chart label.

### 2.1.2 Users, Goals, KPIs, and Data Context

A dashboard brief is a compact specification of the design problem. The user description identifies the intended audience and its level of expertise. The goals describe what the audience wants to understand, compare, monitor, or decide. The KPIs identify the measures that are relevant to these goals. The data context describes the available fields, their types, their possible groupings, the time dimension, and any constraints on the design. These parts should be interpreted together because none of them is sufficient on its own.

For example, the instruction “show sales” does not determine a good visualization. A brief such as “show monthly total sales by product category for a regional manager” gives more useful information. It identifies a time grain, an aggregation, a grouping dimension, and an audience. It can therefore support a more precise decision about the analytical task, chart type, encoding, and level of detail. Explicit context also reduces the risk that a model invents a field, a KPI, or an interaction that is not present in the input.

## 2.2 Visualization Design Principles

Visualization design is based on both empirical findings and practical design guidance. These two forms of knowledge should be kept separate. Controlled studies can provide evidence about how accurately people decode a visual encoding. Design frameworks can provide useful concepts for describing tasks, data, and interfaces. Neither type of source turns every chart choice into a universal rule. The relevant principle for this thesis is that a chart should be selected in relation to the task and the context in which it will be read.

### 2.2.1 Graphical Perception

Cleveland and McGill \cite{cleveland1984graphical} studied how people judge elementary graphical quantities. Their work showed that some visual encodings support more accurate judgments than others. In particular, position along a common scale is generally judged more accurately than angle or area. This result gives a perceptual reason to prefer position- and length-based encodings when users must make precise quantitative comparisons.

The result should not be interpreted as a complete ranking of all chart types. The usefulness of an encoding also depends on the task, the data distribution, the number of categories, the display size, and the user's prior knowledge. A chart recommendation should therefore ask what the user needs to compare or detect. Precision is important for some tasks, while overview, composition, or recognition may be more important for others.

### 2.2.2 Expressiveness and Effectiveness

Mackinlay \cite{mackinlay1986apt} distinguishes between expressiveness and effectiveness. An expressive visualization represents the intended information without adding an unsupported meaning. An effective visualization uses graphical encodings that help users interpret the information accurately and efficiently. These criteria are useful for automated recommendation because a chart can be technically valid and still be a poor design choice.

For example, a pie chart can represent category proportions without a data error. It may nevertheless be ineffective when many categories are present or when users need to compare values that are close to one another. A bar chart may support that comparison more directly because the values can be judged by aligned positions and lengths. This example illustrates why the output of the system should include a rationale and not only a chart name.

### 2.2.3 Task–Chart Fit

The fit between an analytical task and a chart is central to visualization design. Saket et al. \cite{saket2019taskbased} show that the effectiveness of basic visualizations changes across tasks. Kim and Heer \cite{kim2018taskdata} likewise show that task and data distribution can affect the effectiveness of visual encodings. These findings support a context-dependent recommendation process instead of a universal ranking such as “bar charts are always best.”

In practice, an ordered temporal trend is often represented with a line chart because connected positions make continuity and direction visible. A categorical comparison or ranking is often represented with bars because aligned lengths support comparison. A relationship between two quantitative variables is often represented with a scatter plot. These mappings are useful starting points, but they still require checks on data semantics, scale, cardinality, and the user's goal. The project consequently represents task type and chart type as explicit fields and leaves the final choice to the model under the constraints of the brief.

### 2.2.4 Visual Hierarchy and Information Density

A dashboard must guide the user's attention. High-priority information should be easy to find, while secondary detail should remain available without competing with the main message. Visual hierarchy can be created through position, size, grouping, whitespace, contrast, and consistent alignment. The dashboard design-pattern literature describes these choices as recurring solutions to common interface problems \cite{bach2023dashboardpatterns}.

Information density requires a balance. A dashboard with too little information may not support the intended decision. A dashboard with too much information may increase search time and make important changes harder to detect. Density is therefore not simply a matter of counting charts. It also depends on the complexity of each view, the number of encodings, the amount of text, the interaction model, and the user's familiarity with the domain. In this thesis, layout and styling are part of the output because they influence how the selected charts are interpreted together.

### 2.2.5 Readability, Labels, and Scales

Readable charts use titles, labels, legends, annotations, and number formats that match the target audience. Axis labels should communicate the meaning and unit of a measure. Category labels should be understandable and should not expose database names without explanation. A scale should preserve the intended comparison and should not exaggerate or hide a difference through an inappropriate axis range or transformation.

Readability also depends on the available space. A design that works on a large screen may become difficult to read when several views are placed in a small dashboard. This is another reason to treat layout as a design decision rather than as a final decoration. The system should make a reasonable connection between the amount of information, the expected display, and the level of detail required by the user.

### 2.2.6 Color and Accessibility

Color can distinguish categories, show magnitude, communicate status, or draw attention to an important value. It can also create problems when too many colors are used, when similar colors are difficult to separate, or when a critical distinction is communicated only through color. The Web Content Accessibility Guidelines (WCAG) 2.2 require, among other things, that information and relationships are not communicated through color alone and that text has sufficient contrast for reading \cite{wcag2024}.

For dashboard design, this means that color should usually be supported by labels, position, shape, text, or another redundant cue. Color palettes should be selected with the audience and display conditions in mind. Accessibility is not a separate visual layer that can be added after chart selection. It can change the suitability of an encoding and should therefore be included in the recommendation and its rationale.

## 2.3 Analytical Tasks and Chart Types

An analytical task describes what a user wants to do with data. Brehmer and Munzner \cite{brehmer2013typology} distinguish the purpose of a task from the means used to perform it and from the data involved. This distinction is useful because a column type alone does not reveal the user's intention. A date column can support a trend, a comparison between periods, a lookup, or a deviation from a target. The same field can therefore lead to different visual designs depending on the question.

The task vocabulary used in this thesis is informed by visualization task literature and adapted to the project's structured schema. It includes trend, comparison, composition, distribution, correlation, ranking, deviation, part-to-whole, and flow. The vocabulary is an implementation choice that makes evaluation possible. It should not be interpreted as a complete or universal taxonomy of all analytical work.

### 2.3.1 Comparison and Ranking

Comparison asks how values differ between categories, groups, or periods. Ranking adds the requirement that the values should be ordered. Bar charts are common choices because the lengths share a baseline and the categories can be sorted. Grouped bars can support comparison between several series, while a table may be more appropriate when exact values are more important than visual pattern recognition. The number of categories and the available screen space can change which option is readable.

### 2.3.2 Trend

Trend tasks examine change along an ordered dimension, most often time. Line charts are often useful because the connected marks emphasize continuity and direction. Area charts can add a sense of volume, but overlapping or stacked areas can make individual series harder to compare. The time grain must also be considered. Daily, monthly, and yearly aggregation can show different patterns and can lead to different decisions.

### 2.3.3 Composition and Part-to-Whole

Composition tasks ask how components contribute to a total. Stacked bars can show composition across several groups, and normalized stacked bars can show relative proportions. Treemaps can use space to represent parts of a hierarchy. Pie and donut charts can be suitable for a small number of clearly different parts, but they are less suitable when categories are numerous or values are close. The choice is therefore influenced by whether the user needs to compare individual parts, inspect a total, or understand changes in composition.

### 2.3.4 Distribution

Distribution tasks focus on frequency, spread, shape, and outliers. Histograms can show the frequency of binned values, while box plots provide a compact view of median, spread, and possible outliers. Density plots can show the shape of a distribution when the display and audience support that level of detail. A categorical bar chart may be enough when the input already contains aggregated frequencies. The data representation and the intended question must be checked before selecting the chart.

### 2.3.5 Correlation and Relationship

Correlation or relationship tasks examine how two or more variables change together. A scatter plot is a standard choice for two quantitative variables because each observation can be represented by a position in a two-dimensional coordinate system. Additional encodings may show groups or a third variable, but they also increase visual complexity. Identifier-like fields should not be treated as measurements only because they are stored as numbers.

### 2.3.6 Deviation

Deviation tasks focus on the difference from a reference value, target, baseline, or zero. A diverging bar chart can show positive and negative differences, while an annotated line chart can show when a measure moves above or below a target. KPI cards can provide a compact status view when the deviation itself is the primary message. The reference must be explicit; otherwise, the viewer cannot determine what the displayed difference means.

### 2.3.7 Flow and Network Relationships

Flow tasks concern movement between stages, categories, or entities. Sankey diagrams can represent quantities moving between connected nodes, and node-link diagrams can represent relationships in a network. These charts can be useful when the connections are the main subject of analysis, but they can become difficult to read as the number of nodes and links grows. In the current project, flow is part of the task vocabulary, while the available training and evaluation data determine how often this task can be assessed.

These examples are literature-informed design guidance, not fixed rules. A line chart is not automatically correct for every temporal field, and a bar chart is not automatically correct for every comparison. A final recommendation should take the user's goal, the data semantics, the number of values, accessibility, and dashboard context into account. This distinction is important for the thesis because the system is evaluated on context-aware structured recommendations rather than on memorization of a chart lookup table.

## 2.4 Large Language Models

Large language models are neural models that learn statistical patterns in sequences of tokens and use those patterns to generate text. Modern LLMs are commonly based on the Transformer architecture introduced by Vaswani et al. \cite{vaswani2017attention}. The model receives a sequence of tokens, represents their relationships, and predicts a continuation. In this thesis, the LLM is used as a conditional generator: it receives a dashboard brief and produces a structured design recommendation.

### 2.4.1 Transformer Architecture

The main operation in a Transformer is self-attention. In simplified form, each token representation is transformed into a query, a key, and a value. The similarity between a query and the keys determines how strongly the corresponding values contribute to the updated representation. Multi-head attention allows the model to learn several types of relationships in parallel. Transformer blocks also contain feed-forward layers, residual connections, and normalization.

The thesis does not require a mathematical treatment of every Transformer component. The important point is that self-attention allows a model to condition its output on different parts of the input sequence. A dashboard brief can therefore contain user information, goals, KPIs, data columns, and constraints in the same context. The model can use these elements when generating a recommendation, but the architecture alone does not guarantee that the recommendation is correct or that it follows the available data.

### 2.4.2 Autoregressive Generation

Decoder-only LLMs usually generate text autoregressively. At each step, the model predicts a probability distribution for the next token based on the tokens that have already been provided or generated. The selected token is appended to the context, and generation continues until a stopping condition is reached. Sampling settings can change the output, which is relevant when the same prompt is evaluated under different methods or random seeds.

Autoregressive generation is flexible because the output can contain several related fields instead of one classification label. It also creates failure modes. A model can stop too early, repeat content, omit a field, or generate text that looks plausible but is unsupported by the input. These failures motivate the use of explicit output constraints and separate validation steps later in the thesis.

### 2.4.3 Instruction Tuning

Instruction tuning adapts a pretrained language model using examples in which tasks are described through natural-language instructions. Wei et al. \cite{wei2022flan} show that instruction tuning can improve performance on tasks that were not included in the same form during training. The method is important for this thesis because a dashboard brief is expressed as an instruction rather than as a fixed classification input.

Instruction tuning should not be confused with a guarantee of domain expertise. An instruction-tuned model may follow the requested format while still making a poor chart choice or inventing a field. Prompt-only generation therefore provides a useful baseline for measuring how much of the task can be solved from the pretrained and instruction-tuned model without retrieval or task-specific adapter training.

### 2.4.4 Limitations and Hallucination

LLMs can produce fluent statements that are not supported by the input or by an external source. The literature commonly refers to this problem as hallucination in natural-language generation \cite{ji2023hallucination}. In the present use case, a hallucination can be an invented data column, an unsupported KPI, an interaction that is not possible with the given data, or a rationale that cites a principle without applying it to the current brief.

A syntactically valid JSON object does not remove this problem. It only means that the output can be parsed according to a format. The project therefore separates structural validity from semantic correctness and grounding. These distinctions are needed when comparing prompt-only, RAG, fine-tuned, and combined methods.

## 2.5 Parameter-Efficient Fine-Tuning

Full fine-tuning updates all trainable parameters of a pretrained model for a downstream task. This can be expensive for LLMs because the optimizer must maintain additional states and because a separate model copy may be needed for each task. PEFT reduces this cost by keeping most pretrained parameters fixed and learning only a small task-specific component. Adapter-based transfer learning established this general idea by inserting small trainable modules while sharing the original model parameters across tasks \cite{houlsby2019adapter}.

PEFT is especially relevant when a study compares several models, methods, or random seeds. The learned task-specific component can be stored separately from the base model, and the same base model can be reused across conditions. This reduces storage and makes the intervention easier to describe. It does not mean that PEFT is always as accurate as full fine-tuning. The result depends on the task, the data, the model, and the chosen PEFT method.

### 2.5.1 LoRA

Low-Rank Adaptation (LoRA) learns a low-rank update to selected weight matrices while keeping the pretrained weights frozen \cite{hu2022lora}. Let \(W_0\) be a pretrained weight matrix. A LoRA update can be written as

\[
W' = W_0 + \frac{\alpha}{r}BA,
\]

where \(A \in \mathbb{R}^{r \times d*{\mathrm{in}}}\) and \(B \in \mathbb{R}^{d*{\mathrm{out}} \times r}\) are trainable matrices, \(r\) is the adapter rank, and \(\alpha\) is a scaling factor. The rank \(r\) is chosen to be much smaller than the dimensions of the original matrix. During training, \(W_0\) is frozen and only \(A\) and \(B\) are updated.

This factorization reduces the number of trainable parameters and allows a task-specific adapter to be saved without copying the entire base model. It also defines a clear experimental intervention: the base model remains the same, while the learned low-rank update changes the model's behaviour for the target task. LoRA was originally evaluated in several language-model settings, but its performance is not independent of rank, target modules, data, and training configuration.

### 2.5.2 QLoRA

QLoRA combines LoRA with low-bit storage of the frozen base model \cite{dettmers2023qlora}. The base weights are stored in a 4-bit representation and are dequantized for the computation needed during forward and backward passes. The gradients are used to update the LoRA adapters rather than the quantized base weights. QLoRA therefore does not mean that the adapter itself is trained as a 4-bit copy of the model; the important distinction is between the frozen quantized base and the trainable adapter parameters.

The method introduces several techniques to reduce memory use. NormalFloat4 (NF4) is used for the quantized base weights, double quantization reduces the storage needed for quantization constants, and paged optimizers help manage memory peaks during training. The original QLoRA paper reports that this combination can make fine-tuning much larger models possible on a single GPU while retaining strong task performance. Those reported results motivate the method, but they are not a guarantee for every model, dataset, or hardware setup.

QLoRA is therefore best understood as a resource-aware implementation of low-rank adaptation. It retains the parameter-sharing and adapter-storage properties of LoRA while lowering the memory required to load the frozen base model. Quantization can also introduce approximation effects, and the final outcome remains sensitive to rank, learning rate, sequence length, data quality, and hardware precision.

### 2.5.3 Why PEFT for This Thesis

The research task has three properties that support a PEFT design. First, the target is a structured recommendation task rather than the training of a new language model. The model must learn a project-specific mapping from a dashboard brief to a multi-field design object. Second, the study compares separate interventions, including prompt-only generation, retrieval, fine-tuning, and fine-tuning with retrieval. Third, the project is intended to run several model and seed conditions under finite GPU memory and storage. PEFT addresses these properties by keeping the base model reusable and by representing the task-specific intervention as a separate adapter. The decision is therefore supported by both the transfer-learning literature and the experimental design of this thesis.

#### Other PEFT and Memory-Efficient Fine-Tuning Algorithms

LoRA and QLoRA are part of a larger family of adaptation methods. Adapter tuning adds small bottleneck modules to the network while leaving the original parameters fixed \cite{houlsby2019adapter}. Prefix tuning learns continuous task-specific vectors that are treated as virtual tokens in the Transformer's activations \cite{li2021prefixtuning}. Prompt tuning is a simpler related approach that learns soft prompt embeddings and conditions a frozen model through those embeddings \cite{lester2021prompttuning}. These methods store a small task-specific component, but they modify different parts of the computation than LoRA.

Sparse methods update only a selected subset of existing parameters. BitFit, for example, updates bias terms while keeping the remaining Transformer parameters fixed \cite{benzaken2022bitfit}. IA3 instead learns vectors that rescale internal activations in attention and feed-forward blocks \cite{liu2022ia3}. These methods can use very few trainable parameters, but their restricted update structure may or may not be sufficient for a task that requires changes to chart selection, layout, styling, and rationale generation at the same time.

Several methods change how the low-rank budget is allocated or how the update is scaled. AdaLoRA allocates different ranks to different weight matrices according to their estimated importance \cite{zhang2023adalora}. DoRA decomposes a weight into magnitude and direction and uses LoRA for the directional update \cite{liu2024dora}. Its purpose is to reduce the gap between low-rank adaptation and full fine-tuning by giving the update a separate magnitude component. Rank-Stabilized LoRA (RSLoRA) changes the scaling of the low-rank update from a form based on \(\alpha/r\) to a form based on \(\alpha/\sqrt{r}\), with the goal of improving stability at higher ranks \cite{kalajdzievski2023rslora}. These methods address different limitations of standard LoRA; they should not be treated as interchangeable names for the same algorithm.

GaLore takes a different approach. It projects gradients into a low-rank subspace to reduce optimizer memory while still allowing full-parameter learning \cite{zhao2024galore}. Because the model parameters are not frozen in the same way as in LoRA, GaLore is a memory-efficient full-parameter training strategy rather than a PEFT method in the strict sense. This distinction matters for experimental interpretation: a GaLore comparison changes both the adaptation mechanism and the set of parameters that can be updated.

The project contains configuration support for QLoRA, DoRA, RSLoRA, and an optional GaLore training path. These alternatives are useful for ablation studies because they represent different resource and update assumptions. The main experimental comparison, however, uses QLoRA for the fine-tuned condition and combines the resulting adapter with RAG in the combined condition. Keeping this main matrix fixed makes it possible to interpret the effect of retrieval separately from the effect of task-specific adaptation. The alternatives are not presented as universally better methods; their suitability must be tested on the same data and evaluation protocol.

#### Comparing the candidate algorithms against the requirements of this study

The selection was made by checking each candidate against four requirements that follow from the research design rather than from a general ranking of algorithms. The first requirement is that the base model must be loadable and trainable for every profile in the matrix, which spans roughly 1.5 to 27 billion parameters, on a single accelerator. The second is that the learned task component must be a **separable artifact**, so that the fine-tuned condition and the combined fine-tuning-plus-retrieval condition can share one trained object and differ only in retrieval; if the combined condition had to be trained again, its comparison with the fine-tuned condition would carry a second stochastic training trajectory. The third is that the method must change **one** experimental factor, so that a difference between conditions can be attributed to task adaptation rather than to a simultaneous change in the update rule or the parameter budget. The fourth is that the update must be expressive enough to affect several output fields at once — chart selection, encoding, layout, styling and rationale text — rather than only a shallow calibration of the model's behaviour.

Table 2.1 records how each candidate stands against these requirements.

Table 2.1. Candidate adaptation algorithms and the reason each was or was not selected as the primary method.

| Algorithm | What it changes | Separable artifact | Memory profile of the frozen base | Fit to the four requirements |
| --- | --- | --- | --- | --- |
| Full fine-tuning | All parameters | No; a full model copy per condition | Full precision weights plus optimizer state | Fails requirements 1 and 2 at the upper end of the matrix |
| Adapter tuning \cite{houlsby2019adapter} | Inserted bottleneck modules | Yes | Full precision base | Meets 2–4; base-model memory still limits the largest profile |
| Prefix tuning \cite{li2021prefixtuning} | Continuous vectors prepended to the activations | Yes | Full precision base | Meets 2; expressiveness for a multi-field structured target is less established than for a low-rank weight update |
| Prompt tuning \cite{lester2021prompttuning} | Soft prompt embeddings only | Yes | Full precision base | Smallest update surface; reported to depend strongly on model scale, which conflicts with a matrix that includes 1.5B models |
| BitFit \cite{benzaken2022bitfit} | Bias terms only | Yes | Full precision base | Very small update surface; requirement 4 is doubtful for a task that must change several output fields simultaneously |
| IA3 \cite{liu2022ia3} | Rescaling vectors for attention and feed-forward activations | Yes | Full precision base | Same concern as BitFit for requirement 4 |
| LoRA \cite{hu2022lora} | Low-rank update of selected linear layers | Yes | Full precision base | Meets 2–4; base-model memory is the remaining obstacle for the largest profile |
| AdaLoRA \cite{zhang2023adalora} | Low-rank update with an adaptive rank budget | Yes | Full precision base | Adds a budget-allocation policy, so it changes two factors at once; fails requirement 3 as a primary method |
| DoRA \cite{liu2024dora} | Magnitude and direction decomposed, low-rank on direction | Yes | Full precision base | Different parameterisation; belongs in a matched ablation rather than the primary matrix |
| RSLoRA \cite{kalajdzievski2023rslora} | Rank-stabilised scaling of the low-rank update | Yes | Full precision base | Changes the scaling rule; relevant only when ranks are compared |
| GaLore \cite{zhao2024galore} | Full-parameter learning with low-rank gradient projection | No; produces a full model | Reduced optimizer memory, full-precision weights | Fails requirement 2; also changes the set of updatable parameters |
| **QLoRA** \cite{dettmers2023qlora} | **Low-rank update of selected linear layers over a 4-bit quantized frozen base** | **Yes** | **4-bit NF4 with double quantization** | **Meets all four requirements** |

QLoRA is the only candidate in the table that satisfies all four requirements simultaneously. It inherits the separable-adapter property and the expressive low-rank weight update from LoRA \cite{hu2022lora}, and it removes the remaining obstacle — the memory needed to hold the frozen base — by keeping that base in a 4-bit NF4 representation with double quantization \cite{dettmers2023qlora}. The argument is one of feasibility and experimental control under a fixed compute budget. It is not a claim that QLoRA produces better task performance than DoRA, AdaLoRA, RSLoRA, plain LoRA or full fine-tuning; establishing that would require a matched experiment in which the base model, data, prompt, seed, number of update steps, evaluation split and output processing are all held constant, and no such experiment is part of this thesis.

### 2.5.4 Why QLoRA for This Thesis

QLoRA is selected as the main PEFT method because it combines the low-rank task update with a quantized frozen base model. This combination is directly relevant to the resource constraint of the project. The 4-bit base loading, NF4 representation, double quantization, and paged optimizer techniques reduce the memory needed during training, while the LoRA adapter keeps the learned task component small and separable \cite{dettmers2023qlora}. This makes the planned comparison of models, methods, and seeds more feasible than a full-parameter training setup under the same hardware budget. The claim is a feasibility and experimental-control argument; it is not a claim that QLoRA is the best fine-tuning algorithm for all tasks.

QLoRA also supports a clean comparison between the main conditions. Method A uses the base model with the shared prompt. Method B adds retrieved context without an adapter. Method C adds a QLoRA adapter to the base model. Method D reuses the compatible adapter from Method C and adds the same retrieval procedure used by Method B. This factorization makes the intended comparison explicit: retrieval is an inference-time source of context, while QLoRA changes the learned parameters. Reusing an adapter with the same seed for C and D removes one unnecessary source of variation.

The alternative methods remain scientifically relevant. DoRA may change the capacity of the low-rank update by separating magnitude and direction. RSLoRA changes the rank scaling and is relevant when different ranks are compared. AdaLoRA changes the distribution of the parameter budget, and GaLore changes the memory strategy while allowing full-parameter learning. Choosing one of these methods as the main intervention would answer a different research question or introduce a different control problem. They are therefore better treated as optional ablations in this thesis unless the experimental plan is expanded explicitly.

The choice also has limitations. QLoRA can be sensitive to quantization settings, adapter rank, target modules, learning rate, sequence length, and the quality of the supervision. A small adapter may reproduce the formatting and recurring patterns of the training data without learning a robust design principle. In addition, the project-specific training data may contain regularities that make internal diagnostic scores look strong without proving dashboard quality. These limitations motivate the separate evaluation of format, chart and encoding behaviour, grounding, robustness, and usefulness in human evaluation.

## 2.6 Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a generative model with an external knowledge source. Lewis et al. \cite{lewis2020rag} describe RAG as a framework in which a generator conditions its output on information retrieved from non-parametric memory in addition to its learned parameters. The central idea is that information needed for a response does not have to be stored only in the model weights. It can be selected from a collection of documents at inference time.

A typical RAG pipeline starts with a document or guideline corpus. The documents are cleaned and divided into passages, and the passages are stored in an index. For a new input, a retriever scores the passages and selects a small set of candidates. These passages are inserted into the model context together with the original user input. The LLM then generates an answer conditioned on both sources. Each step affects the final result: missing documents, poor chunk boundaries, an unsuitable query, or irrelevant retrieved passages can reduce the usefulness of the generated answer \cite{lewis2020rag}.

The current project uses a small, repository-local collection of visualization guidance and a lexical retrieval procedure. This choice is appropriate for a controlled experiment because the corpus can be inspected, versioned, and kept constant across runs. It also makes the retrieved evidence visible in the run artifacts. The use of a simple retriever is not a claim that lexical retrieval is universally superior to dense retrieval. It is a controlled implementation decision for testing whether explicit design guidance changes the generated recommendation.

### 2.6.1 RAG and Fine-Tuning as Different Interventions

Fine-tuning and RAG add information to a model in different ways. Fine-tuning changes model parameters and can teach recurring task formats, domain patterns, and output behaviour. RAG leaves the base parameters unchanged during inference and supplies explicit context for the current input. The two mechanisms can therefore complement each other, but they can also fail differently. A fine-tuned model may learn a useful pattern but apply it incorrectly, while a RAG system may retrieve the right passage but fail to use it in the final answer.

This distinction motivates the A/B/C/D design of the thesis. The prompt-only condition provides a base reference. The RAG condition tests the addition of retrieved guidance. The fine-tuned condition tests the addition of a QLoRA adapter. The combined condition tests whether retrieval remains useful after task-specific adaptation. The comparison is meaningful only when the prompt construction, model, generation settings, corpus, retriever, and adapter provenance are controlled as described in the methods chapter.

### 2.6.2 RAG Limitations and Grounding

RAG does not guarantee that an answer is grounded. A relevant passage may be absent from the corpus, or retrieval may return an irrelevant passage. The model may ignore a relevant passage, combine several passages incorrectly, or add an unsupported claim. A response can also mention retrieved concepts without applying them to the actual data columns and goals in the brief.

For this reason, retrieval availability and retrieval usefulness should be reported separately. The project records retrieved passages and evaluates grounding as a separate dimension. A grounding score should be interpreted as evidence about support for claims under the chosen corpus and metric. It should not be treated as a complete measure of visual-design quality or as proof that RAG improves every recommendation \cite{lewis2020rag}.

## 2.7 Structured LLM Outputs

Many applications cannot use free-form text directly. They require an output that can be parsed and passed to another software component. JSON is widely used for this purpose because it represents nested objects and arrays in a machine-readable format. A structured output contract makes the expected interface explicit, but it does not make the content automatically correct.

### 2.7.1 JSON Schema and Schema-Constrained Generation

A JSON Schema describes the expected structure of a JSON instance. It can specify property names, data types, required fields, arrays, nested objects, and allowed values. The JSON Schema Draft 2020-12 specification defines the vocabulary and validation rules used to describe such instances \cite{jsonschema2022draft}. A schema can therefore be used after generation to check whether an output is structurally valid.

Schema-constrained generation applies these restrictions during decoding or through a generation library so that the model is less likely to produce invalid structures. Post-hoc validation applies the schema after the model has generated text. The two approaches are related but not identical. Constrained decoding can reduce syntax errors, while post-hoc validation remains necessary when the output is produced by an unconstrained model or when application-specific semantic checks are required.

Recent work on structured generation, including JSONSchemaBench, evaluates structured-output systems across many real-world schemas and separates schema coverage, efficiency, and output quality \cite{geng2025structured}. This distinction is important for the thesis. A system can satisfy the formal shape of a schema while producing a chart type that does not fit the task or a rationale that is not supported by the brief.

### 2.7.2 Parsing, Validation, and Semantic Correctness

The evaluation should distinguish between an output that cannot be parsed, an output that parses but violates the schema, and an output that is schema-valid but semantically incorrect. The first case is a generation or parsing failure. The second case is a structural failure. The third case is a content failure that requires task-specific checks or human judgment.

In this project, the typed models in `src/core/schemas.py` define the expected recommendation object and the parser records failures. The prompt also restricts task and chart values to the project's controlled vocabulary. These checks make the output comparable across methods, but they do not prove that the model selected a good chart, used an available column, or gave a useful rationale. Those questions require additional evaluation layers.

### 2.7.3 Structured Outputs in This Thesis

The target of the thesis is a structured dashboard recommendation, not an open-ended design essay. The six top-level fields provide a stable interface for comparing prompt-only generation, RAG, QLoRA fine-tuning, and the combined condition. They also make it possible to evaluate different aspects separately: chart and encoding choices, layout and styling, interactions, rationales, and structural validity.

The schema should therefore be understood as an interface contract. It controls how the recommendation is represented and how it can be evaluated or consumed by software. It does not replace visualization knowledge, retrieval evidence, or human assessment. This separation between structure and meaning is carried into the evaluation and results chapters.

---

## References Used in Chapter 2

Generated from `docs/thesis/references.bib` by `experiments/scripts/build_chapter_reference_lists.py`. The BibTeX key of each entry is given in brackets.

Bach, B., Freeman, E., Abdul-Rahman, A., Turkay, C., Khan, S., Fan, Y., et al. (2023). *Dashboard Design Patterns*. IEEE Transactions on Visualization and Computer Graphics, 29(1), 342–352. https://doi.org/10.1109/TVCG.2022.3209448 [`bach2023dashboardpatterns`]

Ben Zaken, E., Goldberg, Y., & Ravfogel, S. (2022). *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models*. Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 1–9. https://doi.org/10.18653/v1/2022.acl-short.1 [`benzaken2022bitfit`]

Brehmer, M., & Munzner, T. (2013). *A Multi-Level Typology of Abstract Visualization Tasks*. IEEE Transactions on Visualization and Computer Graphics, 19(12), 2376–2385. https://doi.org/10.1109/TVCG.2013.124 [`brehmer2013typology`]

Cleveland, W. S., & McGill, R. (1984). *Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods*. Journal of the American Statistical Association, 79(387), 531–554. https://doi.org/10.1080/01621459.1984.10478080 [`cleveland1984graphical`]

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. Advances in Neural Information Processing Systems, 36, 10088–10115. https://doi.org/10.52202/075280-0441 [`dettmers2023qlora`]

Few, S. (2006). *Information Dashboard Design: The Effective Visual Communication of Data*. O'Reilly Media. https://www.oreilly.com/lpt/pr/1512 [`few2006dashboard`]

Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N., et al. (2025). *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models*. https://doi.org/10.48550/arXiv.2501.10868 [`geng2025structured`]

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., et al. (2019). *Parameter-Efficient Transfer Learning for NLP*. Proceedings of the 36th International Conference on Machine Learning, 97, 2790–2799. https://proceedings.mlr.press/v97/houlsby19a.html [`houlsby2019adapter`]

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference on Learning Representations. https://openreview.net/forum?id=nZeVKeeFYf9 [`hu2022lora`]

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., et al. (2023). *Survey of Hallucination in Natural Language Generation*. ACM Computing Surveys, 55(12), 1–38. https://doi.org/10.1145/3571730 [`ji2023hallucination`]

JSON Schema (2022). *JSON Schema Draft 2020-12*. JSON Schema specification. https://json-schema.org/draft/2020-12 [`jsonschema2022draft`]

Kalajdzievski, D. (2023). *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*. arXiv:2312.03732. https://arxiv.org/abs/2312.03732 [`kalajdzievski2023rslora`]

Kim, Y., & Heer, J. (2018). *Assessing Effects of Task and Data Distribution on the Effectiveness of Visual Encodings*. Computer Graphics Forum, 37(3), 157–167. https://doi.org/10.1111/cgf.13409 [`kim2018taskdata`]

Lester, B., Al-Rfou, R., & Constant, N. (2021). *The Power of Scale for Parameter-Efficient Prompt Tuning*. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 3045–3059. https://doi.org/10.18653/v1/2021.emnlp-main.243 [`lester2021prompttuning`]

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems, 33, 9459–9474. https://arxiv.org/abs/2005.11401 [`lewis2020rag`]

Li, X. L., & Liang, P. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 4582–4597. https://doi.org/10.18653/v1/2021.acl-long.353 [`li2021prefixtuning`]

Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., et al. (2022). *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning*. Advances in Neural Information Processing Systems, 35, 1950–1965. https://doi.org/10.52202/068431-0142 [`liu2022ia3`]

Liu, S., Wang, C., Yin, H., Molchanov, P., Wang, Y. F., Cheng, K., et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. Proceedings of the 41st International Conference on Machine Learning, 235, 32100–32121. https://proceedings.mlr.press/v235/liu24bn.html [`liu2024dora`]

Mackinlay, J. (1986). *Automating the Design of Graphical Presentations of Relational Information*. ACM Transactions on Graphics, 5(2), 110–141. https://doi.org/10.1145/22949.22950 [`mackinlay1986apt`]

Moritz, D., Wang, C., Nelson, G. L., Lin, H., Smith, A. M., Howe, B., et al. (2019). *Formalizing Visualization Design Knowledge as Constraints: Actionable and Extensible Models in Draco*. IEEE Transactions on Visualization and Computer Graphics, 25(1), 438–448. https://doi.org/10.1109/TVCG.2018.2865240 [`moritz2019draco`]

Munzner, T. (2014). *Visualization Analysis and Design*. CRC Press. https://doi.org/10.1201/b17511 [`munzner2014visualization`]

Saket, B., Endert, A., & Demiralp, C. (2019). *Task-Based Effectiveness of Basic Visualizations*. IEEE Transactions on Visualization and Computer Graphics, 25(7), 2505–2512. https://doi.org/10.1109/TVCG.2018.2829750 [`saket2019taskbased`]

Sarikaya, A., Correll, M., Bartram, L., Tory, M., & Fisher, D. (2019). *What Do We Talk About When We Talk About Dashboards?*. IEEE Transactions on Visualization and Computer Graphics, 25(1), 682–692. https://doi.org/10.1109/TVCG.2018.2864903 [`sarikaya2019dashboards`]

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems, 30, 5998–6008. https://arxiv.org/abs/1706.03762 [`vaswani2017attention`]

Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., et al. (2022). *Finetuned Language Models are Zero-Shot Learners*. International Conference on Learning Representations. https://arxiv.org/abs/2109.01652 [`wei2022flan`]

World Wide Web Consortium (2024). *Web Content Accessibility Guidelines (WCAG) 2.2*. W3C Recommendation. https://www.w3.org/TR/WCAG22/ [`wcag2024`]

Yigitbasioglu, O. M., & Velcu, O. (2012). *A Review of Dashboards in Performance Management: Implications for Design and Research*. International Journal of Accounting Information Systems, 13(1), 41–59. https://doi.org/10.1016/j.accinf.2011.08.002 [`yigitbasioglu2012dashboard`]

Zhang, Q., Chen, M., Bukharin, A., Karampatziakis, N., He, P., Cheng, Y., et al. (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*. International Conference on Learning Representations. https://arxiv.org/abs/2303.10512 [`zhang2023adalora`]

Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., & Tian, Y. (2024). *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection*. Proceedings of the 41st International Conference on Machine Learning, 235, 61121–61143. https://proceedings.mlr.press/v235/zhao24s.html [`zhao2024galore`]
