# Chapter 4 — Dataset Construction and Provenance

## 4.1 Methodological role of the dataset

The objective of this thesis is to generate **structured dashboard-design recommendations** from a textual dashboard brief containing user goals, key performance indicators (KPIs), data fields, and analytical constraints. The generated output is not a rendered dashboard or a chart image. Instead, it is a machine-readable design specification containing a context summary, a KPI-to-chart mapping, layout guidance, styling and accessibility guidance, interaction suggestions, and short design rationales. This task therefore differs from classical chart question answering, chart-image understanding, and pure natural-language-to-visualization (NL2VIS) generation. The dataset had to support both a structured prediction target and a defensible evaluation boundary for four downstream systems: prompt-only generation, retrieval-augmented generation (RAG), parameter-efficient fine-tuning, and fine-tuning combined with retrieval. The use of LoRA and QLoRA in the downstream experiments further increased the importance of a compact but high-quality supervision corpus because parameter-efficient fine-tuning preserves most pretrained parameters and learns a comparatively small set of task-specific parameters \cite{hu2022lora,dettmers2023qlora}. RAG, in turn, separates parametric adaptation from access to external knowledge, so the training data and the retrieval knowledge base need to remain conceptually distinct \cite{lewis2020rag}.

Dashboard design literature additionally emphasizes information hierarchy, readability, and concise communication of key indicators as practical design concerns \cite{few2006dashboard}. The core methodological challenge was that no inspected public dataset directly contained all fields required by the thesis schema. Visualization research has long emphasized that an effective visual design depends on the analytical task, the semantics and types of the underlying data, and the perceptual properties of the chosen graphical encoding \cite{cleveland1984graphical,mackinlay1986apt,munzner2014visualization,brehmer2013typology}. Empirical studies further show that visualization effectiveness varies by task and data distribution, which makes a fixed chart lookup table an insufficient scientific basis for a general dashboard-design corpus \cite{saket2019taskbased,kim2018taskdata}. Consequently, dataset construction was treated as a first-class research method rather than as a preprocessing detail.

The final data methodology followed five principles. First, source-backed analytical content had to remain traceable to its public origin. Second, fields that were deterministically derived from the source had to be explicitly distinguished from fields that were directly observed. Third, LLM-generated presentation content had to be labelled as generated rather than as human or expert gold. Fourth, train, validation, test, and human-evaluation boundaries had to be protected against exact and near-duplicate leakage. Fifth, every frozen release had to be reproducible through manifests, hashes, validation reports, and explicit lineage metadata. These principles are consistent with recommendations for transparent dataset documentation such as Datasheets for Datasets and Data Cards, which emphasize provenance, collection and transformation processes, intended use, limitations, and dataset evolution \cite{gebru2021datasheets,pushkarna2022datacards}.

The authoritative dataset used by the current experiment configuration is `dashboard_v4`, with the exact frozen manifest revision `dashboard_v4_1`. According to the frozen project artifacts, the package contains 2,932 training records, 613 validation records, 274 held-out test records, and a separate 40-item human-evaluation artifact. The 3,819 modeling records consist of 1,819 preserved nvBench-derived records and 2,000 AI-generated train/validation records. The test set and the human-evaluation item list remain derived exclusively from the source-grounded nvBench lineage and were not extended with generated records.

## 4.2 Requirements for a suitable source corpus

A source dataset was considered suitable only if it could contribute reliable supervision to the transformation

\[
\text{dashboard brief} \rightarrow \text{structured dashboard recommendation}.
\]

For the purposes of this thesis, a useful source record should provide as much as possible of the following information: (1) a natural-language analytical intent; (2) a database or table schema; (3) explicitly identifiable dimensions and measures; (4) aggregation semantics; (5) filter, sorting, grouping, or temporal constraints where present; (6) a chart type or visualization specification; and (7) sufficient provenance to reconstruct how the record was obtained. The source did **not** need to provide layout, styling, interactions, personas, or rationales, because these fields could be added later under a clearly separated generated lineage. However, the analytical fields underlying a recommendation were required to be source-grounded or deterministically derivable.

This requirement follows established visualization principles. Mackinlay's formulation of automated graphical presentation distinguishes expressiveness from effectiveness: a visualization must first represent the intended information correctly before it can be optimized for perceptual effectiveness \cite{mackinlay1986apt}. Cleveland and McGill similarly show that graphical encodings differ in perceptual accuracy, which motivates validating chart suitability rather than accepting any syntactically valid chart label \cite{cleveland1984graphical}. Brehmer and Munzner distinguish the _why_, _how_, and _what_ of visualization tasks, making it important to preserve analytical intent separately from the concrete visual representation \cite{brehmer2013typology}. Draco formalizes a related principle computationally by representing visualization design knowledge as constraints, illustrating why design validity should be checked as a set of conditions rather than reduced to JSON validity alone \cite{moritz2019draco}.

### 4.2.1 Chart-understanding datasets considered but not selected as the primary source

Several widely used chart datasets were considered because they contain rich visual or numerical supervision. However, their task direction does not match the thesis problem.

**FigureQA** contains more than one million question-answer pairs grounded in over 100,000 synthetic chart images. Its purpose is visual reasoning over an already rendered figure, including questions about maxima, minima, intersections, and other relationships between plot elements \cite{kahou2018figureqa}. **DVQA** similarly frames bar-chart understanding as visual question answering and evaluates whether a model can extract numeric and semantic information from chart images \cite{kafle2018dvqa}. **PlotQA** expands this line of work to scientific plots and complex numerical reasoning, but it still starts from a plot image and asks the model to infer an answer \cite{methani2020plotqa}. **ChartQA** combines visual chart features and underlying data tables to answer human-written and generated questions involving visual and logical reasoning \cite{masry2022chartqa}.

These datasets are scientifically valuable for chart comprehension, but their primary transformation can be summarized as

\[
\text{existing visualization} + \text{question} \rightarrow \text{answer},
\]

whereas the thesis requires

\[
\text{analytical intent} + \text{data context} \rightarrow \text{visual design recommendation}.
\]

The mismatch is therefore not simply that these datasets contain images. More importantly, their supervision assumes that the visualization already exists. They do not normally provide a source-backed record linking a free-form analytical request to a choice of chart type, aggregation, database fields, grouping, sorting, and constraints in a way that can be transformed directly into the target `GoldItem` schema. For that reason, FigureQA, DVQA, PlotQA, and ChartQA were not used as the principal construction source. Their role in the literature review is to characterize adjacent chart-understanding research rather than the training lineage of the final dataset.

### 4.2.2 Natural-language visualization datasets

Datasets and systems for natural-language visualization are closer to the thesis task. Data2Vis demonstrated that visualization specifications can be learned as a sequence-to-sequence translation problem from data descriptions to Vega-Lite-like outputs \cite{dibia2019data2vis}. NL4DV provides a toolkit that converts natural-language queries into analytic specifications by detecting attributes, tasks, and visualizations \cite{narechania2021nl4dv}. Quda focuses specifically on analytical intent: it contains 14,035 free-form queries annotated with one or more analytical tasks and was designed to improve task recognition for visualization-oriented natural-language interfaces \cite{fu2020quda}. These works support the central premise that natural-language analytical intent can be transformed into structured visualization semantics.

Quda was nevertheless not chosen as the main construction source. It is strong for task intent and paraphrase diversity, but it does not provide, for every record, the complete combination of database schema, SQL/VQL evidence, chart specification, aggregation, encoded fields, grouping, filters, and sorting needed by this thesis. Using Quda as the sole source would therefore have required deriving a large portion of the analytical target instead of preserving it from source evidence.

### 4.2.3 Why nvBench was selected

Among the inspected public sources, **nvBench** was the closest fit to the supervision problem. nvBench was introduced as a large-scale cross-domain benchmark for NL2VIS and contains 25,750 published natural-language/visualization pairs derived from 750 tables across 105 domains \cite{luo2021nvbenchdataset}. The benchmark was constructed by synthesizing NL2VIS examples from NL2SQL resources, including Spider, through an intermediate representation that connects SQL semantics with visualization specifications \cite{luo2021nvbenchsigmod,yu2018spider}. This construction is particularly useful for the thesis because a record can contain natural-language intent, database identity, SQL, VQL, chart information, axes, aggregate expressions, grouping, sorting, filtering, and temporal binning. These fields provide a substantially stronger provenance base than a chart label alone.

The selection of nvBench should not be interpreted as a claim that it is universally the best visualization dataset. Rather, it was the best fit **among the inspected sources for this specific structured supervision problem**. The source offers a bridge from natural language and relational data to visualization specifications, while preserving enough machine-readable analytical evidence to audit whether a transformed record still represents the original query.

The project also inspected **nvBench 2.0**, which extends NL2VIS evaluation to ambiguous queries and multiple valid visualizations \cite{luo2025nvbench2}. Its ambiguity-oriented design is highly relevant for future work because a dashboard request can admit multiple valid chart choices. However, in the final source lineage of `dashboard_v3` and `dashboard_v4`, nvBench 2.0 was not used. Its local manifest remained separate, and the final builder and freeze inputs refer to the original nvBench source. This distinction is important because examining a dataset during source selection does not make it part of the training provenance.

ChartGPT was also considered as related evidence for LLM-based chart generation from natural-language abstractions \cite{tian2025chartgpt}. It demonstrates the growing relevance of LLMs for visualization generation, but the current frozen corpus does not derive its source-grounded analytical records from ChartGPT. It therefore belongs to the methodological context, not to the provenance of the frozen source-backed examples.

## 4.3 Source registration and reproducibility

The raw nvBench source was registered before transformation. The project stored the downloaded repository archive under `data/raw_external/nvbench/` together with a `source_manifest.json`. The manifest records the repository URL, the downloaded reference, the archive name, the license evidence, the download metadata, and the archive SHA-256. The local nvBench archive is identified by the digest

`2c95244aca93aaca689fc954f8ae228c6c17fd47c81e1d7b265c4191cb012e4c`.

The upstream repository was downloaded from the `main` branch rather than from a pinned commit. This is a limitation for external reproducibility because an upstream branch can change over time. The local archive hash nevertheless makes the exact bytes used in the project identifiable. This approach follows the broader dataset-documentation principle that a dataset's origin and transformation history should be recorded explicitly rather than inferred retrospectively \cite{gebru2021datasheets,pushkarna2022datacards}.

The source manifest distinguishes top-level visualization objects from query-level records. The registered source contains 7,247 visualization objects and 25,762 natural-language query records, while the published nvBench paper reports 25,750 natural-language/visualization pairs \cite{luo2021nvbenchdataset}. The construction pipeline operates on natural-language query records because one visualization object can be associated with multiple natural-language formulations. This distinction prevents an incorrect assumption that source-object count and model-record count are interchangeable.

The SQLite databases distributed with nvBench are used as evidence during validation. They are extracted into a rebuildable cache. Cache preparation verifies the source archive hash and does not modify the raw source. Database access is read-only and is used to profile schemas, types, key information, uniqueness, cardinality, and selected query results. This database evidence became important during later quality checks, especially for distinguishing actual numeric measures from identifier-like numeric fields.

## 4.4 Source-faithful transformation into the thesis schema

### 4.4.1 Stable record and group identifiers

The nvBench builder traverses the source deterministically and emits one candidate per natural-language query. Two identifiers are preserved. `source_record_id` identifies the individual query record, while `source_group_id` identifies a related source family. The latter groups records derived from the same underlying visualization or closely related source specification. Splitting is performed at group level rather than only at row level. This design is critical because row-level deduplication alone can leave near-identical paraphrases or alternative query formulations distributed across train and test.

The decision is consistent with evidence from language-model data curation showing that duplicate and near-duplicate content can increase memorization and contaminate evaluation when overlapping examples occur across data partitions \cite{lee2022dedup}. In this thesis, deduplication was therefore treated as part of the validity methodology, not merely as storage optimization.

### 4.4.2 Chart mapping and task inference

The source chart vocabulary was normalized through a versioned mapping configuration. Bar, Pie, Line, Scatter, Stacked Bar, Grouping Line, and Grouping Scatter are mapped to the thesis chart vocabulary. Unsupported labels are rejected instead of being silently coerced. Importantly, the analytical task type is **not** treated as an original nvBench label. It is deterministically inferred from source evidence and recorded as derived lineage. For example, line charts are associated with trend tasks, scatter plots with correlation, pie charts with part-to-whole tasks, stacked bars with composition, and bars primarily with comparison.

This explicit separation between source labels and derived task abstractions is scientifically important. Visualization task taxonomies emphasize that analytical intent and concrete graphical form are distinct levels of description \cite{brehmer2013typology}. Empirical work also demonstrates that the same chart type can have different effectiveness depending on the analytical task \cite{saket2019taskbased,kim2018taskdata}. Therefore, a rule-derived task label is useful for supervision but must not be presented as if nvBench itself contained that task annotation.

### 4.4.3 Parsing SQL, VQL, and constraints

The transformation pipeline parses SQL and VQL to preserve analytical semantics. It extracts aggregate intent, selected fields, grouping, filters, ordering, limits, HAVING conditions, time functions, and VQL binning where they can be represented without ambiguity. Unsupported or ambiguous constructs fail closed. For example, unrepresentable `OR` conditions, ambiguous nested aggregates, malformed binning, and incompatible multi-sort expressions are rejected rather than approximated.

This conservative policy addresses a central risk of dataset transformation: a structurally valid target can still encode a **different** analytical problem from the original source. If a filter is silently dropped, an aggregation scope is changed, or a group field is omitted, the model is trained on a modified task. The thesis therefore distinguishes **technical validity** from **source fidelity**. A record is source-faithful only when the transformed brief and KPI/chart mapping preserve the analytical constraints required by the source SQL/VQL.

### 4.4.4 Canonical `GoldItem` contract

Every accepted record follows the Pydantic `GoldItem` contract. At the highest level, a record contains `item_id`, `brief`, and `recommendation`, with a split assignment. The brief contains users, goals, KPIs, columns, constraints, and extensible metadata for provenance and lineage. The recommendation contains six required concepts: `context_summary`, `kpi_chart_mapping`, `layout`, `styling`, `interactions`, and `rationales`.

The controlled task vocabulary contains nine task types: trend, comparison, composition, distribution, correlation, ranking, deviation, part-to-whole, and flow. The chart schema permits line, bar, stacked bar, grouped bar, area, pie, donut, scatter, heatmap, histogram, box plot, KPI card, table, gauge, sankey, treemap, and map. The fact that a chart type is allowed by the schema does not imply that it occurs in every dataset release; the schema is broader than the original nvBench-derived subset.

Validation is layered. JSON parsing verifies syntactic validity. Pydantic checks enforce types and controlled vocabularies. Completeness requires required fields to exist **and** be non-empty. Semantic validation verifies that KPIs and encodings refer to fields represented in the brief. This distinction follows the general principle that structured output validity is not equivalent to semantic validity: a syntactically valid structured object can still express an incorrect mapping.

## 4.5 Progressive pilot studies

The final construction rules were not designed in one pass. They were developed through a sequence of pilots that exposed failure modes and motivated progressively stricter admission criteria. This failure-mode-oriented strategy is also consistent with behavioral-testing approaches in NLP, which argue that aggregate accuracy alone can conceal systematic capability failures \cite{ribeiro2020checklist}. This iterative procedure is scientifically preferable to selecting a large corpus first and defining quality criteria only after observing downstream model performance.

### 4.5.1 Pilots v1 and v2: structural success was not enough

The first two nvBench pilots read 25,762 query-level candidates and selected 100 records with an approximately balanced distribution across five normalized chart types. The initial structural reports appeared positive, but detailed inspection showed unresolved grouping, insufficient axis typing evidence, and near-duplicate relationships. Between v1 and v2, grouping recovery improved and the number of reported near-duplicate pairs decreased. These changes demonstrated that a record could pass JSON and schema checks while still losing analytically important source information.

### 4.5.2 Pilot v3: fail-closed source preservation

Pilot v3 introduced stricter source preservation. It reported 20,986 technically accepted candidates and 4,776 rejected candidates. The documented rejection categories included 3,944 records with unpreserved filters, 157 ambiguous nested aggregates, 260 categorical scatter axes, 236 aggregate-intent conflicts, 44 chart/query conflicts, 10 unpreserved sorts, and 125 missing group fields. The stricter representation removed aggregate expressions from physical-column lists, restored group fields for stacked bars, eliminated categorical scatter axes, and began preserving filters, sorting, grouping, and time-grain evidence.

The methodological lesson from v3 is that source fidelity must be validated at the level of analytical semantics. A data transformation that merely produces a valid schema is insufficient if the resulting record no longer corresponds to the query that generated it.

## 4.6 Separating source fidelity from dashboard-design suitability

Even the stricter v3 source transformation did not guarantee that every source-faithful record was a good **positive dashboard-design example**. nvBench is an NL2VIS benchmark, not an expert-curated dashboard-design corpus. The project therefore introduced a separate quality layer that evaluates dashboard suitability after source fidelity has been established.

This distinction is motivated by visualization research. Graphical encodings differ in perceptual properties \cite{cleveland1984graphical}; visualization tasks should be considered explicitly \cite{brehmer2013typology}; and empirical effectiveness depends on both task and data characteristics \cite{saket2019taskbased,kim2018taskdata}. Constraint-based systems such as Draco similarly show that an admissible design can be represented through explicit design rules and constraints \cite{moritz2019draco}. The project-specific quality layer operationalizes these broader principles for the thesis schema.

### 4.6.1 Tier A, Tier B, and Tier C

Three quality tiers were introduced:

- **Tier A — high-confidence positive candidate.** The record is technically valid, source-faithful, has a meaningful KPI, has a chart considered suitable under the project rules, preserves required constraints, has no mandatory failure, and obtains a quality score of at least 90.
- **Tier B — diagnostic candidate.** The record may remain source-faithful but is ambiguous or unsuitable as a positive dashboard-design target. Examples include identifier-like measures, weak KPI evidence, uncertain chart suitability, or incomplete design evidence.
- **Tier C — reject.** The record contains a severe contradiction or cannot be transformed reliably.

The weighted quality score combines source fidelity (30 points), KPI validity (20), chart suitability (25), constraint completeness (15), and database-profile support (10). The score is an operational admission rule for this project, not a universal measure of visualization quality.

### 4.6.2 Identifier detection and KPI validity

A recurrent failure mode was treating identifiers as quantitative measures. The project therefore profiles primary keys, foreign keys, unique indexes, cardinality, uniqueness ratios, numeric ratios, ranges, and value patterns. A numeric field is not automatically a meaningful measure. SUM, AVG, MIN, or MAX over a strong identifier can be mathematically defined but analytically meaningless. Such cases are demoted rather than rewritten.

This rule is consistent with the visualization literature's distinction between data semantics and graphical representation: choosing an encoding requires more than a primitive data type \cite{mackinlay1986apt,munzner2014visualization}. It also prevents a fine-tuned model from learning source artifacts such as “numeric-looking ID implies quantitative KPI.”

### 4.6.3 Chart-suitability rules

Chart-specific checks were implemented as project rules grounded in broader visualization principles. Bar charts require a meaningful measure and are rejected when an identifier is used as the measure. Line charts require a meaningful ordered axis, normally temporal or otherwise naturally ordered. Scatter plots require two independently numeric, non-identifier axes. Pie charts are restricted to genuine part-to-whole cases, low-cardinality categorical dimensions, non-negative measure bases, and additive COUNT or SUM measures. AVG, MIN, and MAX are demoted for pie charts under reason code `pie_non_additive_kpi` because the resulting slices do not in general represent additive parts of one whole.

The exact thresholds used by the implementation—such as the category limit for pie charts—are **project-specific policies**, not claims that the visualization literature specifies the identical threshold. The scientific literature supports the broader reasoning that graphical effectiveness depends on the task, encoding, and data distribution \cite{cleveland1984graphical,saket2019taskbased,kim2018taskdata}. Keeping this distinction explicit avoids presenting engineering thresholds as universal visualization laws.

## 4.7 Quality-filtered pilot progression

Pilot v4 applied the quality layer to the 20,986-record technical pool. The pool contained 13,896 Tier-A, 7,071 Tier-B, and 19 Tier-C records. A 95-record pilot was selected entirely from Tier A, but the desired number of scatter examples could not be reached. Instead of filling the quota from Tier B, the pilot remained partial. This choice established the project's “quality over artificial balance” policy.

Pilot v5 tightened the quality rules further and yielded 13,044 Tier-A, 7,923 Tier-B, and 19 Tier-C records. The balanced target again failed because only ten high-confidence scatter groups survived all constraints. The failure was treated as scientifically meaningful rather than as a software error: it showed that the source does not supply enough high-confidence scatter cases to support a perfectly balanced sample under the adopted rules.

Pilot v6 therefore selected a deliberately unbalanced but quality-constrained 100-record sample: 23 bar, 23 line, 22 pie, 10 scatter, and 22 stacked-bar records. All selected items were Tier A and no Tier-B fallback was used. Exact goals and near duplicates were removed, and the validation report contained no duplicate IDs, duplicate briefs, near-duplicate leakage, KPI/SQL conflicts, identifier measures, missing temporal/group evidence, or invalid scatter cases.

An AI-assisted pre-audit of the 100-record pilot accepted 96 records, rejected three, and marked one as uncertain. The three rejected cases were pie charts using non-additive KPI semantics. These cases led to the explicit `pie_non_additive_kpi` rule. Importantly, the AI pre-audit was used as a diagnostic device. The actual rule change was grounded in source evidence, database inspection, and the adopted chart-suitability policy; the AI judgment itself was not treated as an expert gold label.

## 4.8 Construction of the source-grounded `dashboard_v3` corpus

### 4.8.1 Final Tier-A pool

The final quality-pool rebuild contained 21,244 technically valid records. Under quality-rule version `nvbench_quality_v6`, 12,147 records were Tier A, 9,064 Tier B, and 33 Tier C. The difference between this 21,244-record rebuild and the earlier 20,986-record v3 technical pool is preserved in the project history, but a complete row-level reconciliation for all 258 records is not available. The thesis therefore treats the change as a documented rebuild boundary rather than inventing a per-record explanation.

The large-dataset selector consumed Tier-A records only. A record was eligible only if it had a quality score of at least 90, had no mandatory failure, had a non-empty normalized goal, survived exact source-record deduplication, survived evaluation-leakage exclusion, survived normalized-goal and fingerprint checks, respected source-group and per-database caps, and passed the character 3-gram Jaccard near-duplicate threshold of 0.8. The selector never promoted Tier-B or Tier-C records solely to satisfy a quota.

### 4.8.2 Deduplication and group-aware splitting

Deduplication was performed at multiple levels: source record, normalized goal, brief fingerprint, and near-duplicate similarity. The near-duplicate gate uses character 3-gram Jaccard similarity with a threshold of 0.8. Although this threshold is a project-specific operational choice, the decision to remove duplicate and near-duplicate content is supported by language-model data-curation research showing that duplicates increase memorization and that train-test overlap can distort evaluation \cite{lee2022dedup}.

The final split is group-disjoint. Whole `source_group_id` groups are assigned rather than individual rows, with deterministic seed 42. The resulting source-grounded corpus contains 1,819 records: 1,281 training records, 264 validation records, and 274 held-out test records. The three partitions contain 784, 167, and 167 unique source groups respectively. The observed proportions deviate slightly from nominal fractions because preserving group integrity took precedence over exact percentages.

The held-out 274-record test set is therefore best described as an **in-domain, source-group-disjoint held-out test set**. It is not a fully external benchmark because its lineage still originates from nvBench. This distinction is important when interpreting later model performance.

### 4.8.3 Final v3 distribution

The final 1,819-record source-grounded corpus contains 1,379 bar, 109 line, 242 pie, 13 scatter, and 76 stacked-bar records. The corresponding primary task distribution is 1,379 comparison, 109 trend, 242 part-to-whole, 13 correlation, and 76 composition. The imbalance reflects the surviving Tier-A supply after source-group, database, deduplication, and quality constraints. Scatter remained particularly scarce. The project explicitly preferred this evidence-limited distribution to a balanced corpus contaminated with lower-confidence examples.

This decision has consequences for evaluation. Aggregate accuracy alone can hide poor performance on rare classes, so downstream analysis must include macro-averaged and per-class metrics in addition to micro/overall performance. The imbalance is therefore documented as a property of the source-derived corpus rather than corrected invisibly.

## 4.9 Constrained LLM enrichment of missing dashboard fields

### 4.9.1 Why enrichment was required

nvBench provides a strong analytical core but not a complete dashboard-design specification. In particular, it does not authoritatively provide the six presentation-oriented fields required by the thesis output schema: `users`, `context_summary`, `layout`, `styling`, `interactions`, and `rationales`. Treating template-filled values for these fields as source gold would misrepresent the provenance of the dataset.

The project therefore used a constrained LLM-enrichment step. This approach is related to LLM-assisted instruction-data generation such as Self-Instruct, which generates candidate supervision and filters invalid or overly similar examples before training \cite{wang2023selfinstruct}. However, the thesis adopts a stricter provenance boundary: source-backed analytical fields are immutable, and only the six presentation fields may be generated. The generated fields are labelled `llm_generated` and `not_gold=true`.

### 4.9.2 Immutable source fields

The enrichment system is allowed to modify only:

1. `users`,
2. `context_summary`,
3. `layout`,
4. `styling`,
5. `interactions`, and
6. `rationales`.

The following content remains immutable: item identifiers, source record and group identifiers, original goal/query, KPI and aggregation, columns and data types, task and chart mapping, alternatives, encoding, filters, sort, limits, grouping, time grain, source evidence, quality score/tier, provenance, and split. The pipeline computes an immutable projection and SHA-256 fingerprint before and after enrichment; an analytical mutation is rejected.

This design addresses a core risk of synthetic supervision: fluent generated text can accidentally overwrite or reinterpret the source facts it is supposed to explain. By allowing the model to generate only the presentation layer, the dataset keeps a traceable boundary between source-backed semantics and generated dashboard advice.

### 4.9.3 Enrichment model and pilots

The enrichment provider used the OpenAI-compatible model `deepseek-v4-flash-sovereign` with requested reasoning effort `xhigh` and temperature 0.0. A ten-record technical sample produced 10/10 accepted outputs. A 30-record pilot produced 29 accepted and one rejected record. The rejected pilot case involved disagreement between a rationale and the task/chart/encoding evidence.

The full enrichment run processed only the 1,545 source-grounded training and validation records (1,281 train + 264 validation). No test record and no human-evaluation item was processed. The first pass accepted 1,534 records and rejected 11, while maintaining a 100% schema-valid rate and zero immutable-field violations. The run recorded 943 API calls, 617 cache hits, and 15 retries. Ten of the 11 initially rejected records were subsequently resolved by offline revalidation after a narrow validator correction, while the final unresolved item required one targeted API retry. The reconciled result contained all 1,545 expected records with zero permanent rejections.

A 30-record R1 human enrichment audit accepted 29 records and rejected one, exceeding the configured minimum of 27 accepted records. This is best described as a **small human-reviewed quality gate**, not as full human annotation of the dataset. Human-evaluation methodology literature recommends making the scope, rater procedure, and limitations of human judgments explicit \cite{vanderlee2019human}. Accordingly, the thesis does not claim that the six enrichment fields are expert gold.

## 4.10 Freezing `dashboard_v3`

The enriched source-grounded dataset was frozen as `dashboard_v3`. Its frozen train and validation files contain 1,281 and 264 nvBench-derived records with LLM-generated presentation fields. The 274-record test file remained source-grounded and un-enriched. A separate 40-item human-evaluation file was deterministically sampled from the held-out test and is not trainable.

The freeze records source and output hashes, schema version, enrichment specification, split membership, quality-rule version, and lineage. It excludes raw API responses, secrets, `.env` files, provider caches, temporary generations, and other non-release artifacts. The train/validation/test source groups are disjoint, and the test and human-evaluation artifacts are preserved for evaluation-only use.

This release discipline is aligned with the transparency goals of Datasheets and Data Cards: the frozen dataset is not only a collection of JSONL rows but a documented artifact with stated origin, transformations, intended use, and limitations \cite{gebru2021datasheets,pushkarna2022datacards}.

## 4.11 Motivation for the `dashboard_v4` augmentation

The source-grounded v3 corpus remained narrow for full dashboard-design supervision. In train and validation, only five main task/chart families were present, explicit grouping was relatively rare, and there were effectively no multi-KPI dashboard records in the canonical v3 training corpus. This is a natural limitation of using nvBench as the sole analytical source: nvBench was designed for NL2VIS rather than for multi-view dashboard composition \cite{luo2021nvbenchdataset,luo2021nvbenchsigmod}.

The current `dashboard_v4` augmentation was therefore designed to expand **coverage**, not to replace the source-grounded core. The goal was to add tasks and chart types that are structurally supported by the thesis schema, to increase filtered, grouped, and temporal cases, and to introduce multi-KPI dashboard layouts. The expansion also had to preserve the original v3 held-out test so that the benchmark boundary would not move together with the training data.

The decision to generate additional supervision rather than force unsupported transformations from external sources is consistent with the logic of model-assisted data generation: synthetic data can expand task coverage, but it requires explicit filtering, provenance, and separation from independent evaluation \cite{wang2023selfinstruct}. The resulting generated records are therefore labelled `source=llm_generated` and `not_gold=true`.

## 4.12 Controlled generation of 2,000 additional training/validation records

The v4 generator reads only the frozen v3 train and validation partitions as construction context. It does not read the v3 test file or the 40 human-evaluation items. The generation specification is recorded as `dashboard_v4-generation-v1`. The project artifacts record the generation model as `gpt-5.6-luna`, generation mode `codex_agent`, seed 42, batch size 40, and 50 batches.

The generation process attempted 2,915 candidates and accepted exactly 2,000. The remaining 915 candidates were rejected by the near-duplicate gate. Accepted generated records were divided into 1,651 training and 349 validation records. No generated record can be assigned to test or to the human-evaluation artifact.

Candidate validation checks the same structural principles used by the source-grounded schema and adds generation-specific constraints. A generated record must contain non-empty brief fields, a sufficiently detailed constraints field, a valid KPI mapping, permitted task/chart combinations, valid references to source columns, meaningful filters/sorts/grouping/limits, task-appropriate encodings, non-empty layout/styling/interactions/rationales, and a rationale that refers to the actual task and chart. The generation layer uses a controlled task/chart catalogue rather than allowing arbitrary chart names.

The duplicate gate checks item IDs, normalized goals, brief fingerprints, exact record hashes, scenario signatures, and character 3-gram Jaccard similarity against both the v3 base and previously accepted generated records. Candidates at or above similarity 0.8 are rejected. This filtering is methodologically consistent with evidence that duplicate supervision can amplify memorization and compromise train-test independence \cite{lee2022dedup}.

The 2,000 accepted generated records broaden the training distribution. Project reports record 1,215 generated records with explicit filters, 1,373 with grouping, 1,229 with temporal information, and 842 with multiple KPI mappings. The generated coverage spans nine task families and 14 chart types. These numbers describe **controlled generation coverage**, not the natural frequency of dashboard types in the real world.

## 4.13 Semantic repair of generated presentation fields

The first v4 generation pass passed structural admission checks but was not accepted as semantically clean presentation supervision. A dedicated audit found presentation-layer problems in all 2,000 generated records. Common issues included generic user descriptions, generic layouts, generic palettes, insufficiently anchored context summaries, interactions that lacked a purpose, interactions referring to fields not present in a record, chart-principle mismatches, and unsupported status-color semantics.

This result illustrates an important methodological point: schema-valid generated data can still be semantically weak. Similar concerns motivate filtering in self-generated instruction pipelines such as Self-Instruct \cite{wang2023selfinstruct}. In the context of visualization, the problem is particularly important because a structurally valid chart recommendation may still violate task, data, or perceptual considerations \cite{mackinlay1986apt,cleveland1984graphical,moritz2019draco}.

A separate v4.1 semantic-repair workflow therefore revised exactly six presentation fields: `users`, `context_summary`, `layout`, `styling`, `interactions`, and `rationales`. It protected `goals`, `kpis`, `columns`, `constraints`, `task_type`, `chart_type`, and `encoding`. The repair model is recorded as `gpt-5.6-luna` with repair mode `codex_agent_context_aware`. The `users` field was changed in 1,819 records; the other five presentation fields were repaired in all 2,000 records.

The post-repair audit checks semantic anchoring field by field. Context summaries must agree with the brief. Layout blocks must correspond to KPI mappings and state reading order. Styling must state typography, contrast, accessibility, and semantic color policy. Interactions are limited to approved types and must reference fields that exist in the record. Rationales must mention the actual KPI, task, and chart and must avoid unsupported outcome claims. The final repository audit reports 2,000 generated records passing the repaired-field checks with zero remaining reported issues.

The phrase **semantically clean** in the manifest must be interpreted narrowly: it means that the records pass the explicit repository audit. It does **not** mean that independent visualization experts certified all 2,000 examples. This distinction is necessary because the repair remains model-assisted.

## 4.14 Final `dashboard_v4_1` composition and lineage

The current operational dataset is `dashboard_v4`; the exact frozen manifest revision is `dashboard_v4_1`. The final composition is shown in Table 4.1.

| Partition or artifact      | Preserved nvBench-derived v3 | AI-generated v4/v4.1 |     Total | Role                         |
| -------------------------- | ---------------------------: | -------------------: | --------: | ---------------------------- |
| Train                      |                        1,281 |                1,651 | **2,932** | trainable                    |
| Validation                 |                          264 |                  349 |   **613** | tuning/model selection only  |
| Held-out test              |                          274 |                    0 |   **274** | evaluation only              |
| **Modeling total**         |                    **1,819** |            **2,000** | **3,819** | A/B/C/D study                |
| Human-evaluation item file |              40 test-derived |                    0 |    **40** | separate evaluation artifact |
| **Train + validation**     |                    **1,545** |            **2,000** | **3,545** | training/tuning input        |

The final dataset is therefore **mixed-lineage by design**. The 1,819 preserved records retain nvBench source provenance. Within train and validation, their analytical content remains source-grounded while the six presentation fields are LLM-generated by the v3 enrichment workflow. The 2,000 additional v4 records are fully marked as AI-generated supervision and are explicitly not nvBench observations, human gold, or expert gold. Their presentation fields were later repaired under the v4.1 workflow.

This lineage distinction is essential when interpreting model results. Improvements on the held-out 274-record test evaluate generalization to source-grounded nvBench-derived analytical tasks. They do not by themselves validate every generated-only task family introduced in v4, because the test set was intentionally preserved from v3. Claims about broader dashboard quality therefore require complementary human evaluation and, where possible, external evidence.

## 4.15 Leakage control and evaluation boundary

The dataset design enforces a strict evaluation boundary. The source-grounded split is group-disjoint. Test records are not used during v3 enrichment. The v4 generator does not read test or human-evaluation items. Generated records can only enter train or validation. The 40 human-evaluation items are derived from the held-out test but are maintained as a separate evaluation artifact and cannot be used for training.

This boundary was introduced partly in response to the limitations of the project's earlier synthetic-only workflow. In the earlier stage of the thesis, train and test examples were generated by the same deterministic task-to-chart logic, producing a circular evaluation risk: a fine-tuned model could learn the generator's rule rather than a general notion of dashboard quality. The later source-grounded pipeline addresses this weakness by anchoring the held-out test to public nvBench data and by separating generated training augmentation from the frozen evaluation set. This does not make the test fully external, but it materially reduces rule-level circularity relative to the earlier synthetic-only benchmark.

Exact duplicate checks, normalized-goal checks, source-group separation, fingerprint comparison, and near-duplicate similarity are all part of the freeze gates. Dataset-deduplication research provides general support for this practice because duplicate content can increase memorization and because overlap between training and evaluation sets can lead to overly optimistic estimates \cite{lee2022dedup}.

## 4.16 Human spot checks and the role of human evaluation

Human inspection was used at selected construction stages rather than to create a new fully human-labelled dashboard dataset. The nvBench construction process included manual review of stratified samples, while the v3 enrichment stage used a 30-record R1 audit with 29 accepted outputs. These checks serve as quality gates for the construction procedure; they do not transform generated fields into expert gold.

The final thesis evaluation includes a separate human-evaluation study of the four systems. Human evaluation is necessary because layout quality, styling, interaction usefulness, and rationale quality cannot be reduced reliably to source-backed chart labels. Best-practice work in NLG evaluation emphasizes clear rating criteria, multiple raters, reporting of agreement, and careful separation between automatic and human evidence \cite{vanderlee2019human}. The project therefore maintains the 40-item human-evaluation artifact outside the trainable dataset. At the dataset-construction stage, these items represent an **evaluation boundary**, not human-quality results. No preference or inter-rater reliability claim should be made until ratings are actually collected.

## 4.17 Reproducibility and dataset documentation

The final package is designed to be reproducible at the artifact level. The frozen directory contains a manifest, dataset card, schema, hashes, distribution reports, leakage reports, duplicate reports, validation reports, and generation/repair metadata. Material changes create a new dataset version rather than mutating an existing frozen release. The freeze verifies file hashes and preserves the held-out test and human-evaluation artifacts byte-for-byte across the v3-to-v4 transition.

This documentation strategy directly follows the rationale of Datasheets for Datasets and Data Cards: downstream users should be able to identify where a dataset came from, what transformations were performed, what its intended use is, and what its limitations are \cite{gebru2021datasheets,pushkarna2022datacards}. For this thesis, documentation is particularly important because a single `GoldItem` can combine three provenance classes: source-backed values, deterministic derivations, and model-generated annotations.

The final release therefore records not only the data files but also the **lineage of fields**. This makes it possible to state, for example, that a chart type is source-backed, a task type may be deterministically derived, and a layout is LLM-generated. Without this separation, the final corpus could be incorrectly described as a fully human-annotated dashboard dataset, which would exceed the evidence available in the project.

## 4.18 Threats to validity of the dataset methodology

### 4.18.1 Construct validity

The source-backed portion of the corpus is derived from nvBench, which was designed for NL2VIS rather than for full dashboard design \cite{luo2021nvbenchdataset,luo2021nvbenchsigmod}. Therefore, nvBench does not supply authoritative user personas, dashboard layout, styling, interaction design, or full design rationales. These components are model-generated and must be interpreted as supervision annotations rather than ground truth.

The quality score and chart rules are project-specific operational definitions. They are informed by visualization theory and empirical effectiveness studies \cite{cleveland1984graphical,mackinlay1986apt,saket2019taskbased,kim2018taskdata}, but exact thresholds such as the near-duplicate cut-off or maximum category count are engineering decisions. Passing these rules indicates compliance with the defined construction protocol, not proof of universally optimal dashboard design.

### 4.18.2 Internal validity

The original synthetic-only dataset created a rule-level circularity problem because training and testing were generated from the same deterministic chart-selection logic. The current pipeline reduces this risk by using source-grounded nvBench-derived test records and excluding all v4 generated records from test. Nevertheless, the test remains in-domain because train, validation, and test originate from the same upstream nvBench corpus. A fully external benchmark would provide stronger evidence.

LLM enrichment and v4/v4.1 generation can introduce model-specific stylistic patterns. Even at temperature 0, cloud model outputs are not guaranteed to be universally deterministic across provider revisions. The project mitigates this through stored outputs, hashes, prompts/specifications, metadata, and immutable source fields, but the generated content remains model-assisted.

### 4.18.3 External validity

The source-grounded test distribution is dominated by bar/comparison examples and contains very few scatter cases. The broader v4 training corpus introduces additional chart and task families that do not appear in the preserved test. Consequently, test accuracy cannot be interpreted as direct evidence that the model performs equally well on all newly generated task types. Human evaluation and future external benchmarks are required to assess broader dashboard usefulness.

The source corpus is also constrained by the domains and schemas represented in nvBench and ultimately by its NL2SQL origins in resources such as Spider \cite{yu2018spider,luo2021nvbenchsigmod}. Generalization to domains with substantially different data semantics, multi-table business logic, or visual conventions should therefore be treated cautiously.

### 4.18.4 Reproducibility limitations

The local nvBench source archive is hash-pinned, but the upstream repository reference was `main` rather than a specific commit. The exact local bytes can therefore be reproduced from the archived source if available, while independently downloading `main` at a later date may not return the same content. This limitation is stated explicitly instead of being hidden by the local hash.

The v4.1 repair is reproducible at the file, rule, and metadata level, but the semantic corrections were produced by a model. The final audit establishes compliance with the repository rules; it does not constitute independent expert validation of every generated dashboard recommendation.

## 4.19 Chapter summary

The dataset methodology evolved from a simple synthetic benchmark into a mixed-lineage, source-grounded, and explicitly versioned training corpus. The final process can be summarized as follows:

1. Candidate visualization datasets were compared by task direction, data modality, available analytical evidence, and compatibility with the target schema.
2. nvBench was selected as the primary source because it provides natural-language queries together with database, SQL/VQL, chart, aggregation, grouping, sorting, and encoding evidence \cite{luo2021nvbenchdataset,luo2021nvbenchsigmod}.
3. Raw source records were transformed conservatively, with stable identifiers, explicit provenance, and fail-closed parsing of unsupported constraints.
4. Technical validity was separated from source fidelity and from dashboard-design suitability.
5. A quality layer with Tier A/B/C admission, KPI/identifier checks, chart-specific rules, database profiling, deduplication, and leakage controls was developed through pilots v1–v6.
6. A 1,819-record source-grounded corpus was created and split group-safely into 1,281 train, 264 validation, and 274 held-out test records.
7. Only six missing presentation fields in the 1,545 train/validation source records were enriched by an LLM; source-backed analytical fields remained immutable.
8. The v3 package was frozen with manifests, hashes, validation reports, and a separate 40-item human-evaluation artifact.
9. To expand dashboard-design coverage, 2,000 additional AI-generated train/validation records were created without reading test or human-evaluation items, filtered for duplication, and subsequently repaired under a stricter semantic audit.
10. The current `dashboard_v4_1` revision contains 2,932 train, 613 validation, and 274 held-out test records, plus the separate 40-item human-evaluation file.

The central scientific property of the final dataset is therefore not that every field is human gold. Its strength is the **explicit separation of evidence types**: source-grounded analytical semantics, deterministic derivations, and generated presentation annotations are tracked independently. This makes the dataset suitable for controlled comparison of prompting, RAG, fine-tuning, and fine-tuning with RAG while preserving an evaluation boundary that is substantially less circular than the project's original synthetic-only benchmark.
