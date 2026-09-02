# Chapter 4 — Dataset Construction and Provenance

This chapter describes how the data for the dashboard-design recommendation task was constructed and how its provenance was preserved. The task is implemented as a structured transformation from a dashboard brief to a machine-readable recommendation. A brief contains information about the intended users, their goals, relevant key performance indicators (KPIs), available data columns, and analytical constraints. The target is not a rendered dashboard image and not only a chart label. It is a structured object that contains a context summary, KPI-to-chart mappings, encoding information, layout, styling, interactions, and rationales. The dataset must therefore preserve analytical evidence while also representing the presentation fields needed by the downstream models.

Provenance is central to this thesis because the four evaluated systems—prompt-only generation, retrieval-augmented generation (RAG), QLoRA fine-tuning, and QLoRA fine-tuning combined with RAG—must be compared on the same task and under the same data boundary. A dataset that silently mixes source observations, deterministic transformations, and model-generated annotations would make such a comparison difficult to interpret. The construction process consequently separates the source-grounded analytical core from derived labels and generated design annotations. It also protects the held-out test set from the later augmentation steps. These controls reduce the circularity risk that affected the earlier synthetic-only approach, although they do not make the final dataset fully human-annotated or externally independent.

The chapter first defines the dataset requirements and explains why the original nvBench corpus was selected as the primary source. It then describes the target schema, the evidence classes, and the source-faithful transformation of natural-language queries, relational schemas, SQL/VQL, and visualization encodings. The following sections explain the progressive quality checks, the Tier A/B/C policy, deduplication, group-aware splitting, and the construction of the source-grounded `dashboard_v3` corpus. The chapter then documents the constrained LLM enrichment of six presentation fields, the controlled `dashboard_v4` augmentation, and the `dashboard_v4_1` semantic repair. It closes by reporting the final mixed lineage, freeze and versioning procedure, and the threats to validity that bound the interpretation of the dataset.

## 4.1 Task Definition and Dataset Requirements

The dataset represents the supervised transformation

\[
\text{dashboard brief} \longrightarrow \text{structured dashboard recommendation}.
\]

The input side of this transformation is a compact dashboard brief. In the project schema, the brief contains `users`, `goals`, `kpis`, `columns`, and `constraints`. The user field describes the intended audience in a short textual form. The goals describe what that audience should understand, compare, monitor, or decide. KPIs identify the measures that are relevant to those goals. The columns field records the available data fields and their types or roles. Constraints describe conditions such as filtering, grouping, sorting, temporal granularity, refresh expectations, or restrictions on the interpretation of the data. These elements are not independent. For example, the same numerical field may support a different design for a comparison task than for a trend or distribution task, and a KPI may be unsuitable when the available field is an identifier rather than a meaningful measure.

The output is a structured recommendation with six required top-level concepts: `context_summary`, `kpi_chart_mapping`, `layout`, `styling`, `interactions`, and `rationales`. The mapping field connects one or more KPIs with an analytical task, a chart type, alternatives, and an encoding. The layout field describes how the views should be arranged and prioritized. Styling covers typography, contrast, palette, labels, and related accessibility decisions. Interactions describe possible actions such as filtering, sorting, tooltips, or selection. Rationales explain why a mapping or design decision is related to the goal and the available data. The output is therefore broader than a single chart recommendation, even when a record contains only one KPI mapping.

This distinction is important for the choice of a construction source. A rendered dashboard or chart image is not sufficient for the target task because an image does not directly expose the source query, aggregation, data field roles, or constraint scope. Conversely, a source query and a chart label are not sufficient to provide authoritative users, layout, styling, interactions, or rationales. The dataset was therefore designed as a layered object. Source evidence supports the analytical part of a record, while later stages add presentation-oriented fields under a separate lineage label. The final object can be consumed by software, but its schema is a project-specific contract rather than a universal definition of dashboard design.

The construction requirements followed from this task definition. First, every source-derived analytical choice had to remain traceable to the external corpus, its database schema, its query, or a deterministic transformation with a recorded rule. Second, the representation had to be stable enough that all four methods received comparable inputs and were evaluated against the same field names and controlled vocabularies. Third, unsupported or ambiguous source expressions had to fail closed. A transformed record should not silently lose a filter, a group, an aggregation, or a time grain, because such a change would create a different analytical task. Fourth, the data had to support a strict separation between trainable material, validation material, the held-out test, and the separate human-evaluation item list. Finally, the release had to be documented with manifests, hashes, validation reports, and field-level lineage so that the dataset could be inspected independently of the model results.

These requirements also explain why the dataset is not described simply as a collection of gold labels. The source-derived fields are evidence-backed, but some task labels are deterministic abstractions and some presentation fields were generated by an LLM. The generated material can provide useful supervision for learning the output format and for broadening task coverage, but it cannot be treated as expert annotation without independent evidence. This distinction is maintained throughout the chapter.

## 4.2 Source Corpus and Source Registration

### 4.2.1 Why nvBench was selected

The primary source was the original nvBench corpus. nvBench was introduced as a cross-domain natural-language-to-visualization (NL2VIS) benchmark containing 25,750 published natural-language/visualization pairs from 750 tables and 105 domains (Luo, Tang, & Li, 2021). Its construction connects natural-language visualization requests with natural-language-to-SQL resources, including Spider, so that a record can retain both linguistic intent and relational query evidence (Luo et al., 2021; Yu et al., 2018). This relationship was a close fit for the present dataset because the project needed more than a chart name. It needed a source query, fields, aggregation intent, grouping or filtering information, and a visualization encoding that could be inspected after transformation.

The original nvBench records provide several useful evidence layers. A natural-language query describes the analytical request. The associated database identifier links the record to a relational schema. SQL and VQL express the data operation and the intended visual form. The visualization object provides fields such as the chart label, x- and y-axis information, classification or series information, and a short description. Together, these elements make it possible to preserve the relation between an analytical goal, a data field, an aggregation, and a chart. The source also represents operations such as grouping, ordering, filtering, and temporal binning in a form that can be parsed or checked against the database.

nvBench was not used as complete dashboard gold. It is an NL2VIS benchmark, not an expert-curated corpus of multi-view dashboards. It does not provide authoritative personas for the intended users, complete dashboard layouts, styling decisions, interaction designs, or full design rationales for the target schema. These missing fields were not filled by pretending that they were present in the source. Instead, the project preserves the source-backed analytical content and adds the six presentation fields later under explicit LLM-generated lineage. This decision keeps the source contribution useful without overstating what nvBench can support.

Other resources were inspected during source selection, as discussed in Chapter 3. They cover related tasks such as analytic-task recognition, chart generation, or chart understanding, but they do not provide the same combination of natural-language intent, relational schema, SQL/VQL, and visualization evidence for the transformation used here. nvBench 2.0 was also inspected because it addresses ambiguity and multiple valid visualizations (Luo et al., 2025). It is relevant to future extensions, but it was not used in the final `dashboard_v3` or `dashboard_v4` lineage. Inspecting a dataset during source selection does not make it a training source.

The selection was therefore task-specific rather than a general claim that nvBench is the best visualization dataset. Its value for this thesis is that it provides a traceable analytical starting point. The dataset can be transformed conservatively while keeping the original query and visualization evidence available for validation. This property was more important for the present research question than the availability of rendered images or a large set of human ratings for a different task.

### 4.2.2 Registration, license, version, and source counts

The source was registered before the transformation stage in `data/raw_external/nvbench/source_manifest.json`. The manifest records the repository URL as `https://github.com/TsinghuaDatabaseGroup/nvBench`, the downloaded reference as `main`, and the local archive as `nvBench-main.zip`. No upstream commit was pinned. The exact local archive is identified by the SHA-256 digest `2c95244aca93aaca689fc954f8ae228c6c17fd47c81e1d7b265c4191cb012e4c`. The extracted README contains the MIT license statement, and the source manifest records the local license status as confirmed on that basis. The license record documents the evidence found in the project; it is not a substitute for checking the conditions of reuse when the data is redistributed.

The source manifest records 7,247 top-level visualization objects and 25,762 natural-language query records. These values must be distinguished from the 25,750 published natural-language/visualization pairs reported in the nvBench paper (Luo, Tang, & Li, 2021). The difference does not by itself indicate a failed download. A top-level visualization object can contain more than one natural-language query, while the published paper reports the benchmark pair count. The local query count is the count used by the extraction process, whereas the published pair count describes the benchmark as presented in the scholarly source. The chapter therefore reports both counts with their units instead of silently normalizing them.

The local archive also records the number of files and total bytes in the source manifest. These values identify the downloaded package, but the archive digest is the more useful integrity identifier for the transformation because it binds the local source to a specific byte sequence. The use of the `main` branch remains a reproducibility limitation: a future download of the branch may contain different bytes even though the repository URL is unchanged. The archive hash makes the bytes used in this project identifiable, but it does not replace a pinned upstream commit.

The database files distributed with nvBench were treated as validation evidence rather than as a second annotation source. A rebuildable cache was created from the registered archive. Cache preparation checks the source archive digest, protects extraction against path traversal, and leaves the raw archive unchanged. Database access is read-only. It is used to inspect schemas, field types, key information, uniqueness, cardinality, and selected query results. This distinction matters because database profiles can help decide whether a field is a meaningful measure, but they do not create a new ground-truth label when the source query is ambiguous.

## 4.3 Target Schema and Evidence Classes

### 4.3.1 Canonical `GoldItem` contract

Each modelable example is represented as a `GoldItem`. At the top level, the object contains an `item_id`, a `brief`, a `recommendation`, and an optional split assignment. The brief contains the fields described in Section 4.1 and an extensible `extra` object for provenance, lineage, quality, generation, and repair metadata. The recommendation contains the six required concepts `context_summary`, `kpi_chart_mapping`, `layout`, `styling`, `interactions`, and `rationales`. A KPI-to-chart mapping carries a KPI name, an enumerated `task_type`, an enumerated `chart_type`, possible alternatives, and an encoding dictionary. A rationale contains a claim and a principle. This structure allows the evaluation code to inspect the recommendation at field level rather than treating it as one undifferentiated text string.

The controlled task vocabulary contains nine values: `trend`, `comparison`, `composition`, `distribution`, `correlation`, `ranking`, `deviation`, `part_to_whole`, and `flow`. The chart vocabulary in the schema contains 17 values: `line`, `bar`, `stacked_bar`, `grouped_bar`, `area`, `pie`, `donut`, `scatter`, `heatmap`, `histogram`, `box`, `kpi_card`, `table`, `gauge`, `sankey`, `treemap`, and `map`. The schema is intentionally broader than the original nvBench-derived subset. A permitted value is not evidence that a chart or task occurs in every release. For example, the source-grounded v3 corpus contains five primary chart families, while the later generated augmentation uses a larger controlled vocabulary.

Validation was applied in layers. Strict JSONL reading first rejects malformed JSON and non-object rows. Pydantic models and the frozen JSON schema then check the declared field types and the required task and chart enumerations. A completeness check requires the required brief and recommendation fields to be present and non-empty. Semantic checks then inspect whether the selected fields, KPI mappings, encodings, constraints, and chart-specific requirements are mutually compatible. This order separates a parsing failure from a schema failure and from a semantic failure. A record may be valid JSON while still containing an unsupported chart value, an empty recommendation field, an unavailable column, or an invalid analytical mapping.

The contract permits additional metadata at defined extension points. This is useful for retaining raw source evidence, rule versions, quality scores, and generation metadata without changing the core recommendation interface. The extension fields do not weaken the validation boundary. Raw task and chart values are checked against the declared vocabularies, and the required recommendation concepts must still be present. The final dataset therefore has one stable interface for the four downstream systems while preserving the information needed to audit how each record was constructed.

### 4.3.2 Source-grounded, derived, and generated fields

The dataset uses three evidence classes. Source-grounded fields are values that can be traced to nvBench, its source query, its visualization object, or its database/query evidence. Deterministically derived fields are values produced by a documented project rule, such as the normalized chart vocabulary, the task inference, or the KPI-selection abstraction. LLM-generated fields are values produced or rewritten by a language model. The third class includes the six presentation fields in the enriched v3 training and validation records and the broader generated content in the v4 augmentation.

This classification is field-level rather than record-level. A single record may combine several classes. In the source-grounded v3 train and validation records, for example, the goals, KPIs, columns, constraints, provenance, and original chart/encoding evidence remain source-backed, while the task type and KPI selection are derived and the six presentation fields are LLM-generated. The v3 test retains source-grounded analytical lineage but was not sent through the enrichment workflow; its presentation fields remain the values stored in the held-out artifact. The v4 augmentation is different: its brief and analytical specification were generated by the project’s controlled generator, and its presentation fields were later repaired by an LLM. These generated records are not new nvBench observations.

The main field-level distinction is summarized in Table 4.1. The table describes provenance, not quality. A generated field can pass all repository checks and still not be human gold. Similarly, a source-backed field can preserve the original query while remaining subject to the limitations or ambiguity of the upstream corpus.

Table 4.1. Field-level provenance across the preserved v3 and generated v4/v4.1 record families.

| Field or field group | Preserved v3 train/validation | Preserved v3 test | Added v4/v4.1 records | Evidence interpretation |
| --- | --- | --- | --- | --- |
| Goals, KPIs, columns, constraints, and source provenance | Source-grounded | Source-grounded | LLM-generated within the v4 generator | Source evidence for v3; generated supervision for v4 |
| Original chart and encoding evidence | Source-provided | Source-provided | Generated and deterministically validated | A v3 source value is not equivalent to a v4 generated value |
| Task type and KPI-selection abstraction | Deterministically derived | Deterministically derived | Deterministically validated from generated content | Rule-based abstraction, not an original nvBench label |
| `users`, `context_summary`, `layout`, `styling`, `interactions`, and `rationales` | LLM-generated enrichment | Held-out artifact values; not enriched in this workflow | LLM-generated and repaired in v4.1 | Generated annotations, not expert gold |
| Item identifiers, split, and lineage metadata | Deterministically assigned or recorded | Deterministically assigned or recorded | Deterministically assigned or recorded | Release and audit metadata |

This separation follows the general principle that dataset documentation should describe origin, transformation, intended use, and limitations rather than presenting all fields as if they had the same status (Gebru et al., 2021; Pushkarna, Zaldivar, & Kjartansson, 2022). It is also necessary for interpreting later model results. A high score on a generated training target cannot be read as performance against an independent expert reference, and a source-grounded test result cannot be generalized automatically to task families introduced only through the v4 generator.

## 4.4 Source-Faithful Transformation of nvBench

The source transformation was designed to keep the meaning of the original record visible. The implementation reads the raw visualization/query structure, resolves database fields, parses SQL and VQL where possible, and stores the original evidence inside the record’s provenance object. Project-specific normalizations are added alongside that evidence. This avoids replacing the source with a cleaned text-only representation in which it would be impossible to see which values were observed and which were inferred.

### 4.4.1 Stable record and group identifiers

The builder traverses `NVBench.json` in sorted-key order and emits one candidate per natural-language query. This choice treats the query as the modeling unit while retaining the relation to the top-level visualization object. A source record is assigned a stable `source_record_id`, such as `nvbench:{key}:query:{query_index}` or the corresponding axis-qualified form used by the source. A separate `source_group_id` identifies related records derived from the same base visualization or source family. The group identifier is derived from the base key after removing axis-name or sort-specific suffixes where appropriate.

The distinction between record and group is important because one source object can generate several natural-language query records. Two queries may use different wording while referring to the same underlying visualization or analytical configuration. If they were split independently, a model could see a close source counterpart during training and evaluation. Group identifiers are therefore established before selection and split assignment. The group-safe split is complemented by exact and near-duplicate checks, but group assignment is the primary protection against related source records crossing the evaluation boundary.

### 4.4.2 Column and type resolution

The transformer keeps physical source columns separate from derived aggregate expressions. It resolves fields from the relational schema and profiles relevant database columns in read-only mode. The profile can contain a field’s observed type, row-level null and distinct ratios, uniqueness, numerical ratio, range, sign information, and representative values. The transformation also records whether an axis type comes directly from database metadata, from an aggregate expression, or from a documented fallback. These details are retained because a field’s storage type is not always the same as its analytical role.

This distinction is especially important for identifiers. A field may be stored as an integer and still represent a customer, account, invoice, or other entity identifier. Such a field can be used as a categorical dimension or for counting entities, but summing or averaging it is usually not a meaningful KPI. The pipeline therefore records physical fields in `brief.columns` and keeps expressions such as `COUNT(field)` or `SUM(field)` as aggregate evidence. An aggregate expression is not inserted into the physical column list merely because it appears in the SQL projection. This prevents the normalized brief from teaching the model that every numerical-looking token is a valid measure.

### 4.4.3 KPI and aggregation extraction

KPI extraction begins with the aggregate expressions represented in the source query and visualization evidence. When one aggregate axis is unambiguous, it is retained as the primary KPI. When two aggregate axes are present, the y-side aggregate is used as the primary KPI while both aggregate expressions remain represented in the provenance and encoding. If the source does not contain an explicit aggregate, the selector searches for a numeric, non-identifier-like measure using the available axis and database evidence. The source expression and the project’s normalized KPI representation are stored separately so that a later reviewer can distinguish extraction from normalization.

COUNT is handled as an intent-sensitive case. A count of rows, entities, or a categorical field may be a meaningful measure when the query expresses entity-count intent. SUM, AVG, MIN, and MAX applied to a strong identifier are technically executable but can be analytically unsuitable. Such cases receive explicit reason codes, such as `identifier_as_measure`, `meaningless_identifier_aggregation`, or `invalid_identifier_aggregation`, and are not silently repaired into a different KPI. If the source y-field is retained for fidelity but no meaningful KPI can be established, the record can remain diagnostic or be rejected by a later quality gate.

The goal is not to impose one universal KPI policy on all visualization data. It is to prevent a source transformation from converting a questionable expression into an apparently reliable label. The pipeline thus preserves evidence of the original aggregation, records the project interpretation, and lets the quality layer decide whether the record is suitable as a positive dashboard-design example.

### 4.4.4 Filters, grouping, sorting, limits, and time grain

The parser extracts filters, SQL `GROUP BY` terms, sorting, limits, `HAVING` conditions, time functions, and VQL `BIN` expressions when these constructs can be represented without ambiguity. SQL grouping fields are retained as constraints or visual grouping evidence. A time function or a grouped datetime x-field is represented with a time-grain value when the source supports that interpretation. Sorting is stored with its field, expression, and direction. Limits and `HAVING` conditions remain separate so that a row limit is not confused with an aggregation condition.

Unsupported expressions fail closed. Unrepresentable `OR` conditions, subqueries, malformed binning, ambiguous nested aggregates, and multiple or otherwise unsupported sort expressions are rejected instead of being approximated. This policy is conservative by design. Silently dropping a filter changes the population being analyzed. Dropping a group field changes the meaning of a stacked or grouped chart. Moving a limit before an aggregation changes the scope of the result. Similarly, omitting a time grain can change a trend from a monthly view to an unordered collection of dates. A record is accepted only when the normalized representation preserves the relevant source constraint or clearly records the limitation.

Natural-language text is used for narrow consistency checks rather than as authority to rewrite the SQL or VQL. Simple keyword matching can create false positives: a word such as “total” may be an intensifier, and “number of” may not identify the intended aggregation. The implementation therefore uses source fields, query structure, and database evidence as the primary basis for field and constraint decisions. The natural-language query remains available as the source expression of the user goal, but it is not used to invent a physical column that is absent from the source.

### 4.4.5 Chart, encoding, and task mapping

Source chart labels are normalized through the versioned mapping in `src/config/data/nvbench_mapping.yaml`. The mapping converts Bar, Pie, Line, Scatter, Stacked Bar, Grouping Line, and Grouping Scatter into the project vocabulary. Grouping variants retain the base chart type and preserve the series or classification information in the encoding and provenance. Unsupported labels are rejected rather than silently converted to a plausible chart.

The analytical task type is derived separately from the chart and query evidence. The rule configuration maps bars primarily to comparison, pies to part-to-whole, lines to trend, scatter plots to correlation, and stacked bars to composition. Grouped variants retain the grouping evidence and may support comparison across series. Each inferred task is recorded as rule-derived with a rule version, confidence, and short evidence statement. It is not presented as an original nvBench annotation. This distinction follows visualization task taxonomies that separate the purpose of an analysis from the means used to perform it (Brehmer & Munzner, 2013).

The source chart and encoding are therefore not treated as sufficient proof that the record is a good dashboard-design positive. Source fidelity asks whether the transformed record still represents the source query. Chart suitability asks a separate question: whether the source-supported fields and result shape satisfy the project’s criteria for a usable dashboard-design example. This separation is consistent with work showing that graphical effectiveness depends on both the analytical task and the data distribution (Saket, Endert, & Demiralp, 2019; Kim & Heer, 2018). Constraint-based visualization systems such as Draco provide a related methodological example by representing design knowledge as explicit constraints rather than relying on an unexplained chart label (Moritz et al., 2019).

## 4.5 Iterative Validation and Quality Filtering

The construction rules were developed through progressive validation rather than fixed after the first successful JSON export. Early pilots showed that structural acceptance could coexist with missing grouping evidence, invalid or ambiguous aggregates, and chart/query conflicts. Later pilots added source-preservation checks, database profiling, identifier detection, chart-specific rules, and a quality tier. This sequence matters because it records how the project discovered and addressed failure modes before building the final 1,819-record source-grounded corpus.

### 4.5.1 What pilots v1–v3 revealed

Pilots v1 and v2 used the 25,762 local query-level candidates and selected 100 examples with a roughly balanced distribution across five normalized chart types. Their manifests reported zero technical rejections at the initial parsing stage. That result was useful for checking the shape of the corpus, but it did not establish dashboard-design validity. The associated reports recorded unresolved grouping information, fallback axis typing, and near-duplicate relationships. Between v1 and v2, 19 common records changed, nine grouping fields were recovered, and the reported near-duplicate count decreased from 38 to 19. These changes affected the analytical interpretation of grouped records rather than only their formatting.

Pilot v3 introduced a stricter source-preservation representation. It accepted 20,986 candidates technically and rejected 4,776. The documented rejection categories were 3,944 records with unpreserved filters, 157 ambiguous nested aggregates, 260 categorical scatter axes, 236 aggregate-intent conflicts, 44 chart/query conflicts, 10 unpreserved sorts, and 125 missing group fields. The category counts sum to the reported 4,776 rejections. The selected 100-record pilot was created only after these checks, exact normalized-goal deduplication, source-group sampling, database caps, and near-duplicate checks.

The v2-to-v3 comparison shows why the additional validation was necessary. Aggregate expressions in the physical-column list fell from 86 to zero, valid stacked-bar group fields increased from zero to 20, and categorical-scatter warnings fell from eight to zero in the compared pilot records. Accepted filters increased from zero to 15, accepted sorts from zero to 43, accepted time grains from zero to 18, and grouped records from nine to 33. These changes indicate that the transformation began preserving source semantics that the first representation had either omitted or represented ambiguously. A valid JSON object was therefore treated as a necessary condition, not as the final quality criterion.

The pilots were methodological development stages rather than model experiments. Their counts describe source extraction and selection behaviour. They do not measure the performance of prompt-only generation, RAG, QLoRA, or the combined method. Keeping these roles separate avoids using construction diagnostics as if they were downstream experimental results. The failure-mode approach also follows the reasoning behind behavioural testing, where systematic checks are used to reveal errors that aggregate success rates can hide (Ribeiro et al., 2020).

### 4.5.2 Source fidelity versus dashboard suitability

The quality pipeline separates three gates. Technical validity asks whether the record can be parsed, conforms to the `GoldItem` contract, contains the required fields, and uses valid controlled values. Source fidelity asks whether the normalized brief and recommendation preserve the analytical meaning of the original source query, including fields, aggregations, filters, grouping, sorting, limits, and temporal information. Dashboard-design suitability asks whether the source-faithful record is appropriate as a positive example for the specific recommendation task.

These gates answer different questions. A record can be technically valid but lose a source filter. It can preserve the source query but use an identifier as a KPI. It can contain a meaningful KPI and still fail a chart-specific requirement, for example by using an unordered categorical axis for a line chart or by using non-additive aggregation for a part-to-whole chart. The project does not correct such differences by silently changing the source. It records them as failures, warnings, or lower quality tiers so that the source record and its suitability remain distinguishable.

The distinction is also important for the interpretation of the final dataset. Tier A means that a record passed the project’s adopted suitability criteria. It does not mean that the record is universally optimal or that an expert has judged every design choice. The quality layer operationalizes the thesis task using explicit rules informed by visualization research, but it remains a project-specific admission procedure.

### 4.5.3 Identifier, KPI, and chart-suitability checks

Identifier detection combines database and field evidence. The implementation considers primary-key, foreign-key, and unique-index metadata, observed uniqueness and distinctness, numeric-looking entity-reference patterns, and name patterns such as `id`, `key`, `code`, or identity-qualified number fields. The configured strong-identifier conditions use a unique ratio of at least 0.98 and at least 20 distinct values, while weaker evidence can be marked ambiguous. The exact name patterns and thresholds are stored in `src/config/data/nvbench_quality_rules.yaml`; they are operational rules for this corpus, not definitions of identifiers for every database.

KPI checks use the identifier result together with the query aggregation and natural-language intent. A strong identifier used as a SUM, AVG, MIN, or MAX measure is not treated as a meaningful KPI. COUNT can remain valid when the query expresses a count of entities or rows. Aggregation conflicts are evaluated in an axis-aware way. A conflict between the encoded aggregate and the SQL or query intent blocks Tier A even when the database can execute the expression. This prevents the system from equating computational executability with analytical validity.

Chart suitability is checked with chart-specific conditions. Bar charts require meaningful measure evidence and are blocked when the principal measure is a strong identifier. Line charts require a meaningful measure and an ordered x-axis, supported by temporal information or another source-backed ordering. Pie charts are limited to at most eight categories, disallow negative values and identifier-like categories, and use additive COUNT or SUM aggregation for Tier A. AVG, MIN, and MAX are retained as source evidence when present but are demoted under the `pie_non_additive_kpi` policy. Stacked bars require a bounded and unambiguous grouping field.

Scatter checks use the result profile in addition to base-table metadata. The configured minimum is ten distinct values for relevant axes, and the profile is capped at 1,000 rows. Identifier axes are rejected, and the two axes must provide independently numeric observations with sufficient variation. A large underlying table is not enough if the executed query produces only a few paired observations or collapses the variation through aggregation. These rules reflect the general need to consider task, data semantics, and encoding together (Cleveland & McGill, 1984; Mackinlay, 1986; Brehmer & Munzner, 2013), while the exact thresholds remain project-specific engineering decisions.

### 4.5.4 Tier A/B/C policy and pilots v4–v6

The quality layer assigns three tiers. Tier A is a high-confidence positive candidate that is technically valid, source-faithful, suitable under the project rules, free of mandatory failures, and above the configured quality threshold. Tier B is source-faithful or potentially useful material that is uncertain or unsuitable as a positive dashboard-design target but may remain useful for diagnostics. Tier C represents a severe contradiction or a transformation that cannot be relied upon. The tiers do not claim human annotation quality; they describe the project’s operational confidence in using a record for positive supervision.

The quality score has five components. Source fidelity contributes 30 points, KPI validity 20, chart suitability 25, constraint completeness 15, and database-profile support 10. Tier A requires a score of at least 90 and no mandatory failure. The score is therefore a weighted admission mechanism, not an interval scale with a universal interpretation. A score of 95 does not mean that one record is universally 5 points better than another, and a score of 90 does not certify expert-level dashboard quality.

Pilot v4 applied the first quality version to the 20,986-record technical pool. It reported 13,896 Tier A, 7,071 Tier B, and 19 Tier C records. The selected pilot contained 95 records because the desired scatter quota could not be reached under the Tier-A and deduplication constraints. No Tier-B records were inserted to make the chart counts appear balanced. Pilot v5 applied a revised rule set and reported 13,044 Tier A, 7,923 Tier B, and 19 Tier C records. Its requested balanced selection was also not possible because only ten high-confidence scatter examples survived the stricter gates.

Pilot v6 retained the quality-first policy and selected 100 records with 23 bar, 23 line, 22 pie, 22 stacked-bar, and 10 scatter examples. All selected records were Tier A, and the manifest records that no Tier-B fallback was used. The resulting imbalance was accepted because replacing scarce high-confidence scatter examples with lower-confidence records would have weakened the meaning of the positive dataset. This decision follows the principle that a controlled dataset should report a real coverage limitation instead of hiding it through artificial balance.

The later full quality-pool rebuild is a separate construction boundary. It contains 21,244 technically valid records under the final `nvbench_quality_v6` rules, with 12,147 Tier A, 9,064 Tier B, and 33 Tier C records. The earlier strict v3 report contains 20,986 technically accepted records and 4,776 rejected records. The repository does not provide a complete row-level reconciliation for the difference of 258 records. The final corpus therefore uses the later quality-pool rebuild as its source, while the 20,986-to-21,244 change is reported as an unreconciled rebuild boundary rather than as an invented sequence of individual additions and removals.

## 4.6 Selection, Deduplication, and Evaluation Isolation

### 4.6.1 Final source-grounded corpus

The final large-dataset selector consumes Tier-A records only. It does not promote Tier B or Tier C records simply to satisfy a target size. Eligibility requires a quality score of at least 90, no mandatory failure, a non-empty normalized goal, valid source-record identity, and survival of exact source-record, normalized-goal, fingerprint, group, database, leakage, and near-duplicate checks. The selector considers the available scatter records first because scatter is the scarce source chart family. The remaining allocation uses a deterministic target policy and redistributes deficits only when a chart pool is exhausted. These constraints define how many high-confidence source records can be retained; they do not relax the quality rules.

The resulting `dashboard_v3` source-grounded corpus contains 1,819 modeling records. Its chart distribution is 1,379 bar, 109 line, 242 pie, 13 scatter, and 76 stacked-bar records. The corresponding rule-derived task distribution is 1,379 comparison, 109 trend, 242 part-to-whole, 13 correlation, and 76 composition records. This is an imbalanced distribution, especially for scatter. It reflects the surviving Tier-A supply after source-group, database, deduplication, and leakage controls. The imbalance is retained as a documented property of the evidence rather than corrected by inserting weaker examples.

The distribution has methodological consequences. Later model evaluation should report per-class or macro-averaged evidence in addition to overall values, because a large comparison/bar class can dominate an aggregate result. This recommendation is not a downstream result and does not assert that any model performs well or poorly on a particular class. It states how the class distribution should be considered when interpreting later results.

### 4.6.2 Duplicate and near-duplicate control

Deduplication operates at several levels. Exact item identifiers and source-record identifiers are checked first. The selector then compares source groups, normalized goals, brief fingerprints, exact record content, and near-duplicate similarity. A source group can contribute at most two selected records, and a second record is retained only when its wording is sufficiently different and its analytical semantic signature differs in fields such as KPI, aggregation, axis, grouping, filter, sorting, or time grain. This preserves limited within-source variation while reducing the risk that one source family dominates the corpus.

The near-duplicate gate uses character 3-gram Jaccard similarity with a threshold of 0.8. Candidate text at or above this threshold is rejected as a near duplicate under the project policy. The threshold is not presented as a universal definition of duplication. It is a reproducible operational criterion selected for this corpus. Its use is supported by the broader concern that duplicate training data can increase memorization and that overlap between training and evaluation material can lead to overly optimistic estimates (Lee et al., 2022).

Exact row comparison alone would not be enough. Two records can have different identifiers and different serialized JSON while expressing nearly the same goal and constraint structure. Conversely, two records from the same source group can be distinct in wording but share enough analytical structure to create leakage if group membership is ignored. Combining exact checks, semantic signatures, source groups, and near-duplicate similarity makes the boundary more conservative while keeping the individual checks interpretable.

### 4.6.3 Group-disjoint splitting and held-out artifacts

The selected source-grounded records were split with seed 42 using the group-safe `nvbench_large_v2_split_v1_group_safe` procedure. Whole `source_group_id` values were assigned to one partition. The final counts are 1,281 training records, 264 validation records, and 274 held-out test records. The partitions contain 784, 167, and 167 unique source groups respectively. The observed record proportions are approximately 70.42%, 14.51%, and 15.06%, rather than exact nominal 70/15/15 proportions, because source-group integrity takes precedence over exact row fractions.

The 274-record test is in-domain and source-group-disjoint. It is not an external benchmark because it originates from the same registered nvBench corpus as the source-grounded training and validation records. It is nevertheless a meaningful held-out boundary for the project because its groups are not shared with the training or validation partitions and because it is preserved before the later generated augmentation. The test therefore measures generalization within the source domain and source construction process, not generalization to an unrelated external corpus.

A separate file contains 40 human-evaluation items selected from the held-out test. The file is an evaluation instrument, not a fourth modeling split. Its rows overlap with the test by design, remain outside training and validation, and were not used as context for enrichment, augmentation, or repair. The human-evaluation file should therefore not be added to the 3,819 modeling records as if it contained 40 new examples. The distinction between modeling records and evaluation items is maintained in the manifests and in the final split table in Section 4.10.

## 4.7 Constrained LLM Enrichment and the `dashboard_v3` Freeze

### 4.7.1 Why enrichment was needed

The source-faithful transformation provides the analytical content required to connect a natural-language query with fields, aggregations, constraints, and a chart. It does not provide all of the presentation fields required by the thesis output contract. In particular, original nvBench does not authoritatively specify the intended user role, a structured context summary, a dashboard layout, styling and accessibility choices, interactions, or a full rationale for the design. Leaving these fields absent would prevent the record from matching the common recommendation interface. Filling them with generic templates and calling them source values would be worse because it would misrepresent the evidence.

The project therefore introduced a constrained LLM-enrichment stage for the source-grounded training and validation records. This stage is methodologically related to filtered model-generated instruction data, where a model proposes content and a separate process filters invalid or overly similar examples (Wang et al., 2023). The project applies a stricter boundary than unconstrained instruction generation: the source-backed analytical projection is immutable, and the model may write only six presentation-oriented fields. These fields are marked `llm_generated` and `not_gold=true` in the resulting lineage metadata.

### 4.7.2 Writable and immutable fields

The enrichment workflow can modify exactly `users`, `context_summary`, `layout`, `styling`, `interactions`, and `rationales`. These fields form the presentation layer of the recommendation. The model can describe the audience, summarize the analytical context, arrange the views, propose accessible styling, suggest interactions, and explain the design choices. It cannot change the analytical task by rewriting the goal or by replacing the source KPI with a more fluent alternative.

The immutable projection contains the item and source identifiers, source group, source record, original goal/query, KPI and aggregation evidence, physical columns and types, task and chart mapping, alternatives, encodings, filters, sorting, limits, grouping, time grain, source provenance, quality score and tier, and split membership. The implementation computes an immutable projection and a SHA-256 fingerprint before and after enrichment. A change to an immutable field raises an `immutable_source_field_changed` failure. This is a control against a common synthetic-data error: a generated explanation should not silently overwrite the facts that it is supposed to explain.

Test records and the 40 human-evaluation items were excluded from the enrichment input. The full run therefore enriched only the 1,545 train and validation records, consisting of 1,281 training and 264 validation records. The held-out analytical content and evaluation boundary were preserved independently of the generated presentation layer.

### 4.7.3 Model, automatic validation, and human pilot

The enrichment artifacts record the model as `deepseek-v4-flash-sovereign`, the generation temperature as 0.0, and the reasoning effort as requested or configured `xhigh`. The project reports the configuration that was supplied to the provider. It does not claim that the provider independently verified an internal reasoning level. This wording is important because a configuration value is evidence about the request, not proof of how a remote model internally reasoned.

The enrichment workflow used several checks. The payload had to satisfy the strict schema and contain all six required fields. The generated context could not introduce unsupported KPIs, chart types, business numbers, or fields. Interactions had to use approved types and refer to fields present in the record. Rationales had to mention the actual task, chart, and encoded fields without asserting an observed outcome that the source did not provide. The trusted application code, rather than the model, merged only the six permitted fields and recorded their lineage.

A ten-record technical sample passed with 10/10 accepted outputs. A 30-record human R1 pilot accepted 29 records and rejected one because the rationale disagreed with the task, chart, or encoding evidence. The full first pass processed 1,545 records and accepted 1,534 while rejecting 11. It reported a 100% schema-valid rate, zero immutable-field violations, 943 API calls, 617 cache hits, and 15 retries. These values describe the enrichment process and not a final human evaluation of dashboard quality.

The targeted retry and offline revalidation stage resolved ten of the eleven first-pass rejections without recalling accepted records. One remaining case required a targeted retry. The final result reconciled to all 1,545 expected train and validation records, with zero permanent rejections and zero recorded immutable-field violations. The pre-freeze audit also normalized fingerprint metadata for records where the metadata was absent; this was metadata normalization and did not change the source-backed or enriched values.

The 29/30 human result is a limited quality gate for the enrichment procedure. It exceeds the configured minimum of 27 accepted records, but it does not establish that the six fields are correct for every record or that they are expert gold. Human-evaluation guidance stresses the need to report the scope and limitations of human judgments separately from automatic checks (van der Lee et al., 2019). The dataset therefore keeps the R1 result as evidence about a small pilot and retains the `llm_generated` and `not_gold` labels for the full enriched corpus.

### 4.7.4 Frozen v3 composition

The enriched source-grounded corpus was frozen as `dashboard_v3`. The frozen package contains 1,281 training records, 264 validation records, and 274 held-out test records. The 1,545 train and validation records contain the six LLM-generated presentation fields. The test remains outside the enrichment input and retains its source-grounded analytical lineage and held-out artifact values. The separate 40-item human-evaluation file is sampled from the test and is not trainable.

The v3 manifest records `nvbench_large_v2` as the source dataset version, `GoldItem` as the schema version, and `phase3-enrichment-v1` as the enrichment specification. It records source-grounded fields, deterministically derived fields, and LLM-generated enrichment fields separately. It also records the R1 pilot, validation results, leakage checks, exact file-preservation checks, and the exclusion of raw API responses, secrets, provider caches, and temporary generations from the frozen package. This freeze created the stable predecessor that was later augmented in train and validation only.

## 4.8 Controlled `dashboard_v4` Augmentation

### 4.8.1 Motivation and scope

The source-grounded v3 train and validation data provided a defensible analytical core but had narrow coverage for the full dashboard-design task. The 1,545 records covered five primary task/chart families: comparison/bar, part-to-whole/pie, trend/line, composition/stacked-bar, and correlation/scatter. The profile contained 316 records with explicit filters, 83 grouped records, and 275 records with temporal information. It contained no canonical multi-KPI records. This narrow coverage is a consequence of using an NL2VIS source for a dashboard-level task rather than evidence that the source corpus is representative of all dashboard work.

The purpose of v4 was to broaden the training and validation supervision to additional task families, chart types, grouped and temporal cases, filters, and multi-KPI layouts. The augmentation did not replace the source-grounded core and did not modify the held-out test. The project used a controlled generator with a versioned task/chart catalogue and deterministic sampling. This design allows the generated records to be useful for learning broader output behaviour while keeping their generated status visible.

The expansion is not interpreted as a correction of the natural distribution of real dashboards. It is an intervention on the training and validation material. The generated records can increase coverage of task families that are absent from the source-derived test, but they cannot turn the preserved test into a benchmark for those new families. That boundary is important for the interpretation of later experimental results.

### 4.8.2 Generation protocol and provenance

The generation run is recorded as `dashboard_v4-generation-v1`. It used `gpt-5.6-luna` in `codex_agent` mode, seed 42, batches of 40 records, and 50 batches. The generator read only the frozen v3 training and validation material as construction context. It did not read `test.jsonl` or the 40-item human-evaluation file while proposing candidates. No generated candidate could be assigned to the test or human-evaluation artifacts.

The accepted v4 records are marked `source=llm_generated`, carry generation-group metadata, and set `not_gold=true`. Unlike the preserved v3 analytical records, their goals, KPIs, columns, constraints, task/chart specification, and encodings are generated within the controlled scenario generator. These fields are then checked against the project’s controlled schema and generation rules, but validation does not transform them into source observations. The generated records are therefore best described as AI-generated supervision, not as additional nvBench examples.

The generator uses a catalogue of 20 operational domains and a set of controlled scenario features. It varies audiences, decision contexts, operating states, planning windows, time grains, themes, layout patterns, filters, grouping, sorting, limits, and the presence of a second KPI. The purpose of these controls is to create identifiable coverage variation rather than unconstrained text diversity. The catalogue and sampling code are versioned in the repository, while the accepted records preserve their generation index, scenario group, seed, and model metadata.

### 4.8.3 Acceptance, deduplication, and split restriction

The run attempted 2,915 candidates and accepted exactly 2,000. The remaining 915 candidates were rejected as near duplicates under the character 3-gram Jaccard threshold of 0.8. Candidate acceptance also required non-empty brief and recommendation fields, a sufficiently detailed constraint string, valid task/chart combinations, KPI references that resolved to the generated columns, task-appropriate encodings, valid filters and sorting, and complete layout, styling, interaction, and rationale fields. These gates are generation-specific checks; passing them does not establish human or expert gold status.

The 2,000 accepted records were assigned 1,651 to train and 349 to validation. No generated record was assigned to test. The generated records were compared against the preserved v3 train and validation material and against previously accepted generated records using identifiers, normalized goals, brief fingerprints, exact record hashes, scenario signatures, and near-duplicate similarity. This prevents the augmentation from simply repeating a source brief under a new identifier. The generator also applies a batch safety limit so that a target cannot be reached through uncontrolled rejection behaviour.

The accepted records broadened the structural coverage. The generation report records 1,215 records with explicit filters, 1,373 with grouping, 1,229 with temporal information, and 842 with multiple KPI mappings. At the record level, the generated task distribution is shown in Table 4.2. Each record is counted once using its primary mapping, so the counts sum to 2,000.

Table 4.2. Record-level distribution of primary task types in the 2,000 accepted generated records.

| Primary task type | Generated records |
| --- | ---: |
| `correlation` | 300 |
| `comparison` | 300 |
| `trend` | 250 |
| `distribution` | 250 |
| `ranking` | 200 |
| `part_to_whole` | 200 |
| `composition` | 200 |
| `flow` | 150 |
| `deviation` | 150 |
| **Total** | **2,000** |

The generated primary chart distribution is shown in Table 4.3. It contains 14 chart types and also sums to 2,000 records. These counts are record-level primary chart counts, not mapping-level counts.

Table 4.3. Record-level distribution of primary chart types in the 2,000 accepted generated records.

| Primary chart type | Generated records |
| --- | ---: |
| `table` | 351 |
| `bar` | 275 |
| `area` | 223 |
| `heatmap` | 163 |
| `scatter` | 137 |
| `line` | 134 |
| `box` | 127 |
| `histogram` | 123 |
| `grouped_bar` | 105 |
| `stacked_bar` | 93 |
| `donut` | 76 |
| `sankey` | 69 |
| `treemap` | 68 |
| `pie` | 56 |
| **Total** | **2,000** |

Some generated records contain two KPI mappings. The distribution report therefore also contains mapping-level counts that can exceed the record count. It reports 2,842 generated mapping instances, 3,635 train mapping instances, and 752 validation mapping instances. These values are not additional JSONL records. Keeping record-level and mapping-level units separate prevents a multi-KPI record from being counted twice in the dataset-size accounting while still allowing the project to describe the coverage of its individual mappings.

## 4.9 `dashboard_v4_1` Semantic Repair

### 4.9.1 Why structural validity was insufficient

The first v4 generation pass passed its structural admission pipeline, but a presentation-layer audit found issues in all 2,000 generated records. The most common categories affected all 2,000 records: missing or unanchored context constraints, KPI/context mismatch, unsupported context fields, generic layouts, under-described layout blocks, missing styling semantics, interactions without a purpose, generic filler, and rationales that did not provide record-specific evidence. Generic user templates affected 1,819 records and generic palettes affected 1,906 records. Additional issues included sort/task mismatches in 779 records, unjustified status colors in 564 records, interactions referring to nonexistent columns in 250 records, and chart-principle mismatches in 876 records.

These findings show why a schema-valid object cannot be treated as semantically adequate. The object can have the right keys and valid enum values while its presentation fields remain generic or disconnected from the brief. In a dashboard setting, an interaction that points to an absent field is not useful merely because its JSON representation is valid. A rationale that mentions a chart principle without applying it to the record is also not evidence of a meaningful explanation. The audit therefore focused on whether each generated field was anchored to the record’s own brief and mapping.

### 4.9.2 Repair scope and protected fields

The repair workflow is versioned as `dashboard_v4_1-semantic-repair-v1`. It used `gpt-5.6-luna` in `codex_agent_context_aware` mode. Only the same six presentation fields were repairable: `users`, `context_summary`, `layout`, `styling`, `interactions`, and `rationales`. The fields `goals`, `kpis`, `columns`, `constraints`, `task_type`, `chart_type`, and `encoding` were protected. In the generated lineage, these protected fields are generated or generated-and-validated content; protecting them means that the repair stage did not change them, not that they became source-grounded.

The repair report records 2,000 repaired records, zero records already valid before repair, and zero rejected or regenerated records. The `users` field changed in 1,819 records, while each of the other five presentation fields changed in all 2,000 records. The model-generated analytical specification was therefore kept fixed during presentation repair. This separation makes it possible to attribute the repair operation to the presentation layer rather than to an undocumented change in the task, KPI, chart, or encoding.

### 4.9.3 Post-repair validation and interpretation

The post-repair audit checks the fields against the record’s existing goal, KPI, columns, constraints, task, chart, and encoding. Context summaries must agree with the brief. Layout blocks must correspond to the KPI mappings and state a meaningful hierarchy or reading order. Styling must describe typography, contrast, accessibility, and a semantic color policy. Interactions must use approved types, have a purpose, and refer only to fields present in the record. Rationales must mention the actual task, chart, and KPI and must avoid unsupported outcome claims such as causal or observed-performance statements that are not present in the brief.

The frozen validation report records zero schema-invalid and zero semantic-invalid generated records after repair. The repair report and the manifest consequently describe the generated fields as semantically clean under the repository audit. This phrase has a narrow meaning. It means that the implemented predicates passed; it does not mean that independent visualization experts reviewed or certified all 2,000 generated recommendations. The generated records remain `not_gold=true`.

## 4.10 Final Dataset Composition and Mixed Lineage

### 4.10.1 Final split table

The operational package is named `dashboard_v4`, while the exact frozen manifest revision stored inside that package is `dashboard_v4_1`. The final modeling composition is shown in Table 4.4. The preserved v3 columns represent the source-grounded predecessor records, and the v4/v4.1 columns represent the added AI-generated train and validation records.

Table 4.4. Final split composition and lineage of the frozen modeling package.

| Partition or artifact | Preserved nvBench-derived v3 | AI-generated v4/v4.1 | Total | Role |
| --- | ---: | ---: | ---: | --- |
| Train | 1,281 | 1,651 | **2,932** | Trainable |
| Validation | 264 | 349 | **613** | Validation/model selection only |
| Held-out test | 274 | 0 | **274** | Evaluation only |
| Modeling total | **1,819** | **2,000** | **3,819** | Modeling study |
| Human-evaluation item file | 40 test-derived items | 0 | **40** | Separate evaluation instrument |

The arithmetic is deliberate. The modeling total is `1,819 + 2,000 = 3,819`, and the final train and validation totals are `1,281 + 1,651 = 2,932` and `264 + 349 = 613`. The 40 human-evaluation rows are sampled from the 274 held-out test records. They overlap with the test by design and must not be added as 40 new modeling records or treated as an independent fourth split.

### 4.10.2 Record-level and field-level lineage

The final dataset has mixed lineage by design. The 1,819 preserved records retain `source=nvbench` and the source query, database, visualization, and provenance information. The 1,545 train and validation records in this group contain six LLM-generated presentation fields from the v3 enrichment stage. Their goals, KPIs, columns, constraints, and analytical chart evidence remain source-grounded, while task inference and KPI selection remain deterministic derivations. The 274 test records remain in the source-grounded analytical lineage and were not processed by the enrichment run; their held-out recommendation fields were preserved as part of the evaluation artifact.

The 2,000 added records retain `source=llm_generated` and `not_gold=true`. Their goals, KPIs, columns, constraints, task/chart specification, and encodings were created by the controlled v4 generator and accepted by deterministic validation. Their six presentation fields were then changed or confirmed through the v4.1 semantic-repair workflow. The repair protected the generated analytical fields, but it did not change their evidence class. They remain AI-generated supervision rather than nvBench observations, human gold, or expert gold.

The field-level view is important for later analysis. A model can be evaluated on the same output schema across both record families, but the reference status of a field differs between them. The v3 analytical target has a source-backed origin, whereas the v4 analytical target is generated. The presentation fields are generated in the v3 train and validation records and generated-plus-repaired in the v4 records. A result that is strong on the generated portion should therefore be described as performance on the project’s generated supervision, not as independent evidence of dashboard quality.

### 4.10.3 Consequences for interpretation

The preserved 274-record test measures in-domain, source-group-disjoint generalization to the five original source chart families. The test includes bar, pie, line, stacked-bar, and scatter records, with a strong concentration of bar/comparison examples and only two scatter records. It does not directly evaluate generated-only task families such as distribution, ranking, deviation, or flow, because no such records were added to the test. It also does not turn the generated v4 records into independent gold merely because they are held out from some training runs.

The broader v4 training and validation distribution can support experiments on additional task and chart families, but any conclusion about those families requires evidence from an evaluation set that contains them. Human evaluation or an external benchmark can provide complementary evidence for layout, styling, interaction usefulness, and rationale quality. Until such evidence is available, the preserved test should be interpreted as a controlled in-domain test of the source-grounded boundary, not as a complete validation of all dashboard capabilities introduced by the augmentation.

## 4.11 Dataset Freeze, Versioning, and Reproducibility

The active configuration in `src/config/data/dashboard_v4.yaml` uses `dashboard_v4` as the operational dataset identity and points to `data/frozen/dashboard_v4/`. The manifest in that directory records `dashboard_v4_1` as the exact frozen revision, `dashboard_v4` as its parent package, and `dashboard_v4_1-semantic-repair-v1` as the repair version. This distinction prevents a path or configuration name from being confused with the precise materialized state of the data.

The frozen directory contains the train, validation, test, and human-evaluation artifacts together with the `GoldItem` schema, dataset card, manifest, hashes, validation report, duplicate report, leakage report, distribution report, semantic-audit reports, and repair report. The manifest records the v3 prefixes, generated counts, repair scope, protected fields, and the checks used to publish the package. The v4 freeze verified that the v3 train and validation prefixes were unchanged, that the test and human-evaluation files were byte-identical to their parent artifacts, and that generated records entered only train and validation.

The file-level hashes provide an additional reproducibility check. Direct SHA-256 verification performed on 2026-09-02 matched the hashes files for the v3 and v4 train, validation, test, and human-evaluation artifacts. The source archive is separately identified by its own digest in the nvBench source manifest. These hashes do not prove that a future model provider will produce the same output from the same prompt, but they do make the published bytes and the source package auditable after the fact.

The package is treated as write-once. A material change to a frozen JSONL, CSV, schema, manifest, or report should create a new version rather than silently modifying the existing release. The held-out artifacts are copied or verified byte-for-byte across the v3-to-v4 transition, and the generation and repair procedures operate only on train and validation material. Raw API responses, credentials, caches, and temporary generations are excluded from the final package. This separation keeps reproducibility records useful without publishing sensitive or irrelevant execution material.

One metadata inconsistency is recorded in the project audit. The nested human-evaluation hash in the v4 manifest differs from the direct file-level hash and the value in the hashes files, while the actual v3 and v4 human-evaluation files are byte-identical. Because the disputed nested digest is not needed to explain the dataset composition, this chapter does not reproduce it as an authoritative value. The accompanying evidence audit records the conflict and gives priority to the directly recomputed file hash and the dedicated hashes files. Reporting the conflict explicitly is preferable to silently selecting one value.

The unpinned upstream `main` reference remains the main limitation of source reproducibility. The local archive digest identifies the exact bytes used in this project, but an independent researcher who downloads the branch at a later time may obtain a different archive. Reproduction should therefore use the registered archive or an independently verified copy of the same bytes whenever possible.

## 4.12 Limitations and Threats to Validity

### 4.12.1 Construct validity

The source-backed part of the dataset is derived from nvBench, which was designed for NL2VIS rather than full multi-view dashboard design (Luo, Tang, & Li, 2021; Luo et al., 2021). It provides useful evidence about natural-language analytical goals, relational data, aggregations, and chart specifications, but it does not provide authoritative personas, complete dashboard layouts, styling, interaction design, or full rationales. The six presentation fields in the v3 train and validation records are therefore LLM-generated annotations. They are not source gold or expert gold. The 2,000 v4 records are generated supervision at both the analytical and presentation levels, followed by a model-assisted repair of the six presentation fields.

The task and chart rules are also project-specific abstractions. The distinction between task and encoding is supported by visualization research, but the exact quality weights, Tier-A threshold, pie cardinality limit, scatter variation requirements, near-duplicate threshold, and related policies are operational decisions for this project (Cleveland & McGill, 1984; Mackinlay, 1986; Brehmer & Munzner, 2013). Passing the rules means that a record satisfies the adopted construction protocol. It does not prove that the record expresses the only valid design or that the selected chart is optimal for every audience.

Source metadata and heuristic fallbacks can misclassify semantic field roles. A field name may look like an identifier without being one, while a domain-specific identifier may not follow the configured patterns. Database profiles improve the evidence but cannot resolve every domain interpretation. The quality tiers should therefore be read as bounded operational confidence rather than as a universal semantic truth.

### 4.12.2 Internal validity

The project’s earlier synthetic-only workflow created a circularity risk because the same deterministic task-to-chart logic influenced both training and testing. The present pipeline reduces this risk by constructing the held-out test from source-group-disjoint nvBench records and by excluding test and human-evaluation items from v3 enrichment, v4 generation, and v4.1 repair. This is a meaningful improvement in the evaluation boundary, but it is not complete independence: the source-grounded train, validation, and test records still originate from the same upstream corpus and share its construction assumptions.

The v4.1 semantic repair is model-assisted. The post-repair audit reports zero remaining schema and semantic failures under the repository rules, but it does not constitute independent review by visualization experts. The R1 enrichment review covers only 30 records and accepted 29 of them. The final 40-item human-evaluation study has not yet produced ratings. Therefore, no completed human-rating result, inter-rater agreement, or human preference claim can be inferred from the 29/30 enrichment pilot.

The validated hybrid staging branch is another internal boundary. It combined 18 synthetic training records and three synthetic validation records with enriched nvBench material and passed its own checks, but it was not used in the final v3 freeze. The repository does not contain a dedicated final-decision report explaining why these 21 synthetic records were omitted. The final manifests clearly identify the selected nvBench-only v3 inputs, so the final scope is known; the rationale for omitting the staging records remains incompletely documented.

### 4.12.3 External validity

The held-out test is in-domain and imbalanced. It contains only the five original source chart families and includes very few scatter cases. The expanded v4 training and validation records cover nine task families and 14 chart types, but the generated-only task families are absent from the preserved test. Test performance therefore cannot be interpreted as direct evidence that a model generalizes equally well to distribution, ranking, deviation, flow, or the other newly introduced chart/task combinations. Such claims require a matching evaluation set, human assessment, or an external benchmark.

The source domain is also constrained by the schemas and domains represented in nvBench and by the NL2SQL resources used in its construction, including Spider. Real dashboard requests may contain multi-table business logic, organization-specific definitions, missing data, or visual conventions not represented in this corpus. The final dataset should therefore be used to study the controlled comparison defined in this thesis, not as a representative sample of all dashboard design work.

### 4.12.4 Reproducibility validity

The source branch was downloaded as `main` without a pinned upstream commit. The local archive hash makes the bytes used in the project identifiable, but it does not guarantee that an independent future download of the branch will match. The 20,986-to-21,244 quality-pool change also lacks complete row-level reconciliation. Both limitations are documented rather than hidden behind a single final count.

The generated fields are reproducible at the level of stored files, manifests, hashes, prompts or specifications, and rule versions, but the semantic content came from language-model calls. Temperature 0.0 does not guarantee identical outputs across provider revisions or service changes. The v4.1 repair is similarly reproducible as a published artifact and rule-checked result, but its text and layout-like values were produced by a model rather than independently annotated by experts.

Finally, the human-evaluation item file is an input list rather than a completed result. It contains 40 rows sampled from the held-out test, while the final human ratings are not present in the frozen dataset artifacts. The 29/30 value belongs only to the earlier enrichment gate. These distinctions limit the claims that can be made from the dataset alone and motivate the controlled system and experiment methodology described in Chapter 5.

---

## References Used in Chapter 4

Brehmer, M., & Munzner, T. (2013). A multi-level typology of abstract visualization tasks. _IEEE Transactions on Visualization and Computer Graphics, 19_(12), 2376–2385. https://doi.org/10.1109/TVCG.2013.124

Cleveland, W. S., & McGill, R. (1984). Graphical perception: Theory, experimentation, and application to the development of graphical methods. _Journal of the American Statistical Association, 79_(387), 531–554. https://doi.org/10.1080/01621459.1984.10478080

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. _Communications of the ACM, 64_(12), 86–92. https://doi.org/10.1145/3458723

Kim, Y., & Heer, J. (2018). Assessing effects of task and data distribution on the effectiveness of visual encodings. _Computer Graphics Forum, 37_(3), 157–167. https://doi.org/10.1111/cgf.13409

Lee, K., Ippolito, D., Nystrom, A., Zhang, C., Eck, D., Callison-Burch, C., & Carlini, N. (2022). Deduplicating training data makes language models better. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, 8424–8445. https://aclanthology.org/2022.acl-long.577/

Luo, Y., Tang, J., & Li, G. (2021). nvBench: A large-scale synthesized dataset for cross-domain natural language to visualization task. _arXiv_. https://arxiv.org/abs/2112.12926

Luo, Y., Tang, N., Li, G., Chai, C., Li, W., & Qin, X. (2021). Synthesizing natural language to visualization (NL2VIS) benchmarks from NL2SQL benchmarks. In _Proceedings of the 2021 International Conference on Management of Data_, 1235–1247. https://doi.org/10.1145/3448016.3457261

Luo, T., Huang, C., Shen, L., Li, B., Shen, S., Zeng, W., Tang, N., & Luo, Y. (2025). nvBench 2.0: Resolving ambiguity in text-to-visualization through stepwise reasoning. In _Advances in Neural Information Processing Systems, 38_, 138749–138786. https://doi.org/10.52202/085713-4172

Mackinlay, J. (1986). Automating the design of graphical presentations of relational information. _ACM Transactions on Graphics, 5_(2), 110–141. https://doi.org/10.1145/22949.22950

Moritz, D., Wang, C., Nelson, G. L., Lin, H., Smith, A. M., Howe, B., & Heer, J. (2019). Formalizing visualization design knowledge as constraints: Actionable and extensible models in Draco. _IEEE Transactions on Visualization and Computer Graphics, 25_(1), 438–448. https://doi.org/10.1109/TVCG.2018.2865240

Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022). Data cards: Purposeful and transparent dataset documentation for responsible AI. In _Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency_, 1776–1826. https://doi.org/10.1145/3531146.3533231

Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond accuracy: Behavioral testing of NLP models with CheckList. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_, 4902–4912. https://aclanthology.org/2020.acl-main.442/

Saket, B., Endert, A., & Demiralp, C. (2019). Task-based effectiveness of basic visualizations. _IEEE Transactions on Visualization and Computer Graphics, 25_(7), 2505–2512. https://doi.org/10.1109/TVCG.2018.2829750

van der Lee, C., Gatt, A., van Miltenburg, E., Wubben, S., & Krahmer, E. (2019). Best practices for the human evaluation of automatically generated text. In _Proceedings of the 12th International Conference on Natural Language Generation_, 355–368. https://doi.org/10.18653/v1/W19-8643

Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., & Hajishirzi, H. (2023). Self-Instruct: Aligning language models with self-generated instructions. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, 13484–13508. https://aclanthology.org/2023.acl-long.754/

Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., Ma, J., Li, I., Yao, Q., Roman, S., Zhang, Z., & Radev, D. (2018). Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_, 3911–3921. https://aclanthology.org/D18-1425/
