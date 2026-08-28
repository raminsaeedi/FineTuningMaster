# Constrained Encoding and HPC Validation Protocol

## Purpose

Validate the repaired output contract before an expensive 27B HPC run. This protocol covers:

- `encoding` structure and semantic correctness;
- Top-3 recommendation validity;
- RAG retrieval quality;
- fair A-D comparison;
- reproducibility and HPC go/no-go.

This is a staged engineering and scientific validation. The 20-item and 50-item results are pilot evidence, not final thesis results.

## Current project basis

Current artifacts show that inference, QLoRA training, and TF-IDF retrieval execute, but many outputs use a string where `encoding` must be an object. Current Top-3 reporting correctly becomes unavailable when three distinct recommendations are rarely produced. Current RAG coverage shows that passages are returned; it does not prove relevance. The frozen dashboard-v4 test data are synthetic and therefore suitable for internal diagnostics, not independent claims about dashboard quality.

## Experimental design

Run every method with both decoder settings:

| Method | Training | RAG |
| --- | --- | --- |
| A | No | No |
| B | No | Yes |
| C | QLoRA | No |
| D | Same C adapter and seed | Yes |

Decoder settings:

- `U`: normal unconstrained generation.
- `C`: constrained generation using the strict response schema.

This creates eight matched conditions: A-U, A-C, B-U, B-C, C-U, C-C, D-U, D-C. Never compare methods across different decoder settings. Primary method contrasts are A versus B and C versus D within the same decoder setting. Constrained versus unconstrained is a separate within-method analysis.

Keep model revision, tokenizer, prompts, generation parameters, sequence limits, hardware class, dataset IDs, parser, evaluator, and post-processing identical within each comparison. B and D must use the same retriever, KB version, `top_k=3`, and context-budget policy. C and D must use the same adapter for each seed. Do not use method-specific JSON repair.

### Training-loss control

Existing C/D adapters used full-sequence loss. Keep them for the first decoder-only smoke test so the formatting intervention is isolated.

If constrained format passes but semantic encoding remains weak, train one new seed-42 C adapter with:

```text
training.sft.completion_only_loss=true
```

This uses TRL prompt-completion training and excludes prompt tokens from the loss. Re-run both C and D with that exact adapter. Treat full-sequence and completion-only adapters as different experimental conditions. Do not pool their loss values or results. Select one training mode using validation/pilot data, then freeze it before final test and 27B runs.

## Samples and stages

### Stage 1: 20-item smoke test

- Use seed 42.
- Select 20 fixed validation items before generation.
- Stratify across available task types and chart types.
- Keep IDs and selection hash in the run artifacts.
- Do not use final test items.
- Run all eight matched conditions.
- Purpose: catch schema, truncation, adapter, retrieval, and artifact failures.
- Do not perform thesis hypothesis tests on this stage.

### Stage 2: 50-item pilot

- Use seed 42.
- Use 50 fixed validation items disjoint from the smoke sample and final test.
- Freeze code, prompt, schema, retriever, KB, and generation settings before running.
- Run all eight matched conditions.
- Build retrieval relevance judgments for these 50 RAG queries before inspecting method quality results.
- Purpose: estimate semantic behavior and decide whether 27B cost is justified.

### Later thesis runs

- Use seeds 42, 43, and 44.
- Use identical item IDs and settings for A-D.
- Run seed 42 first. Run seeds 43 and 44 only after seed 42 passes artifact checks.
- Report item-level results, each seed separately, and mean plus spread across seeds.

## Strict response contract

Use a dedicated generation schema. Do not rely on the current lenient internal model alone.

Required top-level fields:

- `context_summary`
- `kpi_chart_mapping`
- `layout`
- `styling`
- `interactions`
- `rationales`

Each mapping must require:

- non-empty `kpi`;
- valid `task_type` enum;
- valid `chart_type` enum;
- `encoding` as an object, never a string;
- non-empty string `encoding.x`;
- non-empty string `encoding.y`;
- `encoding.aggregate` as an allowed string or `null`;
- typed `alternatives`;
- no unknown keys in the strict response schema.

Constrained decoding controls syntax and types. It does not prove that selected fields, aggregation, chart, or rationale are correct.

## Outcomes

### Primary outcomes

1. **Strict schema validity per item:** extracted output passes the dedicated strict response schema without repair.
2. **Joint semantic encoding accuracy:** all expected mappings in an item have correct `x`, `y`, and `aggregate` against frozen reference data.
3. **RAG retrieval quality:** Recall@3, MRR, and nDCG@3 against manual relevance judgments for B and D queries.

Primary method effects:

- B minus A: RAG effect without fine-tuning.
- D minus C: RAG effect after fine-tuning.
- C minus A: QLoRA effect without RAG.
- D minus B: QLoRA effect with RAG.

Calculate each effect only within the same decoder setting, seed, and item set.

### Secondary outcomes

- JSON parse rate.
- Required-key rate and completeness.
- `encoding` object-type validity.
- exact `x`, exact `y`, exact `aggregate` accuracy;
- joint exact `x+y` accuracy;
- input-column validity for `x` and `y`;
- valid aggregation-token rate;
- exact chart type and task type;
- Top-1 and valid Top-3 diagnostics;
- missing outputs, runtime errors, prompt overflow, and truncation;
- mean and P95 latency;
- RAG grounding mode and supported-claim rate, reported as secondary only.

All accuracy denominators must include every expected item or mapping. Missing, unparsable, or invalid outputs count as wrong. Never calculate semantic accuracy only over valid outputs.

## Semantic encoding scoring

Freeze normalization rules before the pilot:

- trim surrounding whitespace;
- normalize case only where project enums are case-insensitive;
- require column identifiers to match a column in the input brief;
- use the project aggregation allowlist;
- treat missing `aggregate` and explicit `null` according to the frozen reference;
- do not use fuzzy matching, model-based repair, or manual correction.

Report both mapping-level and item-level scores. Item-level joint encoding is correct only when every expected mapping is correct. This prevents a dashboard with one correct chart and several wrong mappings from being counted as fully correct.

## Top-3 protocol

A Top-3 prediction is supported only when it contains:

- one primary chart;
- at least two alternatives;
- three valid, ordered, distinct chart types after de-duplication;
- no alternative equal to the primary chart.

Use current project rule: headline Top-3 is valid only when at least 80% of scored items support three recommendations. Otherwise report:

- `top_3_valid = false`;
- headline Top-3 as unavailable;
- support rate and supported-subset accuracy as diagnostics.

Current synthetic test references do not provide validated alternative rankings. Therefore Top-3 can show whether the gold primary appears among three recommendations, but cannot prove that alternative order or alternative quality is correct. If alternative quality is a thesis claim, create an independent expert-ranked reference set first. Failure of Top-3 support blocks a Top-3 claim, not an encoding-only HPC study.

## RAG relevance protocol

Create query-level relevance judgments for the 50 pilot briefs:

- judge every candidate KB chunk in a pooled set containing retrieved chunks plus sampled non-retrieved chunks;
- use relevance grades `0 = not relevant`, `1 = partly relevant`, `2 = directly useful`;
- store query ID, chunk ID, grade, judge, guideline, and qrels hash;
- judge without seeing method output quality;
- independently review disagreements or a fixed audit sample.

Metrics:

- **Recall@3:** fraction of judged relevant chunks retrieved in the first three positions.
- **MRR:** reciprocal rank of the first relevant chunk; zero when none is retrieved.
- **nDCG@3:** graded ranking quality using the 0/1/2 relevance grades.

RAG execution gates:

- 100% query coverage;
- exactly three unique chunk IDs per query;
- every chunk ID exists in the frozen KB manifest;
- dataset and KB hashes match recorded manifests;
- all three passages remain represented after prompt budgeting;
- no prompt exceeds its input-token budget.

RAG relevance gates for the 50-item pilot:

- Recall@3 at least 0.80;
- MRR at least 0.70;
- nDCG@3 at least 0.70.

These are pre-run engineering gates, not observed results or universal scientific standards. Report bootstrap 95% confidence intervals and raw per-query judgments. If a gate fails, improve or limit the KB/retriever claim before the 27B run.

## Exact format and execution gates

### 20-item smoke gate

Every constrained A-D condition must have:

- 20/20 predictions;
- zero runtime-error records;
- zero missing outputs;
- zero prompt-budget violations;
- zero truncated outputs;
- JSON parse: 20/20;
- strict schema validity: at least 19/20;
- `encoding` object type: 20/20 outputs;
- correct C-to-D adapter seed and recorded adapter hash;
- complete config, dataset, KB, model, schema, and code provenance.

Any failed condition is no-go for the 50-item pilot. Fix cause, create a new run ID, and rerun all matched conditions affected by the change.

### 50-item pilot gate

Every constrained A-D condition must have:

- 50/50 predictions;
- zero runtime-error records;
- zero missing outputs;
- zero prompt-budget violations;
- zero truncated outputs;
- JSON parse: 50/50;
- strict schema validity: at least 48/50;
- `encoding` object type: 50/50 outputs.

Semantic non-inferiority gate:

- constrained joint semantic encoding accuracy must not be more than 5 percentage points below the matched unconstrained condition.

RAG conditions must also pass all RAG execution and relevance gates. Top-3 must follow its separate claim rule.

## Statistical analysis

- Use item-paired data. Preserve failures as zero outcomes.
- For binary outcomes across A-D, use Cochran's Q as omnibus test.
- For planned pairwise binary contrasts, use exact McNemar tests.
- Apply Holm correction within each outcome family.
- Use paired bootstrap with 10,000 resamples for 95% confidence intervals on accuracy differences, semantic metric differences, retrieval metrics, and latency differences.
- Report effect estimates and confidence intervals, not only p-values.
- Analyze seed variation separately from item-level uncertainty. Do not pool all seed-item rows as independent observations.
- Label 20-item and 50-item analyses exploratory. Confirm final claims on frozen evaluation data with seeds 42, 43, and 44.

## Provenance checklist

Every run must record:

- non-unknown Git commit hash and dirty-state flag;
- command and complete resolved configuration;
- model and tokenizer name plus immutable revision;
- decoder mode, Outlines version, strict-schema hash, and generation parameters;
- dataset version, file hashes, split, ordered item IDs, and sample-selection hash;
- KB version, chunk-file hash, manifest hash, retrieval configuration, and qrels hash;
- seed;
- prompt-template hash and token-budget data;
- adapter path, adapter hash, source C run ID, training seed, and training-data hash for C/D;
- Python, package-lock, CUDA, driver, GPU, and HPC job information;
- prediction, error, metric, and per-item evaluation files.

Do not accept `git_hash: unknown`, null D adapter lineage, changed frozen hashes, or an undocumented dirty code state for a thesis run.

## 27B HPC go/no-go

### Go

Start the 27B seed-42 run only when:

- all 20-item smoke gates pass;
- all 50-item pilot format and execution gates pass;
- constrained decoding passes semantic non-inferiority;
- RAG execution and relevance gates pass for B and D;
- A-D settings are matched except for intended method components;
- decoder and strict schema are version-pinned;
- one clean, reproducible run package can be regenerated from recorded provenance;
- GPU memory, checkpoint, resume, and output-path smoke checks pass on HPC.

After seed 42 passes artifact validation, run seeds 43 and 44 without changing prompts, schema, data, KB, or metrics.

### No-go

Do not start or continue the 27B study when any of these occurs:

- format gate failure;
- `encoding` emitted as a string in a constrained condition;
- missing, truncated, or silently repaired predictions;
- D adapter lineage cannot be verified;
- RAG passages are missing, duplicated, outside the manifest, or not relevant enough;
- comparison settings differ beyond the intended A-D components;
- code, data, KB, schema, or model revision is not reproducible.

Top-3 failure alone permits a 27B encoding/RAG run only if Top-3 is removed from confirmatory claims and reported as unsupported.

## Claim limits

- Constrained decoding supports a format-reliability claim only. It does not prove better dashboard reasoning.
- Exact encoding against synthetic reference data is an internal diagnostic. It does not alone prove real-world dashboard quality.
- TF-IDF retrieval plus Recall@3/MRR/nDCG@3 supports retrieval-relevance claims, not factual correctness or causal improvement.
- B better than A or D better than C supports a RAG effect only under matched settings and paired analysis.
- One seed supports debugging, not stability.
- Smoke and pilot samples support go/no-go decisions, not final thesis conclusions.
- Final usefulness or design-quality claims require independent gold and/or blinded human evaluation.
- A 27B improvement is not a model-size effect unless smaller and larger models use matched data, prompts, decoder, metrics, seeds, and evaluation items.

## Decision record

For each stage, save one signed-off record containing:

- gate name;
- threshold;
- observed value;
- pass/fail;
- artifact path;
- approved next action;
- excluded claims when a non-blocking metric fails.

Never change thresholds after viewing pilot or final-test results. Any change creates a new protocol version and new confirmatory run.
