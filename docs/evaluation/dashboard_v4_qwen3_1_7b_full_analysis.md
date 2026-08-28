# Dashboard v4 Full-Run Analysis

## Scope

- **Checked:** 2026-08-26
- **Model:** Qwen3-1.7B
- **Run root:** `ftm_runtime/runs/dashboard_v4/qwen3_1_7b`
- **Additional comparison:** OLMo-2-0425-1B-Instruct (`ftm_runtime/runs/dashboard_v4/olmo2_1_49b`)
- **Dataset:** frozen `dashboard_v4_1`
- **Evaluation size:** 274 test items per completed run
- **Evaluation tier:** `internal-synthetic`
- **Purpose:** decide whether data, code, metrics, and method results are reasonable before another large run

Related documents:

- [Final-run readiness v4](final_run_readiness_v4.md)
- [Evaluation protocol](evaluation_protocol.md)
- [Dashboard v4 dataset manifest](../../data/frozen/dashboard_v4/manifest.json)
- [Dashboard v4 frozen hashes](../../data/frozen/dashboard_v4/hashes.json)

## Executive conclusion

### Dataset

Dataset quality is good enough for another controlled experiment.

- Row counts are correct.
- Frozen hashes match run manifests.
- Schema, semantic, duplicate, and leakage checks pass.
- Test and human-evaluation data remain protected from generated training data.

### Code and training

Pipeline mechanics work.

- A and B inference completed for seeds 42, 43, and 44.
- C training completed for seeds 42, 43, and 44.
- C evaluation completed for seed 42.
- D inference completed for seed 42.
- B retrieval returns exactly three documents per item.
- No current out-of-memory problem appears in completed runs.
- C validation loss decreases across epochs.

### Final thesis result

Current result is **not enough for a thesis claim that one method produces better dashboards**.

Main reasons:

- Qwen Method D now exists for seed 42, but strict schema validity is 0%.
- C seeds 43 and 44 have no evaluation.
- D has no seeds 43 and 44 evaluation.
- Strict schema validity is only 0% to 1.46%, far below the 90% engineering gate.
- Exact KPI, encoding, and aggregation correctness is also close to zero.
- Robustness is very weak.
- Current chart metrics use synthetic labels and are circular for primary quality claims.
- Independent benchmark, realism evaluation, and human usefulness ratings are not complete.
- Parse confidence intervals in `metrics.json` are calculated incorrectly.
- OLMo A completed, but OLMo B failed after 35/274 items for each seed because RAG prompt headings exceeded the input budget.

### Practical decision

Use dashboard v4 for the next controlled repair and rerun.

Do not present current Qwen A-D or OLMo A-B results as final thesis evidence. Current artifacts are useful as an engineering checkpoint and failure diagnosis.

## 1. Data inventory

### Frozen dataset counts

- Train: **2,932** rows
- Validation: **613** rows
- Test: **274** rows
- Paraphrase test: **274** rows
- Missing-information test: **274** rows
- Human-evaluation items: **40**

### Dataset checks

All main v4 checks pass:

- JSONL schema validation: pass
- Semantic validation: pass
- Duplicate IDs: zero
- Duplicate records: zero
- Duplicate goals and briefs: zero
- Generated train/test overlap: zero
- Generated/human-evaluation overlap: zero
- Test bytes unchanged from parent dataset: pass
- Human-evaluation bytes unchanged from parent dataset: pass
- Run dataset hashes match frozen hashes: pass

### Dataset limitation

Generated records are AI-generated. They are not human or expert gold records.

The internal test labels are generated with the same general rule family used to create the synthetic data. This makes internal chart-type accuracy useful for pipeline diagnostics, but not sufficient for claims about real dashboard quality.

Test chart labels are also strongly imbalanced:

- Bar: 208
- Pie: 36
- Line: 17
- Stacked bar: 11
- Scatter: 2

Macro-F1 must therefore be read with care. Invalid outputs dominate current scores, and rare classes have very small support.

### Dataset metadata issue

`data/eval/robustness_v4/manifest.json` points to v4 data but still contains the label `dataset_version: dashboard_v3`.

Hashes and source counts are correct, so this looks like a metadata-label problem. Correct it before final reproducibility packaging.

## 2. Run inventory

### Available results

- **Method A:** seeds 42, 43, 44 complete
- **Method B:** seeds 42, 43, 44 complete
- **Method C:** seed 42 complete
- **Method C:** seeds 43 and 44 contain training artifacts and adapters, but no predictions or metrics
- **Method D:** seed 42 complete; seeds 43 and 44 not present

### OLMo comparison inventory

- **OLMo A:** seeds 42, 43, 44 complete; 274/274 predictions each
- **OLMo B:** seeds 42, 43, 44 failed; 35/274 predictions each
- **OLMo C:** no results

OLMo B failure message:

```text
Prompt and RAG passage headings exceed the input-token budget; shorten the base prompt or increase max_seq_length.
```

OLMo B therefore cannot be used as a complete quality comparison with Qwen B. It is an execution failure first, not a fair model-quality result.

Every completed A, B, and C seed-42 run contains predictions, per-item evaluation, metrics, configuration, dataset hashes, KB hashes, and environment information.

### Run summary issue

`ftm_runtime/runs/dashboard_v4/matrix_summary.json` currently lists only OLMo A/B seed 44. It omits current Qwen A-D and most OLMo artifacts. Regenerate it before using it as a run index.

### Reproducibility issue

All inspected manifests contain `git_hash: unknown`.

Dataset and configuration hashes exist, but exact code revision is not pinned. This is not acceptable for final thesis packaging.

## 3. Metric meanings

Current metrics have different strictness levels.

- **JSON parse rate:** a JSON object can be extracted from model output.
- **Strict schema validity:** extracted object passes the complete Pydantic schema, including types and allowed enum values.
- **Required-key rate:** required top-level keys exist. This is lenient; values can still have wrong types.
- **Completeness:** required fields are present and non-empty. This does not guarantee valid types or correct values.
- **Exact KPI/task/count/encoding/aggregate metrics:** output matches the synthetic reference structure.
- **Top-1 and Macro-F1:** internal synthetic chart-label diagnostics. They are not independent dashboard-quality evidence.
- **Grounding:** current B results use a lexical proxy, not semantic faithfulness verification.
- **Paraphrase and missing-information metrics:** robustness diagnostics. They depend strongly on strict-valid outputs.

Important: high required-key rate and high completeness do not mean output is usable. Current outputs often contain all keys but fail because `encoding` has the wrong type.

## 4. Method A: prompt-only baseline

All three A runs have 274/274 predictions.

### Seed 42

- JSON parse: **99.64%**
- Strict schema validity: **0.36%**; 1/274
- Required keys: **99.64%**
- Completeness: **0.9964**
- Top-1: **0.36%**
- Macro-F1: **0.1111**
- Average latency: **7.077 seconds**
- P95 latency: **8.514 seconds**
- Paraphrase consistency: **0%**
- Paraphrase accuracy: **0.36%**
- Missing-information clarification: **3.65%**
- Missing-information schema rate: **99.64%**

Exact structured results:

- Exact task classification: 0%
- Exact KPI selection: 0%
- Exact mapping count: 0%
- Exact encoding: 0%
- Exact aggregation: 0%

### Seed 43

- JSON parse: **100%**
- Strict schema validity: **0%**; 0/274
- Required keys: **100%**
- Completeness: **1.0**
- Top-1: **0%**
- Macro-F1: **0**
- Average latency: **7.074 seconds**
- P95 latency: **8.112 seconds**
- Paraphrase consistency: **0%**
- Paraphrase accuracy: **0.36%**
- Missing-information clarification: **2.92%**
- Missing-information schema rate: **99.64%**

Exact structured results are 0% for task, KPI, mapping count, encoding, and aggregation.

### Seed 44

- JSON parse: **100%**
- Strict schema validity: **0.36%**; 1/274
- Required keys: **100%**
- Completeness: **1.0**
- Top-1: **0.36%**
- Macro-F1: **0.0185**
- Average latency: **7.064 seconds**
- P95 latency: **8.322 seconds**
- Paraphrase consistency: **0.36%**
- Paraphrase accuracy: **0.73%**
- Missing-information clarification: **2.92%**
- Missing-information schema rate: **99.64%**

Exact task classification, KPI selection, and mapping count are each only 0.36%. Exact encoding and aggregation are 0%.

### A interpretation

A produces extractable JSON almost every time, but almost never produces a complete valid dashboard object.

This is not a good final baseline result. Latency is reasonable. Coverage is good. Output validity and correctness are not.

## 5. Method B: prompt plus RAG

All three B runs have 274/274 predictions.

RAG wiring works:

- Three retrieved KB documents are present for every item.
- KB hashes match the current KB.
- No retrieval coverage failure appears.

### Seed 42

- JSON parse: **100%**
- Strict schema validity: **1.09%**; 3/274
- Required keys: **100%**
- Completeness: **1.0**
- Top-1: **1.09%**
- Macro-F1: **0.0217**
- Average latency: **6.597 seconds**
- P95 latency: **8.202 seconds**
- Paraphrase consistency: **1.09%**
- Paraphrase accuracy: **1.09%**
- Missing-information clarification: **0%**
- Missing-information schema rate: **100%**
- Grounding: 77.78% supported claims, 22.22% unsupported
- Grounding sample: 3 valid outputs, 9 claims

Exact structured results:

- Task classification: 0.36%
- KPI selection: 1.09%
- Mapping count: 1.09%
- Encoding: 0%
- Aggregation: 0%

### Seed 43

- JSON parse: **99.64%**
- Strict schema validity: **1.46%**; 4/274
- Required keys: **99.64%**
- Completeness: **0.9964**
- Top-1: **1.09%**
- Macro-F1: **0.1125**
- Average latency: **6.670 seconds**
- P95 latency: **8.505 seconds**
- Paraphrase consistency: **0.36%**
- Paraphrase accuracy: **0.73%**
- Missing-information clarification: **0.73%**
- Missing-information schema rate: **100%**
- Grounding: 64.58% supported claims, 35.42% unsupported
- Grounding sample: 4 valid outputs, 13 claims

Exact structured results:

- Task classification: 0.36%
- KPI selection: 1.09%
- Mapping count: 1.09%
- Encoding: 0.36%
- Aggregation: 0%

### Seed 44

- JSON parse: **100%**
- Strict schema validity: **1.09%**; 3/274
- Required keys: **100%**
- Completeness: **1.0**
- Top-1: **1.09%**
- Macro-F1: **0.0217**
- Average latency: **6.627 seconds**
- P95 latency: **8.359 seconds**
- Paraphrase consistency: **1.09%**
- Paraphrase accuracy: **1.46%**
- Missing-information clarification: **0.36%**
- Missing-information schema rate: **100%**
- Grounding: 77.78% supported claims, 22.22% unsupported
- Grounding sample: 3 valid outputs, 9 claims

Exact structured results:

- Task classification: 0.36%
- KPI selection: 1.09%
- Mapping count: 1.09%
- Encoding: 0.36%
- Aggregation: 0%

### B interpretation

B improves strict validity slightly over A:

- A three-seed mean strict validity: approximately **0.24%**
- B three-seed mean strict validity: approximately **1.21%**

This is a measurable improvement, but not a useful final result. B still fails almost every strict output and does not show reliable grounding because grounding is measured on only 3–4 valid outputs per run.

RAG retrieval works. RAG-conditioned dashboard generation does not yet meet the output contract.

## 6. Method C: QLoRA fine-tuning

### C training

C training completed for all three seeds.

Training configuration:

- Three epochs
- 1,101 global steps
- QLoRA 4-bit NF4
- LoRA rank 16
- LoRA alpha 32
- Learning rate 0.0002
- Batch size 1
- Gradient accumulation 8
- Approximately 17.4 million trainable parameters
- NVIDIA A30 execution

Validation loss:

- Seed 42: **0.07062, 0.05818, 0.05664** across epochs
- Seed 43: **0.07071, 0.05840, 0.05640** across epochs
- Seed 44: **0.07126, 0.05852, 0.05661** across epochs

Validation token accuracy reaches approximately 98.4% by the final epoch.

Interpretation:

- Trainer runs correctly.
- Adapter checkpoints are written.
- Validation loss decreases normally.
- No obvious training divergence or overfitting appears.
- Token accuracy is not dashboard quality. It does not replace strict schema and semantic evaluation.

### C seed 42 evaluation

- JSON parse: **97.45%**
- Strict schema validity: **1.46%**; 4/274
- Required keys: **97.45%**
- Completeness: **0.9745**
- Top-1: **1.46%**
- Macro-F1: **0.0635**
- Average latency: **9.376 seconds**
- P95 latency: **10.918 seconds**
- Paraphrase consistency: **1.46%**
- Paraphrase accuracy: **3.65%**
- Missing-information clarification: **0%**
- Missing-information schema rate: **95.99%**

Exact structured results:

- Task classification: 0.73%
- KPI selection: 1.46%
- Mapping count: 1.46%
- Encoding: 1.46%
- Strict encoding: 0.36%
- Aggregation: 1.46%

C improves raw chart-type selection, but it does not produce valid complete dashboard objects often enough.

### C seeds 43 and 44 evaluation status

No predictions or metrics exist for these seeds.

Their inference configuration contains:

- `max_seq_length: 1024`
- `max_new_tokens: 1024`

This is invalid. Generation length must be smaller than sequence length. Seed 42 later used `max_new_tokens: 512` and completed.

The existing seed 43 and 44 adapters may be reusable. First run corrected inference and evaluation. Retraining should not be assumed necessary unless target-format inspection proves a training-data problem.

### C interpretation

C is the strongest current method for raw chart-type choice, but not for strict dashboard output.

It is also slower:

- A average latency: approximately 7.07 seconds
- B average latency: approximately 6.63 seconds
- C seed 42 average latency: approximately 9.38 seconds

Fine-tuning adds approximately 2.3 seconds per item compared with A. This is reasonable engineering behavior, but current quality does not justify the cost yet.

## 7. Method D status

Method D seed 42 now exists and completed 274/274 predictions.

Seed-42 metrics:

- JSON parse: **95.99%**
- Strict schema validity: **0%**; 0/274
- Required keys: **95.99%**
- Completeness: **0.958**
- Exact task, KPI, mapping count, encoding, and aggregation: **0%**
- Strict Top-1: **0%**
- Macro-F1: **0**
- Average latency: **9.654 seconds**
- P95 latency: **11.488 seconds**
- Paraphrase consistency: **0%**
- Paraphrase accuracy: **0.36%**
- Missing-information clarification: **0%**
- Missing-information schema rate: **97.08%**

D uses:

- Qwen C seed-42 adapter path
- `max_new_tokens: 512`
- TF-IDF RAG
- `top_k: 3`
- Current KB hashes
- Full test coverage

RAG retrieval executes for all 274 items, but no valid output contributes grounding claims. D therefore proves combined fine-tuned-plus-RAG execution, not quality improvement.

Raw chart diagnostic:

- Chart JSON extracted for 263/274 items
- Raw chart-type accuracy: **92.78%**

This is below C seed 42 raw chart diagnostic of approximately 97.0%, while D strict schema validity falls from C's 1.46% to 0%. Current evidence shows no C-to-D improvement.

### D provenance issue

D manifest points to the correct Qwen C seed-42 adapter path, but D configuration still says:

```text
adapter_source_experiment: E03_qwen0_5b_ft
```

Manifest fields `source_c_run_id` and `adapter_manifest_hash` are null. This looks like stale metadata, not necessarily wrong adapter loading. Correct metadata before thesis packaging so D-to-C lineage is auditable.

D seeds 43 and 44 remain unavailable. Four-method multi-seed comparison remains incomplete.

## 8. Main output failure

Raw prediction inspection shows why strict validity is so low.

Most failures are not missing JSON. They are wrong field types:

- A: approximately 272–274 outputs per seed have `encoding` as a non-dictionary value.
- B: approximately 269–271 outputs per seed have the same problem.
- C seed 42: 263 outputs have non-dictionary `encoding`; 7 have no JSON object extracted.
- D seed 42: 263 outputs have non-dictionary `encoding`; 11 have no JSON object extracted.

This means model output often looks structurally complete, but fails the schema because `encoding` is a string instead of the required structured object.

This is a model/output-contract problem. It is not only a parser problem.

Check alignment between:

- Training target serialization
- Prompt example
- Pydantic schema
- Output parser
- Exact-match evaluator

All five must describe `encoding` in the same way.

### Raw chart diagnostic

When chart type is extracted from otherwise invalid raw JSON, chart-type accuracy is much higher:

- A: **86.81%, 87.23%, 86.86%** for seeds 42, 43, 44
- B: **90.51%, 90.11%, 89.05%** for seeds 42, 43, 44
- C seed 42: **97.0%**
- D seed 42: **92.78%**

This is useful diagnostic evidence:

- C learned chart-type behavior well on this synthetic test.
- B is slightly better than A on raw chart type.
- D remains below C after adding RAG to fine-tuned generation.
- Full dashboard outputs remain invalid.

Do not replace strict schema metrics with raw chart metrics in the thesis. Raw chart extraction ignores the failed output contract.

## 9. Comparison with mini-dataset goals

The readiness document records these mini/pilot results after the prompt-budget fix:

- A: 90% JSON parse, 90% strict schema, 60% Top-1, Macro-F1 0.2857
- B: 100% JSON parse, 90% strict schema, 75% Top-1, Macro-F1 0.6236
- Mini completeness remained approximately 0.333, below the 0.80 engineering gate

Older tiny-run documentation reports approximately:

- C strict schema: 84%
- D strict schema: 88%
- C completeness: 0.4000
- D completeness: 0.4733

These are historical target/reference results, not identical to current full v4 configuration. They should not be treated as a perfectly matched experiment.

### Full versus mini

Full v4 improves:

- Raw JSON extraction
- Required-key presence
- Completeness
- Training stability
- RAG execution coverage

Full v4 regresses badly in:

- Strict schema validity
- Exact KPI and encoding correctness
- Robustness
- Strict chart metrics

The central result is not “big dataset failed.” The central result is “big-data run produces many complete-looking but schema-invalid outputs.”

## 10. Which method is best?

Answer depends on metric.

### Compact A-B-C comparison

| Method | Evaluated runs | JSON parse mean | Strict schema mean | Completeness mean | Strict Top-1 mean | Macro-F1 mean | Average latency | RAG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A prompt-only | 3/3 | 99.88% | 0.24% | 0.9988 | 0.24% | 0.0432 | 7.07 s | No |
| B prompt + RAG | 3/3 | 99.88% | 1.21% | 0.9988 | 1.09% | 0.0520 | 6.63 s | Yes |
| C QLoRA | 1/3 | 97.45% | 1.46% | 0.9745 | 1.46% | 0.0635 | 9.38 s | No |
| D QLoRA + RAG | 1/1 | 95.99% | 0% | 0.9580 | 0% | 0 | 9.65 s | Yes |

Table notes:

- A and B values are means across seeds 42, 43, and 44.
- C and D values are seed 42 only. C seeds 43 and 44 have training artifacts but no evaluation. D seeds 43 and 44 are not present.
- Macro-F1 and strict Top-1 use internal synthetic labels.
- Completeness is lenient. It does not mean output passes schema validation.
- D is included because seed-42 result now exists.

### Simple experiment-running result

- A running: **yes**, all three seeds completed.
- B running: **yes**, all three seeds completed; three RAG documents retrieved per item.
- C training: **yes**, all three seeds completed.
- C evaluation: **partial**, only seed 42 completed.
- D running: **yes for seed 42**, 274/274 predictions completed.
- D evaluation: **partial**, only seed 42 exists.
- Dataset loading and hash checks: **pass**.
- Qwen seed-42 A-D experiment: **complete at artifact level**.
- Qwen multi-seed A-D experiment: **not complete**.

### Metric-by-metric improvement

| Metric | Best current result | Method | Simple meaning |
| --- | ---: | --- | --- |
| JSON parse | 99.88% mean | A and B | Best extraction reliability. C lower at 97.45%. |
| Required keys | About 99.88% mean | A and B | Keys usually exist, but types can still be wrong. |
| Completeness | 0.9988 mean | A and B | Output usually non-empty. Not proof of valid dashboard. |
| Strict schema validity | 1.46% | C seed 42 / B seed 43 | C and B highest, but all methods fail 90% gate. |
| Strict Top-1 | 1.46% | C seed 42 | Small numerical improvement only; most outputs invalid. |
| Macro-F1 | 0.0635 | C seed 42 | Highest current value, still weak and synthetic-only. |
| Raw chart-type choice | About 97.0% | C seed 42 | Strongest chart-type diagnostic before full schema validation. |
| D raw chart-type choice | 92.78% | D seed 42 | Higher than A/B raw diagnostic, lower than C; still not full-output validity. |
| RAG retrieval | Three documents per item | B and D | RAG wiring works. Final output validity still fails. |
| Latency | 6.63 s mean | B | Fastest completed method. |
| Paraphrase consistency | 1.46% | C seed 42 | Highest current value, still very poor. |
| Missing-information clarification | 3.65% | A seed 42 | Highest value, but too low to show reliable clarification behavior. |

### What improved, and what did not

- **A to B:** RAG raises strict schema mean from about 0.24% to 1.21% and raw chart-type accuracy from about 86.97% to 89.89%. Improvement exists, but remains too small for thesis quality claims.
- **B to C:** QLoRA raises raw chart-type accuracy to about 97.0% and strict Top-1 to 1.46% in seed 42. C also has highest Macro-F1. Full structured output remains invalid in 270/274 cases.
- **C to D:** Adding RAG to fine-tuned C does not improve current seed-42 results. Strict schema falls from 1.46% to 0%; raw chart diagnostic falls from about 97.0% to 92.78%; latency rises from 9.38 to 9.65 seconds.
- **Latency:** B is fastest. C is slowest because fine-tuned inference takes about 9.38 seconds per item.
- **Robustness:** C has highest measured paraphrase values, but only 1.46% consistency. No method is robust yet.
- **RAG:** B and D prove retrieval execution, not dashboard-quality improvement.
- **D:** combined pipeline runs for seed 42, but no quality improvement is visible.

### Best for raw chart type

**C seed 42**: approximately 97% raw chart-type accuracy. D reaches 92.78% on extracted raw chart objects.

This is only a supplementary diagnostic on synthetic labels.

### Best for strict schema

**C seed 42 and B seed 43**: 1.46% strict validity.

This is still far below the 90% gate. No method is good enough here.

### Best for latency

**B**: approximately 6.63 seconds average across seeds.

### Best for completeness

**A and B**: approximately 0.9964–1.0.

This result is lenient. It does not mean outputs are valid.

### Best for RAG execution

**B**: retrieval works and exactly three documents are included per item.

### Best overall thesis method

No method can be selected yet.

C looks strongest for internal raw chart choice. B and D prove RAG wiring. D does not improve C on current seed 42. No method proves valid, useful, or superior dashboards.

## 11. Engineering gate check

Current gate status:

- Coverage 100% for completed runs: **pass**
- Missing predictions zero for completed runs: **pass**
- JSON parse at least 95%: **pass**
- Required keys at least 95%: **pass** for completed runs
- Completeness at least 0.80: **pass**
- RAG retrieves exactly three documents: **pass** for B
- Strict schema at least 90%: **fail**
- Exact KPI/encoding/aggregation correctness: **fail**
- Top-3 support at least 80%: **fail and metric invalid**
- C evaluation history and best checkpoint: **pass for training; evaluation incomplete**
- D adapter loading and RAG execution: **pass for seed 42; provenance metadata incomplete**
- Frozen hashes unchanged: **pass**
- Clean reproducible git revision: **fail**

Top-3 is not currently a useful primary metric. The dashboard v4 test set contains no alternatives in most records, and current outputs almost never contain three recommendations. Report it as unsupported rather than as a quality score.

## 12. Robustness and grounding

Robustness is not reasonable yet.

- A paraphrase consistency: 0–0.36%
- B paraphrase consistency: 0.36–1.09%
- C seed 42 paraphrase consistency: 1.46%
- D seed 42 paraphrase consistency: 0%
- A missing-information clarification: 2.92–3.65%
- B missing-information clarification: 0–0.73%
- C missing-information clarification: 0%
- D missing-information clarification: 0%
- Missing-information schema failure: approximately 95.99–100%

B grounding is also not ready for thesis interpretation:

- The mode is `lexical_proxy`.
- Only 3–4 strict-valid outputs contribute claims.
- Claim coverage is only 1.09–1.46% of the test set.
- Retrieval coverage is high, but valid grounded dashboard output is not.

D has 274 retrieved-document contexts but zero valid grounding claims because zero D outputs pass strict schema validation.

## 13. Qwen versus OLMo A/B comparison

OLMo results use `OLMo-2-0425-1B-Instruct`, recorded as 1.49 billion parameters. Qwen results use Qwen3-1.7B. Both use dashboard v4 hashes, 274 test items, seeds 42/43/44, `max_seq_length: 1024`, and `max_new_tokens: 512` in A/B manifests.

This is useful model comparison, but not a perfectly controlled architecture comparison. Tokenizers, chat templates, parameter counts, and model families differ.

### Method A: prompt-only

| Metric | Qwen3-1.7B A | OLMo 1.49B A | Better result | Interpretation |
| --- | ---: | ---: | --- | --- |
| Evaluated seeds | 3/3 | 3/3 | Tie | Both complete. |
| Prediction coverage | 100% | 100% | Tie | Fair coverage. |
| JSON parse mean | 99.88% | 34.31% | Qwen | OLMo often fails to produce extractable JSON. |
| Strict schema mean | 0.24% | 0% | Qwen | Both fail schema gate; Qwen slightly less bad. |
| Required keys mean | 99.88% | 24.09% | Qwen | Qwen output shape much more complete. |
| Completeness mean | 0.9988 | 0.2739 | Qwen | OLMo outputs often incomplete. |
| Strict Top-1 mean | 0.24% | 5.96% | OLMo | Synthetic diagnostic; based on very few usable outputs. |
| Macro-F1 mean | 0.0432 | 0.0829 | OLMo | Synthetic diagnostic; not valid dashboard-quality proof. |
| Raw chart diagnostic | About 87.0% on 821/822 | About 84.8% on 282/822 | Qwen | OLMo denominator is much smaller; not a clean quality win. |
| Average latency | 7.07 s | 4.85 s | OLMo | OLMo is faster. |

OLMo A has higher field-level exact task/KPI/count and synthetic Top-1 values than Qwen A, but strict schema validity is 0% and JSON extraction is only 34.31%. This means OLMo sometimes makes plausible partial chart decisions, but does not reliably produce a usable dashboard object.

Qwen A is clearly stronger for operational output reliability. OLMo A is faster. Neither prompt-only baseline is thesis-ready for dashboard quality.

### Method B: prompt plus RAG

| Metric | Qwen3-1.7B B | OLMo 1.49B B | Interpretation |
| --- | ---: | ---: | --- |
| Evaluated seeds | 3/3 complete | 0/3 complete | OLMo B comparison incomplete. |
| Prediction coverage mean | 100% | 12.77% | OLMo B failed after 35/274 items. |
| JSON parse mean | 99.88% | 5.59% | Qwen result operationally usable; OLMo result not. |
| Strict schema mean | 1.21% | 0% | Both fail gate; OLMo has no valid outputs. |
| Completeness mean | 0.9988 | 0.0444 | OLMo output coverage is severely incomplete. |
| Strict Top-1 mean | 1.09% | 1.82% | Not comparable because OLMo misses 239/274 items. |
| Macro-F1 mean | 0.0520 | 0.0241 | Not comparable because OLMo run failed. |
| Average latency | 6.63 s | 4.83 s on 35 items | OLMo latency measured only on partial run. |

All OLMo B seeds failed with:

```text
Prompt and RAG passage headings exceed the input-token budget; shorten the base prompt or increase max_seq_length.
```

Each OLMo B seed has 35 predictions and 239 missing items. This is an inference-budget failure, not evidence that OLMo is intrinsically worse at RAG. It does mean current OLMo B result cannot support a scientific A/B model comparison.

Qwen B completed all 274 items because current Qwen RAG prompt fitting truncates passage content while preserving retrieval headings. OLMo B needs model-aware prompt fitting or a larger effective input budget before rerun.

### OLMo comparison conclusion

- Qwen A is much better at producing complete extractable structured output.
- OLMo A is faster and shows some higher partial synthetic chart metrics, but strict schema validity is zero.
- Qwen B completes full RAG evaluation. OLMo B does not.
- No claim that Qwen is universally better than OLMo is justified yet. Current evidence supports only this project-specific engineering finding: **Qwen is more reliable under current dashboard-v4 prompt and schema protocol; OLMo is faster, but OLMo RAG execution is not yet validly evaluated.**
- OLMo C and D are absent, so no fine-tuning comparison exists.

## 14. Independent thesis evidence

Current full-run manifests identify results as `internal-synthetic`.

Still missing or incomplete:

- Independent L1 chart-effectiveness evaluation for current v4 A-D outputs
- Independent benchmark result for current v4 A-D outputs
- L3 realism comparison
- L4 human usefulness ratings
- Multi-seed evaluation for C
- Any evaluation for D

Existing old reports must not be mixed into this full-run comparison:

- `experiments/results/benchmark_v1_eval.md` is a July smoke run with 30 items, one run, 50% JSON parse, 0% schema validity, and 4.55% covered accuracy.
- `experiments/results/l1_independent_results.md` uses cached synthetic v1 predictions and is explicitly marked diagnostic/limited.
- Human-evaluation items exist, but ratings are not present.

These files do not provide current full v4 A-D evidence.

## 15. Reporting and artifact problems

### Confidence interval bug

`metrics_auto.json` point estimates use raw JSON extraction for `json_parse_rate`.

`metrics.json` confidence intervals calculate the parse vector from strict-valid predictions instead. This makes parse confidence intervals wrong.

Example:

- A seed 43 point parse rate: 100%
- Reported parse confidence interval behaves like 0/274 strict-valid outputs

Fix reporting before thesis tables. Current point estimates remain useful diagnostics; current parse CIs do not.

### Stale C seed 42 errors

The C seed 42 root log contains errors from older failed attempts. Final seed-42 artifacts use corrected settings and completed, but stale logs should be separated before packaging.

Decisive old errors:

```text
max_new_tokens must be positive and smaller than max_seq_length (1024 vs 1024).
```

```text
Prompt exceeds the reserved input-token budget: 438 > 434.
```

These old failures explain development history. They should not be counted as failures of the corrected final C seed-42 artifacts.

### Inconsistent metric schemas

Older A/B artifacts do not all contain exactly the same metric fields. Some fields, such as `exact_encoding_strict` and `claim_coverage_rate`, appear only in newer artifacts.

Use one reporting version for final reruns.

## 16. Is the result reasonable for the master's thesis?

### Reasonable claims now

Current artifacts support these limited claims:

- Dashboard v4 dataset construction passed integrity checks.
- Full-data training completed successfully for C seeds 42–44.
- Prompt-only and RAG inference can run across 274 test items.
- RAG retrieves three documents per item.
- Qwen seed-42 A-D artifact generation completes.
- Qwen D loads C seed-42 adapter and executes RAG for 274 items.
- OLMo A completes all three prompt-only seeds.
- QLoRA training loss and validation loss behave normally.
- C improves raw synthetic chart-type selection in the inspected seed-42 diagnostic.
- Current pipeline has a strong schema/output-format failure that must be addressed.

### Claims not supported now

Do not claim yet:

- Method C or D produces better dashboards than A or B.
- D improves fine-tuned generation after adding RAG.
- Qwen is universally better than OLMo.
- RAG improves dashboard quality.
- Outputs are reliable enough for end users.
- Outputs are human-effective.
- Outputs are realistic.
- Current methods generalize beyond synthetic labels.
- Four-method multi-seed comparison is complete.

### Final judgment

Current result is reasonable as an **intermediate engineering experiment**.

Current result is not reasonable as the **final quality result for the master's thesis**.

This does not require perfectionism. Minimum blockers are concrete:

- Fix `encoding` output contract.
- Evaluate C seeds 43 and 44.
- Evaluate D seeds 43 and 44 after fixing adapter provenance metadata.
- Fix parse confidence intervals.
- Pin code revision.
- Run at least one independent evaluation before making quality claims.
- Rerun OLMo B after fixing model-specific RAG prompt-budget handling if OLMo comparison remains a thesis goal.

## 17. Recommended next run

1. Inspect one training target, one prompt example, one raw prediction, Pydantic schema, and evaluator side by side.
2. Make `encoding` representation identical across all five.
3. Use valid inference settings for all methods. Current corrected C setting uses `max_new_tokens: 512` with `max_seq_length: 1024`.
4. Evaluate existing C seed-43 and seed-44 adapters before retraining.
5. Evaluate D seed 42 again after correcting adapter source metadata, then run D seeds 43 and 44 with matching C adapters and RAG.
6. Fix confidence-interval calculation and normalize metric fields.
7. Regenerate `matrix_summary.json`.
8. Record actual git commit hash in every manifest.
9. Rerun A-D with one code revision and one metric schema.
10. Rerun OLMo B after prompt-budget repair if model comparison is required.
11. Run independent benchmark and human evaluation if thesis claims include chart effectiveness or usefulness.

### Minimum go/no-go rule

Do not call next run thesis-ready unless:

- A-D results exist.
- C has all intended seeds evaluated.
- Strict schema validity is near planned 90% gate, or limitation is explicitly accepted and reported.
- Confidence intervals use correct vectors.
- Independent evaluation is reported separately from internal synthetic diagnostics.

## 18. Code correction status (2026-08-28)

Previously identified implementation blockers are now corrected in code:

- strict generation-only response schema for object-valued `encoding`;
- modern, pinned Outlines 1.3.3 decoder path;
- strict schema evaluator aligned with decoder contract;
- separate encoding object, `x`, `y`, `aggregate`, and joint mapping metrics;
- corrected JSON-parse confidence-interval vector;
- optional TRL response-only loss for new C/D adapters;
- supervised RAG Recall@3, MRR@3, and nDCG@3 with manual qrels only.

These changes do not improve historical numbers retroactively and do not prove model quality. They make the next measurements interpretable. Required next evidence is defined in [Constrained Encoding and HPC Validation Protocol](constrained_encoding_hpc_protocol.md): 20 fixed validation items, then a disjoint 50-item pilot, then 27B only after all gates pass.

## 19. Evidence paths

### Current metrics

- [A seed 42 metrics](../../ftm_runtime/runs/dashboard_v4/qwen3_1_7b/A/seed_42/metrics_auto.json)
- [A seed 43 metrics](../../ftm_runtime/runs/dashboard_v4/qwen3_1_7b/A/seed_43/metrics_auto.json)
- [A seed 44 metrics](../../ftm_runtime/runs/dashboard_v4/qwen3_1_7b/A/seed_44/metrics_auto.json)
- [B seed 42 metrics](../../ftm_runtime/runs/dashboard_v4/qwen3_1_7b/B/seed_42/metrics_auto.json)
- [B seed 43 metrics](../../ftm_runtime/runs/dashboard_v4/qwen3_1_7b/B/seed_43/metrics_auto.json)
- [B seed 44 metrics](../../ftm_runtime/runs/dashboard_v4/qwen3_1_7b/B/seed_44/metrics_auto.json)
- [C seed 42 metrics](../../ftm_runtime/runs/dashboard_v4/qwen3_1_7b/C/seed_42/metrics_auto.json)
- [D seed 42 metrics](../../ftm_runtime/runs/dashboard_v4/qwen3_1_7b/D/seed_42/metrics_auto.json)

### OLMo comparison metrics

- [OLMo A seed 42 metrics](../../ftm_runtime/runs/dashboard_v4/olmo2_1_49b/A/seed_42/metrics_auto.json)
- [OLMo A seed 43 metrics](../../ftm_runtime/runs/dashboard_v4/olmo2_1_49b/A/seed_43/metrics_auto.json)
- [OLMo A seed 44 metrics](../../ftm_runtime/runs/dashboard_v4/olmo2_1_49b/A/seed_44/metrics_auto.json)
- [OLMo B seed 42 metrics](../../ftm_runtime/runs/dashboard_v4/olmo2_1_49b/B/seed_42/metrics_auto.json)
- [OLMo B seed 43 metrics](../../ftm_runtime/runs/dashboard_v4/olmo2_1_49b/B/seed_43/metrics_auto.json)
- [OLMo B seed 44 metrics](../../ftm_runtime/runs/dashboard_v4/olmo2_1_49b/B/seed_44/metrics_auto.json)

### C training

- [C seed 42 training metadata](../../ftm_runtime/adapters/dashboard_v4/qwen3_1_7b/C/seed_42/adapter/training_metadata.json)
- [C seed 43 adapter](../../ftm_runtime/adapters/dashboard_v4/qwen3_1_7b/C/seed_43/)
- [C seed 44 adapter](../../ftm_runtime/adapters/dashboard_v4/qwen3_1_7b/C/seed_44/)

### Source and reports

- [Reporting implementation](../../src/evaluation/reporting.py)
- [Dataset validation report](../../data/frozen/dashboard_v4/reports/validation_report.json)
- [Dataset duplicate report](../../data/frozen/dashboard_v4/reports/duplicate_report.json)
- [Dataset leakage report](../../data/frozen/dashboard_v4/reports/leakage_report.json)
- [Old benchmark smoke report](../../experiments/results/benchmark_v1_eval.md)
- [Old limited L1 report](../../experiments/results/l1_independent_results.md)
