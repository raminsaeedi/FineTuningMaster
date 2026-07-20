# Scientific Dataset Validity Audit

Read-only audit of dataset, benchmark, and evaluation validity for the thesis
*Fine-Tuning LLMs for Structured Dashboard Design Recommendations*. Date: 2026-07-05.
No files were modified during the audit.

## Terminology
- **Rule leakage** — train and test share the same deterministic label-generation logic, so
  high test accuracy may only show reproduction of a generator rule, not design quality.
- **Data leakage** — exact/near-duplicate items, labels, or source lineage from training
  appearing in validation/test/benchmark.
- **Memorization** — the model repeats seen training examples instead of generalizing to
  unseen briefs, domains, or task combinations.

## Scope inspected
Dataset generation (`src/data_pipeline/synth_generator.py`, `synth_generator_v2.py`),
split logic (`splits.py`), processed/frozen data (`data/processed/*`,
`data/frozen/dashboard_v2/*`), eval data (`data/eval/*`), metrics
(`src/evaluation/metrics/*`), reporting (`src/evaluation/reporting.py`), pipeline
(`src/pipeline/runner.py`), human eval (`src/evaluation/human/*`), cached predictions
(`experiments/outputs/*`), and existing docs.

## Risk register
| # | Risk | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | Exact duplicate leakage | **Controlled** | `frozen_validation.find_duplicate_ids`; v2 validation report shows 0 duplicate ids |
| 2 | Near-duplicate leakage | **Gap** | only exact fingerprint (`builders/leakage.fingerprint`); no fuzzy/similarity check exists |
| 3 | Source/provenance leakage | **Partial** | provenance only in `brief.extra.source_id`; no cross-set source check |
| 4 | Label-lineage leakage | **Present by design** | train + synthetic test both from the generator rule; no `label_lineage_id` field |
| 5 | **Rule leakage (construct validity)** | **CRITICAL** | Top-1/Top-3/macro-F1/paraphrase-accuracy score vs synthetic gold (`pipeline/runner.py:93-100` references = `recommendation`); `robustness.py` self-documents the circularity |
| 6 | Memorization | **Untested** | no held-out-combination / generalization analysis |
| 7 | Benchmark independent of train? | **Gap → must-build** | L1 CSV exists and is independent; `real_briefs` verified (10 items + provenance doc); but there is **no labeled benchmark dataset** for scoring A/B/C/D and L1 is unused |
| 8 | Metric safety | see below | schema/completeness/consistency/grounding = diagnostic-safe; chart-accuracy = circular |
| 9 | Claims supported today | see policy | usefulness/quality unsupported (0 human ratings); independent chart-selection unsupported (L1 scorer absent) |

## Rule leakage — the central finding
The synthetic generator assigns labels deterministically:
`KEYWORD_TASK → task_type → TASK_CHART[task_type][0] → chart_type`
(`src/data_pipeline/synth_generator.py`). Both `train` and the synthetic `test` split are
produced by this same rule, and the split is a per-row content hash (`splits.py`). Therefore
`top_1_accuracy` / `top_3_accuracy` / `macro_f1` / `paraphrase_accuracy` measure how well the
model **reproduced the generator rule**, not whether it makes good dashboards. The fine-tuned
numbers (E03 ≈ 77.8%, E04 ≈ 86.7% Top-1) are inflated by this construct-validity problem and
must be reported as **internal synthetic diagnostics only**.

## Metric safety
- **Diagnostic-safe (already corrected in code):** `json_parse_rate`; `schema_validity_rate`
  (full Pydantic + strict enums, `schema_compliance.full_schema_valid`); `completeness_score`
  (empty string/list/dict/null counts as incomplete, `completeness_fraction`);
  `paraphrase_consistency`; `grounding` (carries `mode`, defaults to `lexical_proxy`).
- **Circular / internal-only:** `top_1_accuracy`, `top_3_accuracy`, `macro_f1`,
  `paraphrase_accuracy` — all scored against synthetic generator gold.
- **Number state:** committed `experiments/results/comparison_table.csv` holds **legacy,
  pre-correction, single-seed (42)** values with CIs deferred. `postprocess.reparse` allows
  exact offline re-scoring from cached `raw_text` (all 200 predictions carry it).

## Data reality
- Cached predictions target the **old v1** `data/processed/test.jsonl` (50 items, `item_*`
  ids), 4 methods, seed 42 only. Frozen v2 is an 18/3/3 **sample**.
- Robustness re-derivation is **blocked**: only E01 has paraphrased predictions; no
  missing_info predictions (regenerating them needs model inference).
- Human eval: infrastructure complete (`src/evaluation/human/*`; rubric, blind assignment,
  Krippendorff α, Streamlit), assignment built for 50 items, **0 ratings on disk**.
- Independent gold present: `data/eval/l1_chart_effectiveness_v1.csv` (literature-derived,
  independent) — **but no scorer reads it**; `data/eval/real_briefs/items.jsonl` (10 verified
  external briefs, provenance in `docs/datasets/real_briefs_provenance.md`).
- Missing referenced files: `data/eval/task_crosswalk.yaml`, `docs/datasets/sources_table.md`.

## What is safe to claim today
See `scientific_dataset_validity_implementation_plan.md` → *Thesis claim policy*. In short:
synthetic results are internal diagnostics only; independent chart-selection needs the L1
scorer (built in this work) and is limited to covered task types; usefulness/actionability
needs human ratings (none yet); benchmark method-scores need an approval-gated inference pass.
