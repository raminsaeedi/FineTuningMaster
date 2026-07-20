# Scientific Dataset Validity — Implementation Plan

Ordered plan to make the dataset, benchmark, and evaluation defensible. Companion to
`scientific_dataset_validity_audit.md`. Constraints: no training, no model inference (re-score
cached predictions only), no git operations, no attribution, English only.

## Decisions locked
- Run new scorers on the cached **v1** predictions now (offline, single-seed, labeled
  diagnostic); derive provenance **in the report only** (no record mutation / schema change);
  L1 scorer keys by a **task_type-level aggregated set**.
- **task_type lineage:** for cached synthetic v1 predictions `task_type` is generator-derived,
  so L1 scores on them are **diagnostic/limited**; for `benchmark_v1`, `task_type` is assigned
  **independently** (source intent / analytical task / documented expert judgment) and **not**
  from `KEYWORD_TASK`, `TASK_CHART`, or the synthetic generator lineage.
- **Benchmark lock:** `data/eval/benchmark_v1.jsonl` is **evaluation-only** and must never be
  used for training, augmentation, prompt optimization, retriever tuning, hyperparameter
  tuning, or model selection.

## Execution order
Docs → benchmark schema/construction → `benchmark_v1.jsonl` → leakage/provenance checks →
L1 scorer → offline re-score → reports. No model runs.

## MUST-HAVE (M1–M11)
- **M1 — Benchmark schema + construction doc.** `data/eval/benchmark_v1_schema.yaml` +
  `docs/evaluation/benchmark_dataset_construction.md`. Fields per item: `benchmark_id, domain,
  users, goals, kpis, columns, constraints, task_type, acceptable_chart_types (set), rationale,
  source_name, source_type, source_reference, license_or_usage_note, label_source,
  label_confidence, suitable_for_auto_scoring, suitable_for_human_eval, notes`.
  `acceptable_chart_types` lineage = L1 literature table or documented expert judgment, never
  `TASK_CHART`. Doc states the benchmark lock.
- **M2 — Build `data/eval/benchmark_v1.jsonl` (30–50 items).** Verify real public briefs exist
  (`data/eval/real_briefs/items.jsonl` + provenance doc) → reuse as `source_type=real_public`
  (strong evidence); author the remainder as clearly-labeled `source_type=realistic_manual`
  (weaker evidence). `task_type` assigned independently via `data/eval/task_crosswalk.yaml`
  (documented analytical-task crosswalk, distinct from the generator);
  `acceptable_chart_types` from the L1 table where covered (`label_source=literature_L1`) else
  documented expert set (`label_source=manual_expert`). `suitable_for_auto_scoring=true` iff the
  item's task_type is L1-covered; `suitable_for_human_eval=true` for all.
- **M3 — Benchmark validation report.** `src/data_pipeline/benchmark_validation.py` +
  `experiments/scripts/validate_benchmark.py` → `experiments/results/benchmark_dataset_report.md`.
  Ten checks: item count; domain / task_type / chart-label distributions; chart-type coverage;
  auto-scorable count; human-eval-only count; strong-vs-weak evidence split; source/provenance
  leakage vs training; label-lineage leakage vs the generator (no `acceptable_chart_types`
  equals the single `TASK_CHART` mapping). Restates the benchmark lock.
- **M4 — Leakage check.** `src/data_pipeline/leakage_similarity.py` (char-3gram Jaccard,
  dependency-free) + `experiments/scripts/check_dataset_leakage.py` →
  `experiments/results/leakage_report.{json,md}`. Checks exact-id, exact-text, cross-set source,
  cross-set label lineage, near-duplicate similarity; classifies `no_issue | warning | critical
  | rule_leakage`.
- **M5 — Provenance report (derive-only).** `src/data_pipeline/provenance.py` +
  `experiments/scripts/build_provenance_report.py` → `experiments/results/dataset_provenance_report.{json,md}`.
  Derives `item_id, source_name, source_type, license, generation_method, label_source,
  label_lineage_id, split, intended_use, is_synthetic, created_at, notes`; states which items
  are safe for independent eval vs internal-diagnostic-only. No record mutation.
- **M6 — Rule-leakage report.** `experiments/scripts/rule_leakage_report.py` →
  `experiments/results/rule_leakage_report.md`. Demonstrates the shared rule
  (`chart_type == TASK_CHART[task_type][0]`), explains inflation, why row-level split is
  insufficient, required independent tests, and claim limits.
- **M7 — Independent L1 scorer.** `src/evaluation/l1_independent.py` +
  `experiments/scripts/eval_l1_independent.py` → `experiments/results/l1_independent_results.{json,md}`.
  `task_type → set(effective_charts)` from `data/eval/l1_chart_effectiveness_v1.csv`; predicted
  primary chart ∈ set = correct; uncovered excluded and reported (coverage honest — 1.0 is fine).
  Reports coverage_rate, covered_accuracy, n_uncovered, per-task_type / per-chart_type accuracy.
  **Limitation** (in report + claims): validates only chart-selection acceptability for covered
  task types, not layout/styling/interaction/rationale/overall quality; two keying cases
  (synthetic v1 generator-derived = diagnostic; benchmark = independent).
- **M8 — Corrected offline re-score.** `eval_auto.py` per E01–E04 (override
  `output_root=experiments/outputs`, reparse, **no model**) then `aggregate_results.py` →
  refreshed `comparison_table.csv` / `final_report.md` / per-run `eval_per_item.jsonl`;
  chart-accuracy stays tagged internal-circular; single-seed caveat surfaced.
- **M9 — Metric-semantics tests.** Extend `src/tests/test_metrics.py`: parse-fail=wrong (Top-1);
  Top-3 `None` when support < 0.8; empty field ⇒ incomplete; enum-invalid ⇒ schema-invalid.
- **M10 — Memorization vs generalization protocol.**
  `docs/evaluation/memorization_vs_generalization_protocol.md` +
  `experiments/scripts/analyze_generalization.py` (domain × task_type combos in train vs eval,
  novel combos, cached-prediction accuracy on seen vs novel).
- **M11 — Human-eval scientific protocol.** `docs/evaluation/human_eval_scientific_protocol.md`
  (blind, same items across methods, ≥3 ratings/output, rubric anchors, pilot, Krippendorff α,
  Friedman/Wilcoxon+Holm, low-IRR limitation; no usefulness/actionability claim until ratings).

## SHOULD-HAVE
- **S1 — Fill missing referenced files:** `data/eval/task_crosswalk.yaml` (created in M2 as the
  independent labeling crosswalk) and `docs/datasets/sources_table.md`.

## OPTIONAL (not in this pass)
- **O1** embedding near-duplicate upgrade. **O2** Draco `hard.lp` legality check.
- **O3** run A/B/C/D on `benchmark_v1` — needs inference → **approval-gated**.
- **O4** regenerate paraphrase/missing_info predictions for E02–E04 — needs inference → gated.
- **O5** multi-seed 43/44 — needs training → out of scope.

## Verification
1. `pytest` (existing + new unit tests: benchmark validation, L1 scorer, leakage similarity,
   provenance derivation, metric semantics).
2. Run M3–M8 offline; confirm no model load (CPU, seconds).
3. Benchmark report: 30–50 items; non-degenerate distributions; strong-vs-weak split; 0
   source/provenance leakage vs train; 0 label-lineage leakage vs `TASK_CHART`.
4. L1: coverage_rate and n_uncovered reported honestly; covered-accuracy printed with coverage;
   limitation paragraph present.
5. Leakage report on {train/val} vs {benchmark, L1, real_briefs, internal_test}: no critical
   exact leakage; rule-leakage flagged.
6. Refreshed `comparison_table.csv` still tags chart-accuracy internal-circular; single-seed +
   deferred-CI caveats surfaced.

## Thesis claim policy (5 separated tiers — enforced in every report)
1. **Synthetic internal diagnostics** — allows: structured-output production; FT improves
   reproduction of the synthetic task→chart mapping; schema/parse reliability. **Forbidden:**
   any statement or implication that high synthetic Top-1 proves real dashboard-design quality,
   real-dashboard superiority, or usefulness.
2. **Independent chart-selection (L1)** — allows (covered items only, with coverage rate):
   non-circular chart-selection acceptability. Limitation: chart choice only, covered task types
   only; synthetic-v1 keying is generator-derived (diagnostic), benchmark keying is independent.
3. **Real/realistic benchmark evaluation** — allows: method comparison on independent briefs for
   auto-scorable items (once predictions exist; O3 approval-gated), with strong-vs-weak-evidence
   labels. Benchmark lock applies.
4. **Human evaluation** — allows: usefulness, actionability, perceived quality — only with
   collected ratings and acceptable IRR (none yet → no such claims).
5. **Pending evidence** — L3 realism, human ratings, multi-seed, benchmark method-scores:
   explicitly listed as not-yet-available.
**Always report limitations:** synthetic circularity, small benchmark, limited L1 coverage,
model scale, single-seed, missing/low IRR.
