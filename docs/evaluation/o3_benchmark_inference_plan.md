# O3 — Benchmark Inference Plan (PLAN ONLY, not executed)

> `benchmark_v1` is **evaluation-only** (benchmark lock): never used for training,
> augmentation, prompt optimization, retriever tuning, hyperparameter tuning, or model
> selection. This document plans a controlled A/B/C/D evaluation on the benchmark; **running
> the inference is separately gated** (weak local CPU — run on GPU or as a tiny smoke first).

## Context
Goal: a controlled, non-circular evaluation of methods A/B/C/D on
`data/eval/benchmark_v1.jsonl` (30 items; 22 auto-scorable, 8 human-eval-only). The independent
L1 gold + this benchmark are the only non-circular evidence; synthetic metrics stay internal
diagnostics. Three blocking issues (below) are resolved as preflight before any run.

## Blocking issues (B1/B2 implemented as preflight; B3 handled via override)
- **B1 — `benchmark_v1.jsonl` is flat** (no `brief`/`recommendation` wrapper), so
  `dataset._record_to_gold` would read empty briefs. Resolved by a one-off wrapper
  `data/eval/benchmark_v1_infer.jsonl` (`item_id=benchmark_id`,
  `brief={users,goals,kpis,columns,constraints}`, `recommendation={}`) built by
  `experiments/scripts/prepare_benchmark_infer.py`. Verified: `load_gold_items` loads 30 items,
  0 empty briefs.
- **B2 — set-valued benchmark scorer** now exists: `src/evaluation/l1_independent.py::score_benchmark`
  + CLI `experiments/scripts/eval_benchmark.py` (unit-tested with mock predictions).
- **B3 — E04 (ft_rag) has no own adapter.** Both C and D override `method.adapter_path` to the
  existing `experiments/outputs/E03_qwen0_5b_ft_42/adapter/` (verified present). No new training.

## 1. Experiment configs (A/B/C/D)
`src/config/experiment/`: **A** `E01_qwen0_5b_prompt.yaml`, **B** `E02_qwen0_5b_rag.yaml`,
**C** `E03_qwen0_5b_ft.yaml`, **D** `E04_qwen0_5b_ft_rag.yaml`. All: model `qwen2_5_0_5b`, data
`dashboard_v1`, eval `full`, seed 42. Config files are **not edited**; everything is CLI override.

## 2. Adapters for FT / FT+RAG
Both reuse `experiments/outputs/E03_qwen0_5b_ft_42/adapter/` via
`method.adapter_path=...` override. No training.

## 3. Separate storage
`output_root=experiments/outputs/benchmark_v1/`, disjoint from synthetic runs. Scores →
`experiments/results/benchmark_v1_eval.{json,md}` (never overwrites `comparison_table.csv` /
`final_report.md`).

## 4. Output naming convention
`experiment_name` carries a `__benchmark_v1` tag → run dirs
`E0{1..4}_..._benchmark_v1_42` under `experiments/outputs/benchmark_v1/`. The tag + separate
root make old vs new runs unambiguous.

## 5. Metrics VALID on benchmark_v1
- **Benchmark chart-acceptability (primary, non-circular):** model primary chart ∈
  `acceptable_chart_types`, over **auto-scorable** items only; report `coverage_rate`,
  `covered_accuracy`, per-task_type/per-domain, evidence split (strong `real_public` vs weak
  `realistic_manual`), `parse_failures` (`score_benchmark`).
- **Schema validity / json parse / completeness** (`schema_compliance`, reference-free).
- **Latency** (reference-free). **Grounding** for RAG (B/D), labeled `lexical_proxy`.

## 6. Metrics NOT valid / only partial
- **Invalid here:** `top_k_accuracy`, `macro_f1` — need per-KPI gold `recommendation` the
  benchmark lacks; do not run them (they would score against an empty recommendation).
- **Not available:** `robustness` (no benchmark paraphrase/missing_info variants — intentionally
  disabled, see §11).
- **Partial:** chart-acceptability covers the 22 auto-scorable items; the 8 human-eval-only
  items (task types uncovered by L1: composition/part_to_whole/flow) are excluded from
  auto-accuracy and reserved for human eval.

## 7. L1 covered accuracy & coverage on benchmark_v1
Benchmark items carry an **independent** `task_type` + set-valued `acceptable_chart_types`
(lineage `literature_L1`/`manual_expert`, never `TASK_CHART`). `score_benchmark` joins
predictions↔benchmark by `item_id`; **covered** = `suitable_for_auto_scoring=true`; correct =
predicted primary chart ∈ `acceptable_chart_types`; `coverage_rate = n_covered/n_total`,
reported honestly (uncovered excluded, never counted correct). Independent here (task_type is
not generator-derived), unlike the cached-v1 L1 run which stays diagnostic/limited.

## 8. Parse-error accounting
A parse failure or missing primary chart on a **covered** item counts as **wrong** (never
skipped/correct); reported as `parse_failures`.

## 9. Schema validity & completeness
Reuse `schema_compliance` (`full_schema_valid` = full Pydantic + strict enums;
`completeness_fraction` = required keys present AND non-empty). Reference-free; valid as-is.

## 10. Report separation (5 evidence tiers)
`experiments/results/benchmark_v1_eval.md` states its tier and never blends: (1) synthetic
diagnostics — cross-link only; (2) independent L1 (cached v1, diagnostic); (3) **benchmark_v1
evaluation** — this file, covered items only, strong-vs-weak flagged; (4) human eval — pending;
(5) pending evidence (multi-seed, L3). Includes a forbidden-claims box.

## 11. Guards against accidental tuning use
- `benchmark_v1*` referenced by **no** training/val config (`dashboard_v2.yaml` trains only on
  frozen train/val; leakage report confirms).
- Predictions under a separate `output_root`.
- Generation/config identical to synthetic runs → no benchmark-driven hyperparameter/prompt/
  retriever tuning or model selection.
- The wrapper's `recommendation` is empty → cannot serve as training labels.
- **Every benchmark inference command sets `data.paraphrased_file=null data.missing_info_file=null`**
  so no synthetic variant set is ever pulled into a benchmark run (confirmed necessary by the
  smoke test — see §Correction).

## Correction (from the prompt-only smoke test)
The smoke run showed `predictions.jsonl` correctly held only 2 benchmark items, **but**
`infer.py`→`run_inference()` also generated `predictions_paraphrased.jsonl` and
`predictions_missing_info.jsonl` (50 each) from the **default synthetic** files
(`data/processed/test_*.jsonl`), because `max_samples` is not applied to variant loads. All
future benchmark commands therefore **must** disable variant inference explicitly with
`data.paraphrased_file=null data.missing_info_file=null`.

## Preflight / dry-run checks (NO model inference)
1. `pytest` (all green; 102 currently).
2. `python experiments/scripts/validate_benchmark.py` (PASS, 30 items).
3. Read-only asserts: adapter dir exists; configs resolve via `load_cfg` (compose only); KB
   `data/knowledge_base/chunks.jsonl` exists (B/D).
4. Wrapper built and dry-loaded with `load_gold_items` (30 items, 0 empty briefs) — done.
5. `score_benchmark()` unit-tested on mock predictions — done.

## Exact commands (PROPOSED — do not run until approved; GPU recommended)
```
# --- Preflight (no inference) ---
python -m pytest -q
python experiments/scripts/validate_benchmark.py
python experiments/scripts/prepare_benchmark_infer.py     # -> data/eval/benchmark_v1_infer.jsonl (eval-only)

# --- Inference (GATED; ~30 items x 4 methods; CPU ~30-45s/item -> prefer GPU) ---
# REQUIRED: disable synthetic variant inference (see Correction).
BM=data/eval/benchmark_v1_infer.jsonl
ROOT=experiments/outputs/benchmark_v1
NOVAR="data.paraphrased_file=null data.missing_info_file=null"
python experiments/scripts/build_kb.py                     # for B/D (idempotent)
python experiments/scripts/infer.py --experiment E01_qwen0_5b_prompt  --override data.test_file=$BM $NOVAR output_root=$ROOT experiment_name=E01_qwen0_5b_prompt__benchmark_v1
python experiments/scripts/infer.py --experiment E02_qwen0_5b_rag     --override data.test_file=$BM $NOVAR output_root=$ROOT experiment_name=E02_qwen0_5b_rag__benchmark_v1
python experiments/scripts/infer.py --experiment E03_qwen0_5b_ft      --override data.test_file=$BM $NOVAR output_root=$ROOT experiment_name=E03_qwen0_5b_ft__benchmark_v1 method.adapter_path=experiments/outputs/E03_qwen0_5b_ft_42/adapter
python experiments/scripts/infer.py --experiment E04_qwen0_5b_ft_rag  --override data.test_file=$BM $NOVAR output_root=$ROOT experiment_name=E04_qwen0_5b_ft_rag__benchmark_v1 method.adapter_path=experiments/outputs/E03_qwen0_5b_ft_42/adapter

# --- Scoring (offline, no model) ---
python experiments/scripts/eval_benchmark.py --predictions-root $ROOT --benchmark data/eval/benchmark_v1.jsonl
```
Windows/PowerShell: expand `$BM`/`$ROOT`/`$NOVAR` inline; pass the two `data.*=null` flags verbatim.

## Optional tiny smoke (prompt-only; stop after)
```
python experiments/scripts/infer.py --experiment E01_qwen0_5b_prompt --override data.test_file=data/eval/benchmark_v1_infer.jsonl data.max_samples=2 data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1_smoke experiment_name=E01_smoke__benchmark_v1
python experiments/scripts/eval_benchmark.py --predictions-root experiments/outputs/benchmark_v1_smoke --benchmark data/eval/benchmark_v1.jsonl
```
Confirm `predictions.jsonl` has 2 records with `bm_v1_*` ids and **no** `predictions_*` variant
files, then stop.

## Smoke-folder note (disposable)
The earlier `experiments/outputs/benchmark_v1_smoke/` (before this correction) contains
unintended `predictions_paraphrased.jsonl` / `predictions_missing_info.jsonl` (50 each) from
the default synthetic files. Its benchmark `predictions.jsonl` (2 items) is valid but tiny. The
whole smoke folder is **disposable** and should be deleted before/independently of the real O3
run (**deletion pending user approval**).

## Expected output files
- `data/eval/benchmark_v1_infer.jsonl` (wrapper; eval-only).
- `experiments/outputs/benchmark_v1/E0{1..4}_..._benchmark_v1_42/` → `predictions.jsonl`
  **only** (no `predictions_paraphrased.jsonl` / `predictions_missing_info.jsonl`, because the
  variant files are null), plus `config_snapshot.yaml`, `config_hash.txt`, `git_hash.txt`,
  `env.txt`, `manifest.json`, `logs/`.
- `experiments/results/benchmark_v1_eval.{json,md}`.

## Scientific limitations
Small n (30; 22 auto-scorable); single seed (42), no CIs/variance; `realistic_manual` items are
weaker evidence than `real_public`; item-level primary-chart scoring (not per-KPI); 0.5B smoke
model; grounding is a lexical proxy; L1-uncovered task types excluded from auto-accuracy.

## Allowed claims (after the run)
- Independent, non-circular **chart-selection acceptability** of A/B/C/D on covered benchmark
  items, reported **with coverage** and strong-vs-weak evidence split.
- Format reliability (schema validity, parse, completeness) and latency on realistic briefs.

## Forbidden claims
- That benchmark accuracy proves layout/styling/interaction/rationale or overall dashboard
  quality, or usefulness/actionability (needs human eval).
- Any significance/variance claim (single seed) or generalization beyond covered items.
- Reusing `benchmark_v1` for training, augmentation, prompt/retriever/hyperparameter tuning, or
  model selection.

## Rollback / cleanup notes (if a run fails)
- Runs are isolated under `experiments/outputs/benchmark_v1/` (and `_smoke/`) — delete that
  folder to reset; nothing else is touched. `infer.py` is idempotent (caches by `item_id`); a
  partial `predictions.jsonl` resumes on re-run, or delete the run folder to restart clean.
- `benchmark_v1_infer.jsonl` and `benchmark_v1_eval.*` are regenerable; safe to delete. No
  synthetic artifacts, configs, or `benchmark_v1.jsonl` are modified. No git actions.
