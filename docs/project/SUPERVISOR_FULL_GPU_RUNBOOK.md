# Supervisor Full GPU Runbook — Final Thesis Pipeline

End-to-end run on the supervisor's GPU machine: **build dataset → train → synthetic diagnostics
→ benchmark inference → score → aggregate**. The local CPU is too weak; everything heavy runs
here. One entrypoint chains the existing scripts:
```
pwsh experiments/scripts/run_supervisor_full_gpu.ps1
```
(For a benchmark-only run without training, see [`SUPERVISOR_GPU_RUNBOOK.md`](SUPERVISOR_GPU_RUNBOOK.md).)

## Project goal
Fine-tune Qwen2.5-0.5B-Instruct to produce structured dashboard-design recommendations from a
brief (users, goals, KPIs, columns). Compare **A** prompt-only, **B** RAG, **C** fine-tuned
(QLoRA), **D** fine-tuned + RAG. Synthetic accuracy is circular (rule leakage) → internal
diagnostic only; the independent `benchmark_v1` + literature L1 are the non-circular evidence.

## Training-dataset status (before this run)
The final training set does **not** exist yet — the frozen v2 set is only an 18/3/3 **sample**.
Legacy (superseded, circular): `data/processed` 491/59/50, `data/gold.jsonl` 600. Stage 2 below
**builds the real v2 dataset** (~1600/200/200) with the existing generator. `benchmark_v1` is
evaluation-only and is never used for training/tuning/selection.

## Environment setup
Follow **[`REMOTE_RUN.md`](../../REMOTE_RUN.md) → Setup** (Python 3.10+ venv, CUDA-matched
`torch==2.6.0`, `pip install -r requirements-train.txt`, `pip install -e .`). Optional HF cache
via `.env`; pin a GPU with `CUDA_VISIBLE_DEVICES=0`.

## CUDA policy
On a local CPU box, preflight CUDA may **warn** (lightweight review only). For the full GPU run
the `.ps1` calls preflight with `--require-cuda`, so **CUDA-unavailable is a hard fail** and the
pipeline aborts before any training/inference.

## Preflight
```
python experiments/scripts/preflight_supervisor_full_gpu.py --require-cuda
```
Checks imports, CUDA (hard-required here), training-file counts (WARN if still the sample —
Stage 2 fixes it), benchmark files + non-empty briefs, KB chunks, writable output roots.

## Exact commands (what the entrypoint runs)
```
# 2) Build + validate the v2 training dataset
python experiments/scripts/generate_dataset_v2.py --n 2000
python experiments/scripts/freeze_dataset_v2.py
python experiments/scripts/build_perturbations_v2.py
python experiments/scripts/validate_frozen_dataset.py     # READ actual unique counts here

# 3) Knowledge base (RAG for B/D)
python experiments/scripts/build_kb.py

# 4) Train FT adapter — ONLY on data=dashboard_v2 (never benchmark)
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --override data=dashboard_v2
#    -> adapter: experiments/outputs/experiments/E03_qwen0_5b_ft_42/adapter

# 5) Synthetic internal diagnostics (INTERNAL DIAGNOSTIC, circular)
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override data=dashboard_v2
python experiments/scripts/run_experiment.py --experiment E02_qwen0_5b_rag    --override data=dashboard_v2
python experiments/scripts/run_experiment.py --experiment E03_qwen0_5b_ft     --override data=dashboard_v2
python experiments/scripts/run_experiment.py --experiment E04_qwen0_5b_ft_rag --override data=dashboard_v2 method.adapter_path=experiments/outputs/experiments/E03_qwen0_5b_ft_42/adapter

# 6-7) Aggregate + stats
python experiments/scripts/aggregate_results.py           # -> final_report.md, comparison_table.csv
python experiments/scripts/eval_stats.py --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag   # best-effort; see note

# 8) Benchmark inference (INDEPENDENT; variants disabled; separate root)
#    every command carries: data.paraphrased_file=null data.missing_info_file=null
python experiments/scripts/infer.py --experiment E01_qwen0_5b_prompt  --override data.test_file=data/eval/benchmark_v1_infer.jsonl data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1 experiment_name=E01_qwen0_5b_prompt__benchmark_v1
python experiments/scripts/infer.py --experiment E02_qwen0_5b_rag     --override data.test_file=data/eval/benchmark_v1_infer.jsonl data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1 experiment_name=E02_qwen0_5b_rag__benchmark_v1
python experiments/scripts/infer.py --experiment E03_qwen0_5b_ft      --override data.test_file=data/eval/benchmark_v1_infer.jsonl data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1 experiment_name=E03_qwen0_5b_ft__benchmark_v1 method.adapter_path=experiments/outputs/experiments/E03_qwen0_5b_ft_42/adapter
python experiments/scripts/infer.py --experiment E04_qwen0_5b_ft_rag  --override data.test_file=data/eval/benchmark_v1_infer.jsonl data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1 experiment_name=E04_qwen0_5b_ft_rag__benchmark_v1 method.adapter_path=experiments/outputs/experiments/E03_qwen0_5b_ft_42/adapter

# 9) Score benchmark (offline)
python experiments/scripts/eval_benchmark.py --predictions-root experiments/outputs/benchmark_v1 --benchmark data/eval/benchmark_v1.jsonl
```
**Note (stats):** `eval_stats.py` resolves references from the default data config; cross-method
significance is reliable only with multi-seed and matching data wiring — treat single-seed stats
as indicative. The primary synthetic diagnostics are the per-run metrics + `comparison_table.csv`.

## Optional multi-seed — RECOMMENDED IF SUPERVISOR GPU TIME ALLOWS
Not run by default. If GPU time allows, repeat **train + synthetic eval + benchmark inference**
for **seeds 42, 43, 44**, each into **seed-specific output folders** (`experiment_id` already
appends the seed; add a per-seed `output_root` suffix, e.g. `output_root=experiments/outputs/benchmark_v1_s43`).
Example (seed 43; repeat for 44):
```
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --override data=dashboard_v2 seed=43
python experiments/scripts/infer.py --experiment E01_qwen0_5b_prompt --override seed=43 data.test_file=data/eval/benchmark_v1_infer.jsonl data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1_s43 experiment_name=E01_qwen0_5b_prompt__benchmark_v1
# ... E02/E03/E04 likewise (C/D add method.adapter_path=experiments/outputs/experiments/E03_qwen0_5b_ft_43/adapter) ...
```
Report **mean/std/CI only if multi-seed is actually run**. Until then results are **single-seed
(42)** and no variance/significance is implied — do not present multi-seed numbers that were not
produced.

## Resume / restart
Inference and (re-run) are **idempotent, cached per `item_id`** — re-running a command skips
completed items. To restart a stage cleanly, delete only its run folder (under
`experiments/outputs/experiments/<run>` for diagnostics or `experiments/outputs/benchmark_v1/<run>`).
Training does not resume mid-epoch but a 0.5B QLoRA run finishes in one session.

## Dataset-size caveat
The v2 generator draws from ~10 domains with bounded audience/goal/KPI pools; `freeze` de-dupes
by brief fingerprint, so unique items may **cap below 2000**. Read the actual counts from
`data/frozen/dashboard_v2/validation_report.md`; if below target, raise `--n` or expand the
generator seed pools (a generator enhancement, out of scope here). Counts are always reported —
no silent truncation.

## Expected outputs / what to return
See **[`OUTPUTS_TO_RETURN.md`](OUTPUTS_TO_RETURN.md)** (adapter, benchmark runs, benchmark_v1_eval,
final_report.md, comparison_table.csv, stats/, logs, env/GPU info).

## Scientific claims
**Allowed after this run:** independent non-circular **chart-selection acceptability** of
A/B/C/D on covered benchmark items (with coverage + strong-vs-weak split); format reliability
(schema/parse/completeness) and latency on realistic briefs.
**Forbidden:** claiming synthetic accuracy proves real dashboard quality/superiority;
usefulness/actionability (needs human ratings); significance/variance from a single seed;
generalization beyond covered items; any use of `benchmark_v1` for training/tuning/augmentation/
prompt/retriever/hyperparameter/model-selection.
