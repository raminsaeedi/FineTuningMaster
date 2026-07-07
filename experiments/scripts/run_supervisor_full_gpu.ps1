<#
  run_supervisor_full_gpu.ps1 — FULL final thesis pipeline for the supervisor GPU machine.

  Stages (all via EXISTING CLIs; no architecture rewrite):
    preflight(--require-cuda) -> build+validate v2 dataset -> build KB -> train FT adapter
    -> synthetic internal diagnostics (E01-E04) + aggregate/stats  [INTERNAL DIAGNOSTIC]
    -> A/B/C/D benchmark inference on benchmark_v1_infer.jsonl      [INDEPENDENT]
    -> benchmark scoring.

  Guarantees: trains ONLY with data=dashboard_v2 (never benchmark); benchmark inference always
  disables synthetic variants and writes to a separate root; benchmark files are never modified.
  Assumes a fresh clone (experiments/outputs/ empty) so the default output root is clean and the
  existing aggregate/stats scripts work as-is.

  Usage (repo root, venv active, on the GPU machine):
      pwsh experiments/scripts/run_supervisor_full_gpu.ps1
  See docs/project/SUPERVISOR_FULL_GPU_RUNBOOK.md.
#>

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")   # repo root

# --- paths / constants ---
$BM        = "data/eval/benchmark_v1_infer.jsonl"
$BENCH     = "data/eval/benchmark_v1.jsonl"
$BENCH_ROOT = "experiments/outputs/benchmark_v1"
$ADAPTER   = "experiments/outputs/experiments/E03_qwen0_5b_ft_42/adapter"   # default output root
$NOVAR     = @("data.paraphrased_file=null", "data.missing_info_file=null")
$N         = 2000   # generated pool; ~1600/200/200 after the 0.8/0.1/0.1 hash split (unique count reported by the validator)

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "1/9 Preflight (CUDA REQUIRED for the full run)"
python experiments/scripts/preflight_supervisor_full_gpu.py --require-cuda

Step "2/9 Build + validate the v2 training dataset (train/val/internal_test)"
python experiments/scripts/generate_dataset_v2.py --n $N
python experiments/scripts/freeze_dataset_v2.py
python experiments/scripts/build_perturbations_v2.py
python experiments/scripts/validate_frozen_dataset.py
Write-Host "Read data/frozen/dashboard_v2/validation_report.md for the ACTUAL unique counts." -ForegroundColor Yellow

Step "3/9 Build knowledge base (RAG for methods B/D)"
python experiments/scripts/build_kb.py

Step "4/9 Train FT adapter on data=dashboard_v2 (NEVER on benchmark_v1)"
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --override data=dashboard_v2

Step "Verify freshly trained adapter before any FT step"
if (-not (Test-Path (Join-Path $ADAPTER "adapter_config.json"))) {
    Write-Error "Trained adapter not found at $ADAPTER. Training did not complete. Aborting."
    exit 1
}
Write-Host "Adapter present: $ADAPTER" -ForegroundColor Green

Step "5/9 Synthetic internal diagnostics — E01-E04 (INTERNAL DIAGNOSTIC, circular)"
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override data=dashboard_v2
python experiments/scripts/run_experiment.py --experiment E02_qwen0_5b_rag    --override data=dashboard_v2
python experiments/scripts/run_experiment.py --experiment E03_qwen0_5b_ft     --override data=dashboard_v2
python experiments/scripts/run_experiment.py --experiment E04_qwen0_5b_ft_rag --override data=dashboard_v2 method.adapter_path=$ADAPTER

Step "6/9 Aggregate synthetic diagnostics -> final_report.md + comparison_table.csv"
python experiments/scripts/aggregate_results.py

Step "7/9 Statistics (best-effort; single-seed significance is limited)"
try {
    python experiments/scripts/eval_stats.py --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag
} catch {
    Write-Host "eval_stats skipped/failed (expected without multi-seed or matching data wiring): $_" -ForegroundColor Yellow
}

Step "8/9 Benchmark inference A/B/C/D (INDEPENDENT; variants disabled; separate root)"
python experiments/scripts/infer.py --experiment E01_qwen0_5b_prompt  --override data.test_file=$BM $NOVAR output_root=$BENCH_ROOT experiment_name=E01_qwen0_5b_prompt__benchmark_v1
python experiments/scripts/infer.py --experiment E02_qwen0_5b_rag     --override data.test_file=$BM $NOVAR output_root=$BENCH_ROOT experiment_name=E02_qwen0_5b_rag__benchmark_v1
python experiments/scripts/infer.py --experiment E03_qwen0_5b_ft      --override data.test_file=$BM $NOVAR output_root=$BENCH_ROOT experiment_name=E03_qwen0_5b_ft__benchmark_v1 method.adapter_path=$ADAPTER
python experiments/scripts/infer.py --experiment E04_qwen0_5b_ft_rag  --override data.test_file=$BM $NOVAR output_root=$BENCH_ROOT experiment_name=E04_qwen0_5b_ft_rag__benchmark_v1 method.adapter_path=$ADAPTER

Step "9/9 Score benchmark predictions (offline, no model)"
python experiments/scripts/eval_benchmark.py --predictions-root $BENCH_ROOT --benchmark $BENCH

Write-Host "`nDONE. Adapter: $ADAPTER" -ForegroundColor Green
Write-Host "Synthetic diagnostics: experiments/results/final_report.md, comparison_table.csv, stats/" -ForegroundColor Green
Write-Host "Independent benchmark: experiments/results/benchmark_v1_eval.{json,md}" -ForegroundColor Green
Write-Host "Return the outputs listed in docs/project/OUTPUTS_TO_RETURN.md." -ForegroundColor Green
