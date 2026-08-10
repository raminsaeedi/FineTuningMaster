<#
  run_supervisor_full_gpu.ps1 — FULL final thesis pipeline for the supervisor GPU machine.

  SUPERSEDED — the current supported path is RUN_PROFESSOR.md at the repo root:
      python experiments/scripts/check_experiment_release.py   # preflight
      python experiments/scripts/run_final_matrix.py           # A/B/C/D x seeds 42/43/44
      python experiments/scripts/package_results.py            # professor_results.zip
  This legacy script is kept for reference; it now targets the current frozen dataset
  (dashboard_v3) so it can no longer train on the superseded dashboard_v2 set.

  Stages (all via EXISTING CLIs; no architecture rewrite):
    preflight(--require-cuda) -> validate frozen v3 dataset -> build KB -> train FT adapter
    -> synthetic internal diagnostics (E01-E04) + aggregate/stats  [INTERNAL DIAGNOSTIC]
    -> A/B/C/D benchmark inference on benchmark_v1_infer.jsonl      [INDEPENDENT]
    -> benchmark scoring.

  Guarantees: trains ONLY with data=dashboard_v3 (never benchmark); benchmark inference always
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
$DATA      = "data=dashboard_v3"   # current frozen dataset (1281 train / 264 val / 274 test)

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "1/9 Preflight (CUDA REQUIRED for the full run)"
python experiments/scripts/preflight_supervisor_full_gpu.py --require-cuda

Step "2/9 Validate the frozen dashboard_v3 training dataset (train/val/test)"
# dashboard_v3 is a frozen nvBench-derived artifact and is NEVER regenerated. The old
# v2 build commands (generate_dataset_v2 / freeze_dataset_v2 / build_perturbations_v2)
# belong to the superseded v2 lineage and are intentionally not run here.
python experiments/scripts/validate_frozen_dataset.py --frozen-dir data/frozen/dashboard_v3
Write-Host "Read data/frozen/dashboard_v3/validation_report.md for the ACTUAL unique counts." -ForegroundColor Yellow

Step "3/9 Build knowledge base (RAG for methods B/D)"
python experiments/scripts/build_kb.py

Step "4/9 Train FT adapter on data=dashboard_v3 (NEVER on benchmark_v1)"
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --override $DATA

Step "Verify freshly trained adapter before any FT step"
if (-not (Test-Path (Join-Path $ADAPTER "adapter_config.json"))) {
    Write-Error "Trained adapter not found at $ADAPTER. Training did not complete. Aborting."
    exit 1
}
Write-Host "Adapter present: $ADAPTER" -ForegroundColor Green

Step "5/9 Synthetic internal diagnostics — E01-E04 (INTERNAL DIAGNOSTIC, circular)"
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override $DATA
python experiments/scripts/run_experiment.py --experiment E02_qwen0_5b_rag    --override $DATA
python experiments/scripts/run_experiment.py --experiment E03_qwen0_5b_ft     --override $DATA
python experiments/scripts/run_experiment.py --experiment E04_qwen0_5b_ft_rag --override $DATA method.adapter_path=$ADAPTER

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
