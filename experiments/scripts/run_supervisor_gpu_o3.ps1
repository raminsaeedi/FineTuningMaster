<#
  run_supervisor_gpu_o3.ps1 — one-shot O3 benchmark inference for methods A/B/C/D.

  SUPERSEDED — the current supported path is RUN_PROFESSOR.md at the repo root:
      python experiments/scripts/check_experiment_release.py   # preflight
      python experiments/scripts/run_final_matrix.py           # A/B/C/D x seeds 42/43/44
      python experiments/scripts/package_results.py            # professor_results.zip
  This legacy script is kept for reference. It carries no data= override, so the
  configs' own default (data: dashboard_v3 in src/config/config.yaml) applies.

  Runs the EXISTING CLI scripts only. It NEVER trains and NEVER edits benchmark labels.
  Benchmark predictions are written under experiments/outputs/benchmark_v1 (separate from
  synthetic runs). Synthetic variant inference is disabled explicitly.

  Usage (from repo root, on the GPU machine, venv activated):
      pwsh experiments/scripts/run_supervisor_gpu_o3.ps1

  See docs/project/SUPERVISOR_GPU_RUNBOOK.md for full context and claim boundaries.
#>

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")   # repo root

$BM      = "data/eval/benchmark_v1_infer.jsonl"
$ROOT    = "experiments/outputs/benchmark_v1"
# Fixed: this used to be experiments/outputs/E03_qwen0_5b_ft_42/adapter, which does not
# match the default output layout (src/config/config.yaml: output_root
# experiments/outputs/experiments, experiment_id ${experiment_name}_${seed}). Now
# identical to run_supervisor_full_gpu.ps1.
$ADAPTER = "experiments/outputs/experiments/E03_qwen0_5b_ft_42/adapter"
# REQUIRED: disable synthetic variant inference (paraphrase/missing_info default to the
# synthetic files and are NOT limited by max_samples). Keeps a benchmark run to the
# benchmark original set only.
$NOVAR   = @("data.paraphrased_file=null", "data.missing_info_file=null")

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "1/8 Preflight (no inference)"
python experiments/scripts/preflight_supervisor_gpu.py

Step "2/8 Build benchmark inference wrapper (idempotent, eval-only)"
python experiments/scripts/prepare_benchmark_infer.py

Step "3/8 Build knowledge base for RAG methods (idempotent)"
python experiments/scripts/build_kb.py

Step "4/8 Method A - prompt_only (E01)"
python experiments/scripts/infer.py --experiment E01_qwen0_5b_prompt `
  --override data.test_file=$BM $NOVAR output_root=$ROOT experiment_name=E01_qwen0_5b_prompt__benchmark_v1

Step "5/8 Method B - rag (E02)"
python experiments/scripts/infer.py --experiment E02_qwen0_5b_rag `
  --override data.test_file=$BM $NOVAR output_root=$ROOT experiment_name=E02_qwen0_5b_rag__benchmark_v1

Step "Verify FT adapter before methods C/D"
if (-not (Test-Path (Join-Path $ADAPTER "adapter_config.json"))) {
    Write-Error "FT adapter not found at $ADAPTER. Methods C/D require it. Aborting (A/B outputs are kept)."
    exit 1
}
Write-Host "FT adapter found: $ADAPTER" -ForegroundColor Green

Step "6/8 Method C - ft (E03)"
python experiments/scripts/infer.py --experiment E03_qwen0_5b_ft `
  --override data.test_file=$BM $NOVAR output_root=$ROOT experiment_name=E03_qwen0_5b_ft__benchmark_v1 method.adapter_path=$ADAPTER

Step "7/8 Method D - ft_rag (E04)"
python experiments/scripts/infer.py --experiment E04_qwen0_5b_ft_rag `
  --override data.test_file=$BM $NOVAR output_root=$ROOT experiment_name=E04_qwen0_5b_ft_rag__benchmark_v1 method.adapter_path=$ADAPTER

Step "8/8 Score benchmark predictions (offline, no model)"
python experiments/scripts/eval_benchmark.py --predictions-root $ROOT --benchmark data/eval/benchmark_v1.jsonl

Write-Host "`nDONE. Benchmark predictions under $ROOT ; scores in experiments/results/benchmark_v1_eval.{json,md}" -ForegroundColor Green
Write-Host "Return the outputs listed in docs/project/OUTPUTS_TO_RETURN.md." -ForegroundColor Green
