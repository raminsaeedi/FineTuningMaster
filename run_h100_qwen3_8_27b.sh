#!/usr/bin/env bash
# Direct-terminal H100 runner for one frozen experiment selection:
# Qwen3.8-27B, dashboard_v4, seed 42, methods C D A B.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_h100_qwen3_8_27b.sh --pilot
  ./run_h100_qwen3_8_27b.sh --full
  ./run_h100_qwen3_8_27b.sh --inference

Run --pilot first. It uses two train/evaluation items and one training step.
Run --full only after the pilot prints PASS_H100_QWEN3_8_27B_PILOT.

Optional environment variables:
  FTM_METHODS="D B"     Run selected methods; default: "C D A B".
  FTM_ROBUSTNESS=0      Run original test split only; default: 1.
  FTM_INFERENCE_MAX_SEQ_LENGTH=4096
                         Context length for --inference; default: 4096.
  FTM_INFERENCE_MAX_NEW_TOKENS=1024
                         Output cap for --inference; default: 1024.
  FTM_AUTO_RESUME_ATTEMPTS=3
                         Retry same crash-safe command up to N times; default: 1.
USAGE
}

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
INFERENCE_ONLY=0
DEFAULT_METHODS="C D A B"
DEFAULT_ROBUSTNESS=1
case "$1" in
  --pilot) PROFILE=smoke; RUN_KIND=pilot ;;
  --full) PROFILE=final; RUN_KIND=full ;;
  --inference)
    PROFILE=final
    RUN_KIND=full
    INFERENCE_ONLY=1
    DEFAULT_METHODS="C D"
    DEFAULT_ROBUSTNESS=0
    ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: no Conda environment is active." >&2
  echo "Run: conda activate ftm_h100_qwen38" >&2
  exit 1
fi
PY="${CONDA_PREFIX}/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: Python missing: ${PY}" >&2; exit 1; }

export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export FTM_MIN_FREE_GPU_GIB="${FTM_MIN_FREE_GPU_GIB:-35}"
# Direct servers can expose every installed GPU. Use GPU 0 unless the caller
# explicitly selects another single H100 before launching this script.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET=dashboard_v4
MODEL=qwen3_8_27b
SEED=42
read -r -a METHODS <<< "${FTM_METHODS:-${DEFAULT_METHODS}}"
ROBUSTNESS="${FTM_ROBUSTNESS:-${DEFAULT_ROBUSTNESS}}"
[[ "${#METHODS[@]}" -gt 0 ]] || { echo "ERROR: FTM_METHODS is empty." >&2; exit 2; }
for METHOD in "${METHODS[@]}"; do
  case "${METHOD}" in
    A|B|C|D) ;;
    *) echo "ERROR: invalid method in FTM_METHODS: ${METHOD}" >&2; exit 2 ;;
  esac
done
case "${ROBUSTNESS}" in
  0|1) ;;
  *) echo "ERROR: FTM_ROBUSTNESS must be 0 or 1." >&2; exit 2 ;;
esac
AUTO_RESUME_ATTEMPTS="${FTM_AUTO_RESUME_ATTEMPTS:-1}"
[[ "${AUTO_RESUME_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: FTM_AUTO_RESUME_ATTEMPTS must be a positive integer." >&2
  exit 2
}
if [[ "${INFERENCE_ONLY}" == 1 ]]; then
  INFERENCE_MAX_SEQ_LENGTH="${FTM_INFERENCE_MAX_SEQ_LENGTH:-4096}"
  INFERENCE_MAX_NEW_TOKENS="${FTM_INFERENCE_MAX_NEW_TOKENS:-1024}"
  [[ "${INFERENCE_MAX_SEQ_LENGTH}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: FTM_INFERENCE_MAX_SEQ_LENGTH must be a positive integer." >&2
    exit 2
  }
  [[ "${INFERENCE_MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: FTM_INFERENCE_MAX_NEW_TOKENS must be a positive integer." >&2
    exit 2
  }
  (( INFERENCE_MAX_NEW_TOKENS < INFERENCE_MAX_SEQ_LENGTH )) || {
    echo "ERROR: FTM_INFERENCE_MAX_NEW_TOKENS must be smaller than FTM_INFERENCE_MAX_SEQ_LENGTH." >&2
    exit 2
  }
fi
STORAGE_ROOT="${FTM_H100_STORAGE:-${HOME}/Sep_work/Ramin/ftm_qwen3_8_27b}"
WORK_ROOT="${STORAGE_ROOT}/${RUN_KIND}"
export HF_HOME="${STORAGE_ROOT}/huggingface"
CACHE_ROOT="${STORAGE_ROOT}/huggingface/cache"
KB_CHUNKS="${STORAGE_ROOT}/knowledge_base/chunks.jsonl"
RUN_ROOT="${WORK_ROOT}/runs"
MODEL_ROOT="${WORK_ROOT}/adapters"
RESULT_ROOT="${WORK_ROOT}/results"
PACKAGE_ROOT="${WORK_ROOT}/packages"
LOG_ROOT="${WORK_ROOT}/logs"

mkdir -p \
  "${HF_HOME}" "${CACHE_ROOT}" "$(dirname -- "${KB_CHUNKS}")" \
  "${RUN_ROOT}" "${MODEL_ROOT}" "${RESULT_ROOT}" \
  "${PACKAGE_ROOT}" "${LOG_ROOT}"

ADAPTER_DIR="${MODEL_ROOT}/${DATASET}/${MODEL}/C/seed_${SEED}/adapter"
if [[ "${INFERENCE_ONLY}" == 1 ]]; then
  [[ -f "${ADAPTER_DIR}/adapter_config.json" ]] || {
    echo "ERROR: completed C adapter missing: ${ADAPTER_DIR}" >&2
    exit 1
  }
  if [[ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" && \
        ! -f "${ADAPTER_DIR}/adapter_model.bin" ]]; then
    echo "ERROR: completed C adapter has no weights: ${ADAPTER_DIR}" >&2
    exit 1
  fi
fi

LOG_FILE="${LOG_ROOT}/qwen3_8_27b_seed42_${RUN_KIND}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== DIRECT H100 RUN ==="
echo "project: ${PROJECT_ROOT}"
echo "mode: ${RUN_KIND}"
echo "model: ${MODEL}"
echo "dataset: ${DATASET}"
echo "seed: ${SEED}"
echo "methods: ${METHODS[*]}"
echo "robustness variants: ${ROBUSTNESS}"
echo "inference only: ${INFERENCE_ONLY}"
if [[ "${INFERENCE_ONLY}" == 1 ]]; then
  echo "inference context/output: ${INFERENCE_MAX_SEQ_LENGTH} / ${INFERENCE_MAX_NEW_TOKENS}"
fi
echo "storage: ${STORAGE_ROOT}"
echo "log: ${LOG_FILE}"
git rev-parse --short HEAD
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${PY}" - <<'PY'
import importlib.metadata as metadata
import os
import torch

assert torch.__version__.split("+")[0] == "2.6.0", torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert metadata.version("transformers") == "5.7.0"
assert metadata.version("bitsandbytes") == "0.49.2"
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, f"expected one visible GPU, got {torch.cuda.device_count()}"
name = torch.cuda.get_device_name(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
assert "H100" in name.upper(), f"expected H100, got {name}"
assert memory_gib >= 75, f"expected H100 80 GB, got {memory_gib:.1f} GiB"
assert torch.cuda.is_bf16_supported(), f"BF16 unsupported on {name}"
free_gib = torch.cuda.mem_get_info(0)[0] / 2**30
minimum_free_gib = float(os.environ["FTM_MIN_FREE_GPU_GIB"])
assert free_gib >= minimum_free_gib, (
    f"insufficient free VRAM: {free_gib:.1f} GiB; need {minimum_free_gib:.1f} GiB"
)
print(
    f"PASS_GPU_ENV torch={torch.__version__} CUDA={torch.version.cuda} "
    f"GPU={name} memory={memory_gib:.1f}GiB free={free_gib:.1f}GiB BF16=True"
)
PY

if [[ ! -s "${KB_CHUNKS}" ]]; then
  "${PY}" experiments/scripts/build_kb.py --out "${KB_CHUNKS}"
fi

"${PY}" experiments/scripts/check_experiment_release.py \
  --profile "${PROFILE}" \
  --require-cuda \
  --require-training \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --output-root "${RUN_ROOT}" \
  --kb-chunks-path "${KB_CHUNKS}"

command=(
  ./run_experiment.sh
  --profile "${PROFILE}"
  --dataset "${DATASET}"
  --model "${MODEL}"
  --methods "${METHODS[@]}"
  --seed "${SEED}"
  --with-dependencies
  --resume
  --no-paths-file
  --python "${PY}"
  --cache-path "${CACHE_ROOT}"
  --kb-chunks-path "${KB_CHUNKS}"
  --output-data-path "${RUN_ROOT}"
  --output-model-path "${MODEL_ROOT}"
  --results-path "${RESULT_ROOT}"
)

if [[ "${RUN_KIND}" == pilot ]]; then
  command+=(--n-eval-items 2 --n-train-items 2 --max-steps 1)
fi
if [[ "${INFERENCE_ONLY}" == 1 ]]; then
  command+=(
    --mode inference
    --input-model-weights "${ADAPTER_DIR}"
    --override "model.max_seq_length=${INFERENCE_MAX_SEQ_LENGTH}"
    --override "method.generate.max_new_tokens=${INFERENCE_MAX_NEW_TOKENS}"
    --override "method.allow_training_config_mismatch=true"
    --override "method.allow_inference_context_length_mismatch=true"
  )
fi
if [[ "${ROBUSTNESS}" == 0 ]]; then
  command+=(--no-paraphrased --no-missing-info)
fi

run_status=1
for ((attempt=1; attempt<=AUTO_RESUME_ATTEMPTS; attempt++)); do
  echo "=== RUN ATTEMPT ${attempt}/${AUTO_RESUME_ATTEMPTS} ==="
  if "${command[@]}"; then
    run_status=0
    break
  else
    run_status=$?
  fi
  if (( attempt < AUTO_RESUME_ATTEMPTS )); then
    echo "[RESUME] attempt ${attempt} failed; completed work is preserved. Retrying in 60 seconds."
    sleep 60
  fi
done
if (( run_status != 0 )); then
  echo "ERROR: run failed after ${AUTO_RESUME_ATTEMPTS} attempt(s). Re-run same command with --resume."
  exit "${run_status}"
fi

if [[ "${RUN_KIND}" == pilot ]]; then
  echo "PASS_H100_QWEN3_8_27B_PILOT"
  echo "Pilot results: ${RESULT_ROOT}"
  exit 0
fi

"${PY}" experiments/scripts/make_figures.py \
  --dataset "${DATASET}" \
  --results-dir "${RESULT_ROOT}" \
  --out-dir "${RESULT_ROOT}/figures"

PACKAGE="${PACKAGE_ROOT}/professor_results_${DATASET}_${MODEL}_seed42.zip"
"${PY}" experiments/scripts/package_professor_results.py \
  --dataset "${DATASET}" \
  --outputs-root "${RUN_ROOT}/${DATASET}" \
  --results-dir "${RESULT_ROOT}" \
  --out "${PACKAGE}"

echo "PASS_H100_QWEN3_8_27B_FULL"
echo "Results: ${RESULT_ROOT}"
echo "Package: ${PACKAGE}"
echo "Log: ${LOG_FILE}"
