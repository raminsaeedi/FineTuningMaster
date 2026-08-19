#!/usr/bin/env bash
# One Slurm job for the final experiment matrix.
#
# Configure storage through setup_hpc.sh / FTM_ENV_FILE. Override scheduler
# directives at submission time when cluster defaults differ:
#   sbatch --account=ACCOUNT --partition=PARTITION \
#     --gpus-per-task=GPU_TYPE:1 experiments/slurm/job_final_matrix.sh
#
# Runtime overrides:
#   DATASET=dashboard_v4 MODEL=qwen3_1_7b METHOD=C SEED=42 sbatch ...
#   ALL_MODELS=1 SEEDS="42 43 44" sbatch ...

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --job-name=ftm_final
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-${FTM_SCRATCH_ROOT:-}}"
ENV_FILE="${FTM_ENV_FILE:-${HPC_ENV_FILE:-}}"
if [[ -z "${ENV_FILE}" && -n "${SCRATCH_ROOT}" ]]; then
  ENV_FILE="${SCRATCH_ROOT}/hpc_env.sh"
fi
if [[ -z "${ENV_FILE}" || ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: HPC environment file missing." >&2
  echo "       Run setup_hpc.sh, then export FTM_ENV_FILE=/path/to/hpc_env.sh." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

PROJECT_ROOT="${FTM_PROJECT_ROOT:-}"
if [[ -z "${PROJECT_ROOT}" || ! -d "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${PROJECT_ROOT}"

[[ -x "${FTM_VENV_DIR:-}/bin/python" ]] || {
  echo "ERROR: venv missing at ${FTM_VENV_DIR:-<unset>}. Run setup_hpc.sh." >&2
  exit 1
}
# shellcheck disable=SC1091
source "${FTM_VENV_DIR}/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p \
  "${FTM_RESULTS_ROOT:-${FTM_SCRATCH_ROOT:-${PROJECT_ROOT}}/results}/logs" \
  "${FTM_OUTPUT_DATA_PATH}" \
  "${FTM_OUTPUT_MODEL_PATH}" \
  "${FTM_RESULTS_PATH}" \
  "${FTM_KB_ROOT:-${FTM_SCRATCH_ROOT:-${PROJECT_ROOT}/.runtime}/kb}" \
  "${MPLCONFIGDIR:-${FTM_CACHE_ROOT:-${PROJECT_ROOT}/.runtime}/mpl}"

PROFILE="${PROFILE:-final}"
DATASET="${DATASET:-dashboard_v4}"
MODEL="${MODEL:-qwen3_1_7b}"
METHOD="${METHOD:-}"
ALL_MODELS="${ALL_MODELS:-0}"
SEED="${SEED:-}"
SEEDS="${SEEDS:-}"
WITH_DEPENDENCIES="${WITH_DEPENDENCIES:-1}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${SEEDS}" ]]; then
  SEEDS="${SEED:-42}"
fi

LAUNCHER=(./run_experiment.sh
  --profile "${PROFILE}"
  --dataset "${DATASET}"
  --seeds ${SEEDS}
  --cache-path "${FTM_CACHE_PATH}"
  --output-data-path "${FTM_OUTPUT_DATA_PATH}"
  --output-model-path "${FTM_OUTPUT_MODEL_PATH}"
  --results-path "${FTM_RESULTS_PATH}"
)

if [[ -n "${FTM_KB_CHUNKS_PATH:-}" ]]; then
  LAUNCHER+=(--kb-chunks-path "${FTM_KB_CHUNKS_PATH}")
fi
if [[ "${ALL_MODELS}" == "1" ]]; then
  LAUNCHER+=(--all-models)
else
  LAUNCHER+=(--model "${MODEL}")
fi
if [[ -n "${METHOD}" ]]; then
  LAUNCHER+=(--method "${METHOD}")
else
  LAUNCHER+=(--all-methods)
fi
if [[ "${WITH_DEPENDENCIES}" == "1" ]]; then
  LAUNCHER+=(--with-dependencies)
else
  LAUNCHER+=(--no-dependencies)
fi
if [[ "${RESUME}" == "1" ]]; then
  LAUNCHER+=(--resume)
else
  LAUNCHER+=(--no-resume)
fi
[[ "${DRY_RUN}" == "1" ]] && LAUNCHER+=(--dry-run)

echo "=== FineTuningMaster Slurm job ==="
echo "  job id       : ${SLURM_JOB_ID:-n/a}"
echo "  node         : ${SLURMD_NODENAME:-$(hostname)}"
echo "  project root : ${PROJECT_ROOT}"
echo "  venv         : ${FTM_VENV_DIR}"
echo "  env file     : ${ENV_FILE}"
echo "  profile      : ${PROFILE}"
echo "  dataset      : ${DATASET}"
if [[ "${ALL_MODELS}" == "1" ]]; then
  echo "  model        : ALL"
else
  echo "  model        : ${MODEL}"
fi
echo "  method       : ${METHOD:-A B C D}"
echo "  seeds        : ${SEEDS}"
echo "  outputs      : ${FTM_OUTPUT_DATA_PATH}"
echo "  results      : ${FTM_RESULTS_PATH}"
echo "  model cache  : ${FTM_CACHE_PATH}"
echo "  KB chunks    : ${FTM_KB_CHUNKS_PATH:-<unset>}"
echo "  HF_TOKEN     : $([[ -n "${HF_TOKEN:-}" ]] && echo set || echo unset)"
echo "  start        : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
printf '[job]'; printf ' %q' "${LAUNCHER[@]}"; printf '\n'

"${LAUNCHER[@]}"
