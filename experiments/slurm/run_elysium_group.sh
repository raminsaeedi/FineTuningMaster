#!/usr/bin/env bash
# Shared body for RUB Elysium A30/H100 model-group jobs.

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

: "${GROUP_KEY:?GROUP_KEY is required}"
: "${MODEL_KEYS:?MODEL_KEYS is required}"
: "${EXPECTED_GPU:?EXPECTED_GPU is required}"

PATHS_FILE="${PATHS_FILE:-paths.env}"
# shellcheck source=../../scripts/lib/paths.sh
source "${PROJECT_ROOT}/scripts/lib/paths.sh"
load_paths_file "${PROJECT_ROOT}" "${PATHS_FILE}" || exit 1

ENV_FILE="${FTM_ENV_FILE:-${HPC_ENV_FILE:-}}"
if [[ -z "${ENV_FILE}" && -n "${BIG:-}" ]]; then
  ENV_FILE="${BIG}/hpc_env.sh"
fi
if [[ -z "${ENV_FILE}" || ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: hpc_env.sh missing. Run setup_hpc.sh on login node first." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

PY="${FTM_VENV_DIR}/bin/python"
[[ -x "${PY}" ]] || {
  echo "ERROR: Python environment missing: ${PY}" >&2
  exit 1
}

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-8}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== RUB Elysium ${GROUP_KEY} ==="
echo "job: ${SLURM_JOB_ID:-n/a}"
echo "node: ${SLURMD_NODENAME:-$(hostname)}"
echo "models: ${MODEL_KEYS}"
echo "seeds: ${SEEDS:-42 43 44}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
EXPECTED_GPU="${EXPECTED_GPU}" "${PY}" - <<'PY'
import os
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
name = torch.cuda.get_device_name(0)
expected = os.environ["EXPECTED_GPU"]
assert expected.lower() in name.lower(), f"expected {expected}, got {name}"
assert torch.cuda.is_bf16_supported(), f"BF16 unsupported on {name}"
print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPU", name, "BF16", True)
PY

KB_CHUNKS="${FTM_KB_CHUNKS_PATH}"
if [[ ! -s "${KB_CHUNKS}" ]]; then
  "${PY}" experiments/scripts/build_kb.py --out "${KB_CHUNKS}"
fi

DATASET="${DATASET:-dashboard_v4}"
PILOT="${PILOT:-0}"
read -r -a MODELS <<< "${MODEL_KEYS//,/ }"
read -r -a SEED_LIST <<< "${SEEDS:-42 43 44}"
read -r -a METHOD_LIST <<< "${METHODS:-A B C D}"

GROUP_ROOT="${FTM_SCRATCH_ROOT}/groups/${GROUP_KEY}"
if [[ "${PILOT}" == "1" ]]; then
  PROFILE="smoke"
  GROUP_ROOT="${GROUP_ROOT}/pilot"
  SEED_LIST=(42)
else
  PROFILE="final"
fi

RUN_ROOT="${GROUP_ROOT}/runs"
MODEL_ROOT="${GROUP_ROOT}/adapters"
RESULT_ROOT="${GROUP_ROOT}/results"
PACKAGE_ROOT="${GROUP_ROOT}/packages"
mkdir -p "${RUN_ROOT}" "${MODEL_ROOT}" "${RESULT_ROOT}" "${PACKAGE_ROOT}"

for model in "${MODELS[@]}"; do
  [[ -f "src/config/model/${model}.yaml" ]] || {
    echo "ERROR: unknown model profile: ${model}" >&2
    exit 1
  }

  "${PY}" experiments/scripts/check_experiment_release.py \
    --profile "${PROFILE}" \
    --require-cuda \
    --require-training \
    --dataset "${DATASET}" \
    --model "${model}" \
    --output-root "${RUN_ROOT}" \
    --kb-chunks-path "${KB_CHUNKS}"

  command=(
    ./run_experiment.sh
    --profile "${PROFILE}"
    --dataset "${DATASET}"
    --model "${model}"
    --methods "${METHOD_LIST[@]}"
    --seeds "${SEED_LIST[@]}"
    --with-dependencies
    --resume
    --paths-file "${PATHS_FILE}"
    --cache-path "${FTM_CACHE_PATH}"
    --kb-chunks-path "${KB_CHUNKS}"
    --output-data-path "${RUN_ROOT}"
    --output-model-path "${MODEL_ROOT}"
    --results-path "${RESULT_ROOT}"
  )
  if [[ "${PILOT}" == "1" ]]; then
    command+=(--n-eval-items 2 --n-train-items 2 --max-steps 1)
  fi
  "${command[@]}"
done

if [[ "${PILOT}" == "1" ]]; then
  echo "PASS_ELYSIUM_${GROUP_KEY}_PILOT"
  exit 0
fi

"${PY}" experiments/scripts/aggregate_results.py \
  --outputs-root "${RUN_ROOT}/${DATASET}" \
  --out-dir "${RESULT_ROOT}"
"${PY}" experiments/scripts/make_figures.py \
  --dataset "${DATASET}" \
  --results-dir "${RESULT_ROOT}" \
  --out-dir "${RESULT_ROOT}/figures"
"${PY}" experiments/scripts/package_professor_results.py \
  --dataset "${DATASET}" \
  --outputs-root "${RUN_ROOT}/${DATASET}" \
  --results-dir "${RESULT_ROOT}" \
  --out "${PACKAGE_ROOT}/professor_results_${DATASET}_${GROUP_KEY}.zip"

echo "PASS_ELYSIUM_${GROUP_KEY}_FULL"
echo "package: ${PACKAGE_ROOT}/professor_results_${DATASET}_${GROUP_KEY}.zip"
