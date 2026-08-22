#!/usr/bin/env bash
# RUB Elysium: one H100 80 GB, Qwen3-1.7B, methods A/B/C/D, seeds 42/43/44.
# Prepare environment and model cache on a login node before submission.
# Submit with:
#   sbatch --account=<rub-project> experiments/slurm/job_elysium_qwen3_1_7b.sh

#SBATCH --partition=fat_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=ftm_qwen17
#SBATCH --output=experiments/slurm/jobs/%x_%j.out
#SBATCH --error=experiments/slurm/jobs/%x_%j.out

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

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

mkdir -p "${PROJECT_ROOT}/experiments/slurm/jobs"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-8}}"
# RUB compute nodes have no public internet by default. Model must be cached
# on login node before submission.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== RUB Elysium Qwen3-1.7B ==="
echo "job: ${SLURM_JOB_ID:-n/a}"
echo "node: ${SLURMD_NODENAME:-$(hostname)}"
echo "seeds: ${SEEDS:-42 43 44}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PY}" -c 'import torch; assert torch.cuda.is_available(); name = torch.cuda.get_device_name(0); assert "H100" in name, name; assert torch.cuda.is_bf16_supported(); print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPU", name, "BF16", True)'

./run_professor.sh \
  --dataset dashboard_v4 \
  --model qwen3_1_7b \
  --seeds "${SEEDS:-42 43 44}" \
  --methods "A B C D" \
  --gpus 0 \
  --paths-file "${PATHS_FILE}" \
  --skip-setup
