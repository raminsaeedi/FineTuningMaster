#!/usr/bin/env bash
# RUB Elysium A30 24 GB: OLMo2 ~1.49B + Qwen3 1.7B.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=ftm_a30_small
#SBATCH --output=experiments/slurm/jobs/%x_%j.out
#SBATCH --error=experiments/slurm/jobs/%x_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GROUP_KEY=a30_small
export MODEL_KEYS="olmo2_1_49b qwen3_1_7b"
export EXPECTED_GPU=A30
exec bash "${SCRIPT_DIR}/run_elysium_group.sh"
