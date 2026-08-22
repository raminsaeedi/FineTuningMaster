#!/usr/bin/env bash
# RUB Elysium H100 80 GB: Qwen3 8B + Qwen3.8 27B.

#SBATCH --partition=fat_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=ftm_h100_large
#SBATCH --output=experiments/slurm/jobs/%x_%j.out
#SBATCH --error=experiments/slurm/jobs/%x_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GROUP_KEY=h100_large
export MODEL_KEYS="qwen3_8b qwen3_8_27b"
export EXPECTED_GPU=H100
exec bash "${SCRIPT_DIR}/run_elysium_group.sh"
