#!/bin/bash
# ============================================================
# SLURM job: Stage 2 — fine-tune model
# Submitted by: slurm/submit_pipeline.sh
# ============================================================
#SBATCH --job-name=ft-train-{ALGORITHM}-{MODEL_SHORT}
#SBATCH --output={EXPERIMENT_DIR}/logs/train_%j.log
#SBATCH --error={EXPERIMENT_DIR}/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem={MEM}
#SBATCH --time={TIME_TRAIN}
#SBATCH --partition={PARTITION}

# ── Environment ─────────────────────────────────────────────────────────────
module load python/3.11 cuda/12.1 2>/dev/null || true

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

cd {PROJECT_ROOT}

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "Algorithm   : {ALGORITHM}"
echo "Model       : {MODEL_SHORT}"
echo "Experiment  : {EXPERIMENT_ID}"
echo "Start time  : $(date)"
echo "------------------------------------------------------------"

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

python pipeline/train.py \
    --base-config {BASE_CONFIG} \
    --model-config {MODEL_CONFIG} \
    --experiment-config {EXPERIMENT_CONFIG} \
    {EXTRA_OVERRIDES}

EXIT_CODE=$?
echo "------------------------------------------------------------"
echo "End time : $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
