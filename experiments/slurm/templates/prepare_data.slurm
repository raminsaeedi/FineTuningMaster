#!/bin/bash
# ============================================================
# SLURM job: Stage 1 — prepare dataset
# Submitted by: slurm/submit_pipeline.sh
# ============================================================
#SBATCH --job-name=ft-data
#SBATCH --output={EXPERIMENT_DIR}/logs/prepare_data_%j.log
#SBATCH --error={EXPERIMENT_DIR}/logs/prepare_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition={PARTITION}

# ── Environment ─────────────────────────────────────────────────────────────
module load python/3.11 2>/dev/null || true

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

cd {PROJECT_ROOT}

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start time  : $(date)"
echo "Working dir : $(pwd)"
echo "------------------------------------------------------------"

python pipeline/prepare_data.py \
    --base-config {BASE_CONFIG} \
    --model-config {MODEL_CONFIG} \
    --experiment-config {EXPERIMENT_CONFIG} \
    --force

EXIT_CODE=$?
echo "------------------------------------------------------------"
echo "End time : $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
