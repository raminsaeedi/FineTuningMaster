#!/bin/bash
# ============================================================
# SLURM job: Stage 4 — compute evaluation metrics
# Submitted by: slurm/submit_pipeline.sh
# ============================================================
#SBATCH --job-name=ft-eval-{ALGORITHM}-{MODEL_SHORT}
#SBATCH --output={EXPERIMENT_DIR}/logs/evaluate_%j.log
#SBATCH --error={EXPERIMENT_DIR}/logs/evaluate_%j.err
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
echo "Experiment  : {EXPERIMENT_ID}"
echo "Start time  : $(date)"
echo "------------------------------------------------------------"

python pipeline/evaluate.py \
    --experiment-dir {EXPERIMENT_DIR} \
    --mode all

EXIT_CODE=$?
echo "------------------------------------------------------------"
echo "End time : $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
