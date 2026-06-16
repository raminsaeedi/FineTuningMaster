#!/bin/bash
# ============================================================
# SLURM job: Stage 3 — run inference (base + fine-tuned)
# Submitted by: slurm/submit_pipeline.sh
# ============================================================
#SBATCH --job-name=ft-infer-{ALGORITHM}-{MODEL_SHORT}
#SBATCH --output={EXPERIMENT_DIR}/logs/inference_%j.log
#SBATCH --error={EXPERIMENT_DIR}/logs/inference_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem={MEM}
#SBATCH --time={TIME_INFER}
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
echo "Experiment  : {EXPERIMENT_ID}"
echo "Start time  : $(date)"
echo "------------------------------------------------------------"

python pipeline/inference.py \
    --experiment-dir {EXPERIMENT_DIR} \
    --mode both

EXIT_CODE=$?
echo "------------------------------------------------------------"
echo "End time : $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
