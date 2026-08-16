#!/usr/bin/env bash
# Submit the experiment matrix to SLURM as a parallel array job.
#
#     ./scripts/submit_slurm.sh                      # all models x all seeds
#     ./scripts/submit_slurm.sh --model qwen3_8b     # one model, 3 seeds
#     ./scripts/submit_slurm.sh --dry-run            # show the plan + sbatch file
#
# One array task = one (model, seed) pair = one GPU. Tasks are independent:
# each writes only into its own dataset/model/method/seed directory, so they can
# run concurrently, fail independently and be resubmitted individually. Every
# task calls ./run_professor.sh, which does the setup check, preflight, the
# A/B/C/D runs (C trains the adapter, D reuses it) and the aggregation.
#
# After the array finishes, one small dependent job aggregates across all tasks,
# builds the figures and writes the result ZIP.
#
# Options:
#   --dataset NAME      dashboard_v4 (default) | dashboard_v3
#   --model KEY         one model instead of all four
#   --seeds "42 43"     seeds (default: 42 43 44)
#   --methods "A C"     methods (default: A B C D)
#   --partition NAME    SLURM partition (default: $SLURM_PARTITION or gpu)
#   --gpus N            GPUs per task (default: 1; use 2 for a 14B model if needed)
#   --time HH:MM:SS     wall clock per task (default: 24:00:00)
#   --mem SIZE          host RAM per task (default: 64G)
#   --cpus N            CPU cores per task (default: 8)
#   --account NAME      SLURM account, if the cluster requires one
#   --max-parallel N    cap simultaneous tasks (default: unlimited)
#   --no-package        skip the final aggregation/packaging job
#   --dry-run           write and print the job script without submitting
#   -h, --help

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="dashboard_v4"
MODEL=""
SEEDS="42 43 44"
METHODS="A B C D"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS=1
TIME="24:00:00"
MEM="64G"
CPUS=8
ACCOUNT=""
MAX_PARALLEL=""
DO_PACKAGE=1
DRY_RUN=0

ALL_MODELS=(qwen3_1_7b qwen3_8b qwen3_14b llama3_1_8b)

log() { printf '[slurm] %s\n' "$*"; }
die() { printf '[slurm] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --no-package) DO_PACKAGE=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) die "Unknown option: $1 (see --help)" ;;
  esac
done

[[ -f "src/config/data/$DATASET.yaml" ]] || die "Unknown dataset '$DATASET'."

declare -a MODELS=()
if [[ -n "$MODEL" ]]; then
  [[ -f "src/config/model/$MODEL.yaml" ]] || die "Unknown model '$MODEL'."
  MODELS=("$MODEL")
else
  MODELS=("${ALL_MODELS[@]}")
fi
read -r -a SEED_LIST <<< "${SEEDS//,/ }"

# One line per array task: "<model> <seed>". Written to a file so the job script
# needs no bash arithmetic to map SLURM_ARRAY_TASK_ID onto a work item.
JOB_DIR="$PROJECT_ROOT/experiments/slurm/jobs"
mkdir -p "$JOB_DIR"
TASKLIST="$JOB_DIR/tasks_${DATASET}.txt"
: > "$TASKLIST"
for model in "${MODELS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    printf '%s %s\n' "$model" "$seed" >> "$TASKLIST"
  done
done
N_TASKS="$(wc -l < "$TASKLIST" | tr -d ' ')"
[[ "$N_TASKS" -gt 0 ]] || die "Nothing to submit."

# HF_TOKEN is passed through the environment by SLURM (--export below) and is
# never written into the job script, the task list or any log.
needs_token=0
for model in "${MODELS[@]}"; do
  [[ "$model" == llama3_1_8b ]] && needs_token=1
done
if [[ "$needs_token" == 1 && -z "${HF_TOKEN:-}" && "$DRY_RUN" != 1 ]]; then
  die "HF_TOKEN is not set and llama3_1_8b is gated. export HF_TOKEN=\"hf_...\" first."
fi

ARRAY_SPEC="0-$((N_TASKS - 1))"
[[ -n "$MAX_PARALLEL" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${MAX_PARALLEL}"
ACCOUNT_LINE=""
[[ -n "$ACCOUNT" ]] && ACCOUNT_LINE="#SBATCH --account=$ACCOUNT"

RUN_SCRIPT="$JOB_DIR/run_matrix_${DATASET}.sbatch"
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=thesis_${DATASET}
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --array=$ARRAY_SPEC
#SBATCH --output=$JOB_DIR/%x_%A_%a.out
#SBATCH --error=$JOB_DIR/%x_%A_%a.out
$ACCOUNT_LINE
set -euo pipefail
export PYTHONUTF8=1
cd "$PROJECT_ROOT"

# Map this array index to its (model, seed) pair.
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKLIST")
MODEL=\$(echo "\$LINE" | awk '{print \$1}')
SEED=\$(echo "\$LINE" | awk '{print \$2}')
echo "[task \$SLURM_ARRAY_TASK_ID] model=\$MODEL seed=\$SEED on \$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# SLURM already restricts the visible GPUs for this task; run_professor.sh
# performs setup checks, preflight, A/B/C/D and per-task aggregation.
./run_professor.sh \\
  --dataset "$DATASET" \\
  --model "\$MODEL" \\
  --seed "\$SEED" \\
  --methods "$METHODS" \\
  --no-package \\
  --no-figures
EOF
chmod +x "$RUN_SCRIPT"

PACKAGE_SCRIPT="$JOB_DIR/package_${DATASET}.sbatch"
cat > "$PACKAGE_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=thesis_pack_${DATASET}
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=$JOB_DIR/%x_%j.out
$ACCOUNT_LINE
set -euo pipefail
cd "$PROJECT_ROOT"
PY="\$(./scripts/lib/venv_python.sh)"
"\$PY" experiments/scripts/aggregate_results.py \\
  --outputs-root "\${OUTPUT_DATA_PATH:-experiments/outputs/final}/$DATASET" \\
  --out-dir "\${RESULTS_PATH:-experiments/results/final/$DATASET}"
"\$PY" experiments/scripts/make_figures.py --dataset "$DATASET"
"\$PY" experiments/scripts/package_professor_results.py --dataset "$DATASET"
EOF
chmod +x "$PACKAGE_SCRIPT"

log "dataset      : $DATASET"
log "models       : ${MODELS[*]}"
log "seeds        : ${SEED_LIST[*]}"
log "methods      : $METHODS"
log "array tasks  : $N_TASKS (1 GPU each, partition=$PARTITION, time=$TIME)"
log "task list    : $TASKLIST"
log "job script   : $RUN_SCRIPT"
[[ "$DO_PACKAGE" == 1 ]] && log "package job  : $PACKAGE_SCRIPT"

if [[ "$DRY_RUN" == 1 ]]; then
  log "dry run - not submitting. Job script:"
  echo "----------------------------------------------------------------------"
  cat "$RUN_SCRIPT"
  echo "----------------------------------------------------------------------"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch not found: this is not a SLURM machine."

# --export=ALL forwards HF_TOKEN and any paths.env values already exported.
ARRAY_ID="$(sbatch --parsable --export=ALL "$RUN_SCRIPT")"
log "submitted array job $ARRAY_ID"
if [[ "$DO_PACKAGE" == 1 ]]; then
  PACK_ID="$(sbatch --parsable --export=ALL --dependency=afterany:"$ARRAY_ID" "$PACKAGE_SCRIPT")"
  log "submitted packaging job $PACK_ID (runs after the array finishes)"
fi
log "watch:   squeue -u \$USER"
log "logs:    $JOB_DIR/"
log "results: \${RESULTS_PATH:-experiments/results/final/$DATASET}"
