#!/usr/bin/env bash
# ============================================================
# submit_pipeline.sh — Submit the full fine-tuning pipeline as a
# chained SLURM job sequence:
#
#   prepare_data → train → inference → evaluate
#
# Usage:
#   bash slurm/submit_pipeline.sh \
#       --model-config   configs/models/qwen_0_5b.yaml \
#       --experiment-config configs/experiments/qlora.yaml \
#       [--base-config   configs/base.yaml] \
#       [--partition     gpu] \
#       [--override      "training.learning_rate=5e-4"] \
#       [--skip-data]    # skip prepare_data (data already exists) \
#       [--dry-run]      # print sbatch commands without submitting
# ============================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG=""
EXPERIMENT_CONFIG=""
PARTITION="gpu"
EXTRA_OVERRIDES=""
SKIP_DATA=false
DRY_RUN=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-config)          BASE_CONFIG="$2";          shift 2 ;;
        --model-config)         MODEL_CONFIG="$2";         shift 2 ;;
        --experiment-config)    EXPERIMENT_CONFIG="$2";    shift 2 ;;
        --partition)            PARTITION="$2";            shift 2 ;;
        --override)             EXTRA_OVERRIDES="--override $2"; shift 2 ;;
        --skip-data)            SKIP_DATA=true;            shift   ;;
        --dry-run)              DRY_RUN=true;              shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_CONFIG" || -z "$EXPERIMENT_CONFIG" ]]; then
    echo "ERROR: --model-config and --experiment-config are required."
    echo "Usage: bash slurm/submit_pipeline.sh --model-config <path> --experiment-config <path>"
    exit 1
fi

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Derive experiment metadata from Python ───────────────────────────────────
ALGORITHM=$(python -c "
import yaml
cfg = {}
for f in ['$BASE_CONFIG', '$MODEL_CONFIG', '$EXPERIMENT_CONFIG']:
    try:
        with open(f) as fh: d = yaml.safe_load(fh) or {}
        cfg.update(d)
    except: pass
print(cfg.get('algorithm', {}).get('name', 'unknown'))
" 2>/dev/null || echo "unknown")

MODEL_SHORT=$(python -c "
import yaml
cfg = {}
for f in ['$BASE_CONFIG', '$MODEL_CONFIG', '$EXPERIMENT_CONFIG']:
    try:
        with open(f) as fh: d = yaml.safe_load(fh) or {}
        cfg.update(d)
    except: pass
name = cfg.get('model', {}).get('name', '').lower().split('/')[-1]
m = {'qwen2.5-0.5b-instruct':'qwen05b','qwen2.5-1.5b-instruct':'qwen15b','qwen2.5-3b-instruct':'qwen3b'}
print(m.get(name, name[:8]))
" 2>/dev/null || echo "unknown")

MEM=$(python -c "
import yaml
cfg = {}
for f in ['$BASE_CONFIG', '$MODEL_CONFIG', '$EXPERIMENT_CONFIG']:
    try:
        with open(f) as fh: d = yaml.safe_load(fh) or {}
        cfg.update(d)
    except: pass
print(cfg.get('slurm', {}).get('mem', '24G'))
" 2>/dev/null || echo "24G")

TIME_TRAIN=$(python -c "
import yaml
cfg = {}
for f in ['$BASE_CONFIG', '$MODEL_CONFIG', '$EXPERIMENT_CONFIG']:
    try:
        with open(f) as fh: d = yaml.safe_load(fh) or {}
        cfg.update(d)
    except: pass
print(cfg.get('slurm', {}).get('time_train', '02:00:00'))
" 2>/dev/null || echo "02:00:00")

TIME_INFER=$(python -c "
import yaml
cfg = {}
for f in ['$BASE_CONFIG', '$MODEL_CONFIG', '$EXPERIMENT_CONFIG']:
    try:
        with open(f) as fh: d = yaml.safe_load(fh) or {}
        cfg.update(d)
    except: pass
print(cfg.get('slurm', {}).get('time_infer', '01:00:00'))
" 2>/dev/null || echo "01:00:00")

# Use a placeholder experiment dir for log paths (train script creates the real one)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_ID="${ALGORITHM}_${MODEL_SHORT}_${TIMESTAMP}_preview"
EXPERIMENT_DIR="${PROJECT_ROOT}/outputs/experiments/${EXPERIMENT_ID}"
mkdir -p "${EXPERIMENT_DIR}/logs"

# ── Helper: substitute placeholders and submit ───────────────────────────────
submit_job() {
    local template="$1"
    local dependency="$2"

    local job_script="${EXPERIMENT_DIR}/logs/$(basename $template)"
    sed \
        -e "s|{PROJECT_ROOT}|${PROJECT_ROOT}|g" \
        -e "s|{EXPERIMENT_DIR}|${EXPERIMENT_DIR}|g" \
        -e "s|{EXPERIMENT_ID}|${EXPERIMENT_ID}|g" \
        -e "s|{BASE_CONFIG}|${BASE_CONFIG}|g" \
        -e "s|{MODEL_CONFIG}|${MODEL_CONFIG}|g" \
        -e "s|{EXPERIMENT_CONFIG}|${EXPERIMENT_CONFIG}|g" \
        -e "s|{ALGORITHM}|${ALGORITHM}|g" \
        -e "s|{MODEL_SHORT}|${MODEL_SHORT}|g" \
        -e "s|{MEM}|${MEM}|g" \
        -e "s|{TIME_TRAIN}|${TIME_TRAIN}|g" \
        -e "s|{TIME_INFER}|${TIME_INFER}|g" \
        -e "s|{PARTITION}|${PARTITION}|g" \
        -e "s|{EXTRA_OVERRIDES}|${EXTRA_OVERRIDES}|g" \
        "$template" > "$job_script"

    local dep_flag=""
    if [[ -n "$dependency" ]]; then
        dep_flag="--dependency=afterok:${dependency}"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] sbatch ${dep_flag} ${job_script}"
        echo "DRYRUN_JID"
    else
        sbatch $dep_flag "$job_script" | awk '{print $NF}'
    fi
}

# ── Submit pipeline stages ───────────────────────────────────────────────────
echo "========================================================"
echo "  Submitting fine-tuning pipeline"
echo "  Algorithm  : ${ALGORITHM}"
echo "  Model      : ${MODEL_SHORT}"
echo "  Partition  : ${PARTITION}"
echo "  Memory     : ${MEM}"
echo "  Experiment : ${EXPERIMENT_ID}"
echo "========================================================"

PREV_JID=""

if [[ "$SKIP_DATA" == "false" ]]; then
    echo -n "Stage 1 — prepare_data   : "
    DATA_JID=$(submit_job "slurm/templates/prepare_data.slurm" "")
    echo "job $DATA_JID"
    PREV_JID="$DATA_JID"
fi

echo -n "Stage 2 — train           : "
TRAIN_JID=$(submit_job "slurm/templates/train.slurm" "$PREV_JID")
echo "job $TRAIN_JID"
PREV_JID="$TRAIN_JID"

echo -n "Stage 3 — inference       : "
INFER_JID=$(submit_job "slurm/templates/inference.slurm" "$PREV_JID")
echo "job $INFER_JID"
PREV_JID="$INFER_JID"

echo -n "Stage 4 — evaluate        : "
EVAL_JID=$(submit_job "slurm/templates/evaluate.slurm" "$PREV_JID")
echo "job $EVAL_JID"

echo "========================================================"
echo "  Logs   : ${EXPERIMENT_DIR}/logs/"
echo "  Compare: python pipeline/compare.py --outputs-root outputs/experiments"
echo "========================================================"
