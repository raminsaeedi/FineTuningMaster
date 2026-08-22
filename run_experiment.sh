#!/usr/bin/env bash
# Main experiment launcher.
#
# This is a thin wrapper around the existing Hydra-backed entry points. It does
# not train or infer itself. Set values in this block, export the same variable
# names, or override them with command-line options.

set -euo pipefail

# Force UTF-8 for every child process. Some dependencies (e.g. trl's
# chat_template_utils) read packaged UTF-8 files with Python's *locale* encoding;
# on a non-UTF-8 locale that raises UnicodeDecodeError at import time. Harmless
# where UTF-8 is already the default.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Per-machine paths file. One editable file per system decides where model
# caches, run artifacts, adapters/checkpoints and results live, so a
# storage-limited server never writes large files into the repository.
#
#   cp paths.env.example paths.env   # then edit paths.env
#
# Loaded BEFORE the defaults below, so every value it sets is picked up; a
# command-line option still wins over the file. Sourced from paths.env by
# default, or from --paths-file FILE / PATHS_FILE=FILE.
PATHS_FILE="${PATHS_FILE:-paths.env}"
for ((_i = 1; _i <= $#; _i++)); do
  if [[ "${!_i}" == "--paths-file" ]]; then
    _next=$((_i + 1))
    [[ $_next -le $# ]] || { echo "[launcher] ERROR: --paths-file requires a value." >&2; exit 1; }
    PATHS_FILE="${!_next}"
  elif [[ "${!_i}" == --paths-file=* ]]; then
    PATHS_FILE="${!_i#*=}"
  elif [[ "${!_i}" == "--no-paths-file" ]]; then
    PATHS_FILE=""
  fi
done
# shellcheck source=scripts/lib/paths.sh
source "$PROJECT_ROOT/scripts/lib/paths.sh"
load_paths_file "$PROJECT_ROOT" "$PATHS_FILE" || exit 1
apply_cache_paths "$PROJECT_ROOT"
[[ -z "${PATHS_FILE_RESOLVED:-}" ]] || echo "[launcher] paths file: $PATHS_FILE_RESOLVED"

# Optional generated HPC environment. Its location is supplied by the
# machine-specific paths file or the caller; no cluster path is hard-coded.
FTM_ENV_FILE="${FTM_ENV_FILE:-${HPC_ENV_FILE:-}}"
if [[ -z "$FTM_ENV_FILE" && -n "${SCRATCH_ROOT:-}" ]]; then
  FTM_ENV_FILE="$SCRATCH_ROOT/hpc_env.sh"
fi
if [[ -n "$FTM_ENV_FILE" && -f "$FTM_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$FTM_ENV_FILE"
  apply_cache_paths "$PROJECT_ROOT"
  echo "[launcher] HPC environment: $FTM_ENV_FILE"
fi
VENV_PATH="${VENV_PATH:-${FTM_VENV_DIR:-}}"

# One-place configuration. Command-line values win over these defaults.
PROFILE="${PROFILE:-smoke}"                         # smoke | final
MODE="${MODE:-full}"                               # full | inference | train
MODEL="${MODEL:-}"                                 # profile key; empty = all final models
METHOD="${METHOD:-}"                               # one of A/B/C/D
METHODS="${METHODS:-}"                             # space/comma-separated method keys
SEEDS="${SEEDS:-}"                                 # space/comma-separated integers
ALL_MODELS="${ALL_MODELS:-0}"
WITH_DEPENDENCIES="${WITH_DEPENDENCIES:-1}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

# Frozen dataset selection. dashboard_v4 is the final thesis dataset;
# dashboard_v3 stays selectable with --dataset dashboard_v3 (no code change).
DATASET="${DATASET:-dashboard_v4}"
DATA_PATH="${DATA_PATH:-}"                         # empty = data/frozen/<dataset>
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
TEST_DATA_PATH="${TEST_DATA_PATH:-}"
PARAPHRASED_DATA_PATH="${PARAPHRASED_DATA_PATH:-}"     # empty = dataset default
MISSING_INFO_DATA_PATH="${MISSING_INFO_DATA_PATH:-}"   # empty = dataset default
NO_PARAPHRASED=0
NO_MISSING_INFO=0
KB_CHUNKS_PATH="${KB_CHUNKS_PATH:-${FTM_KB_CHUNKS_PATH:-data/knowledge_base/chunks.jsonl}}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"             # local base-model directory
MODEL_ID="${MODEL_ID:-}"                           # Hugging Face ID or local directory
INPUT_MODEL_WEIGHTS="${INPUT_MODEL_WEIGHTS:-}"     # existing PEFT adapter directory
CACHE_PATH="${CACHE_PATH:-${FTM_CACHE_PATH:-}}"    # Hugging Face cache directory

OUTPUT_DATA_PATH="${OUTPUT_DATA_PATH:-${FTM_OUTPUT_DATA_PATH:-}}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-${FTM_OUTPUT_MODEL_PATH:-}}"
RESULTS_PATH="${RESULTS_PATH:-${FTM_RESULTS_PATH:-}}"

PYTHON_BIN="${PYTHON_BIN:-}"   # empty = resolve the Poetry environment below
TRAIN_EXPERIMENT="${TRAIN_EXPERIMENT:-E03_qwen0_5b_ft}"
N_EVAL_ITEMS="${N_EVAL_ITEMS:-2}"
N_TRAIN_ITEMS="${N_TRAIN_ITEMS:-2}"
MAX_STEPS="${MAX_STEPS:-1}"

declare -a EXTRA_OVERRIDES=()
# EXTRA_OVERRIDES_STR lets the paths file add Hydra overrides (an array cannot
# be expressed in a plain env file), e.g. training.sft.save_total_limit=1.
if [[ -n "${EXTRA_OVERRIDES_STR:-}" ]]; then
  read -r -a EXTRA_OVERRIDES <<< "$EXTRA_OVERRIDES_STR"
fi

usage() {
  cat <<'USAGE'
Usage:
  ./run_experiment.sh [options]

Modes and selection:
  --paths-file FILE           Per-machine path settings (default: paths.env)
  --no-paths-file             Ignore paths.env
  --profile smoke|final       Profile (default: smoke)
  --dataset NAME              Frozen dataset: dashboard_v4 (default) or dashboard_v3
  --mode full|inference|train Full matrix, inference-only, or C/QLoRA train
  --model PROFILE             One model profile
  --all-models                All four final model profiles
  --method A|B|C|D            One method
  --methods "A B C D"         Selected methods (commas also accepted)
  --seed N                    One seed
  --seeds "42 43 44"          Selected seeds (commas also accepted)
  --with-dependencies         Train C automatically when D needs it
  --no-dependencies           Do not inject C for D
  --resume / --no-resume      Resume compatible interrupted training

Paths:
  --data-path PATH            Dataset directory containing train/val/test JSONL
  --train-data-path PATH      Training split override
  --val-data-path PATH        Validation split override
  --test-data-path PATH       Test split override
  --paraphrased-data-path PATH  Robustness split override
  --missing-info-data-path PATH Robustness split override
  --no-paraphrased            Disable the paraphrase robustness split
  --no-missing-info           Disable the missing-info robustness split
  --kb-chunks-path PATH       RAG knowledge-base chunks file
  --base-model-path PATH      Local base-model directory
  --model-id ID               Hugging Face model ID (or local directory)
  --input-model-weights PATH  Existing PEFT adapter for C/D inference
  --cache-path PATH           Hugging Face model cache directory
  --output-data-path PATH     Run artifacts / predictions root
  --output-model-path PATH    Adapter and checkpoint root
  --results-path PATH         Aggregated results directory

Other:
  --python PATH               Python executable (default: ./.venv, else poetry run)
  --train-experiment NAME     Existing training config (default: E03_qwen0_5b_ft)
  --n-eval-items N            Smoke evaluation items (default: 2)
  --n-train-items N           Smoke training items (default: 2)
  --max-steps N               Smoke training max steps (default: 1)
  --override KEY=VALUE        Additional Hydra override; repeat it
  --dry-run                   Print resolved commands without running them
  --force                     Re-run compatible cached stages
  -h, --help                  Show this help

Examples:
  ./run_experiment.sh --profile smoke --model qwen2_5_0_5b \
      --all-methods --seed 42 --with-dependencies
  ./run_experiment.sh --profile final --dataset dashboard_v4 --all-models \
      --all-methods --seeds 42 43 44 --with-dependencies --resume
  ./run_experiment.sh --profile final --dataset dashboard_v3 --model qwen3_8b \
      --all-methods --seed 42 --with-dependencies --resume
  ./run_experiment.sh --profile final --model qwen3_8b --method C \
      --data-path /mnt/thesis/data --base-model-path /mnt/models/qwen3-8b \
      --output-data-path /mnt/thesis/runs --output-model-path /mnt/thesis/adapters \
      --results-path /mnt/thesis/results --cache-path /mnt/hf-cache
USAGE
}

die() {
  echo "[launcher] ERROR: $*" >&2
  exit 1
}

need_value() {
  [[ $# -ge 2 ]] || die "Option $1 requires a value."
}

is_absolute_path() {
  [[ "$1" == /* || "$1" =~ ^[A-Za-z]:[\\/].* ]]
}

resolve_path() {
  local raw="$1"
  if is_absolute_path "$raw"; then
    printf '%s' "$raw"
  else
    printf '%s/%s' "$PROJECT_ROOT" "$raw"
  fi
}

hydra_path() {
  local value="${1//\\//}"
  # Hydra treats the colon in a Windows drive letter as override syntax.
  # Quote paths with Hydra-significant characters, including spaces.
  if [[ "$value" =~ [[:space:]:,\[\]\{\}=#] ]]; then
    value="${value//\"/\\\"}"
    printf '"%s"' "$value"
  else
    printf '%s' "$value"
  fi
}

require_file() {
  [[ -f "$1" ]] || die "Required file is missing: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Required directory is missing: $1"
}

normalize_list() {
  local value="$1"
  printf '%s' "${value//,/ }"
}

contains_word() {
  local needle="$1"
  shift
  local word
  for word in "$@"; do
    [[ "$word" == "$needle" ]] && return 0
  done
  return 1
}

# -----------------------------------------------------------------------------
# CLI parsing. Arrays are used when invoking Python so spaces in paths survive.
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage; exit 0
      ;;
    --profile)
      need_value "$@" ; PROFILE="$2"; shift 2
      ;;
    --dataset)
      need_value "$@" ; DATASET="$2"; shift 2
      ;;
    --paths-file)
      need_value "$@" ; shift 2   # already consumed before the defaults block
      ;;
    --paths-file=*|--no-paths-file)
      shift
      ;;
    --mode)
      need_value "$@" ; MODE="$2"; shift 2
      ;;
    --model)
      need_value "$@" ; MODEL="$2"; ALL_MODELS=0; shift 2
      ;;
    --all-models)
      ALL_MODELS=1; MODEL=""; shift
      ;;
    --method)
      need_value "$@" ; METHOD="$2"; METHODS="$2"; shift 2
      ;;
    --methods)
      shift
      [[ $# -gt 0 && "$1" != --* ]] || die "--methods requires at least one method."
      METHODS=""
      while [[ $# -gt 0 && "$1" != --* ]]; do
        METHODS+="${METHODS:+ }$1"; shift
      done
      ;;
    --all-methods)
      METHODS="A B C D"; METHOD=""; shift
      ;;
    --seed)
      need_value "$@" ; SEEDS="$2"; shift 2
      ;;
    --seeds)
      shift
      [[ $# -gt 0 && "$1" != --* ]] || die "--seeds requires at least one seed."
      SEEDS=""
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SEEDS+="${SEEDS:+ }$1"; shift
      done
      ;;
    --with-dependencies) WITH_DEPENDENCIES=1; shift ;;
    --no-dependencies) WITH_DEPENDENCIES=0; shift ;;
    --resume) RESUME=1; shift ;;
    --no-resume) RESUME=0; shift ;;
    --data-path) need_value "$@" ; DATA_PATH="$2"; shift 2 ;;
    --train-data-path) need_value "$@" ; TRAIN_DATA_PATH="$2"; shift 2 ;;
    --val-data-path) need_value "$@" ; VAL_DATA_PATH="$2"; shift 2 ;;
    --test-data-path) need_value "$@" ; TEST_DATA_PATH="$2"; shift 2 ;;
    --paraphrased-data-path) need_value "$@" ; PARAPHRASED_DATA_PATH="$2"; shift 2 ;;
    --missing-info-data-path) need_value "$@" ; MISSING_INFO_DATA_PATH="$2"; shift 2 ;;
    --no-paraphrased) NO_PARAPHRASED=1; PARAPHRASED_DATA_PATH=""; shift ;;
    --no-missing-info) NO_MISSING_INFO=1; MISSING_INFO_DATA_PATH=""; shift ;;
    --kb-chunks-path) need_value "$@" ; KB_CHUNKS_PATH="$2"; shift 2 ;;
    --base-model-path) need_value "$@" ; BASE_MODEL_PATH="$2"; shift 2 ;;
    --model-id) need_value "$@" ; MODEL_ID="$2"; shift 2 ;;
    --input-model-weights) need_value "$@" ; INPUT_MODEL_WEIGHTS="$2"; shift 2 ;;
    --cache-path) need_value "$@" ; CACHE_PATH="$2"; shift 2 ;;
    --output-data-path) need_value "$@" ; OUTPUT_DATA_PATH="$2"; shift 2 ;;
    --output-model-path) need_value "$@" ; OUTPUT_MODEL_PATH="$2"; shift 2 ;;
    --results-path) need_value "$@" ; RESULTS_PATH="$2"; shift 2 ;;
    --python) need_value "$@" ; PYTHON_BIN="$2"; shift 2 ;;
    --train-experiment) need_value "$@" ; TRAIN_EXPERIMENT="$2"; shift 2 ;;
    --n-eval-items) need_value "$@" ; N_EVAL_ITEMS="$2"; shift 2 ;;
    --n-train-items) need_value "$@" ; N_TRAIN_ITEMS="$2"; shift 2 ;;
    --max-steps) need_value "$@" ; MAX_STEPS="$2"; shift 2 ;;
    --override) need_value "$@" ; EXTRA_OVERRIDES+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    *) die "Unknown option: $1 (use --help)." ;;
  esac
done

# -----------------------------------------------------------------------------
# Defaults and selection.
case "$PROFILE" in
  smoke)
    [[ -n "$MODEL" ]] || MODEL="qwen2_5_0_5b"
    [[ -n "$SEEDS" ]] || SEEDS="42"
    [[ -n "$METHODS" ]] || METHODS="A B C D"
    ;;
  final)
    [[ -n "$MODEL" || "$ALL_MODELS" == 1 ]] || ALL_MODELS=1
    [[ -n "$SEEDS" ]] || SEEDS="42 43 44"
    [[ -n "$METHODS" ]] || METHODS="A B C D"
    ;;
  *) die "PROFILE must be smoke or final, got '$PROFILE'." ;;
esac

case "$MODE" in full|inference|train) ;; *) die "MODE must be full, inference, or train." ;; esac

# Dataset selection drives data paths, run/result paths and the Hydra data group.
require_file "$PROJECT_ROOT/src/config/data/$DATASET.yaml"
DATASET_SUFFIX="${DATASET#dashboard_}"
[[ -n "$DATA_PATH" ]] || DATA_PATH="data/frozen/$DATASET"
# The smoke profile is a plumbing check on a 2-item slice: the full 274-item
# robustness splits would dominate its runtime, so they default to off there.
if [[ "$PROFILE" != smoke ]]; then
  if [[ "$NO_PARAPHRASED" == 0 && -z "$PARAPHRASED_DATA_PATH" ]]; then
    PARAPHRASED_DATA_PATH="data/eval/robustness_${DATASET_SUFFIX}/test_paraphrased.jsonl"
  fi
  if [[ "$NO_MISSING_INFO" == 0 && -z "$MISSING_INFO_DATA_PATH" ]]; then
    MISSING_INFO_DATA_PATH="data/eval/robustness_${DATASET_SUFFIX}/test_missing_info.jsonl"
  fi
fi

if [[ "$MODE" == train && -z "$METHOD" && ( -z "$METHODS" || "$METHODS" == "A B C D" ) ]]; then
  METHODS="C"
fi

if [[ -n "$METHOD" ]]; then
  if [[ -n "$METHODS" && "$METHODS" != "$METHOD" && "$METHODS" != "A B C D" ]]; then
    die "Set METHOD or METHODS, not both."
  fi
  METHODS="$METHOD"
fi
METHODS="$(normalize_list "$METHODS")"
read -r -a METHOD_LIST <<< "$METHODS"
[[ "${#METHOD_LIST[@]}" -gt 0 ]] || die "At least one method is required."
for i in "${!METHOD_LIST[@]}"; do
  METHOD_LIST[$i]="${METHOD_LIST[$i]^^}"
  case "${METHOD_LIST[$i]}" in A|B|C|D) ;; *) die "Unknown method '${METHOD_LIST[$i]}'." ;; esac
done
if [[ "$MODE" == train ]]; then
  for method in "${METHOD_LIST[@]}"; do
    [[ "$method" == C ]] || die "MODE=train uses only the existing C/QLoRA trainer; select method C."
  done
fi

SEEDS="$(normalize_list "$SEEDS")"
read -r -a SEED_LIST <<< "$SEEDS"
[[ "${#SEED_LIST[@]}" -gt 0 ]] || die "At least one seed is required."
for seed in "${SEED_LIST[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid seed '$seed'."
done

[[ -n "$OUTPUT_DATA_PATH" ]] || OUTPUT_DATA_PATH="experiments/outputs/$PROFILE"
[[ -n "$OUTPUT_MODEL_PATH" ]] || OUTPUT_MODEL_PATH="$OUTPUT_DATA_PATH"
[[ -n "$RESULTS_PATH" ]] || RESULTS_PATH="experiments/results/$PROFILE/$DATASET"

# -----------------------------------------------------------------------------
# Resolve and validate input paths before starting a run.
DATA_PATH="$(resolve_path "$DATA_PATH")"
require_dir "$DATA_PATH"
[[ -n "$TRAIN_DATA_PATH" ]] || TRAIN_DATA_PATH="$DATA_PATH/train.jsonl"
[[ -n "$VAL_DATA_PATH" ]] || VAL_DATA_PATH="$DATA_PATH/val.jsonl"
[[ -n "$TEST_DATA_PATH" ]] || TEST_DATA_PATH="$DATA_PATH/test.jsonl"
TRAIN_DATA_PATH="$(resolve_path "$TRAIN_DATA_PATH")"
VAL_DATA_PATH="$(resolve_path "$VAL_DATA_PATH")"
TEST_DATA_PATH="$(resolve_path "$TEST_DATA_PATH")"
require_file "$TRAIN_DATA_PATH"
require_file "$VAL_DATA_PATH"
require_file "$TEST_DATA_PATH"

if [[ -n "$PARAPHRASED_DATA_PATH" ]]; then
  PARAPHRASED_DATA_PATH="$(resolve_path "$PARAPHRASED_DATA_PATH")"
  require_file "$PARAPHRASED_DATA_PATH"
fi
if [[ -n "$MISSING_INFO_DATA_PATH" ]]; then
  MISSING_INFO_DATA_PATH="$(resolve_path "$MISSING_INFO_DATA_PATH")"
  require_file "$MISSING_INFO_DATA_PATH"
fi

KB_CHUNKS_PATH="$(resolve_path "$KB_CHUNKS_PATH")"
if contains_word B "${METHOD_LIST[@]}" || contains_word D "${METHOD_LIST[@]}"; then
  # chunks.jsonl is gitignored: it is rebuilt from the tracked guidelines.
  # Under --dry-run this is only a note: on a fresh clone the plan is still
  # valid, and run_professor.sh builds the knowledge base before the real run.
  if [[ ! -f "$KB_CHUNKS_PATH" ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then
      echo "[launcher] NOTE: RAG knowledge base not built yet: $KB_CHUNKS_PATH" >&2
      echo "[launcher]       It is created by: python experiments/scripts/build_kb.py" >&2
    else
      die "RAG knowledge base missing: $KB_CHUNKS_PATH
       Build it once with: python experiments/scripts/build_kb.py"
    fi
  fi
fi

if [[ -n "$BASE_MODEL_PATH" && -n "$MODEL_ID" ]]; then
  die "Set BASE_MODEL_PATH or MODEL_ID, not both."
fi
if [[ -n "$BASE_MODEL_PATH" ]]; then
  BASE_MODEL_PATH="$(resolve_path "$BASE_MODEL_PATH")"
  require_dir "$BASE_MODEL_PATH"
fi
if [[ -n "$MODEL_ID" && -e "$MODEL_ID" ]]; then
  MODEL_ID="$(resolve_path "$MODEL_ID")"
  require_dir "$MODEL_ID"
fi
if [[ -n "$INPUT_MODEL_WEIGHTS" ]]; then
  INPUT_MODEL_WEIGHTS="$(resolve_path "$INPUT_MODEL_WEIGHTS")"
  require_dir "$INPUT_MODEL_WEIGHTS"
  require_file "$INPUT_MODEL_WEIGHTS/adapter_config.json"
fi
if [[ -n "$CACHE_PATH" ]]; then CACHE_PATH="$(resolve_path "$CACHE_PATH")"; fi

OUTPUT_DATA_PATH="$(resolve_path "$OUTPUT_DATA_PATH")"
OUTPUT_MODEL_PATH="$(resolve_path "$OUTPUT_MODEL_PATH")"
RESULTS_PATH="$(resolve_path "$RESULTS_PATH")"

# -----------------------------------------------------------------------------
# Interpreter resolution. A fresh clone must never fall back to a system Python
# that lacks the locked thesis stack, so the project environment wins:
#   1. --python / PYTHON_BIN, when given explicitly
#   2. ./.venv  (created by scripts/bootstrap_remote.sh via poetry.toml)
#   3. `poetry run python`
#   4. system python, with a warning
declare -a PYTHON_CMD=()
if [[ -n "$PYTHON_BIN" ]]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
  PYTHON_CMD=("$PYTHON_BIN")
elif [[ -n "$(venv_python_path "$PROJECT_ROOT")" ]]; then
  PYTHON_CMD=("$(venv_python_path "$PROJECT_ROOT")")
elif [[ -x "$PROJECT_ROOT/.poetry/bin/poetry" ]]; then
  PYTHON_CMD=("$PROJECT_ROOT/.poetry/bin/poetry" run python)
elif command -v poetry >/dev/null 2>&1; then
  PYTHON_CMD=(poetry run python)
elif command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
  PYTHON_CMD=("$(command -v python3 || command -v python)")
  echo "[launcher] WARNING: no Poetry environment found; using $(command -v python3 || command -v python)." >&2
  echo "[launcher]          Run ./scripts/bootstrap_remote.sh for the locked thesis environment." >&2
else
  die "No Python interpreter found. Run ./scripts/bootstrap_remote.sh first."
fi
RUNNER="$PROJECT_ROOT/experiments/scripts/run_final_matrix.py"
TRAINER="$PROJECT_ROOT/experiments/scripts/train.py"
require_file "$RUNNER"
require_file "$TRAINER"

declare -a MODEL_LIST=()
if [[ "$PROFILE" == smoke ]]; then
  [[ -n "$MODEL" ]] || die "Smoke needs one explicit model profile."
  MODEL_LIST=("$MODEL")
elif [[ "$ALL_MODELS" == 1 ]]; then
  MODEL_LIST=(qwen3_1_7b qwen3_8b qwen3_14b llama3_1_8b)
else
  [[ -n "$MODEL" ]] || die "Final needs --model or --all-models."
  MODEL_LIST=("$MODEL")
fi
for model in "${MODEL_LIST[@]}"; do
  require_file "$PROJECT_ROOT/src/config/model/$model.yaml"
done

if [[ "$DRY_RUN" != 1 ]]; then
  mkdir -p "$OUTPUT_DATA_PATH" "$OUTPUT_MODEL_PATH" "$RESULTS_PATH"
  [[ -z "$CACHE_PATH" ]] || mkdir -p "$CACHE_PATH"
fi

# These are the actual Hydra keys consumed by train.py/run_experiment.py.
declare -a DATA_OVERRIDES=(
  "data=$DATASET"
  "data.frozen_dir=$(hydra_path "$DATA_PATH")"
  "data.train_file=$(hydra_path "$TRAIN_DATA_PATH")"
  "data.val_file=$(hydra_path "$VAL_DATA_PATH")"
  "data.test_file=$(hydra_path "$TEST_DATA_PATH")"
)
if [[ -n "$PARAPHRASED_DATA_PATH" ]]; then
  DATA_OVERRIDES+=("data.paraphrased_file=$(hydra_path "$PARAPHRASED_DATA_PATH")")
else
  DATA_OVERRIDES+=("data.paraphrased_file=null")
fi
if [[ -n "$MISSING_INFO_DATA_PATH" ]]; then
  DATA_OVERRIDES+=("data.missing_info_file=$(hydra_path "$MISSING_INFO_DATA_PATH")")
else
  DATA_OVERRIDES+=("data.missing_info_file=null")
fi

declare -a MODEL_OVERRIDES=()
if [[ -n "$BASE_MODEL_PATH" ]]; then
  MODEL_OVERRIDES+=(
    "model.name=$(hydra_path "$BASE_MODEL_PATH")"
    "model.hf_id=$(hydra_path "$BASE_MODEL_PATH")"
    "model.requires_hf_token=false"
  )
elif [[ -n "$MODEL_ID" ]]; then
  MODEL_OVERRIDES+=(
    "model.name=$(hydra_path "$MODEL_ID")"
    "model.hf_id=$(hydra_path "$MODEL_ID")"
  )
fi

declare -a CACHE_OVERRIDES=()
if [[ -n "$CACHE_PATH" ]]; then CACHE_OVERRIDES+=("model.cache_dir=$(hydra_path "$CACHE_PATH")"); fi

declare -a CONFIG_OVERRIDES=(
  "${DATA_OVERRIDES[@]}"
  "${MODEL_OVERRIDES[@]}"
  "${CACHE_OVERRIDES[@]}"
)

print_config() {
  local hf_auth=unset
  [[ -n "${HF_TOKEN:-}" ]] && hf_auth="set (value hidden)"
  echo
  echo "======================================================================"
  echo "RESOLVED EXPERIMENT LAUNCH CONFIGURATION"
  echo "======================================================================"
  printf '  project root       : %s\n' "$PROJECT_ROOT"
  printf '  python             : %s
' "${PYTHON_CMD[*]}"
  printf '  paths file         : %s
' "${PATHS_FILE:-<disabled>}"
  printf '  dataset            : %s
' "$DATASET"
  printf '  profile / mode     : %s / %s\n' "$PROFILE" "$MODE"
  printf '  models             : %s\n' "${MODEL_LIST[*]}"
  printf '  methods            : %s\n' "${METHOD_LIST[*]}"
  printf '  seeds              : %s\n' "${SEED_LIST[*]}"
  printf '  train experiment   : %s\n' "$TRAIN_EXPERIMENT"
  printf '  data path          : %s\n' "$DATA_PATH"
  printf '  train data         : %s\n' "$TRAIN_DATA_PATH"
  printf '  validation data    : %s\n' "$VAL_DATA_PATH"
  printf '  test data          : %s\n' "$TEST_DATA_PATH"
  printf '  paraphrased data   : %s\n' "${PARAPHRASED_DATA_PATH:-disabled}"
  printf '  missing-info data  : %s\n' "${MISSING_INFO_DATA_PATH:-disabled}"
  printf '  KB chunks          : %s\n' "$KB_CHUNKS_PATH"
  printf '  base model path    : %s\n' "${BASE_MODEL_PATH:-<profile config>}"
  printf '  model ID           : %s\n' "${MODEL_ID:-<profile config>}"
  printf '  input weights      : %s\n' "${INPUT_MODEL_WEIGHTS:-<same-run C adapter>}"
  printf '  cache path         : %s\n' "${CACHE_PATH:-<Hugging Face default>}"
  printf '  output data path   : %s\n' "$OUTPUT_DATA_PATH"
  printf '  output model path  : %s\n' "$OUTPUT_MODEL_PATH"
  printf '  results path       : %s\n' "$RESULTS_PATH"
  printf '  HF_TOKEN           : %s\n' "$hf_auth"
  printf '  resume/dependencies: %s / %s\n' "$RESUME" "$WITH_DEPENDENCIES"
  printf '  extra overrides    : %s\n' "${EXTRA_OVERRIDES[*]:-<none>}"
  echo "======================================================================"
}

print_config

run_training() {
  local model="$1"
  local seed="$2"
  local run_id="${DATASET}_${model}_C_seed_${seed}"
  local -a overrides=(
    "output_root=$(hydra_path "$OUTPUT_DATA_PATH")"
    "output_model_path=$(hydra_path "$OUTPUT_MODEL_PATH")"
    "profile=$PROFILE"
    "run_layout=$PROFILE"
    "model=$model"
    "model_key=$model"
    "method_key=C"
    "experiment_name=${model}_C"
    "experiment_id=$run_id"
    "seed=$seed"
    "${CONFIG_OVERRIDES[@]}"
    "${EXTRA_OVERRIDES[@]}"
  )
  if [[ "$PROFILE" == smoke ]]; then
    overrides+=(
      "data.max_samples=$N_TRAIN_ITEMS"
      "model.max_seq_length=512"
      "training.sft.num_train_epochs=1"
      "training.sft.per_device_train_batch_size=1"
      "training.sft.gradient_accumulation_steps=1"
      "training.sft.gradient_checkpointing=false"
      "+training.sft.max_steps=$MAX_STEPS"
      "training.sft.logging_steps=1"
      "training.sft.save_steps=1000"
    )
  fi
  local -a command=("${PYTHON_CMD[@]}" "$TRAINER" --experiment "$TRAIN_EXPERIMENT")
  [[ "$RESUME" == 1 ]] && command+=(--resume)
  command+=(--override "${overrides[@]}")
  echo
  echo "[launcher] TRAIN C: model=$model seed=$seed"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[launcher] [DRY-RUN]'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

if [[ "$MODE" == train ]]; then
  for model in "${MODEL_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      run_training "$model" "$seed"
    done
  done
  echo "[launcher] Training mode complete. Adapters/checkpoints: $OUTPUT_MODEL_PATH"
  exit 0
fi

# Matrix and inference modes delegate to the canonical runner.
declare -a command=(
  "${PYTHON_CMD[@]}" "$RUNNER"
  --profile "$PROFILE"
  --dataset "$DATASET"
  --output-root "$OUTPUT_DATA_PATH"
  --output-model-path "$OUTPUT_MODEL_PATH"
  --results-dir "$RESULTS_PATH"
  --kb-chunks-path "$KB_CHUNKS_PATH"
  --n-eval-items "$N_EVAL_ITEMS"
  --n-train-items "$N_TRAIN_ITEMS"
  --max-steps "$MAX_STEPS"
)
if [[ "$ALL_MODELS" == 1 ]]; then
  command+=(--all-models)
else
  command+=(--model "$MODEL")
fi
if [[ "${#METHOD_LIST[@]}" -eq 1 ]]; then
  command+=(--method "${METHOD_LIST[0]}")
else
  command+=(--methods "${METHOD_LIST[@]}")
fi
if [[ "${#SEED_LIST[@]}" -eq 1 ]]; then
  command+=(--seed "${SEED_LIST[0]}")
else
  command+=(--seeds "${SEED_LIST[@]}")
fi
[[ "$WITH_DEPENDENCIES" == 1 && "$MODE" == full ]] && command+=(--with-dependencies)
[[ "$RESUME" == 1 ]] && command+=(--resume)
[[ "$FORCE" == 1 ]] && command+=(--force)
[[ "$DRY_RUN" == 1 ]] && command+=(--dry-run)
[[ -n "$INPUT_MODEL_WEIGHTS" ]] && command+=(--input-model-weights "$INPUT_MODEL_WEIGHTS")
[[ "$PROFILE" == smoke ]] && command+=(--smoke-source "$VAL_DATA_PATH")
command+=(--override "${CONFIG_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}")
[[ "$MODE" == inference ]] && command+=(--skip-training)

echo "[launcher] Starting canonical experiment runner ($MODE mode)."
if [[ "$DRY_RUN" == 1 ]]; then
  printf '[launcher] [DRY-RUN]'
  printf ' %q' "${command[@]}"
  printf '\n'
else
  "${command[@]}"
fi
