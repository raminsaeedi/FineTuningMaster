#!/usr/bin/env bash
# Shared per-machine path configuration.
#
# Sourced by run_professor.sh, run_experiment.sh and scripts/bootstrap_remote.sh
# so that ONE file -- paths.env -- decides, on every machine, where each large
# or machine-specific thing lives:
#
#   VENV_PATH          the Python environment                (default: ./.venv)
#   HF_HOME            downloaded base model weights          (default: ~/.cache/huggingface)
#   CACHE_PATH         model.cache_dir for training/inference (default: HF default)
#   DATA_PATH          frozen dataset directory               (default: data/frozen/<dataset>)
#   KB_CHUNKS_PATH     RAG chunk index                        (default: data/knowledge_base/chunks.jsonl)
#   OUTPUT_DATA_PATH   predictions/metrics/manifests/logs     (default: experiments/outputs/final)
#   OUTPUT_MODEL_PATH  adapters and trainer checkpoints       (default: = OUTPUT_DATA_PATH)
#   RESULTS_PATH       aggregated tables and figures          (default: experiments/results/<profile>/<dataset>)
#
# Usage in a script:
#     source "$PROJECT_ROOT/scripts/lib/paths.sh"
#     load_paths_file "$PROJECT_ROOT" "$PATHS_FILE"
#
# Values already exported in the environment win over the file only if the file
# does not set them; command-line options always win over both.

# Resolve a possibly relative path against the project root.
paths_resolve() {
  local root="$1" raw="$2"
  case "$raw" in
    "") printf '' ;;
    /*|[A-Za-z]:[\\/]*) printf '%s' "$raw" ;;
    *) printf '%s/%s' "$root" "$raw" ;;
  esac
}

# Source paths.env (or an explicit file) with every assignment exported, so the
# settings reach Python subprocesses too. A missing default file is not an error.
load_paths_file() {
  local root="$1" file="${2:-paths.env}"
  [[ -n "$file" ]] || return 0
  local resolved
  resolved="$(paths_resolve "$root" "$file")"
  if [[ -f "$resolved" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$resolved"
    set +a
    PATHS_FILE_RESOLVED="$resolved"
    return 0
  fi
  if [[ "$file" != "paths.env" ]]; then
    echo "[paths] ERROR: paths file not found: $resolved" >&2
    return 1
  fi
  PATHS_FILE_RESOLVED=""
  return 0
}

# Export the caches so every child process (Poetry, HF, PyTorch) agrees on them.
# Nothing is created here; each consumer creates what it needs.
apply_cache_paths() {
  local root="$1"
  if [[ -n "${HF_HOME:-}" ]]; then
    HF_HOME="$(paths_resolve "$root" "$HF_HOME")"
    export HF_HOME
    mkdir -p "$HF_HOME" 2>/dev/null || true
  fi
  if [[ -n "${CACHE_PATH:-}" ]]; then
    CACHE_PATH="$(paths_resolve "$root" "$CACHE_PATH")"
    export CACHE_PATH
    mkdir -p "$CACHE_PATH" 2>/dev/null || true
  fi
}

# Path of the project's Python interpreter, honouring VENV_PATH.
# Prints nothing when no environment exists yet.
venv_python_path() {
  local root="$1"
  local venv="${VENV_PATH:-$root/.venv}"
  venv="$(paths_resolve "$root" "$venv")"
  if [[ -x "$venv/bin/python" ]]; then printf '%s' "$venv/bin/python"
  elif [[ -x "$venv/Scripts/python.exe" ]]; then printf '%s' "$venv/Scripts/python.exe"
  fi
}
