#!/usr/bin/env bash
# One-command environment setup for a fresh Linux GPU machine.
#
#     ./scripts/bootstrap_remote.sh
#
# Creates ./.venv from poetry.lock (exact thesis versions), installs the
# training stack, and verifies that PyTorch actually sees the GPU. Safe to
# re-run: an existing environment is synced, not rebuilt. Installs no secrets
# and starts no training.
#
# Machine-specific locations (environment, caches, outputs) come from paths.env
# -- see paths.env.example. Nothing here is hard-coded to this repository.
#
# Options:
#   --no-train        Inference/evaluation only (methods A and B)
#   --with-dev        Also install pytest
#   --cpu-ok          Do not fail when no CUDA GPU is visible
#   --python PATH     Interpreter to build the environment with
#   TORCH_CUDA_INDEX  PyTorch wheel channel (default: cu124)
#   -h, --help

set -euo pipefail

# Force UTF-8 for every child process. Some dependencies (e.g. trl's
# chat_template_utils) read packaged UTF-8 files with Python's *locale* encoding;
# on a non-UTF-8 locale that raises UnicodeDecodeError at import time. Harmless
# where UTF-8 is already the default.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

POETRY_VERSION="${POETRY_VERSION:-2.2.1}"   # pinned, tested
MIN_PY_MINOR=11                             # requires-python = >=3.11,<3.14
MAX_PY_MINOR=13
WITH_TRAIN=1
WITH_DEV=0
REQUIRE_CUDA=1
PYTHON_BIN="${PYTHON_BIN:-}"
PATHS_FILE="${PATHS_FILE:-paths.env}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu124}"

log()  { printf '[bootstrap] %s\n' "$*"; }
die()  { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-train) WITH_TRAIN=0; shift ;;
    --with-dev) WITH_DEV=1; shift ;;
    --cpu-ok) REQUIRE_CUDA=0; shift ;;
    --python) [[ $# -ge 2 ]] || die "--python needs a value."; PYTHON_BIN="$2"; shift 2 ;;
    --paths-file) [[ $# -ge 2 ]] || die "--paths-file needs a value."; PATHS_FILE="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

# Per-machine paths: VENV_PATH, HF_HOME, CACHE_PATH, ... (all optional).
# shellcheck source=lib/paths.sh
source "$PROJECT_ROOT/scripts/lib/paths.sh"
load_paths_file "$PROJECT_ROOT" "$PATHS_FILE" || exit 1
apply_cache_paths "$PROJECT_ROOT"
[[ -z "${PATHS_FILE_RESOLVED:-}" ]] || log "paths file: $PATHS_FILE_RESOLVED"
[[ -z "${HF_HOME:-}" ]] || log "HF_HOME: $HF_HOME"

case "$TORCH_CUDA_INDEX" in
  cu118|cu124|cu126) ;;
  *) die "PyTorch 2.6.0 supports cu118, cu124 or cu126; got '$TORCH_CUDA_INDEX'." ;;
esac

# ---------------------------------------------------------------------------
# 1. Supported Python interpreter (Poetry cannot install one).
find_python() {
  local candidate
  for candidate in "$PYTHON_BIN" python3.12 python3.13 python3.11 python3 python; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c "import sys; raise SystemExit(0 if (3,$MIN_PY_MINOR) <= sys.version_info[:2] <= (3,$MAX_PY_MINOR) else 1)" 2>/dev/null; then
      printf '%s' "$(command -v "$candidate")"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(find_python)" || die \
  "No Python 3.$MIN_PY_MINOR-3.$MAX_PY_MINOR found. Install one (e.g. 'sudo apt install python3.12 python3.12-venv') and re-run, or pass --python /path/to/python."
log "python: $PYTHON_BIN ($("$PYTHON_BIN" -c 'import platform;print(platform.python_version())'))"

# ---------------------------------------------------------------------------
# 2. Poetry at the pinned version, in its own environment (never in .venv).
POETRY_HOME_DIR="${POETRY_HOME:-$PROJECT_ROOT/.poetry}"
# Linux/macOS venvs use bin/, Git-Bash-on-Windows venvs use Scripts/.
venv_exe() {
  local root="$1" name="$2"
  if [[ -x "$root/bin/$name" ]]; then printf '%s' "$root/bin/$name"
  elif [[ -x "$root/Scripts/$name.exe" ]]; then printf '%s' "$root/Scripts/$name.exe"
  fi
}
poetry_bin() {
  if command -v poetry >/dev/null 2>&1 &&
     poetry --version 2>/dev/null | grep -q "$POETRY_VERSION"; then
    command -v poetry
  else
    venv_exe "$POETRY_HOME_DIR" poetry
  fi
}

POETRY="$(poetry_bin || true)"
if [[ -z "$POETRY" ]]; then
  if command -v poetry >/dev/null 2>&1; then
    log "found $(poetry --version); installing pinned $POETRY_VERSION locally instead"
  else
    log "poetry not found; installing pinned $POETRY_VERSION into $POETRY_HOME_DIR"
  fi
  "$PYTHON_BIN" -m venv "$POETRY_HOME_DIR"
  POETRY_PY="$(venv_exe "$POETRY_HOME_DIR" python)"
  [[ -n "$POETRY_PY" ]] || die "Could not create the Poetry environment in $POETRY_HOME_DIR."
  "$POETRY_PY" -m pip install --quiet --upgrade pip
  "$POETRY_PY" -m pip install --quiet "poetry==$POETRY_VERSION"
  POETRY="$(venv_exe "$POETRY_HOME_DIR" poetry)"
  [[ -n "$POETRY" ]] || die "Poetry $POETRY_VERSION did not install into $POETRY_HOME_DIR."
fi
log "poetry: $("$POETRY" --version)"

# ---------------------------------------------------------------------------
# 3. Project environment. By default ./.venv (poetry.toml sets
#    virtualenvs.in-project). VENV_PATH in paths.env moves it to another volume:
#    the venv is created there and Poetry is pointed at that interpreter.
TARGET_VENV="$(paths_resolve "$PROJECT_ROOT" "${VENV_PATH:-$PROJECT_ROOT/.venv}")"
if [[ "$TARGET_VENV" != "$PROJECT_ROOT/.venv" ]]; then
  if [[ -z "$(venv_exe "$TARGET_VENV" python)" ]]; then
    log "creating the project environment in $TARGET_VENV"
    "$PYTHON_BIN" -m venv "$TARGET_VENV"
  fi
  ENV_PYTHON="$(venv_exe "$TARGET_VENV" python)"
  [[ -n "$ENV_PYTHON" ]] || die "Could not create the environment in $TARGET_VENV."
  "$POETRY" env use "$ENV_PYTHON" >/dev/null
else
  "$POETRY" env use "$PYTHON_BIN" >/dev/null
fi
log "environment: $("$POETRY" env info --path)"

# ---------------------------------------------------------------------------
# 4. Install exactly what poetry.lock pins.
# `poetry sync` (Poetry 2.x) installs exactly the locked set and removes
# anything else, so a re-run always converges on the same environment.
declare -a INSTALL_ARGS=(sync --no-interaction)
if [[ "$WITH_TRAIN" == 1 ]]; then INSTALL_ARGS+=(--extras train); fi
if [[ "$WITH_DEV" == 1 ]]; then INSTALL_ARGS+=(--with dev); else INSTALL_ARGS+=(--without dev); fi

"$POETRY" check --lock >/dev/null || die "poetry.lock is out of sync with pyproject.toml."
log "installing locked dependencies (this downloads ~3 GB of wheels the first time)"
"$POETRY" "${INSTALL_ARGS[@]}"

RUN_PY=("$POETRY" run python)

# Poetry's lock is platform-stable, but the wheel channel is machine-specific.
# On a CUDA run, replace a CPU/default PyPI torch wheel when necessary and then
# verify that the installed CUDA runtime matches the selected PyTorch channel.
if [[ "$REQUIRE_CUDA" == 1 ]]; then
  expected_torch_cuda="${TORCH_CUDA_INDEX#cu}"
  actual_torch_cuda="$("${RUN_PY[@]}" -c 'import torch; print((torch.version.cuda or "").replace(".", ""))' 2>/dev/null || true)"
  if [[ "$actual_torch_cuda" != "$expected_torch_cuda" ]]; then
    log "installing PyTorch 2.6.0 CUDA wheel (${TORCH_CUDA_INDEX})"
    "${RUN_PY[@]}" -m pip install --force-reinstall "torch==2.6.0" \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}"
  else
    log "PyTorch CUDA wheel matches ${TORCH_CUDA_INDEX}"
  fi
fi
"${RUN_PY[@]}" -m pip check

# ---------------------------------------------------------------------------
# 5+6. Lightweight environment check. Fails before any GPU work is started.
log "verifying the installed stack"
"${RUN_PY[@]}" - "$WITH_TRAIN" "$REQUIRE_CUDA" <<'PYCHECK'
import importlib.util
import sys

with_train = sys.argv[1] == "1"
require_cuda = sys.argv[2] == "1"

problems = []
core = ["torch", "transformers", "pydantic", "hydra", "omegaconf", "numpy",
        "scipy", "sklearn", "pandas", "yaml", "huggingface_hub"]
train = ["peft", "trl", "bitsandbytes", "accelerate", "datasets"]
missing = [m for m in core + (train if with_train else []) if importlib.util.find_spec(m) is None]
if missing:
    problems.append("missing modules: " + ", ".join(missing))

try:
    import torch

    print(f"  torch          : {torch.__version__}")
    print(f"  torch CUDA     : {torch.version.cuda or 'none (CPU-only build)'}")
    cuda = torch.cuda.is_available()
    print(f"  CUDA available : {cuda}")
    if cuda:
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"  VRAM free/total: {free >> 20} / {total >> 20} MiB")
    elif require_cuda:
        problems.append(
            "PyTorch cannot see a CUDA GPU. Check `nvidia-smi`, the NVIDIA driver "
            "(>= 550 for the CUDA 12.4 wheels), and that this is a GPU machine. "
            "Re-run with --cpu-ok to continue anyway (methods C/D will be unusable)."
        )
    if not torch.version.cuda and require_cuda:
        problems.append(
            "The installed torch is a CPU-only build. Install the CUDA build for this "
            "machine, then re-run: "
            "poetry run pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}"
        )
except Exception as exc:  # noqa: BLE001
    problems.append(f"torch import failed: {type(exc).__name__}: {exc}")

if with_train:
    try:
        import bitsandbytes  # noqa: F401

        print("  bitsandbytes   : import ok")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"bitsandbytes import failed: {type(exc).__name__}: {exc}")

if problems:
    print("\n".join(f"  FAIL: {p}" for p in problems))
    raise SystemExit(1)
print("  environment ok")
PYCHECK

log "done."
log "next:"
log "  export HF_TOKEN=\"hf_...\"                       # gated Llama profile"
log "  ./run_professor.sh                              # everything in one command"
log "  ./run_professor.sh --model qwen3_1_7b --seed 42  # one model, one seed"
