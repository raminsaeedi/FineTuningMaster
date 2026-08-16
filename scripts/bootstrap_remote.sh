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
# Options:
#   --no-train        Inference/evaluation only (methods A and B)
#   --with-dev        Also install pytest
#   --cpu-ok          Do not fail when no CUDA GPU is visible
#   --python PATH     Interpreter to build the environment with
#   -h, --help

set -euo pipefail

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

log()  { printf '[bootstrap] %s\n' "$*"; }
die()  { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-train) WITH_TRAIN=0; shift ;;
    --with-dev) WITH_DEV=1; shift ;;
    --cpu-ok) REQUIRE_CUDA=0; shift ;;
    --python) [[ $# -ge 2 ]] || die "--python needs a value."; PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

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
# 3. Project environment in ./.venv (poetry.toml sets virtualenvs.in-project).
"$POETRY" env use "$PYTHON_BIN" >/dev/null
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
            "poetry run pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
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
log "  ./.venv/bin/python experiments/scripts/build_kb.py"
log "  ./.venv/bin/python experiments/scripts/check_experiment_release.py --profile final --all-models --dataset dashboard_v4"
log "  ./run_experiment.sh --profile final --dataset dashboard_v4 --model qwen3_1_7b --all-methods --seed 42 --with-dependencies --resume"
