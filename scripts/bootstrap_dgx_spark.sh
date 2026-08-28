#!/usr/bin/env bash
# Environment setup for NVIDIA DGX Spark / Lenovo ThinkStation PGX (GB10).
#
# This is intentionally separate from the Poetry/x86 path. DGX Spark is ARM64
# and needs the CUDA 13 PyTorch wheel tested on the machine. The project code,
# configs, datasets and metrics are shared unchanged.

set -euo pipefail

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

WITH_TRAIN=1
WITH_DEV=0
REQUIRE_CUDA=1
PYTHON_BIN="${PYTHON_BIN:-}"
PATHS_FILE="${PATHS_FILE:-paths.env}"

# These are the versions observed and tested on the user's DGX Spark.
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu130}"
SPARK_TORCH_VERSION="${SPARK_TORCH_VERSION:-2.13.0}"

log()  { printf '[dgx-spark] %s\n' "$*"; }
die()  { printf '[dgx-spark] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-train) WITH_TRAIN=0; shift ;;
    --with-dev) WITH_DEV=1; shift ;;
    --cpu-ok) REQUIRE_CUDA=0; shift ;;
    --python) [[ $# -ge 2 ]] || die "--python needs a value."; PYTHON_BIN="$2"; shift 2 ;;
    --paths-file) [[ $# -ge 2 ]] || die "--paths-file needs a value."; PATHS_FILE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      printf '%s\n' 'Options: --no-train --with-dev --cpu-ok --python PATH --paths-file FILE'
      exit 0
      ;;
    *) die "Unknown option: $1" ;;
  esac
done

# shellcheck source=lib/paths.sh
source "$PROJECT_ROOT/scripts/lib/paths.sh"
load_paths_file "$PROJECT_ROOT" "$PATHS_FILE" || exit 1
apply_cache_paths "$PROJECT_ROOT"
[[ -z "${PATHS_FILE_RESOLVED:-}" ]] || log "paths file: $PATHS_FILE_RESOLVED"
[[ -z "${HF_HOME:-}" ]] || log "HF_HOME: $HF_HOME"

ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|arm64) ;;
  *) die "This launcher is for DGX Spark ARM64, but uname -m returned '$ARCH'." ;;
esac
[[ "$TORCH_CUDA_INDEX" == "cu130" ]] || die \
  "DGX Spark requires TORCH_CUDA_INDEX=cu130; got '$TORCH_CUDA_INDEX'."

find_python() {
  local candidate resolved
  for candidate in "$PYTHON_BIN" python3.12 python3; do
    [[ -n "$candidate" ]] || continue
    if [[ "$candidate" == */* ]]; then
      [[ -x "$candidate" ]] || continue
      resolved="$candidate"
    else
      command -v "$candidate" >/dev/null 2>&1 || continue
      resolved="$(command -v "$candidate")"
    fi
    if "$resolved" -c 'import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; then
      printf '%s' "$resolved"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(find_python)" || die \
  "No Python 3.11-3.13 found. Use --python /path/to/python3.12."
log "system Python: $PYTHON_BIN ($($PYTHON_BIN -c 'import platform; print(platform.python_version(), platform.machine())'))"

TARGET_VENV="$(paths_resolve "$PROJECT_ROOT" "${VENV_PATH:-$PROJECT_ROOT/.venv}")"
if [[ -e "$TARGET_VENV" && ! -x "$TARGET_VENV/bin/python" ]]; then
  die "VENV_PATH exists but is not a usable Linux venv: $TARGET_VENV. Choose a new path; the old 'venv' is not used."
fi
if [[ ! -x "$TARGET_VENV/bin/python" ]]; then
  log "creating project environment: $TARGET_VENV"
  mkdir -p "$(dirname -- "$TARGET_VENV")"
  "$PYTHON_BIN" -m venv "$TARGET_VENV"
fi
ENV_PYTHON="$TARGET_VENV/bin/python"
[[ -x "$ENV_PYTHON" ]] || die "Could not create the project environment: $TARGET_VENV"

log "upgrading pip tools"
"$ENV_PYTHON" -m pip install --quiet --upgrade pip setuptools wheel

# Install Torch first from the CUDA 13 channel. --no-deps is deliberate:
# requirements-train.txt below supplies the common Python dependencies without
# allowing pip to replace the tested GB10 wheel with an x86/CPU wheel.
log "installing PyTorch $SPARK_TORCH_VERSION from $TORCH_CUDA_INDEX"
"$ENV_PYTHON" -m pip install \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_INDEX" \
  --no-deps --force-reinstall "torch==$SPARK_TORCH_VERSION"

[[ -f "$PROJECT_ROOT/requirements-train.txt" ]] || die "requirements-train.txt is missing."
FILTERED_REQUIREMENTS="$(mktemp)"
TORCH_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$FILTERED_REQUIREMENTS" "$TORCH_REQUIREMENTS"' EXIT

# The exported requirements file is exact for the project, but its Torch and
# x86 Triton/NVIDIA entries are not appropriate for ARM64 GB10. Keep all
# portable pinned packages and omit only those machine-specific lines.
awk -v with_train="$WITH_TRAIN" '
  /^[[:space:]]*(torch|triton|nvidia-[A-Za-z0-9_.-]+)==/ { next }
  with_train == 0 && /^[[:space:]]*(accelerate|bitsandbytes|datasets|peft|trl)==/ { next }
  { print }
' "$PROJECT_ROOT/requirements-train.txt" > "$FILTERED_REQUIREMENTS"

log "installing portable project dependencies"
# This file is a complete exported lock-style requirement set. Installing it
# without dependency resolution is essential: accelerate/bitsandbytes declare
# only a generic torch>=2 requirement, and pip would otherwise replace the
# CUDA 13 GB10 wheel with the first compatible CPU wheel from PyPI.
"$ENV_PYTHON" -m pip install --index-url https://pypi.org/simple --no-deps -r "$FILTERED_REQUIREMENTS"

# The CUDA 13 ARM64 wheel carries runtime requirements that are not present in
# the older x86 requirements export (for example cuda-toolkit, CUDA 13 NVIDIA
# libraries and the matching Triton wheel). Read the exact requirements from
# the installed Torch wheel and install only those requirements. Torch itself
# is deliberately not in this temporary file, so it cannot be replaced by a
# CPU wheel during dependency resolution.
"$ENV_PYTHON" - "$TORCH_REQUIREMENTS" <<'PY'
import sys
from importlib.metadata import requires
from pathlib import Path

requirements = [
    requirement
    for requirement in (requires("torch") or [])
    if "extra ==" not in requirement
]
Path(sys.argv[1]).write_text("\n".join(requirements) + "\n", encoding="utf-8")
print("  torch runtime dependencies:")
for requirement in requirements:
    print(f"    {requirement}")
PY

log "installing Torch runtime dependencies"
"$ENV_PYTHON" -m pip install \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_INDEX" \
  --extra-index-url https://pypi.org/simple \
  --only-binary=:all: \
  -r "$TORCH_REQUIREMENTS"
"$ENV_PYTHON" -m pip check

log "checking PyTorch, CUDA and the real 4-bit path"
"$ENV_PYTHON" - "$REQUIRE_CUDA" "$SPARK_TORCH_VERSION" "$TORCH_CUDA_INDEX" "$WITH_TRAIN" <<'PY'
import platform
import sys

import torch

require_cuda = sys.argv[1] == "1"
expected_torch = sys.argv[2]
expected_channel = sys.argv[3].removeprefix("cu")
with_train = sys.argv[4] == "1"

print(f"  architecture : {platform.machine()}")
print(f"  torch        : {torch.__version__}")
print(f"  torch CUDA   : {torch.version.cuda or 'none'}")
print(f"  CUDA visible : {torch.cuda.is_available()}")

if require_cuda and not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch.")
if require_cuda:
    build = (torch.version.cuda or "").replace(".", "")
    if build != expected_channel:
        raise SystemExit(f"Expected CUDA build {expected_channel}, got {torch.version.cuda!r}.")
    if not torch.__version__.split("+")[0].startswith(expected_torch):
        raise SystemExit(f"Expected Torch {expected_torch}, got {torch.__version__!r}.")

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"  GPU          : {name}")
    print(f"  memory       : {free // (1 << 30)} / {total // (1 << 30)} GiB free/total")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    x = torch.randn((16, 16), device="cuda", dtype=dtype)
    y = x @ x
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all()):
        raise SystemExit("CUDA matrix operation returned non-finite values.")
    print(f"  CUDA matmul  : OK ({dtype})")

    if with_train:
        try:
            import bitsandbytes as bnb
            from bitsandbytes.nn import Linear4bit

            layer = Linear4bit(16, 16, bias=False, compute_dtype=dtype).to("cuda")
            sample = torch.randn((2, 16), device="cuda", dtype=dtype)
            with torch.no_grad():
                output = layer(sample)
            torch.cuda.synchronize()
            if tuple(output.shape) != (2, 16) or not bool(torch.isfinite(output).all()):
                raise SystemExit("bitsandbytes 4-bit forward returned an invalid result.")
            print(f"  bitsandbytes : {getattr(bnb, '__version__', 'installed')} (4-bit forward OK)")
        except Exception as exc:
            raise SystemExit(f"bitsandbytes 4-bit GPU test failed: {type(exc).__name__}: {exc}") from exc

print("  environment  : OK")
PY

log "done"
log "environment: $TARGET_VENV"
log "next smoke run: bash run_professor.sh --dataset dashboard_v4_tiny --model qwen2_5_0_5b --seed 42 --methods 'A B C D' --no-package"
