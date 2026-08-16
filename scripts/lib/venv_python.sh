#!/usr/bin/env bash
# Print the path of the project's Python interpreter, honouring paths.env
# (VENV_PATH). Used by job scripts that must not guess ./.venv.
#
#     PY="$(./scripts/lib/venv_python.sh)"
#     "$PY" experiments/scripts/aggregate_results.py ...

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=paths.sh
source "$SCRIPT_DIR/paths.sh"
load_paths_file "$PROJECT_ROOT" "${PATHS_FILE:-paths.env}" >/dev/null || true
PY="$(venv_python_path "$PROJECT_ROOT")"
if [[ -z "$PY" ]]; then
  echo "[venv_python] ERROR: no project environment found. Run ./scripts/bootstrap_remote.sh" >&2
  exit 1
fi
printf '%s\n' "$PY"
