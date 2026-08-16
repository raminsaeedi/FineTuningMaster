#!/usr/bin/env bash
# One command for the whole remote thesis run.
#
#     ./run_professor.sh
#
# Orchestrates the existing pieces in order and skips whatever is already done:
#
#   scripts/bootstrap_remote.sh          environment (only if missing/incomplete)
#   check_experiment_release.py          preflight: GPU, deps, dataset, HF access
#   build_kb.py                          RAG knowledge base (only if missing/stale)
#   run_experiment.sh                    A/B/C/D x models x seeds, resume enabled
#     -> run_final_matrix.py -> train.py / run_experiment.py -> aggregate_results.py
#   package_professor_results.py         final ZIP
#
# It contains no training or inference logic of its own. Re-running it after an
# interruption continues where the previous run stopped.
#
# Options (everything else has a sensible default):
#   --dataset NAME     dashboard_v4 (default) | dashboard_v3
#   --model KEY        one model instead of all four
#   --seed N           one seed instead of 42 43 44
#   --seeds "42 43"    selected seeds
#   --methods "A C"    selected methods (default: A B C D)
#   --skip-setup       do not run the environment bootstrap
#   --no-package       do not create the result ZIP
#   --cpu-ok           allow a machine without a CUDA GPU (A/B only, slow)
#   --dry-run          print the plan and the resolved commands, run nothing
#   -h, --help

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATASET="${DATASET:-dashboard_v4}"
MODEL=""
SEEDS="42 43 44"
METHODS="A B C D"
SKIP_SETUP=0
DO_PACKAGE=1
CPU_OK=0
DRY_RUN=0

ALL_MODELS=(qwen3_1_7b qwen3_8b qwen3_14b llama3_1_8b)
GATED_MODELS=(llama3_1_8b)

say()  { printf '\n\033[1m[%s]\033[0m %s\n' "run_professor" "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf '\n[run_professor] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) [[ $# -ge 2 ]] || die "--dataset needs a value."; DATASET="$2"; shift 2 ;;
    --model)   [[ $# -ge 2 ]] || die "--model needs a value.";   MODEL="$2";   shift 2 ;;
    --seed)    [[ $# -ge 2 ]] || die "--seed needs a value.";    SEEDS="$2";   shift 2 ;;
    --seeds)   [[ $# -ge 2 ]] || die "--seeds needs a value.";   SEEDS="$2";   shift 2 ;;
    --methods) [[ $# -ge 2 ]] || die "--methods needs a value."; METHODS="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --no-package) DO_PACKAGE=0; shift ;;
    --cpu-ok) CPU_OK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) die "Unknown option: $1 (see --help). Advanced options live in ./run_experiment.sh." ;;
  esac
done

[[ -f "src/config/data/$DATASET.yaml" ]] || die \
  "Unknown dataset '$DATASET'. Available: $(ls src/config/data/dashboard_v*.yaml | xargs -n1 basename | sed 's/.yaml//' | tr '\n' ' ')"

declare -a MODELS=()
if [[ -n "$MODEL" ]]; then
  [[ -f "src/config/model/$MODEL.yaml" ]] || die "Unknown model '$MODEL'. Final models: ${ALL_MODELS[*]}"
  MODELS=("$MODEL")
else
  MODELS=("${ALL_MODELS[@]}")
fi
read -r -a SEED_LIST <<< "${SEEDS//,/ }"
read -r -a METHOD_LIST <<< "${METHODS//,/ }"

run() {  # echo + execute, or only echo in --dry-run
  printf '  $ %s\n' "$*"
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

# ---------------------------------------------------------------------------
say "plan"
info "dataset : $DATASET"
info "models  : ${MODELS[*]}"
info "methods : ${METHOD_LIST[*]}"
info "seeds   : ${SEED_LIST[*]}"
info "resume  : on (completed work is reused, never recomputed)"
info "package : $([[ "$DO_PACKAGE" == 1 ]] && echo yes || echo no)"

# ---------------------------------------------------------------------------
# 1. HF_TOKEN gate. Checked before any expensive step, and never stored,
#    printed, logged or written to any artifact -- only its presence is used.
needs_token=0
for model in "${MODELS[@]}"; do
  for gated in "${GATED_MODELS[@]}"; do
    [[ "$model" == "$gated" ]] && needs_token=1
  done
done
if [[ "$needs_token" == 1 && -z "${HF_TOKEN:-}" ]]; then
  die "HF_TOKEN is not set, and ${GATED_MODELS[*]} is gated.
       Fix: export HF_TOKEN=\"hf_...\"   (approved account required)
       Or run the public models only, e.g.: ./run_professor.sh --model qwen3_8b"
fi

# ---------------------------------------------------------------------------
# 2. Environment. Bootstrap only when ./.venv cannot import the stack.
venv_python() {
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then printf '%s' "$PROJECT_ROOT/.venv/bin/python"
  elif [[ -x "$PROJECT_ROOT/.venv/Scripts/python.exe" ]]; then printf '%s' "$PROJECT_ROOT/.venv/Scripts/python.exe"
  fi
}

say "environment"
PY="$(venv_python || true)"
env_ready=0
if [[ -n "$PY" ]] && "$PY" -c "import torch, transformers, peft, trl, bitsandbytes, accelerate, datasets" 2>/dev/null; then
  env_ready=1
fi
if [[ "$env_ready" == 1 ]]; then
  info "already installed: $PY (skipping bootstrap)"
elif [[ "$SKIP_SETUP" == 1 ]]; then
  die "--skip-setup was given but ./.venv is missing or incomplete. Run ./scripts/bootstrap_remote.sh"
else
  info "installing the locked environment (first run only)"
  declare -a BOOTSTRAP_ARGS=()
  [[ "$CPU_OK" == 1 ]] && BOOTSTRAP_ARGS+=(--cpu-ok)
  run ./scripts/bootstrap_remote.sh "${BOOTSTRAP_ARGS[@]}"
  PY="$(venv_python || true)"
  if [[ "$DRY_RUN" != 1 ]]; then
    [[ -n "$PY" ]] || die "Bootstrap did not produce ./.venv."
  fi
fi
[[ -n "$PY" ]] || PY="python"   # --dry-run before the first install

# ---------------------------------------------------------------------------
# 3. RAG knowledge base. Rebuilt only when absent or no longer matching its
#    manifest; methods B and D need it.
say "rag knowledge base"
# Read-only manifest verification; cheap enough to run in --dry-run too.
kb_ok=0
if "$PY" - <<'PYKB'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
try:
    from src.data_pipeline.kb_builder import verify_kb
    ok, _ = verify_kb(Path("data/knowledge_base"))
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PYKB
then kb_ok=1; fi
if [[ "$kb_ok" == 1 ]]; then
  info "existing knowledge base verified against its manifest (skipping rebuild)"
else
  run "$PY" experiments/scripts/build_kb.py
fi

# ---------------------------------------------------------------------------
# 4. Preflight. Fast (~10 s): no weights are downloaded, nothing is inferred.
say "preflight"
# The `final` profile turns the CUDA/training-stack checks into hard failures;
# `--cpu-ok` downgrades them to warnings via the `smoke` profile.
preflight_profile=final
if [[ "$CPU_OK" == 1 ]]; then
  preflight_profile=smoke
  info "GPU checks downgraded to warnings (--cpu-ok)"
fi
declare -a PREFLIGHT=("$PY" experiments/scripts/check_experiment_release.py
                      --profile "$preflight_profile" --dataset "$DATASET")
if [[ -n "$MODEL" ]]; then PREFLIGHT+=(--model "$MODEL"); else PREFLIGHT+=(--all-models); fi
run "${PREFLIGHT[@]}"

# ---------------------------------------------------------------------------
# 5. Experiments. run_experiment.sh owns the matrix, adapter dependencies,
#    resume/cache identity and the aggregation step.
say "experiments"
declare -a EXPERIMENT=(./run_experiment.sh --profile final --dataset "$DATASET"
                       --with-dependencies --resume)
if [[ -n "$MODEL" ]]; then EXPERIMENT+=(--model "$MODEL"); else EXPERIMENT+=(--all-models); fi
EXPERIMENT+=(--methods "${METHOD_LIST[@]}" --seeds "${SEED_LIST[@]}")
[[ "$DRY_RUN" == 1 ]] && EXPERIMENT+=(--dry-run)

experiment_rc=0
printf '  $ %s\n' "${EXPERIMENT[*]}"
"${EXPERIMENT[@]}" || experiment_rc=$?

OUTPUT_ROOT="${OUTPUT_DATA_PATH:-$PROJECT_ROOT/experiments/outputs/final}"
RESULTS_ROOT="${RESULTS_PATH:-$PROJECT_ROOT/experiments/results/final/$DATASET}"
SUMMARY="$OUTPUT_ROOT/$DATASET/matrix_summary.json"

# ---------------------------------------------------------------------------
# 6. Package. Attempted even after a partial failure, so finished work is
#    always retrievable; completed results are never discarded.
PACKAGE_PATH="$PROJECT_ROOT/professor_results_$DATASET.zip"
if [[ "$DO_PACKAGE" == 1 && "$DRY_RUN" != 1 ]]; then
  say "packaging"
  "$PY" experiments/scripts/package_professor_results.py --dataset "$DATASET" || {
    info "packaging skipped (no result artifacts yet)"
    PACKAGE_PATH="<not created>"
  }
elif [[ "$DO_PACKAGE" == 1 ]]; then
  say "packaging"
  printf '  $ %s\n' "$PY experiments/scripts/package_professor_results.py --dataset $DATASET"
  PACKAGE_PATH="<dry-run>"
else
  PACKAGE_PATH="<disabled>"
fi

# ---------------------------------------------------------------------------
# 7. Report.
if [[ "$DRY_RUN" == 1 ]]; then
  say "dry run complete - nothing was executed"
  exit 0
fi

echo
echo "======================================================================"
if [[ "$experiment_rc" == 0 ]]; then echo "RUN COMPLETE"; else echo "RUN INCOMPLETE (see Failed below)"; fi
echo "======================================================================"
echo
echo "Results:"
echo "  $RESULTS_ROOT"
echo "  $OUTPUT_ROOT/$DATASET/<model>/<A|B|C|D>/seed_<seed>/"
echo
echo "Package:"
echo "  $PACKAGE_PATH"
echo
if [[ -f "$SUMMARY" ]]; then
  "$PY" - "$SUMMARY" <<'PYSUM'
import json, sys
from collections import defaultdict

rows = json.loads(open(sys.argv[1], encoding="utf-8").read()).get("runs", [])
done, failed = defaultdict(set), []
for row in rows:
    key = (str(row.get("model")), str(row.get("seed")))
    status = str(row.get("status", ""))
    if row.get("stage") == "run":
        if status in {"ok", "skipped"}:
            done[key].add(str(row.get("method")))
        else:
            failed.append(f"{row.get('model')} {row.get('method')} seed={row.get('seed')}: {status}")
    elif row.get("stage") == "train" and status not in {"ok", "skipped"}:
        failed.append(f"{row.get('model')} C-training seed={row.get('seed')}: {status}")

print("Completed:")
if done:
    for (model, seed), methods in sorted(done.items()):
        print(f"  {model:<14} seed {seed:<4} {' '.join(sorted(methods))}")
else:
    print("  none")
print()
print("Failed:")
print("  none" if not failed else "\n".join(f"  {item}" for item in failed))
PYSUM
else
  echo "Completed:"
  echo "  (no matrix summary written yet)"
  echo
  echo "Failed:"
  echo "  none"
fi
echo
if [[ "$experiment_rc" != 0 ]]; then
  echo "Re-run the same command to continue; finished runs are reused."
  echo "======================================================================"
  exit "$experiment_rc"
fi
echo "======================================================================"
