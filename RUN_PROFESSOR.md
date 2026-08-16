# Quick Start

**Before you start**, the machine needs: a Linux **NVIDIA GPU** (driver >= 550),
**Python 3.11–3.13** with `venv`, `git`, internet, ~65 GB free disk, and a
Hugging Face token from an account approved for `meta-llama/Llama-3.1-8B-Instruct`.
Nothing else has to be installed by hand.

**Then copy-paste these four lines:**

```bash
git clone https://github.com/raminsaeedi/FineTuningMaster.git
cd FineTuningMaster
export HF_TOKEN="hf_your_token_here"
./run_professor.sh
```

**That is the whole procedure.** One command does all of it, in order:

1. installs every dependency into `./.venv` (exact locked versions),
2. checks GPU, CUDA, PyTorch, dataset and model access,
3. builds the RAG knowledge base,
4. runs all 4 models x methods A/B/C/D x seeds 42/43/44,
5. aggregates the results and draws the figures,
6. writes `professor_results_dashboard_v4.zip` to send back.

**If it stops** — session ended, GPU lost, network died, anything — run the
**same command again**:

```bash
./run_professor.sh
```

It continues where it stopped. Finished models, seeds, adapters and predictions
are reused, never recomputed. Nothing that already succeeded is lost.

**Smaller pieces**, when a session is too short for everything:

```bash
./run_professor.sh --model qwen3_8b             # one model, all three seeds
./run_professor.sh --model qwen3_8b --seed 42   # one model, one seed
./run_professor.sh --dry-run                    # show the plan, run nothing
```

Results land in `experiments/results/final/dashboard_v4/`; the ZIP in the
repository root. Everything below is detail — read it only if needed.

> Shell says `Permission denied`? Use `bash run_professor.sh`, or once:
> `chmod +x run_professor.sh scripts/*.sh scripts/lib/*.sh`.

---

## 1. What gets run

| | |
|---|---|
| Dataset | `dashboard_v4` (2932 train / 613 validation / 274 held-out test) |
| Models | `qwen3_1_7b`, `qwen3_8b`, `qwen3_14b`, `llama3_1_8b` |
| Methods | **A** prompt-only, **B** RAG, **C** QLoRA fine-tuning, **D** QLoRA + RAG |
| Seeds | 42, 43, 44 |
| Total | 4 models x 4 methods x 3 seeds = 48 evaluation runs + 12 fine-tuning runs |

Method C trains one adapter per model and seed. Method D automatically reuses
the C adapter of the **same** model, seed and dataset — never another one.

## 2. What the machine needs

Only these cannot be installed by the script:

- Linux with an **NVIDIA GPU** and driver **>= 550** (CUDA 12.4 wheels)
- **Python 3.11–3.13** with `venv` (e.g. `sudo apt install python3.12 python3.12-venv`)
- `git`, internet access
- Disk: ~65 GB for model weights plus space for adapters and checkpoints
  (see [§6](#6-storage-on-a-small-disk) if the home volume is small)
- A Hugging Face account **approved for `meta-llama/Llama-3.1-8B-Instruct`**
  (the three Qwen models are public)

Everything else — Poetry, the virtual environment, PyTorch, the training stack —
is installed automatically from the committed `poetry.lock`, so the versions are
exactly the ones the thesis was developed with.

## 3. Run it

```bash
export HF_TOKEN="hf_your_token_here"
```

| Goal | Command |
|---|---|
| Everything (default) | `./run_professor.sh` |
| One model, all 3 seeds | `./run_professor.sh --model qwen3_8b` |
| One model, one seed | `./run_professor.sh --model qwen3_8b --seed 42` |
| Continue after an interruption | `./run_professor.sh` (same command again) |
| See the plan without running | `./run_professor.sh --dry-run` |
| Only some methods | `./run_professor.sh --methods "A B"` |
| The earlier dataset | `./run_professor.sh --dataset dashboard_v3` |

Model keys: `qwen3_1_7b`, `qwen3_8b`, `qwen3_14b`, `llama3_1_8b`.
Seeds: `42`, `43`, `44`.

**If the GPU session has a time limit**, run one model and one seed per session.
The commands compose freely and each one resumes:

```bash
./run_professor.sh --model qwen3_1_7b --seed 42
./run_professor.sh --model qwen3_1_7b --seed 43
./run_professor.sh --model qwen3_1_7b --seed 44
```

`./run_professor.sh --help` lists every option.

## 4. Resuming

Re-running the same command **never repeats finished work**. Reused as-is:

- completed runs with their predictions and metrics,
- compatible fine-tuning checkpoints,
- the C adapter that method D needs,
- the RAG knowledge base.

A run is reused only when dataset, model, method, seed, dataset hashes, training
config, inference config and knowledge-base hash all match; otherwise it is
recomputed. A failed model or seed never damages results that already finished —
the end-of-run report lists `Completed:` and `Failed:` separately.

## 5. Results

At the end the script prints where everything is:

```text
experiments/results/final/dashboard_v4/
├── comparison_table.csv / .md        one row per run
├── cross_model/model_method_summary.csv / .md
├── multi_seed_summary.csv / .md      mean and spread across seeds
├── per_model/<model>/                per-model tables
├── figures/                          F1–F5 as PNG + PDF, plus figure_data.csv
└── final_report.md                   readable summary

experiments/outputs/final/dashboard_v4/<model>/<A|B|C|D>/seed_<seed>/
├── predictions.jsonl                 model outputs
├── metrics_auto.json                 scores for this run
├── manifest.json                     dataset/model/method/seed/hashes/adapter/timestamps
├── config_snapshot.yaml, logs/, errors.jsonl
└── adapter/ + checkpoints/           (method C only)
```

The ZIP to send back is created automatically in the repository root:

```text
professor_results_dashboard_v4.zip
```

It holds predictions, metrics, manifests, configs, logs, training metadata,
tables and figures — no model weights, no caches, no credentials.

Rebuild the figures or the ZIP at any time, without a GPU:

```bash
PY="$(./scripts/lib/venv_python.sh)"
"$PY" experiments/scripts/make_figures.py --dataset dashboard_v4
"$PY" experiments/scripts/package_professor_results.py --dataset dashboard_v4
```

## 6. Storage on a small disk

By default, model weights go to `~/.cache/huggingface` and all run artifacts
stay inside the repository. To put them on a bigger volume, edit **one** file:

```bash
cp paths.env.example paths.env
```

```bash
BIG=/mnt/big                    # usually the only line to change
VENV_PATH=$BIG/venv             # the Python environment
HF_HOME=$BIG/hf                 # downloaded model weights (~65 GB)
CACHE_PATH=$BIG/hf-cache
OUTPUT_DATA_PATH=$BIG/runs      # predictions, metrics, logs
OUTPUT_MODEL_PATH=$BIG/adapters # adapters + checkpoints (large)
RESULTS_PATH=$BIG/results       # tables and figures
```

Every script reads `paths.env` automatically. Nothing else changes.

To keep fewer training checkpoints, add to the same file:

```bash
EXTRA_OVERRIDES_STR="training.sft.save_total_limit=1 training.sft.save_steps=200"
```

## 7. Several GPUs, or a cluster

```bash
./run_professor.sh --gpus 0                        # use GPU 0 only
./run_professor.sh --gpus 0,1 --model qwen3_14b    # split the 14B model over two GPUs
```

On SLURM: one array task per (model, seed), one GPU each, running in parallel,
followed automatically by an aggregation + packaging job.

```bash
./scripts/submit_slurm.sh                          # 12 tasks: 4 models x 3 seeds
./scripts/submit_slurm.sh --dry-run                # show the job script, submit nothing
./scripts/submit_slurm.sh --model qwen3_14b --gpus 2 --time 48:00:00 --partition gpu
```

Options: `--partition --gpus --time --mem --cpus --account --max-parallel`.
Job scripts and logs land in `experiments/slurm/jobs/`.

## 8. If something goes wrong

| Message | What to do |
|---|---|
| `Permission denied` on `./run_professor.sh` | `bash run_professor.sh` |
| `HF_TOKEN is not set, and llama3_1_8b is gated` | `export HF_TOKEN="hf_..."`, or run the public models only: `--model qwen3_8b` |
| `PyTorch cannot see a CUDA GPU` | check `nvidia-smi` and the driver version; this must be a GPU machine |
| `The installed torch is a CPU-only build` | `"$(./scripts/lib/venv_python.sh)" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` |
| `No Python 3.11-3.13 found` | install one, or `./scripts/bootstrap_remote.sh --python /path/to/python3.12` |
| `rag knowledge base ... not built` | `"$(./scripts/lib/venv_python.sh)" experiments/scripts/build_kb.py` |
| Disk full | set up `paths.env` (§6), then re-run |
| Session died mid-run | run the same command again |
| `UnicodeDecodeError` while importing `trl` | only happens outside the launchers; prefix the command with `PYTHONUTF8=1` |
| Anything else | re-run with `--dry-run` first; it prints the exact plan without executing anything |

Environment check on its own (fast, downloads nothing, ~10 s):

```bash
"$(./scripts/lib/venv_python.sh)" experiments/scripts/check_experiment_release.py \
    --profile final --all-models --dataset dashboard_v4
```

Its last line must be `PASS`.

---

## Appendix A — what `run_professor.sh` does internally

It only orchestrates; it contains no training or inference code of its own.

1. `scripts/bootstrap_remote.sh` — installs Poetry and `./.venv` from
   `poetry.lock`, then verifies that PyTorch sees the GPU. **Skipped** when the
   environment already imports the stack.
2. `experiments/scripts/build_kb.py` — RAG knowledge base. **Skipped** when the
   existing one still matches its manifest.
3. `experiments/scripts/check_experiment_release.py` — preflight: packages,
   PyTorch/CUDA, dataset files and SHA-256, split counts, model configs, Hugging
   Face access, output permissions, adapter-path logic. Downloads no weights.
4. `run_experiment.sh` → `run_final_matrix.py` → `train.py` / `run_experiment.py`
   → `aggregate_results.py` — the matrix itself, with resume and the C→D adapter
   dependency.
5. `experiments/scripts/make_figures.py` — figures from the aggregated tables.
6. `experiments/scripts/package_professor_results.py` — the ZIP.

Each step can also be run on its own; every one supports `--help`.

## Appendix B — why all four methods use three seeds

Generation is stochastic for every method: `src/config/method/*.yaml` set
`do_sample: true`, `temperature: 0.1`, `top_p: 0.9`, and the seed is applied per
run. A and B therefore produce genuinely different outputs per seed, so reusing
one A/B run across three seeds would misreport run-to-run variability rather
than save duplicate work. C and D additionally carry training randomness.
Retrieval itself (TF-IDF over a fixed knowledge base) is deterministic.

## Appendix C — quick pipeline test (optional, small model)

Checks the whole chain with Qwen2.5-0.5B, 2 items and 1 training step. It is a
plumbing test, not a result:

```bash
./run_experiment.sh --profile smoke --dataset dashboard_v4 \
  --model qwen2_5_0_5b --all-methods --seed 42 --with-dependencies --resume
```

Expected last line: `PASS_QWEN_0_5B_END_TO_END_SMOKE`.

## Appendix D — advanced launcher

`./run_experiment.sh` is the lower-level entry point behind `run_professor.sh`.
It exposes the full matrix selection plus Hydra overrides, separate output roots,
local base-model directories and inference-only mode:

```bash
./run_experiment.sh --help
```

Example — one method, one seed, custom locations:

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_8b --method C --seed 42 --resume \
  --cache-path /mnt/hf-cache --output-model-path /mnt/adapters
```

`dashboard_v3` remains fully runnable (`--dataset dashboard_v3`); its results go
to their own directories and are never mixed with `dashboard_v4`.
