# Running on a GPU Server

One command to train and evaluate. The script handles data prep, training,
inference, evaluation, and packages everything into a ZIP you copy back.

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd master-thesis-finetuning

# 2. Create environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# 3. Install CUDA-matched PyTorch first (adjust cu124 to your CUDA version)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Install all dependencies
pip install -r requirements-train.txt
pip install -e .
```

---

## Validate (no GPU needed)

Confirms data prep and inference pipeline work before using GPU time:

```bash
python experiments/scripts/run_remote.py --mode smoke
```

Expected output: `Smoke passed. Environment is ready for GPU runs.`

---

## Train only (returns adapter ZIP)

```bash
python experiments/scripts/run_remote.py --mode train
```

Produces `thesis_results_<timestamp>.zip` containing the trained adapter,
config snapshot, and logs. Send this ZIP back.

---

## Full run (train + E01–E04 + aggregate + ZIP)

```bash
python experiments/scripts/run_remote.py --mode full --seeds 42 43 44
```

Produces `thesis_results_<timestamp>.zip` with:
```
outputs/                    per-run experiment folders
  E01_qwen0_5b_prompt_42/
    predictions.jsonl
    metrics_auto.json
    logs/
  E03_qwen0_5b_ft_42/
    adapter/                trained LoRA weights
    predictions.jsonl
    metrics_auto.json
    logs/
  ...
results/
  comparison_table.csv
  comparison_seeds.csv
  final_report.md
runtime_summary.json        per-stage timing
```

---

## Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | required | `smoke`, `train`, or `full` |
| `--train-experiment` | `E03_qwen0_5b_ft` | Which training config to use |
| `--experiments` | E01–E04 | Inference experiments (full mode) |
| `--seeds` | `42` | One or more seeds, e.g. `42 43 44` |
| `--output-dir` | `experiments/outputs/experiments` | Override output location |
| `--cache-dir` | none | HF model cache (e.g. `/mnt/data/hf_cache`) |
| `--max-samples` | none | Cap dataset size (e.g. `100` for a quick test) |
| `--smoke-items` | `5` | Items used in smoke inference |

---

## Useful overrides

```bash
# Use a shared model cache on HPC (avoids re-downloading)
python experiments/scripts/run_remote.py --mode full \
    --cache-dir /mnt/data/hf_cache

# Quick test with 50 items and 1 seed
python experiments/scripts/run_remote.py --mode full \
    --max-samples 50 --seeds 42

# Custom output location (e.g. mounted data disk on Azure)
python experiments/scripts/run_remote.py --mode full \
    --output-dir /mnt/data/outputs/experiments \
    --cache-dir /mnt/data/hf_cache
```

---

## Resume interrupted runs

Re-run the same command. Inference is cached per item — completed items are
skipped automatically. Training does not resume mid-epoch, but with a 0.5B
model + QLoRA it completes in one session (~15–30 min on a T4 GPU).

---

## Copy results back

```bash
# From your local machine:
scp user@server:/path/to/repo/thesis_results_<timestamp>.zip .
```
