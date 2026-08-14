# Running the final experiments on a GPU server

This is the single authoritative guide for the remote run. One launcher does
everything: `./run_experiment.sh`. No Hydra knowledge is needed, and adapters
never have to be located by hand.

Final matrix: 4 models x 4 methods (A/B/C/D) x 3 seeds (42/43/44) on the frozen
dataset `dashboard_v4`.

| Method | Meaning |
| --- | --- |
| A | Prompt-only baseline |
| B | RAG (retrieval-augmented) |
| C | QLoRA fine-tuning (trains one adapter per model+seed) |
| D | QLoRA fine-tuning + RAG (reuses the C adapter of the same dataset+model+seed) |

| Model key | Hugging Face ID |
| --- | --- |
| `qwen3_1_7b` | `Qwen/Qwen3-1.7B` |
| `qwen3_8b` | `Qwen/Qwen3-8B` |
| `qwen3_14b` | `Qwen/Qwen3-14B` |
| `llama3_1_8b` | `meta-llama/Llama-3.1-8B-Instruct` (gated: needs an approved account) |

---

## 1. Clone

```bash
git clone https://github.com/raminsaeedi/FineTuningMaster.git
cd FineTuningMaster
```

## 2. Create the Python environment

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install dependencies

Install a CUDA-matched PyTorch build first (adjust `cu124` to the server's CUDA
version), then the rest:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install -r requirements-train.txt
```

```bash
pip install -e .
```

## 4. Set HF_TOKEN

The Qwen3 profiles are public. `meta-llama/Llama-3.1-8B-Instruct` is gated and
needs an approved Hugging Face account plus a token. Set it in the environment
only — never in YAML, manifests or archives.

```bash
export HF_TOKEN="hf_your_token_here"
```

## 5. Build the RAG knowledge base (required for methods B and D)

The guideline documents are in Git; the chunk index is generated locally.

```bash
python experiments/scripts/build_kb.py
```

## 6. Preflight (no model weights are downloaded)

```bash
python experiments/scripts/check_experiment_release.py --profile final --all-models --dataset dashboard_v4
```

It verifies: Python version, core + training packages, CUDA/PyTorch/bitsandbytes,
frozen dataset files, SHA-256 of every dataset file, split counts, robustness
splits, RAG knowledge base, all model configs, Hugging Face access and token,
output-directory writability, and the D-to-C adapter path logic. Continue only
when the last line is `PASS`.

A command-level preview of everything that would run:

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 --all-models --all-methods --seeds 42 43 44 --dry-run
```

## 7. Run one model, one seed (recommended unit of work)

Every command below is safe to interrupt and re-run.

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_1_7b --all-methods --seed 42 \
  --with-dependencies --resume
```

## 8. Run one complete model (all three seeds)

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_1_7b --all-methods \
  --seeds 42 43 44 --with-dependencies --resume
```

## 9. Run all four models

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --all-models --all-methods \
  --seeds 42 43 44 --with-dependencies --resume
```

## 10. Resume after an interruption

Re-run **exactly the same command** with `--resume` (all commands here already
include it). The launcher then:

- skips runs that are already complete **and** compatible (same dataset, model,
  method, seed, dataset hashes, training config, inference config, RAG KB hash);
- reuses completed predictions and metrics;
- resumes a compatible C training checkpoint, and starts fresh if the existing
  checkpoint belongs to a different config;
- reuses the correct completed C adapter for D instead of retraining;
- never accepts a run, cache or adapter from another dataset, model or seed;
- leaves already-finished seeds untouched when another seed fails.

A failed stage is reported in the summary table at the end; other seeds and
models keep their results.

## 11. Where the results are

Per-run artifacts:

```text
experiments/outputs/final/dashboard_v4/<model>/<A|B|C|D>/seed_<seed>/
```

For example `experiments/outputs/final/dashboard_v4/qwen3_8b/C/seed_42/`.

Each run directory contains:

```text
manifest.json              dataset, model, method, seed, base model ID, adapter path,
                           source C run (for D), training/inference config, config hash,
                           dataset hashes, KB hash, cache identity, status, timestamps
config_snapshot.yaml       fully resolved config
config_hash.txt / git_hash.txt / env.txt
cache_identity.json / dataset_hashes.json / kb_hashes.json
predictions.jsonl
predictions_paraphrased.jsonl     (when the robustness split is enabled)
predictions_missing_info.jsonl    (when the robustness split is enabled)
metrics_auto.json
errors.jsonl
logs/
```

Method C additionally contains:

```text
adapter/                   LoRA adapter + tokenizer
adapter/training_metadata.json
checkpoints/               trainer checkpoints (stay on the GPU machine)
resume_metadata.json
```

Aggregated results:

```text
experiments/results/final/dashboard_v4/
├── per_model/<model>/comparison_table.csv, run_index.json
├── cross_model/comparison_by_run.csv
├── cross_model/model_method_summary.csv
├── cross_model/model_method_summary.md
├── statistics/
├── final_run_index.json
├── comparison_table.csv / .md
├── multi_seed_summary.csv / .md
└── final_report.md
```

`dashboard_v3` results, if ever produced, land under
`experiments/outputs/final/dashboard_v3/` and
`experiments/results/final/dashboard_v3/`. V3 and V4 are never mixed.

## 12. Package the results and send them back

```bash
python experiments/scripts/package_professor_results.py --dataset dashboard_v4
```

This writes `professor_results_dashboard_v4.zip` plus
`professor_results_dashboard_v4_manifest.json` in the repository root. The
archive contains manifests, resolved configs, predictions, metrics, error logs,
training metadata, run logs, aggregate CSV/Markdown tables, statistics and the
run index. It excludes base model weights, the Hugging Face cache, trainer
checkpoints and anything secret-like.

---

## Copy-paste command block

### Qwen3 1.7B

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_1_7b --all-methods \
  --seeds 42 43 44 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_1_7b --all-methods --seed 42 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_1_7b --all-methods --seed 43 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_1_7b --all-methods --seed 44 --with-dependencies --resume
```

### Qwen3 8B

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_8b --all-methods \
  --seeds 42 43 44 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_8b --all-methods --seed 42 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_8b --all-methods --seed 43 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_8b --all-methods --seed 44 --with-dependencies --resume
```

### Qwen3 14B

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_14b --all-methods \
  --seeds 42 43 44 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_14b --all-methods --seed 42 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_14b --all-methods --seed 43 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_14b --all-methods --seed 44 --with-dependencies --resume
```

### Llama 3.1 8B Instruct (needs HF_TOKEN)

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model llama3_1_8b --all-methods \
  --seeds 42 43 44 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model llama3_1_8b --all-methods --seed 42 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model llama3_1_8b --all-methods --seed 43 --with-dependencies --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model llama3_1_8b --all-methods --seed 44 --with-dependencies --resume
```

### Entire four-model matrix

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --all-models --all-methods \
  --seeds 42 43 44 --with-dependencies --resume
```

### Single scenario

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_8b --method C --seed 42 --resume
```

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_8b --method D --seed 42 --with-dependencies --resume
```

With `--with-dependencies`, method D trains the matching C adapter first when it
does not exist yet. Without it, D aborts rather than using a foreign adapter.

---

## Seed handling for A/B versus C/D — and why

All four methods are executed with seeds 42, 43 and 44.

The reason is in the generation config (`src/config/method/*.yaml`): every
method, including A and B, generates with `do_sample: true`, `temperature: 0.1`,
`top_p: 0.9`. Sampling is stochastic, and the seed is set per run
(`src/utils/seed.py`), so A and B produce genuinely different outputs per seed.
Reusing one A/B run across three seeds would therefore misreport run-to-run
variability, not save an identical computation. Retrieval (TF-IDF over a fixed
KB) is deterministic, but it is only one part of the pipeline.

C and D additionally carry training randomness, so they need the three seeds
regardless.

If A/B were made deterministic later (`do_sample: false`), a single A/B run per
model could be referenced across seeds — that change is not part of this run,
and no result here silently reuses a run across seeds.

---

## Optional: local pipeline smoke test

Small model, two evaluation items, one training step. A plumbing check, not a
thesis result:

```bash
./run_experiment.sh --profile smoke --dataset dashboard_v4 \
  --model qwen2_5_0_5b --all-methods --seed 42 --with-dependencies --resume
```

Expected final line: `PASS_QWEN_0_5B_END_TO_END_SMOKE`.

---

## Selecting the other dataset

`dashboard_v4` is the default final thesis dataset. `dashboard_v3` remains
runnable without any source change:

```bash
./run_experiment.sh --profile final --dataset dashboard_v3 \
  --all-models --all-methods --seeds 42 43 44 --with-dependencies --resume
```

```bash
python experiments/scripts/package_professor_results.py --dataset dashboard_v3
```

---

## Custom storage locations (optional)

If model caches, adapters or results must live on a different volume:

```bash
./run_experiment.sh --profile final --dataset dashboard_v4 \
  --model qwen3_14b --all-methods --seeds 42 43 44 \
  --with-dependencies --resume \
  --cache-path /mnt/hf-cache \
  --output-data-path /mnt/thesis/runs \
  --output-model-path /mnt/thesis/adapters \
  --results-path /mnt/thesis/results
```

These paths are passed through to training and inference, not merely printed.

## Full launcher help

```bash
./run_experiment.sh --help
```
