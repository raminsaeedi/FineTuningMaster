# Professor GPU run

The repository contains a smoke profile and a four-model final profile. Run all
commands from the project root. The final runner does not modify Git state.

## Install

```bash
git clone <REPOSITORY_URL>
cd master-thesis-finetuning
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
python experiments/scripts/build_kb.py
```

PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Set Hugging Face authentication in the environment only. Do not put the token
in YAML, command history, manifests, or result archives.

```bash
export HF_TOKEN="hf_your_token_here"
```

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

The Qwen profiles are public. The Llama 3.1 profile is gated and needs an
approved account plus `HF_TOKEN`.

## Preflight

Check all four final model profiles, frozen data, the knowledge base, CUDA,
training packages, and repository metadata without downloading model weights:

```bash
python experiments/scripts/check_experiment_release.py --profile final --all-models
```

Continue only when the final line is `PASS`. A missing gated-model token is a
deliberate `FAIL`; no token value is printed.

## Optional local smoke

This uses only Qwen2.5-0.5B-Instruct, two validation items, two training items,
one optimizer step, and short generation. It is a pipeline check, not a thesis
result.

```bash
python experiments/scripts/run_final_matrix.py --profile smoke \
  --model qwen2_5_0_5b --all-methods --seed 42 \
  --with-dependencies --resume
```

Expected status: `PASS_QWEN_0_5B_END_TO_END_SMOKE`.

## Final matrix

Run methods A/B/C/D for Qwen3 1.7B, Qwen3 8B, Qwen3 14B, and Llama 3.1 8B
Instruct across seeds 42, 43, and 44:

```bash
python experiments/scripts/run_final_matrix.py --profile final \
  --all-models --all-methods --seeds 42 43 44 \
  --with-dependencies --resume
```

One-model example:

```bash
python experiments/scripts/run_final_matrix.py --profile final \
  --model qwen3_8b --all-methods --seeds 42 43 44 \
  --with-dependencies --resume
```

One experiment example (method D, with its same-model/same-seed C adapter):

```bash
python experiments/scripts/run_final_matrix.py --profile final \
  --model qwen3_8b --method D --seed 42 \
  --with-dependencies --resume
```

Re-run the same command after interruption. Completed compatible runs and
adapters are reused; compatible checkpoints are resumed. Method D never falls
back to another model or seed.

## Results and handoff

Run artifacts are isolated at:

```text
experiments/outputs/final/dashboard_v3/<model>/<A|B|C|D>/seed_<seed>/
```

Each run records resolved config, config hash, dataset hashes, model ID and
revision, method, seed, training/inference settings, KB hashes, adapter source
identity, cache identity, timestamps, hardware, package versions, predictions,
metrics, and logs. Adapter weights/checkpoints remain on the GPU machine.

Aggregated results are written to:

```text
experiments/results/final/dashboard_v3/
├── per_model/<model>/
├── cross_model/
└── statistics/
```

Package the safe evidence after the matrix completes:

```bash
python experiments/scripts/package_professor_results.py
```

This creates `professor_results.zip` and `professor_results_manifest.json`.
The package includes predictions, metrics, manifests, resolved configs, logs,
training metadata, and aggregate tables. It excludes model weights,
checkpoints, Hugging Face caches, credentials, and other secret-like files.
