# Running the experiments

Clone, install, verify, run, send results back. No project knowledge required.

---

## 1. Requirements

- Linux or Windows
- Python 3.10 or newer (developed on 3.12)
- NVIDIA GPU with CUDA — required for the fine-tuning runs (methods C and D)
- ~15 GB free disk (dataset ~10 MB, base model + adapters + outputs make up the rest)
- No Hugging Face login needed for the default model (`Qwen/Qwen2.5-0.5B-Instruct` is public).
  A gated model would need `hf auth login` first.

---

## 2. Clone and install

```bash
git clone <REPOSITORY_URL>
cd master-thesis-finetuning
python -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the package plus the training stack:

```bash
pip install -e ".[train]"
```

Build the retrieval knowledge base (needed by methods B and D):

```bash
python experiments/scripts/build_kb.py
```

---

## 3. Verify the package

```bash
python experiments/scripts/check_experiment_release.py --require-cuda --require-training
```

Expected: `PASS` on the last line.

Any `FAIL` line states exactly what is missing. Do not continue until this passes.

---

## 4. Optional smoke test

Runs all four methods end-to-end on a tiny slice, in a few minutes, to prove the
pipeline works before the long run. It uses a small model and two training steps,
so the output quality is meaningless — only completion matters.

```bash
python experiments/scripts/run_smoke.py
```

Expected: `PASS_QWEN_0_5B_END_TO_END_SMOKE`.

Smoke artifacts land in `outputs/smoke/` and are not thesis results.

---

## 5. Run the final experiments

One command runs methods A, B, C and D across seeds 42, 43 and 44:

```bash
python experiments/scripts/run_final_matrix.py
```

It trains the fine-tuned adapter for each seed and then runs all four methods.
Method D automatically uses the method C adapter of the **same seed** — no paths
to look up by hand. If a matching adapter is missing, the run stops with a clear
message instead of using the wrong one.

This is the long step: 12 runs plus 3 fine-tunes.

To use a different model, change `model:` in `src/config/matrix/final.yaml`, or:

```bash
python experiments/scripts/run_final_matrix.py --model <model_config_name>
```

Model configs live in `src/config/model/`. No source code needs editing.

Then aggregate:

```bash
python experiments/scripts/aggregate_results.py --outputs-root experiments/outputs/final
```

---

## 6. Resume after an interruption

Re-run the exact same command:

```bash
python experiments/scripts/run_final_matrix.py
```

Finished runs are skipped, trained adapters are reused, and inference resumes
per item. Nothing completed is repeated. Add `--force` only if you deliberately
want to recompute everything.

---

## 7. Results

Everything is written to:

```
experiments/outputs/final/<experiment>_<seed>/
experiments/results/
```

Per run: `predictions*.jsonl`, `errors*.jsonl`, `metrics_auto.json`, `metrics.json`,
`eval_per_item.jsonl`, `manifest.json`, `config_snapshot.yaml`, `config_hash.txt`,
`git_hash.txt`, `env.txt`, `logs/`, and for the fine-tuned runs
`adapter/training_metadata.json`.

Aggregate: comparison tables, final report and statistical results under
`experiments/results/`.

---

## 8. Send the results back

```bash
python experiments/scripts/package_results.py --outputs-root experiments/outputs/final
```

This writes `professor_results.zip` in the repository root, containing only the
files listed above. Model weights, checkpoints, caches and any secrets are
excluded, so the archive stays small.

Send that one ZIP file back.
