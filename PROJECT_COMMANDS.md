# Project Command Handbook

Operational guide for current master's-thesis repository.

Commands below verified against current Python code and CLI help on 2026-08-13.
Final experiments use frozen `data/frozen/dashboard_v4/` (default; select with
`--dataset dashboard_v4`). `data/frozen/dashboard_v3/` stays selectable with
`--dataset dashboard_v3`. Do not regenerate, edit, or replace either frozen
package during thesis experiments. Remote GPU instructions: `RUN_PROFESSOR.md`.

## Which command do I need?

| I want to...                | Command                                                                                                                                                                                                                                                   | Cost                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| verify repository           | `python experiments/scripts/check_experiment_release.py --require-cuda --require-training --model qwen2_5_0_5b --output-root experiments/outputs/final`                                                                                                   | cheap, GPU check only          |
| verify frozen data          | same release-check command                                                                                                                                                                                                                                | cheap/local                    |
| test everything locally     | `python experiments/scripts/run_smoke.py`                                                                                                                                                                                                                 | cheap smoke, tiny GPU training |
| train C                     | `python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final`                                                                                                            | GPU-expensive                  |
| resume C                    | same command with `--resume` before `--override`                                                                                                                                                                                                          | GPU-expensive                  |
| run A                       | `python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final`                                                                                               | model inference                |
| run B                       | `python experiments/scripts/run_experiment.py --experiment E02_qwen0_5b_rag --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final`                                                                                                  | model inference                |
| run C                       | `python experiments/scripts/run_experiment.py --experiment E03_qwen0_5b_ft --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final`                                                                                                   | model inference                |
| run D                       | `python experiments/scripts/run_experiment.py --experiment E04_qwen0_5b_ft_rag --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final method.adapter_source_experiment=E03_qwen0_5b_ft`                                              | model inference                |
| evaluate cached predictions | `python experiments/scripts/eval_auto.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final`                                                                                                    | cheap/local                    |
| aggregate results           | `python experiments/scripts/aggregate_results.py --outputs-root experiments/outputs/final --out-dir experiments/results`                                                                                                                                  | cheap/local                    |
| statistics                  | `python experiments/scripts/eval_stats.py --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag --out-dir experiments/results/stats/seed_42 --override output_root=experiments/outputs/final seed=42 model=qwen2_5_0_5b` | cheap/local                    |
| human evaluation            | `python experiments/scripts/build_human_eval.py --dataset dashboard_v4 --model qwen3_8b --seed 42 --n-items 40 --n-raters 6 --ratings-per-output 3`                                                                                         | manual                         |
| run full tests              | `pytest -q`                                                                                                                                                                                                                                               | cheap/moderate CPU             |

## Cost labels

- **LOCAL / CHEAP:** no model generation; seconds to minutes on CPU.
- **SMOKE:** tiny Qwen-0.5B pipeline check; tiny C training slice.
- **MODEL INFERENCE:** loads a Hugging Face model; GPU preferred, CPU possible but slower.
- **GPU-EXPENSIVE:** fine-tuning or full model inference.
- **MANUAL:** human rating work.
- **ONE-TIME / CONDITIONAL:** run only when a required resource is missing or a new environment is prepared.

## A. Environment setup

### Requirements

- Python 3.10 or newer. Repository was developed with Python 3.12.
- Base stack supports inference, evaluation, statistics, and aggregation.
- CUDA GPU required for QLoRA training C.
- Default model is public `Qwen/Qwen2.5-0.5B-Instruct`; Hugging Face login is not needed.
- A gated/private model requires external `hf auth login`. Never put tokens in this file or repository.
- CPU-only machines can run validation, cached evaluation, aggregation, statistics, and human-rating tools. They are not the recommended environment for C training.

### Environment (Poetry is authoritative)

Dependencies live in `pyproject.toml` and are pinned in `poetry.lock`. Python
3.11-3.13 is required. One command builds `./.venv` with the exact locked
versions, including the QLoRA training stack:

```bash
./scripts/bootstrap_remote.sh
```

Variants: `--no-train` (inference/evaluation only), `--with-dev` (pytest),
`--cpu-ok` (no GPU), `--python /path/to/python3.12`.

Run project commands inside that environment:

```bash
./.venv/bin/python experiments/scripts/<script>.py ...
```

`./run_experiment.sh` resolves `./.venv` (then `poetry run`) automatically.

`requirements-train.txt` is a GENERATED pip export of the lockfile, kept only
for machines where Poetry cannot be installed. Never edit it by hand:

```bash
poetry export -f requirements.txt --extras train --without-hashes -o requirements-train.txt
```

### Optional dependencies

`bootstrap_remote.sh` installs Poetry into `./.poetry` when it is not already on
PATH; in that case call it as `./.poetry/bin/poetry`.

```bash
poetry install --extras human        # Streamlit human-rating app
poetry install --with dev            # pytest
poetry install --extras constrained  # optional constrained decoding
poetry install --extras rag-dense    # optional dense retriever
poetry install --extras galore       # optional GaLore training
poetry install --extras enrichment   # dataset enrichment (OpenAI-compatible SDK)
```

G-Eval is optional and uses an OpenAI-compatible endpoint. It requires an
externally supplied `OPENAI_API_KEY`; default A/B/C/D runs do not require it.

## B. Repository / release check

Run after cloning and installing, before any final run:

```powershell
python experiments/scripts/check_experiment_release.py --require-cuda --require-training --model qwen2_5_0_5b --output-root experiments/outputs/final
```

This cheap check verifies Python/packages, CUDA, frozen dataset files and
hashes, split counts, RAG guidelines/chunks/manifest, experiment configs,
selected model config, output writability, and absence of staging/processed
data dependencies in final configs.

Expected final line:

```text
PASS
```

On a CPU-only machine, omit `--require-cuda --require-training` to receive
warnings instead of a hard failure. That does not certify GPU readiness.

## C. Frozen dataset verification

Authoritative package:

```text
data/frozen/dashboard_v3/
```

Current frozen counts:

- Train: 1,281
- Validation: 264
- Test: 274
- Separate human-evaluation file: 40 items

Verify hashes and counts with the release check:

```powershell
python experiments/scripts/check_experiment_release.py --model qwen2_5_0_5b --output-root experiments/outputs/final
```

The frozen manifest status is
`PASS_DASHBOARD_V3_FROZEN_READY_FOR_EXPERIMENTS`. Frozen JSONL files,
human-evaluation CSV, schema, reports, and `hashes.json` are thesis inputs.
Do not edit or regenerate them.

`validate_frozen_dataset.py` exists for dataset validation, but it writes
`validation_report.md` and can write hashes. Do not use it in normal final
workflow because final workflow must leave frozen package untouched.

## D. RAG knowledge base

Sources:

```text
data/knowledge_base/guidelines/
```

Built resources:

```text
data/knowledge_base/chunks.jsonl
data/knowledge_base/kb_manifest.json
```

Current state: ready. Release check verified three guideline documents and 41
chunks.

Verify as part of release check:

```powershell
python experiments/scripts/check_experiment_release.py --model qwen2_5_0_5b --output-root experiments/outputs/final
```

Rebuild only if KB files are missing or guidelines deliberately changed:

```powershell
python experiments/scripts/build_kb.py
```

`build_kb.py` is local/cheap and writes only KB output and manifest. It does
not touch frozen data. Method B and Method D require RAG. Method A and Method
C do not use RAG.

Actual default retriever: TF-IDF, `top_k: 3`, reading
`data/knowledge_base/chunks.jsonl`.

## E. Local Qwen-0.5B smoke test

Run complete cheap smoke check:

```powershell
python experiments/scripts/run_smoke.py
```

Defaults: three validation items, eight frozen training items, two optimizer
steps, seed 42. Output:

```text
outputs/smoke/dashboard_v3_qwen0_5b/
```

Expected status:

```text
PASS_QWEN_0_5B_END_TO_END_SMOKE
```

Smoke checks dataset loading from frozen train/validation data, model loading,
tiny QLoRA training, adapter save/load, Methods A/B/C/D, TF-IDF RAG, parsing,
schema metrics, automatic metrics, and run artifacts. It does not use frozen
test split, 40-item human-evaluation file, or robustness variants.

This is a pipeline check, not thesis evidence. It uses GPU-backed model code
for C. Do not replace it with full 1,281-record CPU training.

Use `--keep` to preserve existing smoke output; default execution starts clean
by removing only its smoke output directory.

## F. Fine-tuning / training C

Train one model and one seed:

```powershell
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

Meaning:

- `--experiment E03_qwen0_5b_ft`: selects Method C experiment config.
- `model=qwen2_5_0_5b`: selects a file in `src/config/model/`.
- `seed=42`: selects one independent random seed.
- `output_root=experiments/outputs/final`: selects final-results root.

Adapter output:

```text
experiments/outputs/final/E03_qwen0_5b_ft_42/adapter/
```

Change only seed for other independent runs:

```powershell
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --override model=qwen2_5_0_5b seed=43 output_root=experiments/outputs/final
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --override model=qwen2_5_0_5b seed=44 output_root=experiments/outputs/final
```

Each command runs one training job. It does not start other methods or seeds.
Training needs CUDA and `[train]` extra.

## G. Resume interrupted training

Normal training does not silently resume. Without resume flag, it does not
select an old checkpoint.

### Automatic latest-checkpoint resume

```powershell
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --resume --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

`--resume` searches expected run's `checkpoints/`, selects newest valid numeric
`checkpoint-N`, and forwards it to native trainer resume API. Optimizer,
scheduler, RNG, and trainer state must all be present for a checkpoint to count
as valid; an incomplete one is skipped with a logged reason and the previous
complete checkpoint is used. Full validity rules, the OOM behaviour and
`--no-resume`: section L2.

### Explicit checkpoint resume

List available checkpoints:

```powershell
Get-ChildItem experiments/outputs/final/E03_qwen0_5b_ft_42/checkpoints -Directory -Filter checkpoint-* | Sort-Object Name
```

Resume selected checkpoint. Replace `<STEP>` with existing checkpoint number:

```powershell
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --resume-from experiments/outputs/final/E03_qwen0_5b_ft_42/checkpoints/checkpoint-<STEP> --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

Resume rejects incompatible model, seed, dataset version/hash, or training
configuration metadata. Legacy checkpoints without sufficient metadata are
rejected for automatic discovery; explicit path inside expected run is
required. Missing or invalid checkpoints fail instead of starting from
scratch.

Manifest records:

```text
resumed
resume_checkpoint
initial_global_step
final_global_step
resume_timestamp
```

## H. Method A — prompt-only

```powershell
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

- Training required: no
- Adapter required: no
- RAG required: no
- Output: `experiments/outputs/final/E01_qwen0_5b_prompt_42/`

## I. Method B — RAG

```powershell
python experiments/scripts/run_experiment.py --experiment E02_qwen0_5b_rag --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

- Training required: no
- Adapter required: no
- RAG required: yes
- Output: `experiments/outputs/final/E02_qwen0_5b_rag_42/`
- Retriever: TF-IDF over `data/knowledge_base/chunks.jsonl`, `top_k=3`

## J. Method C — fine-tuned

Train C first using section F, then run inference/evaluation:

```powershell
python experiments/scripts/run_experiment.py --experiment E03_qwen0_5b_ft --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

- Training required beforehand: yes
- Adapter required: yes
- RAG required: no
- Output: `experiments/outputs/final/E03_qwen0_5b_ft_42/`

Method C resolves its adapter to selected run's `adapter/` directory and
validates adapter metadata before loading it.

## K. Method D — fine-tuned + RAG

```powershell
python experiments/scripts/run_experiment.py --experiment E04_qwen0_5b_ft_rag --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final method.adapter_source_experiment=E03_qwen0_5b_ft
```

- Training required again: no
- C adapter required: yes
- RAG required: yes
- Output: `experiments/outputs/final/E04_qwen0_5b_ft_rag_42/`

D must reuse C adapter for same model, dataset, and seed. Current resolver
constructs:

```text
<output_root>/E03_qwen0_5b_ft_<seed>/adapter/
```

Method D validates adapter existence and recorded base model, seed, and dataset
version before loading. Seed 43 resolves seed-43 C adapter; it never silently
falls back to seed 42.

## L. Inference only

Run inference without automatic evaluation:

```powershell
python experiments/scripts/infer.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

Output is `predictions.jsonl` plus optional
`predictions_paraphrased.jsonl` and `predictions_missing_info.jsonl` under the
run directory. Use this when generation should be separated from metric
computation. Inference loads a model and is not metrics-only work.

Inference is cached per item. Re-running same command fills missing items
instead of discarding completed predictions.

### L1. Batched inference (optional, off by default)

Applies to inference for A, B, C (after the adapter is loaded) and D.

**Default safe mode.** Inference generates one item per `generate` call
(`batch_size = 1`). This is the regime every existing result was produced under.
No command changes, no config changes, and the composed config — and therefore
`config_hash` — is byte-identical to before the feature existed. Nothing needs
to be set to stay here.

**Optional batch mode.** Batching is configured under the method's existing
`inference` group and requires two keys, because one of them is an
acknowledgement:

```powershell
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final +method.inference.batch_size=4 +method.inference.allow_nonequivalent_batching=true
```

The runner and the shell launcher expose the same thing as flags:
`--inference-batch-size 4 --allow-nonequivalent-batching`. Setting the batch
size without the acknowledgement is a hard error, not a silent fallback.

**Scientific limitations.** Batched output is *not* item-identical to sequential
output, for two independent reasons:

- **Sampling.** The methods run `do_sample: true`. Every row of a batch draws
  from one shared RNG stream inside `generate`, so an item's sample depends on
  which items sit next to it. Measured on 20 items of
  `data/frozen/dashboard_v4/test.jsonl` with Qwen2.5-0.5B-Instruct:
  **6 of 20 items produced different text** at `batch_size=4`. With
  `do_sample=false` the same fixture matched byte-for-byte.
- **Numerics.** Batching left-pads prompts and changes every matmul shape, so
  even greedy decoding is only usually — never provably — identical.

Control measurement on the same fixture: two *sequential* passes over those 20
items also differed on 6 of 20 items, because inference does not set a torch
seed (`set_seeds` is called by `train.py` only). Under `do_sample: true` no
per-item equality claim can be made for any two runs, batched or not — which is
why the equality check above is only meaningful with `--greedy`, and why
batching stays opt-in regardless of what it measures.

Two further consequences: `latency_ms` in batch mode is the batch wall time
divided by the batch size (amortized, not measured per item), so the latency
metric is not comparable with sequential runs; and enabling batching changes
`config_hash`, so batched predictions can never be appended to a sequential
result file — the cache guard refuses the mix. The resolved regime is recorded
in `manifest.json` under `inference_batching`.

**Smoke benchmark** (small, local, no run directory is touched):

```powershell
python experiments/scripts/benchmark_batch_inference.py --experiment E01_qwen0_5b_prompt --model qwen2_5_0_5b --n-items 8 --batch-size 4 --max-new-tokens 32 --greedy
```

**Validation before enabling.** Run the 20-item fixture with the *real*
generation settings and require exact equality (exit code 0; exit code 2 means
outputs differ):

```powershell
python experiments/scripts/benchmark_batch_inference.py --experiment E01_qwen0_5b_prompt --model qwen2_5_0_5b --n-items 20 --batch-size 4
```

Only if that reports `EXACT MATCH` may the feature be enabled for a real run:

```powershell
./run_experiment.sh --profile final --model qwen3_8b --all-methods --seeds 42 43 44 --with-dependencies --resume --inference-batch-size 4 --allow-nonequivalent-batching
```

With `do_sample: true` that check does not pass, so batching is a development
and throughput-experiment tool here, not a final-thesis-run setting.

### L2. Crash-safe resume

Applies to inference for A, B, C, D and to method C training. An interruption
may cost the remaining work; it must never cost the finished work.

**Guarantees**

- Completed JSONL records are never rewritten, reordered, or duplicated. A
  resumed run appends only the missing item ids, in dataset order.
- An interrupted append leaves a truncated final line. That fragment — and only
  that fragment — is removed before the resume set is computed, because the next
  append would otherwise fuse onto it and destroy the following record too. The
  removed bytes are kept in a `predictions.jsonl.partial-<utc>.bak` sibling.
- Robustness variants resume independently: `predictions_paraphrased.jsonl` and
  `predictions_missing_info.jsonl` each track their own completed ids.
- `inference_status.json` records `running | interrupted | failed | completed`,
  the completed item ids per variant, the config hash, the source-code hash,
  timestamps, every repair, and every retry. It is written atomically
  (temporary file plus `os.replace`) at job boundaries — not per generated item,
  which would add an fsync to every generation for a file that is provenance,
  not data. `predictions*.jsonl` stays the authority for what is done.
- A regenerated item is recorded as a retry with
  `"bitwise_equality_verified": false`. Under `do_sample: true` it is a fresh
  sample, not a reproduction of the lost one (see L1).
- `--resume` refuses artifacts from a different run. Gated fields: model
  identity and revision, dataset version and file hashes, knowledge-base hash,
  seed, method, config hash, and source-code hash. Runs created before this
  feature carry no status file and are therefore never gated by it.
- `--no-resume` always starts fresh: previous `predictions*.jsonl`,
  `errors*.jsonl` and `inference_status.json` are **moved** to
  `<run>/_stale_cache/fresh_<utc>/`, never deleted, and training starts at
  step 0.

**Checkpoint compatibility rules (method C)**

A checkpoint is resumable only when all of these hold; anything else is skipped
with a logged reason and the previous valid checkpoint is used instead:

| Requirement | Why |
| --- | --- |
| `trainer_state.json` with a numeric `global_step` | otherwise the run restarts at 0 |
| adapter or model weights | nothing to resume from |
| `adapter_config.json` / `config.json` | the checkpoint must describe itself |
| `optimizer.pt` | Adam moments; losing them changes the trajectory |
| `scheduler.pt` | otherwise the LR schedule restarts |
| `rng_state.pth` (or `rng_state_*.pth`) | otherwise the data order restarts |
| `checkpoint_complete.json` parses, when present | written last and atomically, so a partially written checkpoint never has a valid one |
| `resume_metadata.json` matches | experiment, model, seed, dataset version, dataset hash, training config hash |

Checkpoint frequency stays at the configured default (`save_strategy: epoch`).
Make it denser for a long or preemptible job through the normal config system:

```bash
--override training.sft.save_strategy=steps training.sft.save_steps=25
```

**Limitations after CUDA out of memory**

Nothing is reconfigured automatically. Sequence length, batch size and precision
stay exactly as configured, because silently shrinking them would produce a run
whose results no longer match its own recorded configuration. On OOM the trainer
keeps the last valid checkpoint, writes `training_status.json` with
`failure_reason: cuda_out_of_memory` and the last valid `global_step`, and
prints the exact resume command. If the run genuinely does not fit, change the
configuration deliberately and start a **new** run — a different batch size is a
different experiment, not a continuation of this one.

**Exact smoke-test command** (2 items per method, no training, temporary output
root — nothing under `experiments/outputs/final/` is touched):

```bash
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/_resume_smoke experiment_id=smoke_A data.eval_max_samples=2 method.generate.max_new_tokens=24 method.generate.do_sample=false data.paraphrased_file=null data.missing_info_file=null
```

**Exact resume command** — re-run the identical command. Resume is the default;
finished items are skipped and only the missing ones are generated:

```bash
./run_experiment.sh --profile final --model qwen3_8b --all-methods --seeds 42 43 44 --with-dependencies --resume
```

For one interrupted training run, the trainer prints this itself on failure:

```bash
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --resume --override model=qwen3_8b seed=42 output_root=experiments/outputs/final
```

**Measured overhead** (Windows, CPU, 274-record / 703 KB predictions file):

| Operation | Cost | Frequency |
| --- | --- | --- |
| truncated-line scan | 0.75 ms median | once per variant file |
| status update (read ids + atomic write) | 7.4 ms median | once per variant file |
| source-code hash | 40.5 ms | once per process |
| checkpoint completion marker | 4.2 ms median, 112 bytes | once per checkpoint |
| checkpoint validation | 1.4 ms median | once per resume, per candidate |

End to end over 274 items the difference is below the noise floor: 29.4 ms
baseline versus 28.4 ms with crash-safe resume, against a baseline spread of
18–38 ms. Against real generation (~3.4 s per item on this CPU) the total added
cost of roughly 80 ms per experiment is about 0.01 %. No GPU memory is added and
no CUDA synchronization is introduced: none of the new modules import torch.

## M. Evaluation only / re-evaluation

Evaluate existing predictions without model loading or inference:

```powershell
python experiments/scripts/eval_auto.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

This reads cached predictions, reparses current `raw_text`, joins frozen
references, computes configured metrics and available robustness metrics, and
writes:

```text
metrics_auto.json
metrics.json
eval_per_item.jsonl
```

Use after parser or metric changes. CPU only; no model inference. Keep
overrides identical to original run so command points to same run and test
split.

## N. Robustness evaluation

Current external robustness files:

```text
data/eval/robustness_v3/test_paraphrased.jsonl
data/eval/robustness_v3/test_missing_info.jsonl
data/eval/robustness_v3/manifest.json
```

They contain 274 records each, pair by `item_id`, and live outside
`data/frozen/`. They already exist. Manifest records frozen test source hash
and read-only source use.

If missing, rebuild with deterministic offline command:

```powershell
python experiments/scripts/build_perturbations_v3.py
```

This reads `data/frozen/dashboard_v3/test.jsonl` strictly read-only and writes
only `data/eval/robustness_v3/`. No LLM, network, or randomness.

`run_experiment.py` and `infer.py` automatically generate original plus
available paraphrased and missing-information predictions because
`dashboard_v3.yaml` points to these files. `eval_auto.py` computes robustness
metrics from cached variant predictions. No separate robustness evaluator is
required; robustness is wired into normal inference/evaluation path.

## O. Run a different seed

Example using Method A. Seeds run independently and need not be consecutive:

```powershell
# seed 42
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final

# seed 43
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=43 output_root=experiments/outputs/final

# seed 44
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=44 output_root=experiments/outputs/final
```

Apply same seed replacement to B, C, and D. Train matching C seed before C
inference. Keep `method.adapter_source_experiment=E03_qwen0_5b_ft` for D.

## P. Run a different model

Model configs live in:

```text
src/config/model/
```

### AVAILABLE NOW

| Config name    | Hugging Face model           |
| -------------- | ---------------------------- |
| `qwen2_5_0_5b` | `Qwen/Qwen2.5-0.5B-Instruct` |

Change model through Hydra override:

```powershell
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

### NEEDS CONFIG BEFORE USE

No other model YAML exists today. A larger final model needs a new config under
`src/config/model/` before use. Do not type an unconfigured model name and
assume it works.

## Q. Aggregate experiment results

```powershell
python experiments/scripts/aggregate_results.py --outputs-root experiments/outputs/final --out-dir experiments/results
```

Cheap/local. Reads each immediate run directory containing
`metrics_auto.json`; does not train, load a model, or perform inference.

Outputs:

```text
experiments/results/comparison_table.csv
experiments/results/comparison_table.md
experiments/results/comparison_seeds.csv
experiments/results/multi_seed_summary.csv
experiments/results/multi_seed_summary.md
experiments/results/final_report.md
```

Rows preserve experiment, method, model, seed, dataset version/hash, metrics,
run path, and config hash. Missing robustness/grounding metrics remain
`NA`/null, not zero. Multi-seed output keeps raw seed columns and does not pool
prediction rows.

## R. Statistics for one seed

After A/B/C/D completed for exactly one model, seed, output root, test split,
and protocol:

```powershell
python experiments/scripts/eval_stats.py --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag --out-dir experiments/results/stats/seed_42 --override output_root=experiments/outputs/final seed=42 model=qwen2_5_0_5b
```

The command verifies exact run manifests, model/config, seed, dataset
version/hash, protocol, test item IDs, prediction IDs, and C/D adapter wiring
before computing paired statistics.

Tests retained by current implementation:

- Cochran's Q plus exact paired McNemar post-hoc tests with Holm correction for binary Top-1 correctness.
- Friedman plus paired Wilcoxon signed-rank post-hoc tests with Holm correction for schema completeness.
- Paired effect sizes and bootstrap intervals where implemented.

If paired inputs are incompatible, command fails with:

```text
INCOMPATIBLE_PAIRED_RUNS
```

It does not silently intersect mismatched rows or substitute another seed/model.

Outputs under selected statistics directory:

```text
stats_report.json
posthoc_mcnemar.csv
posthoc_wilcoxon.csv
per_method_ci.csv
```

## S. Multi-seed aggregation

Run each seed separately for A/B/C/D. Then aggregate shared root:

```powershell
python experiments/scripts/aggregate_results.py --outputs-root experiments/outputs/final --out-dir experiments/results
```

For per-seed paired statistics, use separate output directories:

```powershell
python experiments/scripts/eval_stats.py --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag --out-dir experiments/results/stats/seed_42 --override output_root=experiments/outputs/final seed=42 model=qwen2_5_0_5b
python experiments/scripts/eval_stats.py --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag --out-dir experiments/results/stats/seed_43 --override output_root=experiments/outputs/final seed=43 model=qwen2_5_0_5b
python experiments/scripts/eval_stats.py --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag --out-dir experiments/results/stats/seed_44 --override output_root=experiments/outputs/final seed=44 model=qwen2_5_0_5b
```

`multi_seed_summary.csv` and `.md` keep raw values for seeds 42, 43, and 44,
then report descriptive `n_seeds`, mean, standard deviation, and supported
confidence intervals. Three seeds are repeated runs, not independent item
observations. Never concatenate all prediction rows across seeds and run paired
tests as if they were one sample.

## T. Human evaluation

Run after final A/B/C/D predictions exist for one fixed dataset, model and
seed. Do not mix models, seeds or legacy E01-E04 run directories.

### 1. Build blind assignment

```powershell
python experiments/scripts/build_human_eval.py --dataset dashboard_v4 --model qwen3_8b --seed 42 --n-items 40 --n-raters 6 --ratings-per-output 3
```

Outputs:

```text
experiments/results/human_eval/dashboard_v4/qwen3_8b/seed_42/items.jsonl
experiments/results/human_eval/dashboard_v4/qwen3_8b/seed_42/assignment.json
experiments/results/human_eval/dashboard_v4/qwen3_8b/seed_42/study_manifest.json
```

Current builder reads the four canonical `predictions.jsonl` files from
`experiments/outputs/final/<dataset>/<model>/<A-D>/seed_<seed>/`, checks their
compatibility, and uses the canonical item list configured by the dataset.
The old `--experiments` command is legacy and intentionally rejected.

### 2. Start rating application

```powershell
python experiments/scripts/run_human_eval.py --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8b/seed_42 --port 8501
```

Open Streamlit URL shown by app. Ratings save under the study's `ratings/`
directory. Requires `[human]` extra.

### 3. Collect ratings

Raters use assigned IDs and score outputs without seeing method identity.
Assignment covers A/B/C/D outputs and balances rating workload.

### 4. Compute reliability and statistics

```powershell
python experiments/scripts/compute_irr.py --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8b/seed_42
```

Outputs include Krippendorff's ordinal alpha, per-system means, and paired
human-score statistics:

```text
irr_alphas.json
irr_alphas.csv
system_means.json
system_means.csv
human_stats.json
```

## U. Tests

Fast focused resume/statistics regression:

```powershell
pytest -q src/tests/test_training_resume.py src/tests/test_multi_run_statistics.py
```

Run after changes to checkpoint handling, run discovery, compatibility gates,
or aggregation.

Full suite:

```powershell
pytest -q
```

Tests are local/CPU and do not launch full model training or final inference.

## V. Package results for professor / thesis

Preview package:

```powershell
python experiments/scripts/package_results.py --outputs-root experiments/outputs/final --results-dir experiments/results --out professor_results.zip --dry-run
```

Create archive after reviewing preview:

```powershell
python experiments/scripts/package_results.py --outputs-root experiments/outputs/final --results-dir experiments/results --out professor_results.zip
```

Archive:

```text
professor_results.zip
```

Preserve and send predictions, metrics, manifests, config snapshots and hashes,
training metadata, statistics, human-evaluation results, logs, and environment
metadata. Do not package base-model/Hugging Face caches, adapter weights unless
specifically required, temporary checkpoints, `.env`, API secrets, or tokens.

Package script excludes these by design and writes `PACKAGE_MANIFEST.json` with
file hashes.

## Normal final thesis workflow

Use one model, one seed, and one operation per command. Example uses currently
available Qwen-0.5B config, seed 42, and `experiments/outputs/final`.

### STEP 1 — Clone/install

```bash
./scripts/bootstrap_remote.sh
```

### STEP 2 — Preflight

```powershell
python experiments/scripts/check_experiment_release.py --require-cuda --require-training --model qwen2_5_0_5b --output-root experiments/outputs/final
```

### STEP 3 — Verify frozen dataset

Preflight verifies frozen files, SHA-256 hashes, and 1,281/264/274 counts. Do
not run a dataset builder.

```powershell
python experiments/scripts/check_experiment_release.py --model qwen2_5_0_5b --output-root experiments/outputs/final
```

### STEP 4 — Verify/build KB

KB is already ready. Preflight verifies three guidelines, 41 chunks, and
manifest. Rebuild only if missing or deliberately changed:

```powershell
python experiments/scripts/build_kb.py
```

### STEP 5 — Optional local 0.5B smoke

```powershell
python experiments/scripts/run_smoke.py
```

### FOR MODEL `qwen2_5_0_5b`, SEED 42

### STEP 6 — Train C

```powershell
python experiments/scripts/train.py --experiment E03_qwen0_5b_ft --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

### STEP 7 — Run A

```powershell
python experiments/scripts/run_experiment.py --experiment E01_qwen0_5b_prompt --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

### STEP 8 — Run B

```powershell
python experiments/scripts/run_experiment.py --experiment E02_qwen0_5b_rag --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

### STEP 9 — Run C

```powershell
python experiments/scripts/run_experiment.py --experiment E03_qwen0_5b_ft --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final
```

### STEP 10 — Run D

```powershell
python experiments/scripts/run_experiment.py --experiment E04_qwen0_5b_ft_rag --override model=qwen2_5_0_5b seed=42 output_root=experiments/outputs/final method.adapter_source_experiment=E03_qwen0_5b_ft
```

Repeat Steps 6–10 independently with `seed=43`, then `seed=44`. Do not chain
seeds into one training process. For interrupted C, use section G resume before
continuing C/D.

### STEP 11 — Aggregate

```powershell
python experiments/scripts/aggregate_results.py --outputs-root experiments/outputs/final --out-dir experiments/results
```

### STEP 12 — Per-seed statistics

```powershell
python experiments/scripts/eval_stats.py --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag --out-dir experiments/results/stats/seed_42 --override output_root=experiments/outputs/final seed=42 model=qwen2_5_0_5b
```

Repeat with `seed_43`/`seed=43` and `seed_44`/`seed=44` after those runs exist.

### STEP 13 — Robustness

Robustness is generated and evaluated by normal `run_experiment.py` because
current external variant files exist. If absent:

```powershell
python experiments/scripts/build_perturbations_v3.py
```

Then rerun each affected method's `run_experiment.py` command.

### STEP 14 — Human evaluation

```powershell
python experiments/scripts/build_human_eval.py --dataset dashboard_v4 --model qwen3_8b --seed 42 --n-items 40 --n-raters 6 --ratings-per-output 3
python experiments/scripts/run_human_eval.py --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8b/seed_42 --port 8501
python experiments/scripts/compute_irr.py --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8b/seed_42
```

Run rating app interactively between build and compute commands.

### STEP 15 — Package final results

```powershell
python experiments/scripts/package_results.py --outputs-root experiments/outputs/final --results-dir experiments/results --out professor_results.zip --dry-run
python experiments/scripts/package_results.py --outputs-root experiments/outputs/final --results-dir experiments/results --out professor_results.zip
```

## What I should NOT run anymore

The following scripts remain for provenance and development history. They are
**HISTORICAL / DEVELOPMENT ONLY — DO NOT RUN FOR `dashboard_v3` FINAL
EXPERIMENTS**.

### Dataset generation, freezing, and legacy splits

- `experiments/scripts/generate_dataset.py`
- `experiments/scripts/generate_dataset_v2.py`
- `experiments/scripts/build_data.py`
- `experiments/scripts/build_hybrid_dataset.py`
- `experiments/scripts/freeze_dataset_v2.py`

These target old synthetic, processed, staging, or v2 workflows. Final configs
point to `data/frozen/dashboard_v3/`.

### nvBench candidate, quality-pool, pilot, and repair workflows

- `experiments/scripts/build_nvbench_candidates.py`
- `experiments/scripts/prepare_nvbench_cache.py`
- `experiments/scripts/rebuild_nvbench_quality_pool.py`
- `experiments/scripts/rebuild_nvbench_quality_pool_final.py`
- `experiments/scripts/repair_nvbench_rules.py`
- `experiments/scripts/run_nvbench_pilot.py`
- `experiments/scripts/run_nvbench_pilot_v4.py`
- `experiments/scripts/run_nvbench_pilot_v5.py`
- `experiments/scripts/run_nvbench_pilot_v6.py`
- `experiments/scripts/compare_nvbench_pilots.py`
- `experiments/scripts/compare_nvbench_pilot_v2_v3.py`
- `experiments/scripts/run_nvbench_large_v1.py`
- `experiments/scripts/build_nvbench_large_v2.py`

### Enrichment and external-data workflows

- `experiments/scripts/run_enrichment_sample.py`
- `experiments/scripts/run_enrichment_full.py`
- `experiments/scripts/run_enrichment_targeted_retry.py`
- `experiments/scripts/test_enrichment_connection.py`
- `experiments/scripts/rebuild_enrichment_audit_template.py`
- `experiments/scripts/register_external_sources.py`

Do not call enrichment APIs or rebuild quality pools for final experiments.
Frozen enrichment is already included in `dashboard_v3`.

### Old perturbation and benchmark workflows

- `experiments/scripts/build_perturbations.py`
- `experiments/scripts/build_perturbations_v2.py`
- `experiments/scripts/build_benchmark.py`
- `experiments/scripts/prepare_benchmark_infer.py`
- `experiments/scripts/eval_benchmark.py`
- `experiments/scripts/validate_benchmark.py`

Current final robustness uses only `build_perturbations_v3.py` and external
`data/eval/robustness_v3/`. Benchmark scripts are separate diagnostics, not
A/B/C/D thesis results.

### Legacy reporting backfill

`experiments/scripts/build_run_reports.py` is a legacy backfill tool. It writes
reporting artifacts for saved runs and labels them diagnostic-only. Use
`eval_auto.py` for current re-evaluation and `aggregate_results.py` for current
aggregation.

Do not delete historical scripts. Do not use them to modify frozen data.

## Current command inventory

### Normal final workflow

- `check_experiment_release.py`: release/environment/data/KB/config verification.
- `build_kb.py`: conditional local KB build.
- `run_smoke.py`: tiny A/B/C/D pipeline smoke test.
- `train.py`: one C training run with optional resume.
- `infer.py`: one inference-only run.
- `eval_auto.py`: metrics-only evaluation of cached predictions.
- `run_experiment.py`: one end-to-end inference plus evaluation run.
- `build_perturbations_v3.py`: conditional external robustness-set build.
- `aggregate_results.py`: per-run and multi-seed aggregation.
- `eval_stats.py`: paired statistics for one exact model/seed/protocol.
- `build_human_eval.py`: blind assignment generation.
- `run_human_eval.py`: Streamlit rating app.
- `compute_irr.py`: human-rating reliability and statistics.
- `package_results.py`: filtered professor/thesis ZIP.
- `benchmark_batch_inference.py`: sequential vs. batched inference on a fixed
  item slice; reports speedup, peak GPU memory and per-item output equality.
  Writes no predictions and touches no run directory (section L1).

### Batch/remote convenience wrappers

These exist but are not normal independent-run workflow:

- `run_final_matrix.py`: batch A/B/C/D across matrix seeds, default 42/43/44; use only when intentionally running a batch. `--dry-run` prints plan without execution.
- `run_all.py`: inference/evaluation batch across experiments and seeds; never trains and assumes adapters already exist.
- `run_remote.py`: remote `smoke`, `train`, or `full` orchestrator; bundles operations and packaging. Direct commands above give clearer one-model/one-seed provenance.

## Important paths

Paths under `experiments/outputs/final/`, `experiments/results/stats/`,
`experiments/results/human_eval/`, and `professor_results.zip` are generated
by commands above. Angle-bracket path components are placeholders.

| Purpose                     | Path                                                                                                        |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| frozen Train                | `data/frozen/dashboard_v4/train.jsonl`                                                                      |
| frozen Validation           | `data/frozen/dashboard_v4/val.jsonl`                                                                        |
| frozen Test                 | `data/frozen/dashboard_v4/test.jsonl`                                                                       |
| human evaluation data       | `data/frozen/dashboard_v4/human_eval_test_items_40.csv`                                                     |
| dataset manifest            | `data/frozen/dashboard_v4/manifest.json`                                                                    |
| dataset hashes              | `data/frozen/dashboard_v4/hashes.json`                                                                      |
| RAG sources                 | `data/knowledge_base/guidelines/`                                                                           |
| built KB                    | `data/knowledge_base/chunks.jsonl`, `data/knowledge_base/kb_manifest.json`                                  |
| external robustness data    | `data/eval/robustness_v4/`                                                                                  |
| model configs               | `src/config/model/`                                                                                         |
| experiment configs          | `src/config/experiment/`                                                                                    |
| method configs              | `src/config/method/`                                                                                        |
| final matrix config         | `src/config/matrix/final.yaml`                                                                              |
| adapters                    | `experiments/outputs/final/<dataset>/<model>/C/seed_<seed>/adapter/`                                        |
| predictions                 | `experiments/outputs/final/<dataset>/<model>/<A-D>/seed_<seed>/predictions*.jsonl`                         |
| automatic metrics           | `experiments/outputs/final/<dataset>/<model>/<A-D>/seed_<seed>/metrics_auto.json`                          |
| per-run reports             | `experiments/outputs/final/<dataset>/<model>/<A-D>/seed_<seed>/metrics.json`, `eval_per_item.jsonl`        |
| per-seed statistics         | `experiments/results/stats/<dataset>/seed_<seed>/`                                                          |
| final result reports        | `experiments/results/comparison_table.*`, `multi_seed_summary.*`, `comparison_seeds.csv`, `final_report.md` |
| human-evaluation study      | `experiments/results/human_eval/<dataset>/<model>/seed_<seed>/`                                             |
| human ratings/statistics    | `<study-dir>/ratings/` and `<study-dir>/analysis/`                                                          |
| professor package           | `professor_results.zip`                                                                                     |

## Model / seed / method explained simply

### Model

The LLM being tested. Current available config is Qwen2.5-0.5B-Instruct.
Selection comes from `src/config/model/`.

### Seed

Randomness control for one run. Seeds 42, 43, and 44 are separate repeated
runs used to measure training variability.

### Method

- A: prompt-only base model.
- B: base model plus TF-IDF RAG.
- C: QLoRA fine-tuned model with adapter.
- D: same fine-tuned model plus TF-IDF RAG.

### Adapter

Small QLoRA parameter files produced by C. Base model remains separate.

### Why D uses C adapter

D measures adding RAG to same fine-tuned model. D must consume C adapter for
same model, dataset, and seed.

### Why seeds are separate

Independent seed runs measure training variation without forcing all work into
one long chained process. Failed seed can resume or rerun without changing
other seeds.

## Final operational guarantees

- Final commands use frozen `dashboard_v4` by default (`dashboard_v3` remains
  selectable via `--dataset`); no final command regenerates either package.
- Outputs/results are keyed on the dataset, so v3 and v4 never mix.
- A/B/C/D runs execute independently under selected output roots.
- D resolves and validates matching C adapter by model/seed/dataset metadata.
- Paired statistics require matching test IDs, dataset hashes, model/config, seed, and protocol.
- Multi-seed aggregation keeps seed values separate and does not pool predictions as independent observations.
- Cached evaluation does not rerun model inference.
- Packaging excludes base-model caches, weights, checkpoints, secrets, and `.env` files.
