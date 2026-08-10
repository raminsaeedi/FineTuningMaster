# Outputs to Return (after the GPU run)

Applies to the **current supported path** (`run_final_matrix.py` →
[`RUN_PROFESSOR.md`](../../RUN_PROFESSOR.md), outputs under `experiments/outputs/final/`) and to
the legacy flows: the **full pipeline** (`run_supervisor_full_gpu.ps1` →
[`SUPERVISOR_FULL_GPU_RUNBOOK.md`](SUPERVISOR_FULL_GPU_RUNBOOK.md)) and the **benchmark-only**
run (`run_supervisor_gpu_o3.ps1` → [`SUPERVISOR_GPU_RUNBOOK.md`](SUPERVISOR_GPU_RUNBOOK.md)).
A single ZIP of the paths below is ideal. Items marked *(full run)* apply only when training +
synthetic diagnostics were run.

> **Easiest way — let the packager do it.** Items 1–8 below are collected automatically:
> ```
> python experiments/scripts/package_results.py --outputs-root experiments/outputs/final
> ```
> (use `--outputs-root experiments/outputs/experiments` for the legacy supervisor `.ps1` flow.
> Each invocation packages **one** run root, so for the separate benchmark root run it again
> with a different `--out`, e.g.
> `--outputs-root experiments/outputs/benchmark_v1 --out professor_results_benchmark.zip`,
> otherwise the second run overwrites the first ZIP. `--dry-run` lists the contents without
> writing.) This writes **`professor_results.zip`** at the repo root, plus a
> `PACKAGE_MANIFEST.json` listing every file with size and sha256. Model weights, checkpoints,
> caches and anything that looks like a secret are **excluded by design** — do not add them
> back by hand. Only items 9–11 have to be attached manually.

## Required
1. **Training summary** *(full run)* — `.../E03_qwen0_5b_ft_42/adapter/training_metadata.json`.
   **Do not send model weights or checkpoints** (`adapter_model.safetensors`, `*.bin`,
   `checkpoints/`) — the packager deliberately excludes them and they are not needed.
   *Collected by `package_results.py`.*
2. **`experiments/outputs/benchmark_v1/`** — all four benchmark run folders
   (`E0{1..4}_..._benchmark_v1_42/`), each with `predictions*.jsonl`, `errors*.jsonl`,
   `metrics_auto.json`, `eval_per_item.jsonl`, `manifest.json`, `config_snapshot.yaml`,
   `config_hash.txt`, `git_hash.txt`, `env.txt`, `logs/`.
   *Collected by `package_results.py --outputs-root experiments/outputs/benchmark_v1`.*
3. **`experiments/results/benchmark_v1_eval.json`** — machine-readable benchmark scores.
   *Collected by `package_results.py` (all of `experiments/results/` is packaged).*
4. **`experiments/results/benchmark_v1_eval.md`** — human-readable benchmark report
   (the independent, non-circular evidence). *Collected by `package_results.py`.*

## Synthetic internal diagnostics *(full run — label as internal diagnostic, circular)*
5. **`experiments/outputs/final/E0{1..4}_..._<seed>/`** (legacy `.ps1` flow:
   `experiments/outputs/experiments/E0{1..4}_..._42/`) — synthetic diagnostic runs
   (`predictions*.jsonl`, `errors*.jsonl`, `metrics_auto.json`, `manifest.json`, `logs/`).
   *Collected by `package_results.py --outputs-root <that root>`.*
6. **`experiments/results/final_report.md`** — synthetic comparison (NOT real-quality evidence).
   *Collected by `package_results.py`.*
7. **`experiments/results/comparison_table.csv`** — synthetic per-run metrics.
   *Collected by `package_results.py`.*
8. **`experiments/results/stats/`** — statistics reports if `eval_stats` produced them
   (single-seed significance is indicative only). *Collected by `package_results.py`.*
9. **`data/frozen/dashboard_v3/validation_report.md`** + **`hashes.json`** — the frozen training
   dataset's actual counts + integrity hashes. **Not** collected by `package_results.py`
   (it packages only run outputs and `experiments/results/`) — attach these two files manually.

## On failure
10. The failing run's **`logs/`** plus the console output / error text.
    (`logs/` is inside the ZIP; the console text has to be pasted in manually.)

## Environment / GPU info (reproducibility)
11. `nvidia-smi` output and GPU/driver/CUDA versions — **attach manually**. `pip freeze` is
    already inside the ZIP as each run's `env.txt`.

## Notes
- `benchmark_v1` is **evaluation-only** — never used for training, augmentation, prompt
  optimization, retriever tuning, hyperparameter tuning, or model selection.
- Do **not** modify `benchmark_v1.jsonl` (labels, task types, chart sets) or any benchmark file.
- Synthetic results (items 5–7) are **internal diagnostics** (circular); do not present them as
  real dashboard-design quality. Usefulness/actionability requires human ratings (none yet).
- Model weights, checkpoints, HuggingFace caches and anything that looks like a secret are
  excluded from `professor_results.zip` on purpose. Do not re-add them; they are not required
  for any result in the thesis.
- If multi-seed was run, include the seed-tagged folders (e.g. `..._43/`,
  `experiments/outputs/benchmark_v1_s43/`) and one `benchmark_v1_eval.*` per seed root; report
  mean/std/CI only for seeds actually run.
