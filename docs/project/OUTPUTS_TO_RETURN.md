# Outputs to Return (after the GPU run)

Applies to both the **full pipeline** (`run_supervisor_full_gpu.ps1` →
[`SUPERVISOR_FULL_GPU_RUNBOOK.md`](SUPERVISOR_FULL_GPU_RUNBOOK.md)) and the **benchmark-only**
run (`run_supervisor_gpu_o3.ps1` → [`SUPERVISOR_GPU_RUNBOOK.md`](SUPERVISOR_GPU_RUNBOOK.md)).
A single ZIP of the paths below is ideal. Items marked *(full run)* apply only when training +
synthetic diagnostics were run.

## Required
1. **Trained adapter** *(full run)* — `experiments/outputs/experiments/E03_qwen0_5b_ft_42/adapter/`
   (`adapter_config.json`, `adapter_model.safetensors`, tokenizer, `training_metadata.json`).
2. **`experiments/outputs/benchmark_v1/`** — all four benchmark run folders
   (`E0{1..4}_..._benchmark_v1_42/`), each with `predictions.jsonl`, `manifest.json`,
   `config_snapshot.yaml`, `config_hash.txt`, `git_hash.txt`, `env.txt`, `logs/`.
3. **`experiments/results/benchmark_v1_eval.json`** — machine-readable benchmark scores.
4. **`experiments/results/benchmark_v1_eval.md`** — human-readable benchmark report
   (the independent, non-circular evidence).

## Synthetic internal diagnostics *(full run — label as internal diagnostic, circular)*
5. **`experiments/outputs/experiments/E0{1..4}_..._42/`** — synthetic diagnostic runs
   (`predictions.jsonl`, `metrics_auto.json`, `manifest.json`, `logs/`).
6. **`experiments/results/final_report.md`** — synthetic comparison (NOT real-quality evidence).
7. **`experiments/results/comparison_table.csv`** — synthetic per-run metrics.
8. **`experiments/results/stats/`** — statistics reports if `eval_stats` produced them
   (single-seed significance is indicative only).
9. **`data/frozen/dashboard_v2/validation_report.md`** + **`hashes.json`** — the built training
   dataset's actual counts + integrity hashes.

## On failure
10. The failing run's **`logs/`** plus the console output / error text.

## Environment / GPU info (reproducibility)
11. `nvidia-smi` output, `pip freeze` (or a run's `env.txt`), GPU/driver/CUDA versions.

## Notes
- `benchmark_v1` is **evaluation-only** — never used for training, augmentation, prompt
  optimization, retriever tuning, hyperparameter tuning, or model selection.
- Do **not** modify `benchmark_v1.jsonl` (labels, task types, chart sets) or any benchmark file.
- Synthetic results (items 5–7) are **internal diagnostics** (circular); do not present them as
  real dashboard-design quality. Usefulness/actionability requires human ratings (none yet).
- If multi-seed was run, include the seed-tagged folders (e.g. `..._43/`,
  `experiments/outputs/benchmark_v1_s43/`) and one `benchmark_v1_eval.*` per seed root; report
  mean/std/CI only for seeds actually run.
