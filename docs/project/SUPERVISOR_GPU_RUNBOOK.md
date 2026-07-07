# Supervisor GPU Runbook — O3 Benchmark Inference

> **This is the benchmark-only path (no training).** For the full final pipeline
> (build dataset → train → synthetic diagnostics → benchmark → score → aggregate), use
> [`SUPERVISOR_FULL_GPU_RUNBOOK.md`](SUPERVISOR_FULL_GPU_RUNBOOK.md).

## Project goal
The thesis fine-tunes a small LLM (Qwen2.5-0.5B-Instruct) to produce **structured dashboard
design recommendations** (chart selection + layout/styling/interactions) from a user brief
(users, goals, KPIs, data columns). Four methods are compared: **A** prompt-only, **B** RAG,
**C** fine-tuned (QLoRA), **D** fine-tuned + RAG. Synthetic accuracy is circular (rule leakage)
and counts only as internal diagnostics; the **independent benchmark** (`benchmark_v1`) and the
literature L1 gold are the non-circular evidence. This runbook covers **O3**: running A/B/C/D on
the benchmark on a GPU machine (the local CPU is too weak). **No training is required.**

## What is already prepared (in the repo)
- `data/eval/benchmark_v1.jsonl` — 30 independent items (22 auto-scorable), eval-only.
- `data/eval/benchmark_v1_infer.jsonl` — inference wrapper (rebuildable).
- `experiments/scripts/eval_benchmark.py` — independent benchmark scorer (set-valued).
- `experiments/outputs/E03_qwen0_5b_ft_42/adapter/` — trained LoRA adapter (reused by C and D).
- `data/knowledge_base/chunks.jsonl` — RAG KB (rebuildable).
- Plan + claim policy: `docs/evaluation/o3_benchmark_inference_plan.md`,
  `docs/evaluation/scientific_dataset_validity_implementation_plan.md`.

## Environment setup
Follow **[`REMOTE_RUN.md`](../../REMOTE_RUN.md) → Setup** (venv, CUDA-matched
`torch==2.6.0`, `pip install -r requirements-train.txt`, `pip install -e .`). Optional HF cache
via `.env` (`TRANSFORMERS_CACHE`), and `CUDA_VISIBLE_DEVICES=0` to pin one GPU. Do **not** run
`REMOTE_RUN.md`'s train/full modes for O3 — this runbook is inference-only on the benchmark.

## GPU preflight (no inference)
```
python experiments/scripts/preflight_supervisor_gpu.py
```
Checks imports, CUDA, benchmark files, non-empty briefs, FT adapter, KB chunks, writable output.
Resolve any `FAIL` before continuing (a CUDA `WARN` on a CPU box is expected — run on GPU).

## One-shot entrypoint (recommended)
```
pwsh experiments/scripts/run_supervisor_gpu_o3.ps1
```
Runs preflight → wrapper → KB → A → B → (verify adapter) → C → D → benchmark scoring, writing
everything under `experiments/outputs/benchmark_v1/`. It never trains and never edits the
benchmark.

## Exact commands (equivalent, if running manually)
All benchmark runs use `data/eval/benchmark_v1_infer.jsonl`, **disable synthetic variants**
(`data.paraphrased_file=null data.missing_info_file=null`), and write to a **separate output
root** `experiments/outputs/benchmark_v1`.
```
# preflight + one-time prep
python experiments/scripts/preflight_supervisor_gpu.py
python experiments/scripts/prepare_benchmark_infer.py
python experiments/scripts/build_kb.py

# A/B/C/D inference (C/D reuse the E03 adapter)
python experiments/scripts/infer.py --experiment E01_qwen0_5b_prompt  --override data.test_file=data/eval/benchmark_v1_infer.jsonl data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1 experiment_name=E01_qwen0_5b_prompt__benchmark_v1
python experiments/scripts/infer.py --experiment E02_qwen0_5b_rag     --override data.test_file=data/eval/benchmark_v1_infer.jsonl data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1 experiment_name=E02_qwen0_5b_rag__benchmark_v1
python experiments/scripts/infer.py --experiment E03_qwen0_5b_ft      --override data.test_file=data/eval/benchmark_v1_infer.jsonl data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1 experiment_name=E03_qwen0_5b_ft__benchmark_v1 method.adapter_path=experiments/outputs/E03_qwen0_5b_ft_42/adapter
python experiments/scripts/infer.py --experiment E04_qwen0_5b_ft_rag  --override data.test_file=data/eval/benchmark_v1_infer.jsonl data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1 experiment_name=E04_qwen0_5b_ft_rag__benchmark_v1 method.adapter_path=experiments/outputs/E03_qwen0_5b_ft_42/adapter
```

## Benchmark scoring (offline, no model)
```
python experiments/scripts/eval_benchmark.py --predictions-root experiments/outputs/benchmark_v1 --benchmark data/eval/benchmark_v1.jsonl
```
Writes `experiments/results/benchmark_v1_eval.{json,md}` (coverage, covered accuracy,
per-task_type/per-domain, evidence split, parse failures; schema/parse/completeness).

## Optional multi-seed — ONLY IF SUPERVISOR APPROVES
Not part of the default O3 run. If approved, repeat inference per seed into seed-tagged names,
then score each root separately:
```
# example for seed 43 (repeat for 44); keep the same null-variant flags
python experiments/scripts/infer.py --experiment E01_qwen0_5b_prompt --override seed=43 data.test_file=data/eval/benchmark_v1_infer.jsonl data.paraphrased_file=null data.missing_info_file=null output_root=experiments/outputs/benchmark_v1 experiment_name=E01_qwen0_5b_prompt__benchmark_v1
# ... E02/E03/E04 likewise (C/D add method.adapter_path=...E03...adapter) ...
```
Multi-seed enables variance/CIs; without it, no significance claims.

## Resume instructions
Inference is **idempotent and cached per `item_id`** — re-running the same command skips
completed items and continues. To restart a method cleanly, delete only its run folder under
`experiments/outputs/benchmark_v1/`. Nothing outside that root is touched.

## Expected outputs
- `experiments/outputs/benchmark_v1/E0{1..4}_..._benchmark_v1_42/` → `predictions.jsonl`
  **only** (no `predictions_paraphrased.jsonl` / `predictions_missing_info.jsonl`),
  `config_snapshot.yaml`, `config_hash.txt`, `git_hash.txt`, `env.txt`, `manifest.json`, `logs/`.
- `experiments/results/benchmark_v1_eval.{json,md}`.
See **[`OUTPUTS_TO_RETURN.md`](OUTPUTS_TO_RETURN.md)** for exactly what to send back.

## Scientific limitations
Small n (30; 22 auto-scorable); single seed (42) unless multi-seed approved (no CIs otherwise);
`realistic_manual` items are weaker evidence than `real_public`; item-level primary-chart
scoring (not per-KPI); 0.5B smoke model; grounding is a lexical proxy; L1-uncovered task types
(composition/part_to_whole/flow) are excluded from auto-accuracy and reserved for human eval.

## Allowed claims (after this run)
- Independent, non-circular **chart-selection acceptability** of A/B/C/D on covered benchmark
  items, reported **with coverage** and strong-vs-weak evidence split.
- Format reliability (schema validity, parse, completeness) and latency on realistic briefs.

## Forbidden claims
- That benchmark accuracy proves layout/styling/interaction/rationale or overall dashboard
  quality, or usefulness/actionability (needs human evaluation).
- Any significance/variance claim from a single seed; generalization beyond covered items.
- Reusing `benchmark_v1` for training, augmentation, prompt optimization, retriever tuning,
  hyperparameter tuning, or model selection (**evaluation-only**).
