# Experiment comparison report (backfilled)

> This is a backfilled legacy internal-synthetic diagnostic report. It is not the final thesis-valid independent evaluation report. L1 human-effectiveness, L3 realism, and L4 human evaluation are pending.

Backfilled from `experiments/outputs` — 4 run(s). All numbers are carried from existing `metrics_auto.json` (legacy, pre-Task-7) and are **internal-synthetic diagnostics only**. `top1%` is synthetic (circular) — not a validity claim. L1 human-effectiveness / L3 realism / L4 human evaluation are pending.

## Per-run (legacy synthetic diagnostics)

| experiment_id | method | model | seed | json_parse% | schema_valid% | complete | top1%(legacy,synthetic) | macro_f1(legacy) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E01_qwen0_5b_prompt_42 | prompt_only | Qwen/Qwen2.5-0.5B-Instruct | 42 | 66.0 | 66.0 | 0.66 | 15.62 | 0.0338 |
| E02_qwen0_5b_rag_42 | rag | Qwen/Qwen2.5-0.5B-Instruct | 42 | 52.0 | 44.0 | 0.4667 | 11.54 | 0.0251 |
| E03_qwen0_5b_ft_42 | ft | Qwen/Qwen2.5-0.5B-Instruct | 42 | 54.0 | 54.0 | 0.54 | 77.78 | 0.1496 |
| E04_qwen0_5b_ft_rag_42 | ft_rag | Qwen/Qwen2.5-0.5B-Instruct | 42 | 84.0 | 84.0 | 0.84 | 86.67 | 0.3317 |
