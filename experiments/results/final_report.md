# Experiment comparison report

Generated from `experiments/outputs` — 4 run(s).

## Per-run metrics

| experiment_id | method | model | seed | n | json_parse% | schema_valid% | complete | top1% | n_fail | top3_ok | top3% | top3_support | n_3rec | n_alt | macro_f1 | latency_s | para_stab% | para_acc% | clarify% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E01_qwen0_5b_prompt_42 | prompt_only | Qwen/Qwen2.5-0.5B-Instruct | 42 | 50 | 66 | 20 | 0.54 | 10 | 18 | False |  | 0 | 0 | 0 | 0.034 | 28712.7 |  |  |  |
| E02_qwen0_5b_rag_42 | rag | Qwen/Qwen2.5-0.5B-Instruct | 42 | 50 | 52 | 14 | 0.44 | 6 | 24 | False |  | 0 | 0 | 0 | 0.025 | 30700 |  |  |  |
| E03_qwen0_5b_ft_42 | ft | Qwen/Qwen2.5-0.5B-Instruct | 42 | 50 | 54 | 18 | 0.54 | 14 | 41 | False |  | 0 | 0 | 9 | 0.15 | 42389.2 |  |  |  |
| E04_qwen0_5b_ft_rag_42 | ft_rag | Qwen/Qwen2.5-0.5B-Instruct | 42 | 50 | 84 | 28 | 0.84 | 26 | 35 | False |  | 0 | 0 | 15 | 0.332 | 44710.3 |  |  |  |

## Across seeds (mean / std per model+method)

| model | method | n_mean | n_std | n_count | json_parse%_mean | json_parse%_std | json_parse%_count | schema_valid%_mean | schema_valid%_std | schema_valid%_count | complete_mean | complete_std | complete_count | top1%_mean | top1%_std | top1%_count | n_fail_mean | n_fail_std | n_fail_count | top3_ok_mean | top3_ok_std | top3_ok_count | top3%_mean | top3%_std | top3%_count | top3_support_mean | top3_support_std | top3_support_count | n_3rec_mean | n_3rec_std | n_3rec_count | n_alt_mean | n_alt_std | n_alt_count | macro_f1_mean | macro_f1_std | macro_f1_count | latency_s_mean | latency_s_std | latency_s_count | para_stab%_mean | para_stab%_std | para_stab%_count | para_acc%_mean | para_acc%_std | para_acc%_count | clarify%_mean | clarify%_std | clarify%_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | ft | 50 | nan | 1 | 54 | nan | 1 | 18 | nan | 1 | 0.54 | nan | 1 | 14 | nan | 1 | 41 | nan | 1 | 0 | nan | 1 | nan | nan | 0 | 0 | nan | 1 | 0 | nan | 1 | 9 | nan | 1 | 0.15 | nan | 1 | 42389.2 | nan | 1 | nan | nan | 0 | nan | nan | 0 | nan | nan | 0 |
| Qwen/Qwen2.5-0.5B-Instruct | ft_rag | 50 | nan | 1 | 84 | nan | 1 | 28 | nan | 1 | 0.84 | nan | 1 | 26 | nan | 1 | 35 | nan | 1 | 0 | nan | 1 | nan | nan | 0 | 0 | nan | 1 | 0 | nan | 1 | 15 | nan | 1 | 0.332 | nan | 1 | 44710.3 | nan | 1 | nan | nan | 0 | nan | nan | 0 | nan | nan | 0 |
| Qwen/Qwen2.5-0.5B-Instruct | prompt_only | 50 | nan | 1 | 66 | nan | 1 | 20 | nan | 1 | 0.54 | nan | 1 | 10 | nan | 1 | 18 | nan | 1 | 0 | nan | 1 | nan | nan | 0 | 0 | nan | 1 | 0 | nan | 1 | 0 | nan | 1 | 0.034 | nan | 1 | 28712.7 | nan | 1 | nan | nan | 0 | nan | nan | 0 | nan | nan | 0 |
| Qwen/Qwen2.5-0.5B-Instruct | rag | 50 | nan | 1 | 52 | nan | 1 | 14 | nan | 1 | 0.44 | nan | 1 | 6 | nan | 1 | 24 | nan | 1 | 0 | nan | 1 | nan | nan | 0 | 0 | nan | 1 | 0 | nan | 1 | 0 | nan | 1 | 0.025 | nan | 1 | 30700 | nan | 1 | nan | nan | 0 | nan | nan | 0 | nan | nan | 0 |

> `top1%` is over ALL items with a reference (parse failures count as wrong; see `n_fail`). `top3_ok=False` means fewer than 80% of items carried 3 distinct ordered recommendations (`top3_support`), so `top3%` is reported as empty - see `src/evaluation/metrics/topk_accuracy.py`.

## Evidence tier & limitations (read before citing)

- **INTERNAL SYNTHETIC DIAGNOSTICS ONLY.** `top1%`, `top3%` and `macro_f1` here are scored against the synthetic generator's own labels (shared `KEYWORD_TASK -> task_type -> TASK_CHART` lineage). High values reflect reproduction of the generator rule, **not** real dashboard-design quality. See `experiments/results/rule_leakage_report.md`.
- **Independent chart-selection** is in `experiments/results/l1_independent_results.md` (covered items only, coverage reported) — do not substitute the circular `top1%` for it.
- **Usefulness / actionability / real-dashboard quality:** NOT supported here (require human evaluation; no ratings collected yet).
- **Single seed (42) only**; confidence intervals deferred. No multi-seed variance is available, so significance claims are not supported from this table alone.
