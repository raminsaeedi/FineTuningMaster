# Memorization vs Generalization — Held-out Combination Analysis

> INTERNAL DIAGNOSTIC (generator mapping, synthetic gold, single seed). Not evidence of real dashboard-design quality.

- train `(domain, task_type)` combos: 79
- test combos: 62
- **novel** combos (in test, not train): 0 → []

| method | seen acc (n) | novel acc (n) |
| --- | --- | --- |
| prompt_only | 0.0854 (199) | None (0) |
| rag | 0.0804 (199) | None (0) |
| ft | 0.1709 (199) | None (0) |
| ft_rag | 0.2412 (199) | None (0) |

> A large seen≫novel gap suggests memorization of seen combinations; comparable accuracy suggests generalization of the mapping. Interpret with the small-n caveat and only as an internal diagnostic.

> **Finding:** the synthetic generator emits a closed set of `(domain, task_type)` combinations — the test split contains **no** held-out combinations, so held-out-combination generalization **cannot be tested on synthetic data**. Use the independent `benchmark_v1` (broader, independent combinations) for this — an approval-gated inference pass.