# Rule-Leakage Report (construct-validity risk)

## The shared rule

The synthetic generator assigns chart labels deterministically:

    KEYWORD_TASK -> task_type -> TASK_CHART[task_type][0] -> chart_type

(`src/data_pipeline/synth_generator.py`). Both the training split and the synthetic
`internal_test` split are produced by this **same** rule, and the split is a per-row
content hash (`src/data_pipeline/splits.py`).

## Evidence: synthetic gold conforms to the rule

Fraction of gold `kpi_chart_mapping` entries whose `chart_type` equals
`TASK_CHART[task_type][0]`:

| split           | conforming | rate   |
| --------------- | ---------- | ------ |
| `train`         | 72/72      | 100.0% |
| `val`           | 11/11      | 100.0% |
| `internal_test` | 11/11      | 100.0% |

Overall: **100.0%** of 94 gold entries follow the single deterministic
mapping. (Any residue below 100% comes only from multi-chart tasks / normalisation, not
from independent labelling.)

## Why this inflates fine-tuned results

A model fine-tuned on the training split learns the generator's `task_type -> chart_type`
mapping. Measuring Top-1/Top-3/macro-F1 on the synthetic test split then rewards
**reproducing that mapping**, not producing good dashboards. High fine-tuned accuracy
(e.g. E03/E04) is therefore substantially a measure of rule reproduction.

## Why a row-level split is not enough

The hash split guarantees that no _item_ appears in both train and test, but it does
**not** break the shared _label-generation logic_. Train and test remain the same
function of `task_type`, so the construct being measured is "did the model relearn the
generator", not "did the model learn dashboard-design quality".

## Independent tests required

1. **Independent L1** chart-selection scorer against literature-derived effective sets
   (`experiments/scripts/eval_l1_independent.py`) — covered items only, coverage reported.
2. **Independent benchmark** (`data/eval/benchmark_v1.jsonl`) with a label lineage disjoint
   from the generator — method comparison gated behind an approved inference pass.
3. **Human evaluation** for usefulness/actionability/quality (no ratings yet).

## Claim limits

Synthetic chart-selection accuracy is an **internal diagnostic only**. It must never be
presented as evidence of real dashboard-design quality, real-dashboard superiority, or
usefulness. See `docs/evaluation/scientific_dataset_validity_implementation_plan.md`
(Thesis claim policy).
