"""Rule-leakage report: demonstrate the shared synthetic label rule and its impact.

Programmatically confirms that the synthetic gold labels are produced by the
deterministic mapping `KEYWORD_TASK -> task_type -> TASK_CHART[task_type][0]`, so a
row-level train/test split does NOT remove the shared label-generation logic.
Writes experiments/results/rule_leakage_report.md. No model inference.

    python experiments/scripts/rule_leakage_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.frozen_validation import read_jsonl_strict
from src.data_pipeline.synth_generator import TASK_CHART
from src.evaluation.metrics.base import chart_token

SYNTHETIC_FILES = (
    ("train", "data/frozen/dashboard_v2/train.jsonl"),
    ("val", "data/frozen/dashboard_v2/val.jsonl"),
    ("internal_test", "data/frozen/dashboard_v2/internal_test.jsonl"),
)


def _rule_conformance(records) -> tuple:
    """Return (n_entries, n_following_rule) for chart == TASK_CHART[task][0]."""
    n = ok = 0
    for r in records:
        for m in (r.get("recommendation") or {}).get("kpi_chart_mapping", []) or []:
            if not isinstance(m, dict):
                continue
            task = m.get("task_type")
            chart = chart_token(m.get("chart_type"))
            if task in TASK_CHART:
                n += 1
                if chart == chart_token(TASK_CHART[task][0]):
                    ok += 1
    return n, ok


def main() -> None:
    per_split = {}
    total_n = total_ok = 0
    for split, rel in SYNTHETIC_FILES:
        recs, _ = read_jsonl_strict(_PROJECT_ROOT / rel)
        n, ok = _rule_conformance(recs)
        per_split[split] = (n, ok)
        total_n += n
        total_ok += ok

    rule_rows = "\n".join(
        f"| `{split}` | {ok}/{n} | {round(100.0 * ok / n, 1) if n else 'n/a'}% |"
        for split, (n, ok) in per_split.items()
    )
    overall_pct = round(100.0 * total_ok / total_n, 1) if total_n else "n/a"

    md = f"""# Rule-Leakage Report (construct-validity risk)

## The shared rule
The synthetic generator assigns chart labels deterministically:

    KEYWORD_TASK -> task_type -> TASK_CHART[task_type][0] -> chart_type

(`src/data_pipeline/synth_generator.py`). Both the training split and the synthetic
`internal_test` split are produced by this **same** rule, and the split is a per-row
content hash (`src/data_pipeline/splits.py`).

## Evidence: synthetic gold conforms to the rule
Fraction of gold `kpi_chart_mapping` entries whose `chart_type` equals
`TASK_CHART[task_type][0]`:

| split | conforming | rate |
| --- | --- | --- |
{rule_rows}

Overall: **{overall_pct}%** of {total_n} gold entries follow the single deterministic
mapping. (Any residue below 100% comes only from multi-chart tasks / normalisation, not
from independent labelling.)

## Why this inflates fine-tuned results
A model fine-tuned on the training split learns the generator's `task_type -> chart_type`
mapping. Measuring Top-1/Top-3/macro-F1 on the synthetic test split then rewards
**reproducing that mapping**, not producing good dashboards. High fine-tuned accuracy
(e.g. E03/E04) is therefore substantially a measure of rule reproduction.

## Why a row-level split is not enough
The hash split guarantees that no *item* appears in both train and test, but it does
**not** break the shared *label-generation logic*. Train and test remain the same
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
"""
    out = _PROJECT_ROOT / "experiments/results/rule_leakage_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Rule-leakage report: overall rule-conformance {overall_pct}% -> {out}")


if __name__ == "__main__":
    main()
