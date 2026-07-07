"""Run the independent L1 chart-selection scorer over cached predictions.

Scores each method's cached predictions against the independent L1 literature table
(`data/eval/l1_chart_effectiveness_v1.csv`). Offline — reparses cached `raw_text`,
no model inference.

IMPORTANT (keying case): these are the cached SYNTHETIC v1 predictions, whose gold
`task_type` is generator-derived. L1 scores here are therefore DIAGNOSTIC / LIMITED
(they test chart choice against independent literature, but the task label shares the
generator lineage). Independent L1 scoring requires benchmark_v1 predictions
(approval-gated inference).

    python experiments/scripts/eval_l1_independent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.schemas import GenerationResult
from src.data_pipeline.dataset import load_gold_items
from src.evaluation.l1_independent import DEFAULT_L1_CSV, load_effective_sets, score_l1
from src.inference.postprocess import reparse
from src.utils.io import read_jsonl, write_json

GOLD = "data/processed/test.jsonl"          # v1 gold the cached predictions target
OUTPUTS_ROOT = "experiments/outputs"
METHOD_RUNS = {
    "prompt_only": "E01_qwen0_5b_prompt_42",
    "rag": "E02_qwen0_5b_rag_42",
    "ft": "E03_qwen0_5b_ft_42",
    "ft_rag": "E04_qwen0_5b_ft_rag_42",
}


def _load_predictions(run_dir: Path):
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        return None
    return [reparse(GenerationResult(**r)) for r in read_jsonl(path)]


def main() -> None:
    l1_path = _PROJECT_ROOT / DEFAULT_L1_CSV
    effective_sets = load_effective_sets(l1_path)

    gold_items = load_gold_items(_PROJECT_ROOT / GOLD)
    references = [
        {"item_id": it.item_id, "recommendation": it.recommendation.model_dump(mode="json")}
        for it in gold_items
    ]

    per_method = {}
    for method, run in METHOD_RUNS.items():
        results = _load_predictions(_PROJECT_ROOT / OUTPUTS_ROOT / run)
        if results is None:
            per_method[method] = {"status": "no_predictions"}
            continue
        per_method[method] = score_l1(results, references, effective_sets)

    payload = {
        "l1_gold_file": DEFAULT_L1_CSV,
        "gold_file": GOLD,
        "keying_case": "synthetic_v1_generator_derived_task_type (diagnostic/limited)",
        "covered_task_types": sorted(effective_sets.keys()),
        "per_method": per_method,
    }
    write_json(payload, _PROJECT_ROOT / "experiments/results/l1_independent_results.json")

    lines = ["# Independent L1 Chart-Selection Results", "",
             f"L1 gold (independent, literature): `{DEFAULT_L1_CSV}` "
             "(Saket 2019 + Kim & Heer 2018).",
             f"Predictions gold join: `{GOLD}` (cached v1 synthetic test).", "",
             "> **Keying case — diagnostic/limited.** These cached predictions are on the "
             "SYNTHETIC v1 test set, whose `task_type` is generator-derived. The chart set is "
             "independent (literature), but the task label shares the generator lineage, so "
             "these L1 numbers are a diagnostic, not a fully independent claim. Independent L1 "
             "requires `benchmark_v1` predictions (approval-gated inference).", "",
             "> **L1 limitation.** L1 validates only chart-selection *acceptability for covered "
             "task types*. It does NOT validate layout, styling, interaction, rationale, or "
             "overall dashboard-design quality. Uncovered items are excluded from accuracy "
             "(never counted correct); coverage is reported below.", "",
             "| method | coverage_rate | n_covered | n_uncovered | covered_accuracy |",
             "| --- | --- | --- | --- | --- |"]
    for method in METHOD_RUNS:
        m = per_method.get(method, {})
        if m.get("status") == "no_predictions":
            lines.append(f"| {method} | (no predictions) | | | |")
            continue
        lines.append(f"| {method} | {m['coverage_rate']} | {m['n_covered']} | "
                     f"{m['n_uncovered']} | {m['covered_accuracy']} |")
    lines += ["", "## Per-method detail (per task_type accuracy on covered items)", ""]
    for method in METHOD_RUNS:
        m = per_method.get(method, {})
        if m.get("status") == "no_predictions":
            continue
        lines.append(f"### {method}")
        lines.append(f"- coverage_rate={m['coverage_rate']} "
                     f"(covered {m['n_covered']} of {m['n_gold_kpi']} gold KPI entries; "
                     f"uncovered {m['n_uncovered']})")
        lines.append(f"- covered_accuracy={m['covered_accuracy']}")
        for t, v in m["per_task_type"].items():
            lines.append(f"  - `{t}`: {v['correct']}/{v['covered']} = {v['accuracy']}")
        if m["uncovered_task_types"]:
            lines.append(f"- uncovered task types (excluded): {m['uncovered_task_types']}")
        lines.append("")

    (_PROJECT_ROOT / "experiments/results/l1_independent_results.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("L1 independent results -> experiments/results/l1_independent_results.md")


if __name__ == "__main__":
    main()
