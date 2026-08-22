"""Run the independent L1 chart-selection scorer over cached predictions.

Scores each method's cached predictions against the independent L1 literature table
(`data/eval/l1_chart_effectiveness_v1.csv`). Offline — reparses cached `raw_text`,
no model inference.

IMPORTANT (keying case): the chart set is independent (literature), but `task_type`
comes from whichever gold file is passed. With the default (cached SYNTHETIC v1)
the task label is generator-derived, so those scores are DIAGNOSTIC / LIMITED.
Fully independent L1 scoring requires benchmark_v1 predictions (approval-gated
inference). The recorded `keying_case` states which case a report belongs to.

    python experiments/scripts/eval_l1_independent.py
    python experiments/scripts/eval_l1_independent.py \
        --gold data/frozen/dashboard_v4/test.jsonl \
        --outputs-root experiments/outputs/final/dashboard_v4 \
        --run prompt_only=qwen2_5_0_5b/A/seed_42 \
        --out-prefix experiments/results/l1_independent_v4
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    """Paths are overridable so the same scorer runs on the final matrix layout
    (``<root>/<dataset>/<model>/<method>/seed_<n>``), not only on the cached v1
    runs. Defaults reproduce the previous behaviour exactly."""
    p = argparse.ArgumentParser(description="Independent L1 chart-selection scorer (offline)")
    p.add_argument("--gold", default=GOLD, help="gold jsonl the predictions were produced on")
    p.add_argument("--outputs-root", default=OUTPUTS_ROOT)
    p.add_argument("--run", action="append", default=[], metavar="METHOD=RELDIR",
                   help="repeatable; run directory (relative to --outputs-root) per method")
    p.add_argument("--out-prefix", default="experiments/results/l1_independent_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gold = args.gold
    outputs_root = args.outputs_root
    method_runs = dict(METHOD_RUNS)
    if args.run:
        method_runs = {}
        for entry in args.run:
            if "=" not in entry:
                raise SystemExit(f"--run expects METHOD=RELDIR, got: {entry}")
            method, _, rel = entry.partition("=")
            method_runs[method.strip()] = rel.strip()

    l1_path = _PROJECT_ROOT / DEFAULT_L1_CSV
    effective_sets = load_effective_sets(l1_path)

    gold_items = load_gold_items(_PROJECT_ROOT / gold)
    references = [
        {"item_id": it.item_id, "recommendation": it.recommendation.model_dump(mode="json")}
        for it in gold_items
    ]

    per_method = {}
    for method, run in method_runs.items():
        results = _load_predictions(_PROJECT_ROOT / outputs_root / run)
        if results is None:
            per_method[method] = {"status": "no_predictions"}
            continue
        per_method[method] = score_l1(results, references, effective_sets)

    payload = {
        "l1_gold_file": DEFAULT_L1_CSV,
        "gold_file": gold,
        "predictions_root": outputs_root,
        "runs": method_runs,
        "keying_case": (
            "synthetic_v1_generator_derived_task_type (diagnostic/limited)"
            if gold == GOLD else
            f"gold={gold}: task_type lineage of that gold file applies (see its dataset card)"
        ),
        "covered_task_types": sorted(effective_sets.keys()),
        "per_method": per_method,
    }
    write_json(payload, _PROJECT_ROOT / f"{args.out_prefix}.json")

    lines = ["# Independent L1 Chart-Selection Results", "",
             f"L1 gold (independent, literature): `{DEFAULT_L1_CSV}` "
             "(Saket 2019 + Kim & Heer 2018).",
             f"Predictions gold join: `{gold}`; predictions root: `{outputs_root}`.", "",
             f"> **Keying case.** {payload['keying_case']} The chart set is independent "
             "(literature), but the `task_type` label comes from the gold file above, so read "
             "these numbers with that lineage in mind. Fully independent L1 requires "
             "`benchmark_v1` predictions (approval-gated inference).", "",
             "> **L1 limitation.** L1 validates only chart-selection *acceptability for covered "
             "task types*. It does NOT validate layout, styling, interaction, rationale, or "
             "overall dashboard-design quality. Uncovered items are excluded from accuracy "
             "(never counted correct); coverage is reported below.", "",
             "| method | coverage_rate | n_covered | n_uncovered | covered_accuracy |",
             "| --- | --- | --- | --- | --- |"]
    for method in method_runs:
        m = per_method.get(method, {})
        if m.get("status") == "no_predictions":
            lines.append(f"| {method} | (no predictions) | | | |")
            continue
        lines.append(f"| {method} | {m['coverage_rate']} | {m['n_covered']} | "
                     f"{m['n_uncovered']} | {m['covered_accuracy']} |")
    lines += ["", "## Per-method detail (per task_type accuracy on covered items)", ""]
    for method in method_runs:
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

    out_md = _PROJECT_ROOT / f"{args.out_prefix}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"L1 independent results -> {out_md}")


if __name__ == "__main__":
    main()
