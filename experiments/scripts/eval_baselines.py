"""Trivial chart-selection baselines + prompt-leakage stratification (offline).

Two reference points the method comparison (A/B/C/D) cannot be interpreted
without, and one honesty check. Nothing here runs a model, writes into a run
directory, or touches frozen data — it only reads gold splits plus cached
`predictions.jsonl` files and writes one report pair under `experiments/results/`.

1. `majority_class` baseline — always predicts the single most frequent primary
   chart of the **training** split (never the test split, so the baseline itself
   is not fitted on test). On a skewed label distribution this is the floor any
   method must beat; without it a top-1 number cannot be read as "learning".

2. `chart_word_copy` baseline — copies a chart type named literally in the brief
   text, else falls back to the majority chart. nvBench-derived briefs often
   contain the answer ("... in a pie chart"), so this quantifies how much of
   top-1 is retrievable from the prompt surface alone.

3. Prompt-leakage stratification — each method's cached top-1, split into items
   whose brief names a chart type and items where it does not. The non-leaking
   subset is the part of top-1 that reflects an actual chart-choice decision.

Both baselines are scored with the SAME metric code as the methods
(`TopKAccuracy`, `MacroF1ChartType`), so the numbers are directly comparable.

    python experiments/scripts/eval_baselines.py \
        --gold data/frozen/dashboard_v4/test.jsonl \
        --train data/frozen/dashboard_v4/train.jsonl \
        --outputs-root experiments/outputs/final/dashboard_v4 \
        --run prompt_only=qwen2_5_0_5b/A/seed_42 \
        --run rag=qwen2_5_0_5b/B/seed_42
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.schemas import ChartType, DesignOutput, GenerationResult  # noqa: E402
from src.data_pipeline.dataset import load_gold_items  # noqa: E402
from src.evaluation.metrics.base import predicted_charts, reference_charts  # noqa: E402
from src.evaluation.metrics.macro_f1 import MacroF1ChartType  # noqa: E402
from src.evaluation.metrics.topk_accuracy import TopKAccuracy  # noqa: E402
from src.inference.postprocess import reparse  # noqa: E402
from src.utils.io import read_jsonl, write_json  # noqa: E402

# Surface forms that name a chart type in a brief -> ChartType token. Only
# unambiguous wordings are listed; anything else counts as "no chart named".
_CHART_WORDS: list[tuple[str, str]] = [
    (r"stacked\s+bar", ChartType.STACKED_BAR.value),
    (r"grouped\s+bar", ChartType.GROUPED_BAR.value),
    (r"\bpie\b", ChartType.PIE.value),
    (r"\bdonut\b", ChartType.DONUT.value),
    (r"\bscatter\b", ChartType.SCATTER.value),
    (r"\bhistogram\b", ChartType.HISTOGRAM.value),
    (r"\bheat\s?map\b", ChartType.HEATMAP.value),
    (r"\btreemap\b", ChartType.TREEMAP.value),
    (r"\bbox\s?plot\b", ChartType.BOX.value),
    (r"\bline\b", ChartType.LINE.value),
    (r"\barea\b", ChartType.AREA.value),
    (r"\bbar\b", ChartType.BAR.value),
    (r"\btable\b", ChartType.TABLE.value),
    (r"\bgauge\b", ChartType.GAUGE.value),
    (r"\bsankey\b", ChartType.SANKEY.value),
    (r"\bmap\b", ChartType.MAP.value),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trivial baselines + prompt-leakage stratification")
    p.add_argument("--gold", default="data/frozen/dashboard_v4/test.jsonl")
    p.add_argument("--train", default="data/frozen/dashboard_v4/train.jsonl",
                   help="split the majority chart is taken from (never the test split)")
    p.add_argument("--outputs-root", default="experiments/outputs")
    p.add_argument("--run", action="append", default=[], metavar="METHOD=RELDIR",
                   help="repeatable; cached run directory per method (relative to --outputs-root)")
    p.add_argument("--out-prefix", default="experiments/results/baselines_and_leakage")
    return p.parse_args()


def brief_text(brief) -> str:
    parts = [brief.users or "", " ".join(brief.goals or []), " ".join(brief.kpis or []),
             brief.constraints or ""]
    return " ".join(parts).lower()


def named_chart(text: str) -> str | None:
    """The chart type named literally in the brief text, if any."""
    for pattern, token in _CHART_WORDS:
        if re.search(pattern, text):
            return token
    return None


def as_prediction(item_id: str, chart: str, method: str) -> GenerationResult:
    """Wrap a baseline chart choice as a GenerationResult so the real metric
    classes score it exactly like a model output."""
    parsed = DesignOutput(kpi_chart_mapping=[{
        "kpi": "", "task_type": "comparison", "chart_type": chart,
    }])
    return GenerationResult(
        item_id=item_id, method_name=method, model_name="baseline",
        raw_text="", parsed=parsed,
    )


def majority_chart(train_path: Path) -> tuple[str, int, int]:
    counts: collections.Counter = collections.Counter()
    for item in load_gold_items(train_path):
        charts = reference_charts({"recommendation": item.recommendation.model_dump(mode="json")})
        if charts:
            counts[charts[0]] += 1
    chart, hits = counts.most_common(1)[0]
    return chart, hits, sum(counts.values())


def score(results, references) -> dict:
    top_k = TopKAccuracy().compute(results, references)
    return {
        "top_1_accuracy": top_k["top_1_accuracy"],
        "n": top_k["n"],
        "macro_f1": MacroF1ChartType().compute(results, references)["macro_f1"],
    }


def stratified_top1(results, references, leaking_ids: set[str]) -> dict:
    """Top-1 split by whether the brief names a chart type."""
    leaking = [r for r in references if r["item_id"] in leaking_ids]
    clean = [r for r in references if r["item_id"] not in leaking_ids]
    return {
        "all": score(results, references),
        "brief_names_chart": score(results, leaking),
        "brief_names_no_chart": score(results, clean),
    }


def main() -> None:
    args = parse_args()
    items = load_gold_items(_PROJECT_ROOT / args.gold)
    references = [
        {"item_id": it.item_id, "recommendation": it.recommendation.model_dump(mode="json")}
        for it in items
    ]
    texts = {it.item_id: brief_text(it.brief) for it in items}
    named = {item_id: named_chart(text) for item_id, text in texts.items()}
    leaking_ids = {item_id for item_id, chart in named.items() if chart is not None}

    chart, hits, n_train = majority_chart(_PROJECT_ROOT / args.train)
    majority_results = [as_prediction(it.item_id, chart, "majority_class") for it in items]
    copy_results = [
        as_prediction(it.item_id, named[it.item_id] or chart, "chart_word_copy") for it in items
    ]

    payload = {
        "gold_file": args.gold,
        "train_file": args.train,
        "n_test_items": len(items),
        "prompt_leakage": {
            "definition": "brief text (users/goals/kpis/constraints) names a chart type literally",
            "n_brief_names_chart": len(leaking_ids),
            "n_brief_names_no_chart": len(items) - len(leaking_ids),
            "leakage_rate": round(100.0 * len(leaking_ids) / len(items), 2) if items else None,
        },
        "majority_class": {
            "chart": chart,
            "fitted_on": args.train,
            "train_share": round(100.0 * hits / n_train, 2) if n_train else None,
            **stratified_top1(majority_results, references, leaking_ids),
        },
        "chart_word_copy": {
            "fallback_chart": chart,
            **stratified_top1(copy_results, references, leaking_ids),
        },
        "methods": {},
    }

    root = _PROJECT_ROOT / args.outputs_root
    for entry in args.run:
        if "=" not in entry:
            raise SystemExit(f"--run expects METHOD=RELDIR, got: {entry}")
        method, _, rel = entry.partition("=")
        path = root / rel.strip() / "predictions.jsonl"
        if not path.exists():
            payload["methods"][method.strip()] = {"status": f"no predictions at {path}"}
            continue
        results = [reparse(GenerationResult(**row)) for row in read_jsonl(path)]
        entry_payload = stratified_top1(results, references, leaking_ids)
        # How often the method simply echoed the chart named in the brief.
        echo = matched = 0
        by_id = {r.item_id: r for r in results}
        for item_id in leaking_ids:
            result = by_id.get(item_id)
            preds = predicted_charts(result) if result is not None else []
            if not preds:
                continue
            matched += 1
            echo += int(preds[0] == named[item_id])
        entry_payload["echo_of_named_chart"] = {
            "n_scored": matched,
            "rate": round(100.0 * echo / matched, 2) if matched else None,
        }
        entry_payload["run_dir"] = str((root / rel.strip()).resolve())
        payload["methods"][method.strip()] = entry_payload

    write_json(payload, _PROJECT_ROOT / f"{args.out_prefix}.json")

    leak = payload["prompt_leakage"]
    lines = [
        "# Trivial baselines and prompt-leakage stratification",
        "",
        f"Test gold: `{args.gold}` ({len(items)} items). Majority chart fitted on "
        f"`{args.train}` (never on test).",
        "",
        f"**Prompt leakage:** {leak['n_brief_names_chart']} of {len(items)} briefs "
        f"({leak['leakage_rate']}%) name a chart type literally. On those items a method can "
        "reach the gold chart by copying the brief instead of choosing a chart, so top-1 over "
        "all items is an upper bound on chart-choice ability. The "
        "`brief_names_no_chart` column is the decision-only part.",
        "",
        f"**Majority-class floor:** always `{payload['majority_class']['chart']}` "
        f"({payload['majority_class']['train_share']}% of training primaries). A method whose "
        "top-1 is below this floor has not learned chart selection on this label distribution.",
        "",
        "| system | top1 all % | top1 brief-names-chart % | top1 brief-names-no-chart % | macro_f1 all |",
        "| --- | --- | --- | --- | --- |",
    ]
    rows = [("majority_class (baseline)", payload["majority_class"]),
            ("chart_word_copy (baseline)", payload["chart_word_copy"])]
    rows += [(name, value) for name, value in payload["methods"].items()
             if "status" not in value]
    for name, value in rows:
        lines.append(
            f"| {name} | {value['all']['top_1_accuracy']} "
            f"| {value['brief_names_chart']['top_1_accuracy']} "
            f"| {value['brief_names_no_chart']['top_1_accuracy']} "
            f"| {value['all']['macro_f1']} |"
        )
    skipped = {name: value["status"] for name, value in payload["methods"].items()
               if "status" in value}
    if skipped:
        lines += ["", "Runs without cached predictions (not scored):", ""]
        lines += [f"- `{name}`: {status}" for name, status in skipped.items()]
    lines += [
        "",
        f"Subset sizes: brief-names-chart n={leak['n_brief_names_chart']}, "
        f"brief-names-no-chart n={leak['n_brief_names_no_chart']}. A small "
        "brief-names-no-chart subset makes that column noisy — report it with its n, "
        "and do not draw significance from it alone.",
        "",
        "`echo_of_named_chart` (in the JSON) is, per method, how often its primary chart equals "
        "the chart named in the brief.",
        "",
    ]
    out_md = _PROJECT_ROOT / f"{args.out_prefix}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Baselines + leakage stratification -> {out_md}")


if __name__ == "__main__":
    main()
