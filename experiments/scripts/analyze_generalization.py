"""Held-out combination analysis: memorization vs generalization probe.

Computes `(domain, task_type)` combinations seen in training vs present in the
evaluation set, identifies novel combinations, and compares cached-prediction
accuracy on seen vs novel combinations.

The cached predictions target the v1 synthetic test set, so training/eval here use the
matching v1 generation (`data/processed/{train,test}.jsonl`). This is an INTERNAL
DIAGNOSTIC on the generator mapping — not evidence of real dashboard-design quality.

    python experiments/scripts/analyze_generalization.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.schemas import GenerationResult
from src.evaluation.l1_independent import _primary_chart_by_kpi
from src.evaluation.metrics.base import chart_token
from src.inference.postprocess import reparse
from src.utils.io import read_jsonl, write_json

TRAIN = "data/processed/train.jsonl"
TEST = "data/processed/test.jsonl"
OUTPUTS_ROOT = "experiments/outputs"
METHOD_RUNS = {
    "prompt_only": "E01_qwen0_5b_prompt_42",
    "rag": "E02_qwen0_5b_rag_42",
    "ft": "E03_qwen0_5b_ft_42",
    "ft_rag": "E04_qwen0_5b_ft_rag_42",
}


def _domain(rec: dict) -> str:
    reco = rec.get("recommendation", {}) or {}
    cs = reco.get("context_summary", {}) or {}
    return cs.get("domain") or (rec.get("brief", {}).get("extra", {}) or {}).get("domain") or "?"


def _combos(records):
    combos = set()
    for r in records:
        dom = _domain(r)
        for m in (r.get("recommendation") or {}).get("kpi_chart_mapping", []) or []:
            if isinstance(m, dict) and m.get("task_type"):
                combos.add((dom, m["task_type"]))
    return combos


def _load_predictions(run_dir: Path):
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        return None
    return [reparse(GenerationResult(**r)) for r in read_jsonl(path)]


def main() -> None:
    train = read_jsonl(_PROJECT_ROOT / TRAIN)
    test = read_jsonl(_PROJECT_ROOT / TEST)
    train_combos = _combos(train)
    test_combos = _combos(test)
    novel = sorted(test_combos - train_combos)

    per_method = {}
    for method, run in METHOD_RUNS.items():
        preds = _load_predictions(_PROJECT_ROOT / OUTPUTS_ROOT / run)
        if preds is None:
            per_method[method] = {"status": "no_predictions"}
            continue
        pred_by_id = {p.item_id: p for p in preds}
        seen = {"n": 0, "correct": 0}
        nov = {"n": 0, "correct": 0}
        for r in test:
            dom = _domain(r)
            item_id = r.get("item_id", "")
            primary = _primary_chart_by_kpi(pred_by_id.get(item_id))
            for m in (r.get("recommendation") or {}).get("kpi_chart_mapping", []) or []:
                if not isinstance(m, dict) or not m.get("task_type"):
                    continue
                bucket = nov if (dom, m["task_type"]) in novel else seen
                bucket["n"] += 1
                gold = chart_token(m.get("chart_type"))
                pred = primary.get(str(m.get("kpi", "")).strip().lower())
                if pred is not None and pred == gold:
                    bucket["correct"] += 1
        per_method[method] = {
            "seen_acc": round(seen["correct"] / seen["n"], 4) if seen["n"] else None,
            "seen_n": seen["n"],
            "novel_acc": round(nov["correct"] / nov["n"], 4) if nov["n"] else None,
            "novel_n": nov["n"],
        }

    payload = {
        "note": "internal diagnostic on the generator mapping; not real design quality; single seed (42)",
        "n_train_combos": len(train_combos),
        "n_test_combos": len(test_combos),
        "n_novel_combos": len(novel),
        "novel_combos": [list(c) for c in novel],
        "per_method": per_method,
    }
    write_json(payload, _PROJECT_ROOT / "experiments/results/generalization_report.json")

    lines = ["# Memorization vs Generalization — Held-out Combination Analysis", "",
             "> INTERNAL DIAGNOSTIC (generator mapping, synthetic gold, single seed). Not "
             "evidence of real dashboard-design quality.", "",
             f"- train `(domain, task_type)` combos: {len(train_combos)}",
             f"- test combos: {len(test_combos)}",
             f"- **novel** combos (in test, not train): {len(novel)} → {[list(c) for c in novel]}", "",
             "| method | seen acc (n) | novel acc (n) |", "| --- | --- | --- |"]
    for method in METHOD_RUNS:
        m = per_method.get(method, {})
        if m.get("status") == "no_predictions":
            lines.append(f"| {method} | (no predictions) | |")
            continue
        lines.append(f"| {method} | {m['seen_acc']} ({m['seen_n']}) | {m['novel_acc']} ({m['novel_n']}) |")
    lines += ["", "> A large seen≫novel gap suggests memorization of seen combinations; "
              "comparable accuracy suggests generalization of the mapping. Interpret with the "
              "small-n caveat and only as an internal diagnostic."]
    if not novel:
        lines += ["", "> **Finding:** the synthetic generator emits a closed set of "
                  "`(domain, task_type)` combinations — the test split contains **no** "
                  "held-out combinations, so held-out-combination generalization **cannot be "
                  "tested on synthetic data**. Use the independent `benchmark_v1` (broader, "
                  "independent combinations) for this — an approval-gated inference pass."]
    (_PROJECT_ROOT / "experiments/results/generalization_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"Generalization analysis: {len(novel)} novel combos -> "
          "experiments/results/generalization_report.md")


if __name__ == "__main__":
    main()
