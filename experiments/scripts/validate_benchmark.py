"""Validate `data/eval/benchmark_v1.jsonl` and write the benchmark report.

Runs the ten required checks and writes
`experiments/results/benchmark_dataset_report.md`. No model inference.

    python experiments/scripts/validate_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.benchmark_validation import (
    chart_type_coverage,
    distributions,
    evidence_split,
    label_lineage_check,
    scoring_split,
    source_leakage,
    validate_item,
)
from src.data_pipeline.frozen_validation import read_jsonl_strict
from src.data_pipeline.synth_generator import TASK_CHART
from src.evaluation.l1_independent import DEFAULT_L1_CSV, load_effective_sets
from src.evaluation.metrics.base import chart_token
from src.utils.io import read_jsonl

BENCHMARK = "data/eval/benchmark_v1.jsonl"
TRAIN_FILES = (
    "data/frozen/dashboard_v2/train.jsonl",
    "data/frozen/dashboard_v2/val.jsonl",
    "data/processed/train.jsonl",
)
LOCK = ("EVALUATION-ONLY (benchmark lock): never use for training, augmentation, prompt "
        "optimization, retriever tuning, hyperparameter tuning, or model selection.")


def _dist_md(title: str, dist: dict) -> str:
    lines = [f"### {title}", ""]
    for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        lines.append(f"- `{k}`: {v}")
    return "\n".join(lines) + "\n"


def _generator_sets() -> dict:
    out = {}
    for task, (primary, alts) in TASK_CHART.items():
        out[task] = {chart_token(primary)} | {chart_token(a) for a in alts}
    return out


def _train_briefs() -> list:
    briefs = []
    for rel in TRAIN_FILES:
        p = _PROJECT_ROOT / rel
        if p.exists():
            for r in read_jsonl(p):
                if isinstance(r.get("brief"), dict):
                    briefs.append(r["brief"])
    return briefs


def main() -> None:
    bpath = _PROJECT_ROOT / BENCHMARK
    items, parse_errors = read_jsonl_strict(bpath)
    effective_sets = load_effective_sets(_PROJECT_ROOT / DEFAULT_L1_CSV)
    gen_sets = _generator_sets()
    train_briefs = _train_briefs()

    schema_problems = {it.get("benchmark_id", f"row{i}"): probs
                       for i, it in enumerate(items) if (probs := validate_item(it))}
    dist = distributions(items)
    coverage = chart_type_coverage(items)
    evidence = evidence_split(items)
    scoring = scoring_split(items)
    leaked = source_leakage(items, train_briefs)
    lineage = label_lineage_check(items, effective_sets, gen_sets)

    hard_ok = (not parse_errors and not schema_problems and not leaked
               and lineage["label_source_ok"] and not lineage["l1_mismatch"])

    md = [
        "# Benchmark Dataset Report — `benchmark_v1`", "",
        f"> {LOCK}", "",
        f"**Hard checks: {'PASS' if hard_ok else 'FAIL'}**", "",
        "## 1. Item count", "",
        f"- items: {len(items)}",
        f"- JSON parse errors: {len(parse_errors)}",
        f"- schema-invalid items: {len(schema_problems)}", "",
        _dist_md("2. Domain distribution", dist["domain"]),
        _dist_md("3. Task-type distribution", dist["task_type"]),
        _dist_md("4. Chart-label distribution (acceptable_chart_types)", dist["chart_label"]),
        "## 5. Chart-type coverage", "",
        f"- covered chart types ({len(coverage['covered'])}): {coverage['covered']}",
        f"- not covered ({len(coverage['not_covered'])}): {coverage['not_covered']}", "",
        "## 6-7. Auto-scorable vs human-eval", "",
        f"- auto-scorable (task_type L1-covered): {scoring['auto_scorable']}",
        f"- human-eval suitable: {scoring['human_eval']}",
        f"- human-eval-ONLY (not auto-scorable): {scoring['human_eval_only']}", "",
        "## 8. Evidence strength", "",
        f"- strong (real_public + literature_L1): {evidence['strong']}",
        f"- weak (realistic_manual or manual_expert): {evidence['weak']}", "",
        "## 9. Source/provenance leakage vs training", "",
        f"- training briefs checked: {len(train_briefs)}",
        f"- benchmark items colliding with training (fingerprint): {len(leaked)} {leaked}", "",
        "## 10. Label-lineage leakage vs synthetic generator", "",
        f"- all labels sourced from literature_L1/manual_expert (never generator): "
        f"{lineage['label_source_ok']}",
        f"- literature_L1 items mismatching the independent L1 table: "
        f"{len(lineage['l1_mismatch'])} {lineage['l1_mismatch']}",
        f"- acceptable sets identical to the generator's TASK_CHART set (informational): "
        f"{len(lineage['identical_to_generator_set'])} {lineage['identical_to_generator_set']}", "",
        "> `task_type` is assigned independently via `data/eval/task_crosswalk.yaml`; "
        "`acceptable_chart_types` come from the independent L1 literature table (covered "
        "tasks) or documented expert judgment (uncovered tasks). Neither is derived from "
        "`TASK_CHART`/`KEYWORD_TASK`. Overlap with the generator set is possible but not "
        "derivation; identity is reported above for transparency.", "",
    ]
    if schema_problems:
        md += ["## Schema problems", ""]
        for bid, probs in list(schema_problems.items())[:20]:
            md.append(f"- `{bid}`: {probs}")
        md.append("")

    out = _PROJECT_ROOT / "experiments/results/benchmark_dataset_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"Benchmark validation {'PASS' if hard_ok else 'FAIL'} -> {out}")
    if not hard_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
