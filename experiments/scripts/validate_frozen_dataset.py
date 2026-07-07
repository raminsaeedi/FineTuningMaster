"""Validate the frozen v2 dataset and write the report + hashes.

Runs the checks in DATASET_V2_IMPLEMENTATION_PLAN.md §5 over the frozen files and
writes:

    data/frozen/dashboard_v2/validation_report.md
    data/frozen/dashboard_v2/hashes.json   (only if all hard checks pass)

Hard checks (fail => no hashes.json written): JSON parsing, Pydantic schema +
enums, non-empty required fields, duplicate ids, train/val vs eval leakage.
Distributions and fingerprint near-duplicates are reported as diagnostics.

Usage:
    python experiments/scripts/validate_frozen_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.frozen_validation import (
    distributions,
    find_duplicate_fingerprints,
    find_duplicate_ids,
    leakage_report,
    read_jsonl_strict,
    sha256_of_file,
    validate_record,
)
from src.data_pipeline.synth_generator_v2 import GENERATOR_VERSION
from src.utils.io import write_json

FROZEN_FILES = ("train.jsonl", "val.jsonl", "internal_test.jsonl",
                "test_paraphrased.jsonl", "test_missing_info.jsonl")
EVAL_REAL_BRIEFS = "data/eval/real_briefs_v1.jsonl"


def _fmt_dist(title: str, dist: dict) -> str:
    lines = [f"### {title}", ""]
    if not dist:
        lines.append("_(none)_\n")
        return "\n".join(lines)
    for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate frozen dataset v2")
    p.add_argument("--frozen-dir", default="data/frozen/dashboard_v2")
    args = p.parse_args()
    frozen_dir = (_PROJECT_ROOT / args.frozen_dir) if not Path(args.frozen_dir).is_absolute() else Path(args.frozen_dir)

    report: list[str] = ["# Frozen Dataset v2 — Validation Report", "",
                         f"Generator: `{GENERATOR_VERSION}`", ""]
    hard_ok = True
    file_records: dict[str, list[dict]] = {}
    hashes: dict[str, dict] = {}

    # Per-file: parse, schema/enum/non-empty, duplicate ids, hash.
    for fname in FROZEN_FILES:
        path = frozen_dir / fname
        report.append(f"## `{fname}`")
        if not path.exists():
            report.append("- status: **MISSING** (skipped)\n")
            continue
        records, parse_errors = read_jsonl_strict(path)
        file_records[fname] = records
        hashes[fname] = {"sha256": sha256_of_file(path), "n_records": len(records)}

        schema_problems = []
        for i, r in enumerate(records):
            for prob in validate_record(r):
                schema_problems.append(f"  - record {i} (`{r.get('item_id','?')}`): {prob}")
        dup_ids = find_duplicate_ids(records)
        dup_fps = find_duplicate_fingerprints(records)

        report.append(f"- records: {len(records)}")
        report.append(f"- JSON parse errors: {len(parse_errors)}")
        report.append(f"- schema/enum/non-empty problems: {len(schema_problems)}")
        report.append(f"- duplicate item_id: {len(dup_ids)}")
        report.append(f"- duplicate brief fingerprint: {len(dup_fps)} (diagnostic)")
        report.append(f"- sha256: `{hashes[fname]['sha256']}`")
        if parse_errors:
            hard_ok = False
            report += [f"  - {e}" for e in parse_errors[:10]]
        if schema_problems:
            hard_ok = False
            report += schema_problems[:10]
        if dup_ids:
            hard_ok = False
            report.append(f"  - duplicate ids: {dup_ids[:10]}")
        report.append("")

    # Distributions over the trainable + diagnostic splits.
    core = (file_records.get("train.jsonl", []) + file_records.get("val.jsonl", [])
            + file_records.get("internal_test.jsonl", []))
    dist = distributions(core)
    report.append("## Distributions (train + val + internal_test)\n")
    report.append(_fmt_dist("Domain", dist["domain"]))
    report.append(_fmt_dist("task_type", dist["task_type"]))
    report.append(_fmt_dist("chart_type", dist["chart_type"]))

    # Leakage: {train, val} must not overlap {internal_test, real_briefs}.
    train_val = file_records.get("train.jsonl", []) + file_records.get("val.jsonl", [])
    eval_recs = list(file_records.get("internal_test.jsonl", []))
    rb_path = _PROJECT_ROOT / EVAL_REAL_BRIEFS
    if rb_path.exists():
        rb_records, _ = read_jsonl_strict(rb_path)
        # real briefs are DashboardBrief-only; wrap so leakage_report can read .brief
        eval_recs += [{"item_id": r.get("item_id", ""), "brief": r} for r in rb_records]
    leak = leakage_report(train_val, eval_recs)
    report.append("## Leakage check (train/val vs internal_test + real_briefs)\n")
    report.append(f"- item_id overlap: {len(leak['item_id_overlap'])}")
    report.append(f"- fingerprint overlap: {len(leak['fingerprint_overlap'])}")
    if leak["item_id_overlap"] or leak["fingerprint_overlap"]:
        hard_ok = False
        report.append(f"  - overlapping ids: {leak['item_id_overlap'][:10]}")
        report.append(f"  - overlapping fingerprints (ids): {leak['fingerprint_overlap'][:10]}")
    report.append("")

    report.append("## Result\n")
    report.append(f"**Hard checks: {'PASS' if hard_ok else 'FAIL'}**")
    report.append("")
    report.append("> `train.jsonl` and `val.jsonl` are the only trainable files. "
                  "`internal_test.jsonl`, the perturbation sets, "
                  "`data/eval/l1_chart_effectiveness_v1.csv` and "
                  "`data/eval/real_briefs_v1.jsonl` are NEVER used for training.")

    frozen_dir.mkdir(parents=True, exist_ok=True)
    (frozen_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")

    if hard_ok:
        write_json({"generator_version": GENERATOR_VERSION, "files": hashes},
                   frozen_dir / "hashes.json")
        print(f"Validation PASS. Wrote validation_report.md + hashes.json to {frozen_dir}")
    else:
        print(f"Validation FAIL. Wrote validation_report.md to {frozen_dir} (hashes.json NOT written).")
        sys.exit(1)


if __name__ == "__main__":
    main()
