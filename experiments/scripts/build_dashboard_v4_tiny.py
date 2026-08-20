"""Build a reproducible, GPU-friendly derivative of frozen dashboard_v4.

The parent dataset is never modified.  This development-only child dataset has
100 train, 50 validation, and 50 canonical held-out test examples.  Sampling is
deterministic random stratification by chart type, so every chart type available
in a source split occurs several times.  A separate 50-item sports holdout is
reserved for cross-domain evaluation and is excluded from tiny train/validation.

Examples:

    python experiments/scripts/build_dashboard_v4_tiny.py
    python experiments/scripts/build_dashboard_v4_tiny.py --verify
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.frozen_validation import (  # noqa: E402
    find_duplicate_fingerprints,
    find_duplicate_ids,
    leakage_report,
    sha256_of_file,
    validate_record,
)


DATASET_VERSION = "dashboard_v4_tiny_1"
PARENT_DATASET_VERSION = "dashboard_v4_1"
DEFAULT_SEED = 20_260_820
DEFAULT_SOURCE = Path("data/frozen/dashboard_v4")
DEFAULT_OUT = Path("data/frozen/dashboard_v4_tiny")

# Explicit allow-list.  This prevents broad text matching from accidentally
# classifying unrelated database names as sports examples.
SPORTS_DATABASE_IDS = frozenset({
    "baseball_1",
    "game_1",
    "game_injury",
    "match_season",
    "soccer_1",
    "soccer_2",
    "sports_competition",
    "university_basketball",
})

HUMAN_EVAL_FIELDS = (
    "item_id",
    "input_brief",
    "source_evidence",
    "provenance",
    "reviewer_id",
    "goal_fidelity",
    "chart_appropriate",
    "encoding_correct",
    "source_fidelity",
    "overall_rating",
    "error_category",
    "review_comment",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _PROJECT_ROOT / value


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    _write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def _chart_type(record: dict) -> str:
    mappings = record.get("recommendation", {}).get("kpi_chart_mapping") or []
    if not mappings or not isinstance(mappings[0], dict):
        raise ValueError(f"{record.get('item_id', '?')}: no primary chart mapping")
    chart = mappings[0].get("chart_type")
    if not isinstance(chart, str) or not chart:
        raise ValueError(f"{record.get('item_id', '?')}: missing chart_type")
    return chart


def _task_type(record: dict) -> str:
    mappings = record.get("recommendation", {}).get("kpi_chart_mapping") or []
    return str(mappings[0].get("task_type", "unknown")) if mappings else "unknown"


def _provenance(record: dict) -> dict:
    value = record.get("brief", {}).get("extra", {}).get("provenance", {})
    return value if isinstance(value, dict) else {}


def _db_id(record: dict) -> str:
    return str(_provenance(record).get("db_id") or "")


def _is_sports(record: dict) -> bool:
    return _db_id(record) in SPORTS_DATABASE_IDS


def _copy_with_split(record: dict, split: str) -> dict:
    copied = copy.deepcopy(record)
    copied["split"] = split
    return copied


def _stratified_sample(
    records: list[dict],
    *,
    size: int,
    rng: random.Random,
    minimum_per_chart: int,
) -> list[dict]:
    """Random sample with a coverage floor for every available chart type."""
    if size <= 0:
        raise ValueError("sample size must be positive")
    if len(records) < size:
        raise ValueError(f"need {size} records, found only {len(records)}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[_chart_type(record)].append(record)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    for chart in sorted(groups):
        candidates = sorted(groups[chart], key=lambda item: str(item["item_id"]))
        n_take = min(minimum_per_chart, len(candidates))
        for record in rng.sample(candidates, n_take):
            selected.append(record)
            selected_ids.add(str(record["item_id"]))

    if len(selected) > size:
        raise ValueError(
            f"coverage floor selects {len(selected)} records, exceeding requested {size}"
        )

    remaining = [record for record in records if str(record["item_id"]) not in selected_ids]
    remaining = sorted(remaining, key=lambda item: str(item["item_id"]))
    selected.extend(rng.sample(remaining, size - len(selected)))
    return sorted(selected, key=lambda item: str(item["item_id"]))


def _all_parent_records(source: Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    by_split = {split: _read_jsonl(source / f"{split}.jsonl") for split in ("train", "val", "test")}
    by_id: dict[str, dict] = {}
    for records in by_split.values():
        for record in records:
            item_id = str(record.get("item_id") or "")
            if not item_id or item_id in by_id:
                raise ValueError(f"parent contains missing or duplicate item_id: {item_id!r}")
            by_id[item_id] = record
    return by_split, by_id


def _histogram(records: Iterable[dict], fn) -> dict[str, int]:
    return dict(sorted(Counter(fn(record) for record in records).items()))


def _human_eval_source_evidence(record: dict) -> dict:
    mapping = (record.get("recommendation", {}).get("kpi_chart_mapping") or [{}])[0]
    provenance = _provenance(record)
    return {
        "task_type": mapping.get("task_type"),
        "chart_type": mapping.get("chart_type"),
        "encoding": mapping.get("encoding"),
        "source_record_id": provenance.get("source_record_id"),
        "source_group_id": provenance.get("source_group_id"),
        "db_id": provenance.get("db_id"),
        "source_query": provenance.get("nl_query"),
        "source_vql": (provenance.get("vis_query") or {}).get("VQL"),
    }


def _write_human_eval_template(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_EVAL_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({
                "item_id": record["item_id"],
                "input_brief": json.dumps(record["brief"], ensure_ascii=False),
                "source_evidence": json.dumps(_human_eval_source_evidence(record), ensure_ascii=False),
                "provenance": json.dumps(_provenance(record), ensure_ascii=False),
            })


def _validation_payload(
    splits: dict[str, list[dict]],
    *,
    source_index: dict[str, dict],
) -> dict:
    all_records = [record for records in splits.values() for record in records]
    schema_problems = {
        name: [
            f"{record.get('item_id', '?')}: {problem}"
            for record in records
            for problem in validate_record(record)
        ]
        for name, records in splits.items()
    }
    duplicate_ids = find_duplicate_ids(all_records)
    duplicate_briefs = find_duplicate_fingerprints(all_records)
    train_val = splits["train"] + splits["val"]
    eval_records = splits["test"] + splits["sports_test"]
    heldout_overlap = leakage_report(train_val, eval_records)
    test_sports_overlap = leakage_report(splits["test"], splits["sports_test"])

    parent_mismatches: list[str] = []
    for record in all_records:
        item_id = str(record.get("item_id") or "")
        parent = source_index.get(item_id)
        if parent is None:
            parent_mismatches.append(f"{item_id}: absent from dashboard_v4 parent")
            continue
        expected = copy.deepcopy(parent)
        expected["split"] = record.get("split")
        if expected != record:
            parent_mismatches.append(f"{item_id}: content differs from parent")

    sports_ids = [_db_id(record) for record in splits["sports_test"]]
    non_sports_train_val = [
        record["item_id"]
        for record in train_val
        if _is_sports(record)
    ]

    checks = {
        "all_jsonl_records_schema_valid": not any(schema_problems.values()),
        "duplicate_item_ids_zero": not duplicate_ids,
        "duplicate_brief_fingerprints_zero": not duplicate_briefs,
        "train_val_vs_evaluation_item_id_overlap_zero": not heldout_overlap["item_id_overlap"],
        "train_val_vs_evaluation_fingerprint_overlap_zero": not heldout_overlap["fingerprint_overlap"],
        "test_vs_sports_item_id_overlap_zero": not test_sports_overlap["item_id_overlap"],
        "test_vs_sports_fingerprint_overlap_zero": not test_sports_overlap["fingerprint_overlap"],
        "all_records_preserved_from_parent_except_split_label": not parent_mismatches,
        "sports_test_uses_only_allowlisted_sports_databases": all(db in SPORTS_DATABASE_IDS for db in sports_ids),
        "train_val_contains_no_allowlisted_sports_examples": not non_sports_train_val,
        "train_size_is_100": len(splits["train"]) == 100,
        "val_size_is_50": len(splits["val"]) == 50,
        "test_size_is_50": len(splits["test"]) == 50,
        "sports_test_size_is_50": len(splits["sports_test"]) == 50,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "counts": {name: len(records) for name, records in splits.items()},
        "schema_problems": schema_problems,
        "duplicate_item_ids": duplicate_ids,
        "duplicate_brief_fingerprints": duplicate_briefs,
        "train_val_vs_evaluation_leakage": heldout_overlap,
        "test_vs_sports_leakage": test_sports_overlap,
        "parent_mismatches": parent_mismatches,
        "sports_examples_in_train_or_val": non_sports_train_val,
        "checks": checks,
    }


def _dataset_card() -> str:
    return """# Dataset Card — dashboard_v4_tiny_1

## Purpose

`dashboard_v4_tiny_1` is a development-only, reproducible Kaggle/GPU smoke dataset derived from frozen `dashboard_v4_1`. It is not a replacement for the thesis dataset and must not be used for final thesis claims.

## Splits

- Train: 100 records from parent `train.jsonl`.
- Validation: 50 records from parent `val.jsonl`.
- Canonical test: 50 records from parent `test.jsonl`.
- Sports test: 50 cross-domain records selected from explicit sports database identifiers across the parent dataset. It is excluded from tiny Train/Validation and is not the parent dataset's canonical test split.

Sampling uses seed `20260820`, random selection inside chart-type strata, and a minimum of three examples per available chart type. When a source split contains fewer than three items of a chart type, all available items are retained. The parent canonical test has only five chart types; therefore the tiny canonical test has only those five chart types. Train and Validation cover all fourteen chart types present in their parent splits.

## Integrity

All record content is copied from the immutable parent dataset. Only `split` is relabeled to `sports_test` for the cross-domain evaluation file. `manifest.json`, `hashes.json`, and the files in `reports/` record lineage, deterministic sampling, validation, distributions, leakage checks, and SHA-256 hashes.

## Human evaluation

`human_eval_test_items_50.csv` and `human_eval_sports_test_items_50.csv` are blank reviewer templates. They contain input briefs and source-backed evidence, but no human ratings. Fill them only after model predictions are generated.

## Use

Train only with `train.jsonl`; use `val.jsonl` for development validation; use `test.jsonl` for the small in-domain check; use `sports_test.jsonl` only for the separate cross-domain diagnostic.
"""


def _human_eval_guide() -> str:
    return """# Tiny Dataset Human-Evaluation Guide

Use one CSV per evaluation condition. Each row is one fixed input brief.

Score these fields after reviewing model output:

1. `goal_fidelity` — recommendation addresses stated user goal.
2. `chart_appropriate` — selected chart fits task and variables.
3. `encoding_correct` — x/y, aggregation, grouping, filters, sorting match source evidence.
4. `source_fidelity` — no unsupported data claims or invented source facts.
5. `overall_rating` — overall usefulness.

Use reviewer IDs, record an error category, and explain disagreements in `review_comment`. Do not use either human-evaluation CSV for training.
"""


def _hash_manifest(out: Path) -> dict:
    files = [
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "sports_test.jsonl",
        "human_eval_test_items_50.csv",
        "human_eval_sports_test_items_50.csv",
        "human_evaluation_guide.md",
        "schema.json",
        "manifest.json",
        "dataset_card.md",
        "reports/sampling_report.json",
        "reports/validation_report.json",
        "reports/leakage_report.json",
        "reports/distribution_report.json",
    ]
    return {
        "hash_algorithm": "SHA-256",
        "dataset_version": DATASET_VERSION,
        "parent_dataset_version": PARENT_DATASET_VERSION,
        "files": {
            name: {
                "sha256": sha256_of_file(out / name),
                "bytes": (out / name).stat().st_size,
            }
            for name in files
        },
        "line_endings": "LF",
    }


def build(*, source: Path, out: Path, seed: int) -> None:
    if out.exists():
        raise FileExistsError(
            f"output already exists: {out}. Verify it with --verify or choose --out."
        )
    required = [source / f"{split}.jsonl" for split in ("train", "val", "test")]
    required.extend([source / "schema.json", source / "manifest.json", source / "hashes.json"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing parent files:\n- " + "\n- ".join(missing))

    parent_by_split, source_index = _all_parent_records(source)
    rng = random.Random(seed)

    non_sports = {
        split: [record for record in records if not _is_sports(record)]
        for split, records in parent_by_split.items()
    }
    selected_train = _stratified_sample(
        non_sports["train"], size=100, rng=rng, minimum_per_chart=3
    )
    selected_val = _stratified_sample(
        non_sports["val"], size=50, rng=rng, minimum_per_chart=3
    )
    selected_test = _stratified_sample(
        non_sports["test"], size=50, rng=rng, minimum_per_chart=3
    )

    sports_candidates = [
        record
        for records in parent_by_split.values()
        for record in records
        if _is_sports(record)
    ]
    selected_sports = _stratified_sample(
        sports_candidates, size=50, rng=rng, minimum_per_chart=3
    )

    splits = {
        "train": [_copy_with_split(record, "train") for record in selected_train],
        "val": [_copy_with_split(record, "val") for record in selected_val],
        "test": [_copy_with_split(record, "test") for record in selected_test],
        "sports_test": [_copy_with_split(record, "sports_test") for record in selected_sports],
    }
    validation = _validation_payload(splits, source_index=source_index)
    if validation["status"] != "PASS":
        raise RuntimeError("pre-write validation failed:\n" + json.dumps(validation, indent=2))

    out.mkdir(parents=True, exist_ok=False)
    reports = out / "reports"
    reports.mkdir()
    for name, records in splits.items():
        _write_jsonl(out / f"{name}.jsonl", records)

    shutil.copyfile(source / "schema.json", out / "schema.json")
    _write_human_eval_template(out / "human_eval_test_items_50.csv", splits["test"])
    _write_human_eval_template(out / "human_eval_sports_test_items_50.csv", splits["sports_test"])
    _write_text(out / "human_evaluation_guide.md", _human_eval_guide())
    _write_text(out / "dataset_card.md", _dataset_card())

    parent_split_by_id = {
        str(record["item_id"]): split
        for split, records in parent_by_split.items()
        for record in records
    }
    sampling_report = {
        "status": "PASS",
        "dataset_version": DATASET_VERSION,
        "parent_dataset_version": PARENT_DATASET_VERSION,
        "selection_seed": seed,
        "algorithm": "random stratification by chart type; sorted item ids before sampling",
        "minimum_per_available_chart_type": 3,
        "sports_database_ids": sorted(SPORTS_DATABASE_IDS),
        "canonical_source_splits": {
            "train": "parent train.jsonl minus allowlisted sports examples",
            "val": "parent val.jsonl minus allowlisted sports examples",
            "test": "parent test.jsonl minus allowlisted sports examples",
        },
        "sports_test_source": "all parent splits, allowlisted sports examples only",
        "sports_parent_split_counts": _histogram(
            splits["sports_test"], lambda record: parent_split_by_id[str(record["item_id"])]
        ),
        "selected_item_ids": {
            name: [record["item_id"] for record in records]
            for name, records in splits.items()
        },
        "parent_file_sha256": {
            path.name: sha256_of_file(path)
            for path in required
            if path.is_file()
        },
    }
    distribution_report = {
        "status": "PASS",
        "chart_types": {
            name: _histogram(records, _chart_type)
            for name, records in splits.items()
        },
        "task_types": {
            name: _histogram(records, _task_type)
            for name, records in splits.items()
        },
        "sports_database_ids": _histogram(splits["sports_test"], _db_id),
        "parent_test_chart_types": _histogram(parent_by_split["test"], _chart_type),
        "note": (
            "The canonical parent test has five chart types only. Sports candidates "
            "contain bar, line, pie, and stacked_bar; no allowlisted sports scatter record exists."
        ),
    }
    leakage = {
        "status": "PASS",
        "train_val_vs_test_and_sports": validation["train_val_vs_evaluation_leakage"],
        "test_vs_sports": validation["test_vs_sports_leakage"],
        "sports_examples_in_train_or_val": validation["sports_examples_in_train_or_val"],
    }
    _write_json(reports / "sampling_report.json", sampling_report)
    _write_json(reports / "validation_report.json", validation)
    _write_json(reports / "leakage_report.json", leakage)
    _write_json(reports / "distribution_report.json", distribution_report)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "parent_dataset_version": PARENT_DATASET_VERSION,
        "status": "PASS_DASHBOARD_V4_TINY_READY_FOR_KAGGLE_SMOKE",
        "purpose": "development-only reproducible small-GPU pipeline validation",
        "not_for_thesis_final_results": True,
        "parent_frozen_dir": str(source.relative_to(_PROJECT_ROOT).as_posix()),
        "selection": {
            "seed": seed,
            "algorithm": sampling_report["algorithm"],
            "minimum_per_available_chart_type": 3,
            "sports_database_ids": sorted(SPORTS_DATABASE_IDS),
        },
        "counts": {name: len(records) for name, records in splits.items()},
        "checks": validation["checks"],
        "lineage": {
            "parent_dataset_unchanged": True,
            "records_copied_without_llm_generation": True,
            "sports_split_relabel_only": True,
            "sports_test_is_cross_domain_diagnostic_not_canonical_parent_test": True,
        },
        "not_for_training": [
            "test.jsonl",
            "sports_test.jsonl",
            "human_eval_test_items_50.csv",
            "human_eval_sports_test_items_50.csv",
        ],
        "reports": {
            "sampling": "reports/sampling_report.json",
            "validation": "reports/validation_report.json",
            "leakage": "reports/leakage_report.json",
            "distribution": "reports/distribution_report.json",
        },
        "hashes_file": "hashes.json",
        "line_endings": "LF",
    }
    _write_json(out / "manifest.json", manifest)
    _write_json(out / "hashes.json", _hash_manifest(out))
    print(f"PASS_DASHBOARD_V4_TINY_BUILT: {out}")


def verify(*, source: Path, out: Path) -> None:
    required = [
        out / "train.jsonl",
        out / "val.jsonl",
        out / "test.jsonl",
        out / "sports_test.jsonl",
        out / "manifest.json",
        out / "hashes.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing tiny-dataset files:\n- " + "\n- ".join(missing))

    _, source_index = _all_parent_records(source)
    splits = {
        name: _read_jsonl(out / f"{name}.jsonl")
        for name in ("train", "val", "test", "sports_test")
    }
    validation = _validation_payload(splits, source_index=source_index)
    if validation["status"] != "PASS":
        raise RuntimeError("dataset validation failed:\n" + json.dumps(validation, indent=2))

    hashes = json.loads((out / "hashes.json").read_text(encoding="utf-8"))
    hash_problems = []
    for name, expected in (hashes.get("files") or {}).items():
        path = out / name
        if not path.exists():
            hash_problems.append(f"missing hashed file: {name}")
            continue
        actual = sha256_of_file(path)
        if actual != expected.get("sha256"):
            hash_problems.append(f"hash mismatch: {name}")
    if hash_problems:
        raise RuntimeError("integrity check failed:\n- " + "\n- ".join(hash_problems))
    print(f"PASS_DASHBOARD_V4_TINY_VERIFIED: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or verify tiny dashboard_v4 derivative")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _resolve(args.source)
    out = _resolve(args.out)
    if args.verify:
        verify(source=source, out=out)
    else:
        build(source=source, out=out, seed=args.seed)


if __name__ == "__main__":
    main()
