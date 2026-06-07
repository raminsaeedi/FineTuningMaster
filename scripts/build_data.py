"""Build the processed dataset with deterministic, hash-based splits.

Reads the gold pool (``data/gold.jsonl`` when present, otherwise the superseded
``data/raw_legacy/{train,val,test}.jsonl`` fallback), merges it into one pool,
assigns a content-based ``item_id`` and a hash split to every example, and writes
the canonical splits to data/processed/.

The previous 80/10/10 file boundaries are intentionally discarded: the split is
now a deterministic function of each item's content, which guarantees that test
items never leak into training as the dataset grows. The synthetic gold data is
regenerable, so re-deriving membership here is safe.

Usage:
    python scripts/build_data.py
    python scripts/build_data.py --raw-dir data --out-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import compute_item_id, load_pool
from src.data.splits import assign_split
from src.utils.io import read_jsonl, write_jsonl

RAW_FILES = ("train.jsonl", "val.jsonl", "test.jsonl")
PERTURBATION_FILES = ("test_paraphrased.jsonl", "test_missing_info.jsonl")


def _gold_to_record(item) -> dict:
    # mode="json" serialises TaskType/ChartType enums to plain strings.
    return {
        "item_id": item.item_id,
        "split": item.split,
        "brief": item.brief.model_dump(mode="json"),
        "recommendation": item.recommendation.model_dump(mode="json"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build processed dataset with hash splits")
    p.add_argument("--raw-dir", default="data")
    p.add_argument("--out-dir", default="data/processed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = (_PROJECT_ROOT / args.raw_dir) if not Path(args.raw_dir).is_absolute() else Path(args.raw_dir)
    out_dir = (_PROJECT_ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    # Prefer the principled generated pool (data/gold.jsonl) when present;
    # otherwise fall back to the superseded legacy splits under data/raw_legacy/.
    legacy_dir = raw_dir / "raw_legacy"
    gold_path = raw_dir / "gold.jsonl"
    if gold_path.exists():
        raw_paths = [gold_path]
        print(f"Using generated gold pool: {gold_path}")
    else:
        raw_paths = [legacy_dir / name for name in RAW_FILES]
        print(f"Using legacy raw files under: {legacy_dir}")
    pool = load_pool(raw_paths)
    if not pool:
        raise SystemExit(f"No gold records found under {raw_dir}")

    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for item in pool:
        buckets[item.split].append(_gold_to_record(item))

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, records in buckets.items():
        write_jsonl(records, out_dir / f"{split}.jsonl")

    # Optional perturbation pass-through: align by file order to the original
    # test file and re-key each row to its base item's content-based id, so the
    # robustness metric can pair original and perturbed predictions.
    base_test = read_jsonl(legacy_dir / "test.jsonl") if (legacy_dir / "test.jsonl").exists() else []
    base_ids = [compute_item_id(dict(r.get("brief", {}))) for r in base_test]
    perturbation_counts: dict[str, int] = {}
    for fname in PERTURBATION_FILES:
        src_path = raw_dir / fname
        if not src_path.exists():
            continue
        rows = read_jsonl(src_path)
        out_rows = []
        for i, row in enumerate(rows):
            base_id = base_ids[i] if i < len(base_ids) else compute_item_id(dict(row.get("brief", {})))
            out_rows.append({
                "item_id": base_id,
                "split": assign_split(base_id),
                "brief": row.get("brief", {}),
                "recommendation": row.get("recommendation", {}),
            })
        write_jsonl(out_rows, out_dir / fname)
        perturbation_counts[fname] = len(out_rows)

    counts = Counter({split: len(records) for split, records in buckets.items()})
    print("=" * 56)
    print("PROCESSED DATASET BUILT")
    print("=" * 56)
    print(f"  Source dir : {raw_dir}")
    print(f"  Output dir : {out_dir}")
    print(f"  Pool size  : {len(pool)} unique items")
    print(f"  train/val/test : {counts['train']}/{counts['val']}/{counts['test']}")
    for fname, n in perturbation_counts.items():
        print(f"  {fname}: {n}")
    print("=" * 56)


if __name__ == "__main__":
    main()
