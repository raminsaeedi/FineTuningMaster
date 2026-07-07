"""Freeze validated candidate items from raw_batches into the v2 frozen splits.

Reads every ``*.jsonl`` under ``data/frozen/dashboard_v2/raw_batches/``, drops
schema-invalid candidates and duplicates (by item_id and by brief fingerprint),
assigns the deterministic content-hash split, and writes:

    data/frozen/dashboard_v2/train.jsonl          (split "train")
    data/frozen/dashboard_v2/val.jsonl            (split "val")
    data/frozen/dashboard_v2/internal_test.jsonl  (split "test")

Stored split values remain "train"/"val"/"test" for schema/loader compatibility.

Usage:
    python experiments/scripts/freeze_dataset_v2.py
    python experiments/scripts/freeze_dataset_v2.py --frozen-dir data/frozen/dashboard_v2
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.dataset import load_pool
from src.data_pipeline.frozen_v2 import (
    STORED_SPLIT_TO_FILE,
    bucket_by_split,
    dedupe_by_fingerprint,
    gold_to_record,
)
from src.data_pipeline.frozen_validation import validate_record
from src.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze v2 candidates into deterministic splits")
    p.add_argument("--frozen-dir", default="data/frozen/dashboard_v2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frozen_dir = (_PROJECT_ROOT / args.frozen_dir) if not Path(args.frozen_dir).is_absolute() else Path(args.frozen_dir)
    raw_dir = frozen_dir / "raw_batches"
    batch_paths = sorted(raw_dir.glob("*.jsonl"))
    if not batch_paths:
        raise SystemExit(f"No candidate batches found under {raw_dir}. Run generate_dataset_v2.py first.")

    # load_pool assigns content-based ids (kept if present) + hash split, and
    # de-duplicates by item_id across all batches.
    pool = load_pool(batch_paths)

    # Drop schema-invalid candidates before freezing (report, no silent drop).
    valid, invalid = [], []
    for it in pool:
        problems = validate_record(gold_to_record(it))
        (valid if not problems else invalid).append((it, problems))
    kept_items = [it for it, _ in valid]

    # Second de-dup pass: distinct ids describing the same brief content.
    deduped, fp_dropped = dedupe_by_fingerprint(kept_items)

    buckets = bucket_by_split(deduped)
    frozen_dir.mkdir(parents=True, exist_ok=True)
    for split, items in buckets.items():
        records = [gold_to_record(it) for it in items]
        write_jsonl(records, frozen_dir / STORED_SPLIT_TO_FILE[split])

    counts = Counter({s: len(v) for s, v in buckets.items()})
    print("=" * 56)
    print("DATASET V2 FROZEN")
    print("=" * 56)
    print(f"  raw batches       : {len(batch_paths)}")
    print(f"  pool (unique ids) : {len(pool)}")
    print(f"  schema-invalid    : {len(invalid)} (dropped)")
    print(f"  fingerprint dups  : {len(fp_dropped)} (dropped)")
    print(f"  train/val/internal_test : {counts['train']}/{counts['val']}/{counts['test']}")
    for it, problems in invalid[:5]:
        print(f"    ! dropped {it.item_id}: {problems[0]}")
    print("  NOTE: run validate_frozen_dataset.py to produce hashes.json + report.")
    print("=" * 56)


if __name__ == "__main__":
    main()
