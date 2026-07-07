"""Generate source-conditioned v2 candidate items into the raw staging folder.

Candidates are written to ``data/frozen/dashboard_v2/raw_batches/`` FIRST. The
frozen train/val/internal_test files are produced only later by
``freeze_dataset_v2.py`` (after validation). This script never touches the frozen
files.

Sample/dry-run first: the default ``--mode sample`` is fully offline and makes no
paid API calls. Use ``--dry-run`` to preview counts without writing anything.

Usage:
    python experiments/scripts/generate_dataset_v2.py --n 24 --seed 42
    python experiments/scripts/generate_dataset_v2.py --n 24 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.synth_generator_v2 import GENERATOR_VERSION, generate_candidates
from src.utils.io import write_jsonl

DEFAULT_OUT = "data/frozen/dashboard_v2/raw_batches"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate v2 candidate items (staging only)")
    p.add_argument("--n", type=int, default=24, help="number of candidate items")
    p.add_argument("--seed", type=int, default=42, help="deterministic generation seed")
    p.add_argument("--mode", default="sample", choices=("sample", "api"),
                   help="'sample' = offline (default); 'api' is not implemented")
    p.add_argument("--dry-run", action="store_true",
                   help="generate and report only; do not write any file")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = generate_candidates(n=args.n, seed=args.seed, mode=args.mode)

    domains = Counter((it["brief"].get("extra", {}) or {}).get("domain") for it in items)

    print("=" * 56)
    print("DATASET V2 — CANDIDATE GENERATION (staging)")
    print("=" * 56)
    print(f"  generator     : {GENERATOR_VERSION}")
    print(f"  mode          : {args.mode}")
    print(f"  n candidates  : {len(items)}")
    print(f"  domains       : {dict(domains)}")

    if args.dry_run:
        print("  DRY-RUN: nothing written.")
        print("=" * 56)
        return

    out_dir = (_PROJECT_ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_path = out_dir / f"batch_seed{args.seed}_n{args.n}.jsonl"
    write_jsonl(items, batch_path)
    print(f"  written       : {batch_path}")
    print("  NOTE: frozen files are NOT produced here — run freeze_dataset_v2.py.")
    print("=" * 56)


if __name__ == "__main__":
    main()
