"""Generate a larger, principled synthetic gold dataset.

    python scripts/generate_dataset.py --n 600

Writes data/gold.jsonl ({brief, recommendation} per line) with CORRECT
task->chart labels grounded in visualization principles. Then run
`python scripts/build_data.py` to assign deterministic hash splits.

This replaces the previous near-random labels; the old data/{train,val,test}.jsonl
are left untouched but build_data.py prefers data/gold.jsonl when it exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.synth_generator import generate_dataset
from src.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate principled synthetic gold data")
    p.add_argument("--n", type=int, default=600, help="Number of items to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/gold.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = _PROJECT_ROOT / out

    items = generate_dataset(n=args.n, base_seed=args.seed)
    write_jsonl(items, out)

    print("=" * 56)
    print("SYNTHETIC GOLD DATASET GENERATED")
    print("=" * 56)
    print(f"  Items  : {len(items)}")
    print(f"  Output : {out}")
    print("  Next   : python scripts/build_data.py")
    print("=" * 56)


if __name__ == "__main__":
    main()
