"""Prepare (extract) the nvBench database cache from the registered archive.

Verifies the registered ``databases.zip`` SHA-256, then extracts it into a
rebuildable cache under ``data/cache_external/nvbench/databases/``. Never writes
inside ``data/raw_external/``.

Usage:
    python experiments/scripts/prepare_nvbench_cache.py --dry-run
    python experiments/scripts/prepare_nvbench_cache.py
    python experiments/scripts/prepare_nvbench_cache.py --force
    python experiments/scripts/prepare_nvbench_cache.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.nvbench_cache import prepare_cache, verify_cache  # noqa: E402

DEFAULT_RAW_DIR = "data/raw_external/nvbench"
DEFAULT_CACHE_BASE = "data/cache_external/nvbench"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare the nvBench database cache.")
    p.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    p.add_argument("--cache-base", default=DEFAULT_CACHE_BASE)
    p.add_argument("--dry-run", action="store_true", help="Verify and plan without writing.")
    p.add_argument("--force", action="store_true", help="Rebuild even if the cache exists.")
    p.add_argument("--verify", action="store_true", help="Read-only check of the existing cache.")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def main() -> None:
    args = parse_args()
    raw_dir = _resolve(args.raw_dir)
    cache_base = _resolve(args.cache_base)
    source_manifest = raw_dir / "source_manifest.json"

    if args.verify:
        result = verify_cache(cache_base)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result.get("ok") else 1)

    if not source_manifest.exists():
        raise SystemExit(f"source manifest not found: {source_manifest}")

    timestamp = datetime.now(timezone.utc).isoformat()
    result = prepare_cache(
        raw_dir=raw_dir,
        source_manifest_path=source_manifest,
        cache_base=cache_base,
        timestamp=timestamp,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
