"""Build nvBench candidate GoldItems into a staging directory (never frozen).

Maps the registered nvBench source into train/val augmentation candidates and
writes accepted/rejected records plus full provenance and distribution reports.
Writes ONLY to the user-specified output directory and refuses to write inside
any frozen dataset.

Usage:
    python experiments/scripts/build_nvbench_candidates.py --dry-run
    python experiments/scripts/build_nvbench_candidates.py \
        --out data/staging/nvbench_pilot --limit 100 --seed 42 \
        --one-query-per-group --stratify-by-chart
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder  # noqa: E402
from src.data_pipeline.frozen_validation import sha256_of_file  # noqa: E402
from src.data_pipeline.nvbench_source import (  # noqa: E402
    apply_limit,
    item_chart,
    select_one_per_group,
)
from src.utils.io import write_json, write_jsonl  # noqa: E402

DEFAULT_NVBENCH_JSON = "data/raw_external/nvbench/extracted/nvBench-main/NVBench.json"
DEFAULT_CACHE_ROOT = "data/cache_external/nvbench/databases"
DEFAULT_CACHE_MANIFEST = "data/cache_external/nvbench/cache_manifest.json"
DEFAULT_MAPPING = "src/config/data/nvbench_mapping.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build nvBench candidate GoldItems.")
    p.add_argument("--out", default=None, help="Staging output directory (required unless --dry-run).")
    p.add_argument("--nvbench-json", default=DEFAULT_NVBENCH_JSON)
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--mapping", default=DEFAULT_MAPPING)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--one-query-per-group", action="store_true")
    p.add_argument("--stratify-by-chart", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _record(item) -> dict:
    return {
        "item_id": item.item_id,
        "split": item.split,
        "brief": item.brief.model_dump(mode="json"),
        "recommendation": item.recommendation.model_dump(mode="json"),
    }


def _select(items, args):
    if args.one_query_per_group:
        items = select_one_per_group(items, seed=args.seed)
    items = apply_limit(items, args.limit, stratify_by_chart=args.stratify_by_chart, seed=args.seed)
    return items


def _distributions(items) -> dict:
    return {
        "n_items": len(items),
        "chart_distribution": dict(collections.Counter(item_chart(it) for it in items)),
        "task_distribution": dict(
            collections.Counter(it.recommendation.kpi_chart_mapping[0].task_type.value for it in items)
        ),
        "split_distribution": dict(collections.Counter(it.split for it in items)),
        "source_distribution": {"nvbench": len(items)},
    }


def _source_hashes(nvbench_json: Path) -> dict:
    hashes = {"nvbench_json_sha256": sha256_of_file(nvbench_json)}
    cache_manifest = _resolve(DEFAULT_CACHE_MANIFEST)
    if cache_manifest.exists():
        cm = json.loads(cache_manifest.read_text(encoding="utf-8"))
        hashes["cache_archive_sha256"] = cm.get("source_archive_sha256")
        hashes["cache_tool_version"] = cm.get("cache_tool_version")
    else:
        hashes["cache_archive_sha256"] = None
    return hashes


def main() -> None:
    args = parse_args()
    nvbench_json = _resolve(args.nvbench_json)
    cache_root = _resolve(args.cache_root)

    builder = NvBenchBuilder(
        nvbench_json_path=nvbench_json,
        cache_root=cache_root if cache_root.exists() else None,
        mapping_path=_resolve(args.mapping),
    )
    result = builder.build()
    selected = _select(result.accepted, args)
    dist = _distributions(selected)

    print(f"nvBench candidates: accepted={len(result.accepted)} rejected={len(result.rejections)}")
    print(f"After selection    : n={len(selected)}")
    print(f"  chart : {dist['chart_distribution']}")
    print(f"  task  : {dist['task_distribution']}")
    print(f"  split : {dist['split_distribution']}")
    print(f"  db_metadata_available={result.stats['db_metadata_available']}")

    if args.dry_run:
        print("[dry-run] nothing written.")
        return

    if not args.out:
        raise SystemExit("--out is required unless --dry-run")
    out_dir = _resolve(args.out)
    if "frozen" in out_dir.parts:
        raise SystemExit(f"refusing to write into a frozen dataset path: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl([_record(it) for it in selected], out_dir / "accepted.jsonl")
    write_jsonl(result.rejections, out_dir / "rejected.jsonl")
    write_json(dist, out_dir / "distribution_report.json")

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": "nvbench",
        "usage_tier": "train_aug",
        "selection": {
            "limit": args.limit,
            "seed": args.seed,
            "one_query_per_group": args.one_query_per_group,
            "stratify_by_chart": args.stratify_by_chart,
        },
        "counts": {
            "accepted_total": len(result.accepted),
            "rejected_total": len(result.rejections),
            "selected": len(selected),
        },
        "mapping_version": result.stats["mapping_version"],
        "task_rule_version": result.stats["task_rule_version"],
        "rejection_reasons": result.stats["rejection_reasons"],
        "db_metadata_available": result.stats["db_metadata_available"],
        "source_hashes": _source_hashes(nvbench_json),
        "distributions": dist,
    }
    write_json(manifest, out_dir / "manifest.json")
    print(f"Wrote candidates + reports to {out_dir}")


if __name__ == "__main__":
    main()
