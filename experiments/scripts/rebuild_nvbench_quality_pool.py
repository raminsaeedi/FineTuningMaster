"""Rebuild ONLY the nvBench quality-tiering pool under the corrected rules.

This is not a pilot: no sampling, no ``accepted.jsonl``, no manual-audit
template. It reruns ``build_quality_pool`` (unchanged) over the full
technically-valid nvBench candidate pool and writes the tier_a/b/c diagnostic
artifacts + a summary, so downstream pilot builds (v4/v5/v6, all untouched by
this script) can be re-run against the corrected quality gate on demand.

Writes only to ``--out`` (default: a standalone location, not any pilot
version's own quality_pool/ subdirectory, so no Pilot v1-v6 artifact is
touched).

Usage:
    python experiments/scripts/rebuild_nvbench_quality_pool.py \
        --out data/staging/dashboard_v3/nvbench_quality_pool
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder  # noqa: E402
from src.data_pipeline.frozen_validation import sha256_of_file  # noqa: E402
from src.data_pipeline.nvbench_pilot import _record  # noqa: E402
from src.data_pipeline.nvbench_profile import DbProfiler  # noqa: E402
from src.data_pipeline.nvbench_quality import build_quality_pool, load_quality_config  # noqa: E402
from src.data_pipeline.nvbench_source import DbMetadataResolver, item_chart, item_group_id, load_mapping  # noqa: E402
from src.utils.io import write_json  # noqa: E402

DEFAULT_NVBENCH_JSON = "data/raw_external/nvbench/extracted/nvBench-main/NVBench.json"
DEFAULT_CACHE_ROOT = "data/cache_external/nvbench/databases"
DEFAULT_MAPPING = "src/config/data/nvbench_mapping.yaml"
DEFAULT_QUALITY_CONFIG = "src/config/data/nvbench_quality_rules.yaml"
DEFAULT_PROFILE_CACHE = "data/cache_external/nvbench/field_profiles.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild only the nvBench quality pool (no pilot sampling).")
    p.add_argument("--out", default="data/staging/dashboard_v3/nvbench_quality_pool")
    p.add_argument("--nvbench-json", default=DEFAULT_NVBENCH_JSON)
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--mapping", default=DEFAULT_MAPPING)
    p.add_argument("--quality-config", default=DEFAULT_QUALITY_CONFIG)
    p.add_argument("--profile-cache", default=DEFAULT_PROFILE_CACHE)
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _write_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out)
    if "frozen" in out_dir.parts:
        raise SystemExit(f"refusing to write into a frozen path: {out_dir}")
    if any(p.startswith("nvbench_pilot_v") for p in out_dir.parts):
        raise SystemExit(f"refusing to write into a pilot version directory: {out_dir}")

    mapping = load_mapping(_resolve(args.mapping))
    cfg = load_quality_config(str(_resolve(args.quality_config)))
    cache_root = _resolve(args.cache_root)
    resolver_cache_root = str(cache_root) if cache_root.exists() else None
    resolver = DbMetadataResolver(resolver_cache_root)
    profiler = DbProfiler(resolver, cache_path=_resolve(args.profile_cache))

    builder = NvBenchBuilder(
        str(_resolve(args.nvbench_json)), cache_root=resolver_cache_root, mapping_path=str(_resolve(args.mapping)),
    )
    build_result = builder.build()
    pool = build_quality_pool(build_result.accepted, mapping, resolver, profiler, cfg)
    profiler.save_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    tier_a_path = out_dir / "tier_a_candidates.jsonl"
    tier_b_path = out_dir / "tier_b_diagnostics.jsonl"
    tier_c_path = out_dir / "tier_c_rejected.jsonl"
    summary_path = out_dir / "quality_pool_summary.json"

    _write_jsonl([_record(it) for it in pool["tier_a"]], tier_a_path)
    _write_jsonl([_record(it) for it in pool["tier_b"]], tier_b_path)
    _write_jsonl([_record(it) for it in pool["tier_c"]], tier_c_path)
    write_json(pool["summary"], summary_path)

    # Unique Tier-A source groups per chart (a group can contribute several
    # query-variant Tier-A candidates, so this is distinct from the raw count).
    unique_groups_per_chart: dict = {}
    for it in pool["tier_a"]:
        chart = item_chart(it)
        unique_groups_per_chart.setdefault(chart, set()).add(item_group_id(it))
    unique_groups_per_chart = {c: len(g) for c, g in unique_groups_per_chart.items()}

    scores = [q["quality_score"] for q in pool["quality_by_id"].values()]
    demoted_by_pie_rule = sum(
        1 for q in pool["quality_by_id"].values() if "pie_non_additive_kpi" in q.get("failed_rules", [])
    )

    report = {
        "rule_version": cfg.get("rule_version"),
        "total_technically_valid_candidates": len(build_result.accepted),
        "tier_a_count": pool["summary"]["tier_a_count"],
        "tier_b_count": pool["summary"]["tier_b_count"],
        "tier_c_count": pool["summary"]["tier_c_count"],
        "tier_a_by_chart": {c: v.get("A", 0) for c, v in pool["summary"]["tier_by_chart"].items()},
        "unique_tier_a_groups_by_chart": unique_groups_per_chart,
        "pie_non_additive_kpi_demoted_count": demoted_by_pie_rule,
        "quality_score_range": {"min": min(scores) if scores else None, "max": max(scores) if scores else None},
        "hashes": {
            "tier_a": sha256_of_file(tier_a_path),
            "tier_b": sha256_of_file(tier_b_path),
            "tier_c": sha256_of_file(tier_c_path),
        },
    }
    write_json(report, out_dir / "rebuild_report.json")

    print(json.dumps(report, indent=2))
    print(f"[rebuild] wrote quality pool to {out_dir}")


if __name__ == "__main__":
    main()
