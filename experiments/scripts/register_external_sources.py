"""Register manually downloaded external datasets under data/raw_external/.

Computes a per-file SHA-256 inventory and writes small, trackable manifests for
each source. It is read-only with respect to the raw files: it never extracts
archives, never downloads anything, and never modifies a raw file.

Outputs written:
    data/raw_external/nvbench/source_manifest.json
    data/raw_external/nvbench/SHA256SUMS.txt
    data/raw_external/nvbench2/source_manifest.json
    data/raw_external/nvbench2/SHA256SUMS.txt
    data/raw_external/quda_pending/source_inspection_report.json

License policy (established from downloaded files only):
    nvBench v1   -> MIT if its README/LICENSE asserts it (confirmed).
    nvBench 2.0  -> pending unless an explicit *active* license is found.
    Quda         -> pending / absent (corpus not provisioned).

Usage:
    python experiments/scripts/register_external_sources.py
    python experiments/scripts/register_external_sources.py --raw-dir data/raw_external
    python experiments/scripts/register_external_sources.py --verify
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.source_registry import (  # noqa: E402
    build_source_manifest,
    inspect_quda,
    nvbench_v1_stats,
    verify_source,
    write_sha256sums,
)
from src.utils.io import write_json  # noqa: E402

# Repository URLs are external facts not derivable from local files; license
# status is still driven only by on-disk evidence.
NVBENCH_REPO = "https://github.com/TsinghuaDatabaseGroup/nvBench"
NVBENCH2_REPO = "https://github.com/HKUSTDial/nvBench-2.0"
NVBENCH2_HF = "https://huggingface.co/datasets/TianqiLuo/nvBench2.0"
# Quda corpus is not provisioned; URL left unset until the source is verified.
QUDA_REPO = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register external raw datasets (read-only).")
    p.add_argument("--raw-dir", default="data/raw_external")
    p.add_argument(
        "--verify",
        action="store_true",
        help="Re-hash and diff against existing manifests without writing anything.",
    )
    return p.parse_args()


def _resolve(raw_dir: str) -> Path:
    path = Path(raw_dir)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _register(raw_dir: Path, timestamp: str) -> None:
    nvbench_dir = raw_dir / "nvbench"
    nvbench2_dir = raw_dir / "nvbench2"
    quda_dir = raw_dir / "quda_pending"

    # nvBench v1 -- MIT confirmed only if its files assert it.
    manifest, files = build_source_manifest(
        nvbench_dir,
        source_name="nvbench",
        repository_url=NVBENCH_REPO,
        timestamp=timestamp,
        stats_fn=nvbench_v1_stats,
    )
    write_json(manifest, nvbench_dir / "source_manifest.json")
    write_sha256sums(files, nvbench_dir / "SHA256SUMS.txt")
    _print_summary("nvbench", manifest)

    # nvBench 2.0 -- pending unless an active license is found on disk.
    manifest2, files2 = build_source_manifest(
        nvbench2_dir,
        source_name="nvbench2",
        repository_url=NVBENCH2_REPO,
        timestamp=timestamp,
        extra_references=[NVBENCH2_HF],
        license_override_to_pending=True,
    )
    write_json(manifest2, nvbench2_dir / "source_manifest.json")
    write_sha256sums(files2, nvbench2_dir / "SHA256SUMS.txt")
    _print_summary("nvbench2", manifest2)

    # Quda -- pending / absent.
    report = inspect_quda(quda_dir, repository_url=QUDA_REPO, timestamp=timestamp)
    write_json(report, quda_dir / "source_inspection_report.json")
    print(
        f"  quda_pending : corpus_present={report['corpus_present']} "
        f"n_files={report['n_files']} license={report['license']['status']}"
    )


def _print_summary(name: str, manifest: dict) -> None:
    lic = manifest["license"]
    print(
        f"  {name:12s}: n_files={manifest['n_files']} "
        f"total_bytes={manifest['total_bytes']} "
        f"license={lic['status']}"
        + (f" ({lic['license']})" if lic["license"] else "")
    )


def _verify(raw_dir: Path) -> int:
    exit_code = 0
    for name in ("nvbench", "nvbench2"):
        result = verify_source(raw_dir / name)
        status = "OK" if result["ok"] else "MISMATCH"
        print(f"  {name:12s}: {status}")
        for key in (
            "missing_files",
            "added_files",
            "changed_hashes",
            "changed_sizes",
            "malformed_checksum_entries",
        ):
            items = result[key]
            if items:
                exit_code = 1
                print(f"      {key}: {len(items)}")
                for item in items[:10]:
                    print(f"        - {item}")
        if not result["checksums_found"]:
            print("      (no SHA256SUMS.txt found; run without --verify first)")
    return exit_code


def main() -> None:
    args = parse_args()
    raw_dir = _resolve(args.raw_dir)
    if not raw_dir.exists():
        raise SystemExit(f"raw external dir not found: {raw_dir}")

    if args.verify:
        print(f"Verifying external sources under {raw_dir} (read-only):")
        raise SystemExit(_verify(raw_dir))

    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"Registering external sources under {raw_dir}:")
    _register(raw_dir, timestamp)
    print("Done. Manifests and checksums written.")


if __name__ == "__main__":
    main()
