"""Package the final multi-model evidence for the professor handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.scripts.package_results import (
    _print_large_file_warning,
    build_manifest,
    collect_result_files,
    collect_run_files,
    write_package,
)

def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package final professor results")
    parser.add_argument("--outputs-root", default="experiments/outputs/final/dashboard_v3")
    parser.add_argument("--results-dir", default="experiments/results/final/dashboard_v3")
    parser.add_argument("--out", default="professor_results.zip")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _write_external_manifest(manifest: dict, archive: Path) -> Path:
    payload = dict(manifest)
    payload["archive"] = archive.name
    payload["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    target = archive.with_name("professor_results_manifest.json")
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs_root = _resolve(args.outputs_root)
    results_dir = _resolve(args.results_dir)
    archive = _resolve(args.out)
    entries = collect_run_files(outputs_root) + collect_result_files(results_dir)
    if not entries:
        raise SystemExit(
            f"No safe result artifacts found under {outputs_root} or {results_dir}. "
            "Run the final matrix first."
        )

    manifest = build_manifest(entries)
    print(f"Packaging {manifest['counts']['total_files']} safe artifact file(s).")
    _print_large_file_warning(manifest)
    if args.dry_run:
        print("Dry run - nothing written.")
        return

    write_package(entries, manifest, archive)
    manifest_path = _write_external_manifest(manifest, archive)
    print(f"Created: {archive}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
