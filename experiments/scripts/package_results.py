"""Package the result artifacts a supervisor has to send back — nothing else.

Produces ``professor_results.zip`` at the repo root containing only the small,
shareable evidence of a GPU run:

    outputs/<run_id>/predictions*.jsonl      cached model outputs
    outputs/<run_id>/errors*.jsonl           per-item failures
    outputs/<run_id>/metrics_auto.json       computed metrics
    outputs/<run_id>/metrics.json            metrics (legacy/eval writer)
    outputs/<run_id>/eval_per_item.jsonl     per-item evaluation records
    outputs/<run_id>/manifest.json           run identity + provenance
    outputs/<run_id>/config_snapshot.yaml    resolved config
    outputs/<run_id>/config_hash.txt
    outputs/<run_id>/git_hash.txt
    outputs/<run_id>/env.txt                 pip freeze
    outputs/<run_id>/logs/**                 log files
    outputs/<run_id>/adapter/training_metadata.json   training SUMMARY only
    results/**                               comparison tables, report, stats
    PACKAGE_MANIFEST.json                    every file with size + sha256

Model weights, checkpoints, HuggingFace caches and anything that looks like a
secret are excluded (see ``_EXCLUSION_RULES``) — the ZIP stays small enough to
e-mail and carries no credentials.

Examples
--------
  python experiments/scripts/package_results.py
  python experiments/scripts/package_results.py --dry-run
  python experiments/scripts/package_results.py \\
      --outputs-root experiments/outputs --out supervisor_run.zip
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_MB = 1_000_000
_LARGE_FILE_BYTES = 25 * _MB

# Defaults mirror src/config/config.yaml (`output_root`) and aggregate_results.py.
_DEFAULT_OUTPUTS_ROOT = "experiments/outputs/experiments"
_DEFAULT_RESULTS_DIR = "experiments/results"
_DEFAULT_OUT = "professor_results.zip"

# ── What goes in ──────────────────────────────────────────────────────────────

# Per-run files taken verbatim (see src/utils/artifacts.py for the contract).
_RUN_FILE_NAMES = {
    "metrics_auto.json",
    "metrics.json",
    "eval_per_item.jsonl",
    "manifest.json",
    "config_snapshot.yaml",
    "config_hash.txt",
    "git_hash.txt",
    "env.txt",
    "cache_identity.json",
    "dataset_hashes.json",
    "kb_hashes.json",
    "resume_metadata.json",
}
_RUN_FILE_GLOBS = ("predictions*.jsonl", "errors*.jsonl")

# Inside adapter/ only the training summary is shipped — never the weights.
_ADAPTER_FILE_NAMES = {"training_metadata.json"}

# ── What stays out ────────────────────────────────────────────────────────────

# Directory names that are dropped wherever they appear in a path.
_EXCLUDED_DIR_NAMES = {
    "checkpoints",          # intermediate trainer state (GB-sized)
    "__pycache__",
    ".git",
    ".cache",               # generic + HuggingFace cache roots
    "hf_cache",
    "huggingface",
    "hub",
    "base_model",
    "_stale_cache",
}

# Binary / weight artifacts, matched on the file name.
_EXCLUDED_FILE_GLOBS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.pyc",
)

# Anything that may carry a credential (case-insensitive substring match).
_SECRET_SUBSTRINGS = ("key", "token", "secret")

# Human-readable copy of the rules, embedded in PACKAGE_MANIFEST.json.
_EXCLUSION_RULES = [
    "model weights: adapter_model.safetensors, adapter_model.bin, *.safetensors, *.bin, *.pt, *.pth, *.ckpt",
    "checkpoint directories: checkpoints/**",
    "base-model / HuggingFace caches: .cache/, hf_cache/, huggingface/, hub/, base_model/, models--*/",
    "secrets: .env*, any name containing 'key', 'token' or 'secret' (case-insensitive)",
    "python bytecode: __pycache__/**, *.pyc",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Package result artifacts into professor_results.zip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--outputs-root", default=_DEFAULT_OUTPUTS_ROOT,
                   help=f"Per-run artifact root (default: {_DEFAULT_OUTPUTS_ROOT})")
    p.add_argument("--results-dir", default=_DEFAULT_RESULTS_DIR,
                   help=f"Aggregate results directory (default: {_DEFAULT_RESULTS_DIR})")
    p.add_argument("--out", default=_DEFAULT_OUT,
                   help=f"Output ZIP path (default: {_DEFAULT_OUT})")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be included plus the total size; write nothing")
    return p.parse_args(argv)


# ── Selection ─────────────────────────────────────────────────────────────────

def exclusion_reason(rel_path: Path | str) -> str | None:
    """Return why ``rel_path`` must not be packaged, or ``None`` if it is safe.

    ``rel_path`` is relative to the packaged root. Matching is case-insensitive
    so Windows-cased names (``API_KEY.txt``) are caught too.
    """
    parts = [part.lower() for part in Path(rel_path).parts]
    if not parts:
        return "empty path"

    for part in parts[:-1]:
        if part in _EXCLUDED_DIR_NAMES:
            return f"excluded directory '{part}'"
        if part.startswith("models--"):
            return "HuggingFace model cache"
        for needle in _SECRET_SUBSTRINGS:
            if needle in part:
                return f"possible secret directory (contains '{needle}')"

    name = parts[-1]
    if name == ".env" or name.startswith(".env"):
        return "environment file"
    for pattern in _EXCLUDED_FILE_GLOBS:
        if fnmatch.fnmatch(name, pattern):
            return f"binary artifact (matches '{pattern}')"
    for needle in _SECRET_SUBSTRINGS:
        if needle in name:
            return f"possible secret (name contains '{needle}')"
    return None


def is_wanted_run_file(rel_path: Path | str) -> bool:
    """True when a per-run file belongs in the supervisor package."""
    rel = Path(rel_path)
    parent_parts = [part.lower() for part in rel.parts[:-1]]
    name = rel.name

    if "logs" in parent_parts:
        return True
    if "adapter" in parent_parts:
        return name in _ADAPTER_FILE_NAMES
    if name in _RUN_FILE_NAMES:
        return True
    return any(fnmatch.fnmatch(name, g) for g in _RUN_FILE_GLOBS)


def collect_files(root: Path, arc_prefix: str, wanted=None) -> list[tuple[Path, str]]:
    """Walk ``root`` and return ``(source_path, archive_name)`` pairs.

    Exclusions are applied before ``wanted`` so a forbidden file can never be
    pulled in by an inclusion rule.
    """
    entries: list[tuple[Path, str]] = []
    if not root.exists():
        return entries
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if exclusion_reason(rel) is not None:
            continue
        if wanted is not None and not wanted(rel):
            continue
        entries.append((path, str(PurePosixPath(arc_prefix, *rel.parts))))
    return entries


def collect_run_files(outputs_root: Path) -> list[tuple[Path, str]]:
    return collect_files(outputs_root, "outputs", is_wanted_run_file)


def collect_result_files(results_dir: Path) -> list[tuple[Path, str]]:
    return collect_files(results_dir, "results")


# ── Manifest ──────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(entries: list[tuple[Path, str]]) -> dict:
    """Describe every packaged file (archive name, size, sha256) plus counts."""
    files = [
        {
            "path": arcname,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path, arcname in entries
    ]
    total_bytes = sum(f["size_bytes"] for f in files)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {
            "total_files": len(files),
            "run_artifact_files": sum(1 for f in files if f["path"].startswith("outputs/")),
            "aggregate_result_files": sum(1 for f in files if f["path"].startswith("results/")),
        },
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / _MB, 2),
        "excluded_by_design": _EXCLUSION_RULES,
        "files": files,
    }


def write_package(entries: list[tuple[Path, str]], manifest: dict, out_path: Path) -> Path:
    """Write the ZIP with the manifest as its first, top-level member."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
        for path, arcname in entries:
            zf.write(path, arcname)
    return out_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _large_files(manifest: dict) -> list[dict]:
    return [f for f in manifest["files"] if f["size_bytes"] > _LARGE_FILE_BYTES]


def _print_large_file_warning(manifest: dict) -> None:
    large = _large_files(manifest)
    if not large:
        return
    print(f"\n  WARNING: {len(large)} file(s) larger than {_LARGE_FILE_BYTES // _MB} MB were included:")
    for f in large:
        print(f"    {f['size_bytes'] / _MB:8.1f} MB  {f['path']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs_root = _resolve(args.outputs_root)
    results_dir = _resolve(args.results_dir)
    out_path = _resolve(args.out)

    run_entries = collect_run_files(outputs_root)
    if not run_entries:
        raise SystemExit(
            f"No run artifacts found under {outputs_root}.\n"
            "Run the experiments first, e.g.:\n"
            "  python experiments/scripts/run_remote.py --mode full\n"
            "or point --outputs-root at the directory that holds the run folders."
        )

    result_entries = collect_result_files(results_dir)
    if not result_entries:
        print(f"  NOTE: no aggregate results under {results_dir} - "
              "run experiments/scripts/aggregate_results.py to add them.")

    entries = run_entries + result_entries
    manifest = build_manifest(entries)

    print("=" * 56)
    print("PACKAGING RESULTS" + (" (dry run)" if args.dry_run else ""))
    print("=" * 56)
    for f in manifest["files"]:
        print(f"  {f['size_bytes'] / _MB:8.2f} MB  {f['path']}")
    print("-" * 56)
    print(f"  Run artifact files   : {manifest['counts']['run_artifact_files']}")
    print(f"  Aggregate result files: {manifest['counts']['aggregate_result_files']}")
    print(f"  Uncompressed total   : {manifest['total_mb']:.1f} MB")

    _print_large_file_warning(manifest)

    if args.dry_run:
        print("\n  Dry run - nothing written.")
        return

    write_package(entries, manifest, out_path)
    size_mb = out_path.stat().st_size / _MB
    print(f"\n  Created: {out_path}  ({size_mb:.1f} MB)")
    print("=" * 56)


if __name__ == "__main__":
    main()
