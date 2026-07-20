"""Deterministic, rebuildable cache preparation for the nvBench databases.

Extracts the registered ``databases.zip`` from the immutable raw-source tree
into a rebuildable cache under ``data/cache_external/nvbench/databases/``.

Guarantees:
- The raw source (``data/raw_external/nvbench/``) is never written to.
- The archive SHA-256 is verified against the registered source manifest before
  extraction.
- Zip-slip / path-traversal entries are rejected.
- Extraction happens in a temporary directory and is finalized atomically.
- Normal runs are idempotent; ``--verify`` is strictly read-only.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data_pipeline.frozen_validation import sha256_of_file

CACHE_TOOL_VERSION = "1.0.0"
DATABASE_CACHE_FORMAT = "sqlite"

# Junk entries shipped inside the upstream archive that must never be extracted.
_JUNK_PREFIXES = ("__MACOSX/",)
_JUNK_BASENAMES = (".DS_Store",)
# Only the database subtree is cached; leading prefix is stripped on extraction.
_SOURCE_PREFIX = "database/"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_junk(name: str) -> bool:
    if any(name.startswith(p) for p in _JUNK_PREFIXES):
        return True
    base = name.rsplit("/", 1)[-1]
    return base.startswith("._") or base in _JUNK_BASENAMES


def _is_within(base: Path, target: Path) -> bool:
    """True iff ``target`` resolves to a path inside ``base`` (zip-slip guard)."""
    try:
        base_res = base.resolve()
        target_res = target.resolve()
        return target_res == base_res or base_res in target_res.parents
    except OSError:
        return False


def read_source_manifest(source_manifest_path: str | Path) -> Dict:
    return json.loads(Path(source_manifest_path).read_text(encoding="utf-8"))


def resolve_archive(
    raw_dir: str | Path, manifest: Dict, archive_basename: str = "databases.zip"
) -> Tuple[Path, str, str]:
    """Locate the databases archive via the registered manifest ``files`` list.

    The databases ship as a nested ``databases.zip`` inside the extracted source
    tree, so its SHA-256 is taken from the per-file inventory (not the outer
    repository archive). Returns ``(archive_path, expected_sha256, relative_path)``.
    """
    for entry in manifest.get("files", []) or []:
        rel = str(entry.get("relative_path", ""))
        if rel.rsplit("/", 1)[-1] == archive_basename:
            return Path(raw_dir) / rel, str(entry["sha256"]), rel
    raise ValueError(
        f"source manifest has no file entry for {archive_basename!r}; "
        "cannot locate the databases archive."
    )


def verify_archive(archive_path: str | Path, expected_sha: str) -> Tuple[bool, str]:
    """Verify the archive SHA-256 (read-only). Returns ``(ok, actual_sha)``."""
    actual = sha256_of_file(archive_path)
    return actual == expected_sha, actual


def _plan_members(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    """Non-junk file members under the database subtree, sorted by name."""
    members = [
        zi
        for zi in zf.infolist()
        if not zi.is_dir()
        and not _is_junk(zi.filename)
        and zi.filename.startswith(_SOURCE_PREFIX)
    ]
    members.sort(key=lambda zi: zi.filename)
    return members


# --------------------------------------------------------------------------- #
# cache manifest
# --------------------------------------------------------------------------- #
def _cache_root(cache_base: Path) -> Path:
    return cache_base / "databases"


def _manifest_path(cache_base: Path) -> Path:
    return cache_base / "cache_manifest.json"


def _count_tree(root: Path) -> Tuple[int, int]:
    """Return (file_count, byte_count) for a directory tree."""
    n, total = 0, 0
    for p in root.rglob("*"):
        if p.is_file():
            n += 1
            total += p.stat().st_size
    return n, total


def build_cache_manifest(
    *,
    archive_relative_path: str,
    archive_sha256: str,
    timestamp: str,
    file_count: int,
    byte_count: int,
) -> Dict:
    return {
        "source_archive_relative_path": archive_relative_path,
        "source_archive_sha256": archive_sha256,
        "extraction_timestamp": timestamp,
        "extracted_file_count": file_count,
        "extracted_byte_count": byte_count,
        "database_cache_format": DATABASE_CACHE_FORMAT,
        "cache_tool_version": CACHE_TOOL_VERSION,
    }


# --------------------------------------------------------------------------- #
# extraction (zip-slip safe, atomic)
# --------------------------------------------------------------------------- #
def _extract_safely(zf: zipfile.ZipFile, members: List[zipfile.ZipInfo], dest: Path) -> Tuple[int, int]:
    """Extract ``members`` into ``dest`` with a zip-slip guard. Returns counts."""
    dest.mkdir(parents=True, exist_ok=True)
    file_count, byte_count = 0, 0
    for zi in members:
        rel = zi.filename[len(_SOURCE_PREFIX):]
        if not rel:
            continue
        out_path = dest / rel
        if not _is_within(dest, out_path):
            raise ValueError(f"unsafe zip entry (path traversal): {zi.filename!r}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(zi, "r") as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        file_count += 1
        byte_count += out_path.stat().st_size
    return file_count, byte_count


# --------------------------------------------------------------------------- #
# top-level operations
# --------------------------------------------------------------------------- #
def prepare_cache(
    *,
    raw_dir: str | Path,
    source_manifest_path: str | Path,
    cache_base: str | Path,
    timestamp: str,
    force: bool = False,
    dry_run: bool = False,
) -> Dict:
    """Verify and extract the nvBench databases into a rebuildable cache.

    Idempotent: if the cache already exists with a matching archive SHA-256 and
    consistent counts, extraction is skipped unless ``force`` is set.
    ``dry_run`` performs verification and planning without writing anything.
    """
    cache_base = Path(cache_base)
    manifest = read_source_manifest(source_manifest_path)
    archive_path, expected_sha, archive_rel = resolve_archive(raw_dir, manifest)

    if not archive_path.exists():
        raise FileNotFoundError(f"archive not found: {archive_path}")

    sha_ok, actual_sha = verify_archive(archive_path, expected_sha)
    if not sha_ok:
        raise ValueError(
            f"archive SHA-256 mismatch for {archive_rel}: "
            f"expected {expected_sha}, got {actual_sha}"
        )

    cache_root = _cache_root(cache_base)
    manifest_path = _manifest_path(cache_base)
    already = manifest_path.exists() and cache_root.exists()

    with zipfile.ZipFile(archive_path) as zf:
        members = _plan_members(zf)
        planned_files = len(members)
        planned_bytes = sum(zi.file_size for zi in members)

        if dry_run:
            return {
                "action": "dry-run",
                "archive_relative_path": archive_rel,
                "archive_sha256": actual_sha,
                "sha_verified": True,
                "planned_file_count": planned_files,
                "planned_byte_count": planned_bytes,
                "cache_root": cache_root.as_posix(),
                "already_prepared": already,
                "would_skip": already and not force,
            }

        # Idempotency: skip re-extraction when already valid and not forced.
        if already and not force:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            fc, bc = _count_tree(cache_root)
            if (
                existing.get("source_archive_sha256") == actual_sha
                and existing.get("extracted_file_count") == fc
                and existing.get("extracted_byte_count") == bc
            ):
                return {"action": "skipped-idempotent", **existing, "cache_root": cache_root.as_posix()}

        # Extract into a temp dir, then atomically finalize.
        tmp_dir = cache_base / ".tmp_extract"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        cache_base.mkdir(parents=True, exist_ok=True)
        file_count, byte_count = _extract_safely(zf, members, tmp_dir)

    if cache_root.exists():
        shutil.rmtree(cache_root)
    os.replace(tmp_dir, cache_root)

    cache_manifest = build_cache_manifest(
        archive_relative_path=archive_rel,
        archive_sha256=actual_sha,
        timestamp=timestamp,
        file_count=file_count,
        byte_count=byte_count,
    )
    manifest_path.write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
    return {"action": "extracted", **cache_manifest, "cache_root": cache_root.as_posix()}


def verify_cache(cache_base: str | Path) -> Dict:
    """Read-only check that the cache matches its manifest counts."""
    cache_base = Path(cache_base)
    cache_root = _cache_root(cache_base)
    manifest_path = _manifest_path(cache_base)

    if not manifest_path.exists():
        return {"ok": False, "reason": "no cache_manifest.json; run preparation first"}
    if not cache_root.exists():
        return {"ok": False, "reason": "cache databases/ directory missing"}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fc, bc = _count_tree(cache_root)
    ok = (
        manifest.get("extracted_file_count") == fc
        and manifest.get("extracted_byte_count") == bc
    )
    return {
        "ok": ok,
        "cache_root": cache_root.as_posix(),
        "manifest_file_count": manifest.get("extracted_file_count"),
        "actual_file_count": fc,
        "manifest_byte_count": manifest.get("extracted_byte_count"),
        "actual_byte_count": bc,
        "database_cache_format": manifest.get("database_cache_format"),
        "cache_tool_version": manifest.get("cache_tool_version"),
    }
