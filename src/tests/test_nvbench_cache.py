"""Tests for the nvBench database cache preparation utility.

Uses small in-memory ZIP fixtures only; never extracts the real archive.
"""

import json
import zipfile

import pytest

from src.data_pipeline.frozen_validation import sha256_of_file
from src.data_pipeline.nvbench_cache import (
    prepare_cache,
    verify_cache,
)

_TS = "2026-07-20T00:00:00+00:00"


def _make_archive(tmp_path, entries):
    """Write a tiny databases.zip and a matching source_manifest.json.

    Mirrors the real manifest shape: the archive SHA-256 is taken from the
    per-file ``files`` inventory (the databases ship as a nested databases.zip).
    """
    raw_dir = tmp_path / "raw" / "nvbench"
    archive_rel = "extracted/nvBench-main/databases.zip"
    archive = raw_dir / archive_rel
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    manifest = {
        "source": "nvbench",
        "files": [
            {
                "relative_path": archive_rel,
                "size_bytes": archive.stat().st_size,
                "sha256": sha256_of_file(archive),
            }
        ],
    }
    (raw_dir / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw_dir


_GOOD_ENTRIES = {
    "database/a.sqlite": b"SQLite-a",
    "database/b/b.sqlite": b"SQLite-b-nested",
    "database/b/schema.sql": b"CREATE TABLE t(x);",
    "__MACOSX/._a.sqlite": b"junk",
    "database/.DS_Store": b"junk",
}


def test_prepare_extracts_and_writes_manifest(tmp_path):
    raw_dir = _make_archive(tmp_path, _GOOD_ENTRIES)
    cache_base = tmp_path / "cache" / "nvbench"
    result = prepare_cache(
        raw_dir=raw_dir,
        source_manifest_path=raw_dir / "source_manifest.json",
        cache_base=cache_base,
        timestamp=_TS,
    )
    assert result["action"] == "extracted"
    # junk (__MACOSX, .DS_Store) excluded -> 3 real files.
    assert result["extracted_file_count"] == 3
    assert result["database_cache_format"] == "sqlite"
    assert (cache_base / "databases" / "a.sqlite").read_bytes() == b"SQLite-a"
    assert (cache_base / "databases" / "b" / "b.sqlite").exists()
    assert not (cache_base / "databases" / ".DS_Store").exists()
    manifest = json.loads((cache_base / "cache_manifest.json").read_text())
    assert manifest["cache_tool_version"]
    assert manifest["source_archive_relative_path"] == "extracted/nvBench-main/databases.zip"


def test_sha_mismatch_aborts(tmp_path):
    raw_dir = _make_archive(tmp_path, _GOOD_ENTRIES)
    manifest_path = raw_dir / "source_manifest.json"
    m = json.loads(manifest_path.read_text())
    m["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_cache(
            raw_dir=raw_dir,
            source_manifest_path=manifest_path,
            cache_base=tmp_path / "cache",
            timestamp=_TS,
        )


def test_zip_slip_is_rejected(tmp_path):
    entries = {"database/../evil.sqlite": b"pwned", "database/ok.sqlite": b"ok"}
    raw_dir = _make_archive(tmp_path, entries)
    with pytest.raises(ValueError, match="path traversal"):
        prepare_cache(
            raw_dir=raw_dir,
            source_manifest_path=raw_dir / "source_manifest.json",
            cache_base=tmp_path / "cache",
            timestamp=_TS,
        )


def test_dry_run_writes_nothing(tmp_path):
    raw_dir = _make_archive(tmp_path, _GOOD_ENTRIES)
    cache_base = tmp_path / "cache" / "nvbench"
    result = prepare_cache(
        raw_dir=raw_dir,
        source_manifest_path=raw_dir / "source_manifest.json",
        cache_base=cache_base,
        timestamp=_TS,
        dry_run=True,
    )
    assert result["action"] == "dry-run"
    assert result["planned_file_count"] == 3
    assert result["sha_verified"] is True
    assert not cache_base.exists()


def test_idempotent_second_run_skips(tmp_path):
    raw_dir = _make_archive(tmp_path, _GOOD_ENTRIES)
    cache_base = tmp_path / "cache" / "nvbench"
    kwargs = dict(
        raw_dir=raw_dir,
        source_manifest_path=raw_dir / "source_manifest.json",
        cache_base=cache_base,
        timestamp=_TS,
    )
    prepare_cache(**kwargs)
    again = prepare_cache(**kwargs)
    assert again["action"] == "skipped-idempotent"


def test_force_rebuilds(tmp_path):
    raw_dir = _make_archive(tmp_path, _GOOD_ENTRIES)
    cache_base = tmp_path / "cache" / "nvbench"
    kwargs = dict(
        raw_dir=raw_dir,
        source_manifest_path=raw_dir / "source_manifest.json",
        cache_base=cache_base,
        timestamp=_TS,
    )
    prepare_cache(**kwargs)
    forced = prepare_cache(force=True, **kwargs)
    assert forced["action"] == "extracted"


def test_verify_cache_read_only(tmp_path):
    raw_dir = _make_archive(tmp_path, _GOOD_ENTRIES)
    cache_base = tmp_path / "cache" / "nvbench"
    prepare_cache(
        raw_dir=raw_dir,
        source_manifest_path=raw_dir / "source_manifest.json",
        cache_base=cache_base,
        timestamp=_TS,
    )
    ok = verify_cache(cache_base)
    assert ok["ok"] is True

    # tamper: delete a cached file -> verify must report mismatch, write nothing.
    (cache_base / "databases" / "a.sqlite").unlink()
    bad = verify_cache(cache_base)
    assert bad["ok"] is False
    assert bad["actual_file_count"] == 2


def test_verify_without_cache(tmp_path):
    result = verify_cache(tmp_path / "missing")
    assert result["ok"] is False
