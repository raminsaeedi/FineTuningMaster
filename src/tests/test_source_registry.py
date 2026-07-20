"""Tests for the external-source registration utility.

All tests use temporary fixture directories only. They never hash the real
datasets under data/raw_external/.
"""

import hashlib
import json

from src.data_pipeline.source_registry import (
    REGISTRATION_OUTPUT_NAMES,
    build_source_manifest,
    detect_license,
    detect_references,
    detect_revision,
    inspect_quda,
    nvbench_v1_stats,
    parse_sha256sums,
    verify_source,
    walk_files,
    write_sha256sums,
)

_TS = "2026-07-20T00:00:00+00:00"


def _make_source(root, *, readme="", with_archive=True):
    """Build a tiny fake source tree and return the root path."""
    (root / "extracted" / "repo-main").mkdir(parents=True)
    (root / "extracted" / "repo-main" / "data.json").write_text('{"a": 1}', encoding="utf-8")
    (root / "extracted" / "repo-main" / "nested").mkdir()
    (root / "extracted" / "repo-main" / "nested" / "b.txt").write_text("hello", encoding="utf-8")
    if readme:
        (root / "extracted" / "repo-main" / "README.md").write_text(readme, encoding="utf-8")
    if with_archive:
        (root / "archive").mkdir()
        (root / "archive" / "repo-main.zip").write_bytes(b"PK\x03\x04fake-zip-bytes")
    return root


# --------------------------------------------------------------------------- #
# hashing + inventory
# --------------------------------------------------------------------------- #
def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    payload = b"some raw bytes " * 100
    f.write_bytes(payload)
    files = walk_files(tmp_path)
    assert len(files) == 1
    assert files[0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert files[0]["size_bytes"] == len(payload)
    assert files[0]["relative_path"] == "x.bin"


def test_walk_files_is_sorted_and_recursive(tmp_path):
    _make_source(tmp_path)
    files = walk_files(tmp_path)
    rels = [f["relative_path"] for f in files]
    assert rels == sorted(rels)
    assert "extracted/repo-main/data.json" in rels
    assert "extracted/repo-main/nested/b.txt" in rels
    assert "archive/repo-main.zip" in rels


def test_registration_outputs_excluded_from_inventory(tmp_path):
    _make_source(tmp_path)
    for name in REGISTRATION_OUTPUT_NAMES:
        (tmp_path / name).write_text("should not be hashed", encoding="utf-8")
    rels = {f["relative_path"] for f in walk_files(tmp_path)}
    assert rels.isdisjoint(REGISTRATION_OUTPUT_NAMES)


def test_write_and_parse_sha256sums_roundtrip(tmp_path):
    _make_source(tmp_path)
    files = walk_files(tmp_path)
    out = tmp_path / "SHA256SUMS.txt"
    write_sha256sums(files, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    for line in lines:
        assert len(line.split("  ", 1)[0]) == 64
    recorded, malformed = parse_sha256sums(out)
    assert not malformed
    assert recorded == {f["relative_path"]: f["sha256"] for f in files}


def test_parse_sha256sums_flags_malformed(tmp_path):
    p = tmp_path / "SHA256SUMS.txt"
    p.write_text("not-a-valid-line\n" + "a" * 64 + "  ok.txt\n", encoding="utf-8")
    recorded, malformed = parse_sha256sums(p)
    assert "ok.txt" in recorded
    assert malformed == ["not-a-valid-line"]


# --------------------------------------------------------------------------- #
# license detection
# --------------------------------------------------------------------------- #
def test_license_confirmed_from_active_readme(tmp_path):
    _make_source(tmp_path, readme="## License\nThis project is under the MIT license.")
    info = detect_license(tmp_path)
    assert info["status"] == "confirmed"
    assert info["license"] == "MIT"


def test_license_pending_when_only_commented_out(tmp_path):
    readme = "# Repo\n<!-- ## License\nCC BY-SA 4.0 applies. -->\nBody text."
    _make_source(tmp_path, readme=readme)
    info = detect_license(tmp_path)
    assert info["status"] == "pending"
    assert info["license"] is None
    assert "commented-out" in info["evidence"]


def test_license_unknown_when_absent(tmp_path):
    _make_source(tmp_path, readme="# Repo\nNo license statement here.")
    info = detect_license(tmp_path)
    assert info["status"] == "unknown"
    assert info["license"] is None


def test_license_file_confirmed(tmp_path):
    _make_source(tmp_path)
    (tmp_path / "LICENSE").write_text("MIT License\n\nPermission is hereby granted...", encoding="utf-8")
    info = detect_license(tmp_path)
    assert info["status"] == "confirmed"
    assert info["license"] == "MIT"


# --------------------------------------------------------------------------- #
# references + revision
# --------------------------------------------------------------------------- #
def test_detect_references_scrapes_urls(tmp_path):
    _make_source(
        tmp_path,
        readme="See https://huggingface.co/datasets/x/y and https://arxiv.org/abs/2503.12880.",
    )
    refs = detect_references(tmp_path)
    assert "https://huggingface.co/datasets/x/y" in refs
    assert "https://arxiv.org/abs/2503.12880" in refs


def test_detect_revision_records_ref_not_commit(tmp_path):
    _make_source(tmp_path)
    files = walk_files(tmp_path)
    rev = detect_revision(tmp_path, files)
    assert rev["downloaded_ref"] == "main"
    assert rev["pinned_commit"] is None
    assert rev["archive_name"] == "repo-main.zip"
    assert len(rev["archive_sha256"]) == 64


# --------------------------------------------------------------------------- #
# nvBench v1 stats
# --------------------------------------------------------------------------- #
def test_nvbench_v1_stats_distinguishes_counts(tmp_path):
    payload = {
        "1": {"nl_queries": ["q1", "q2", "q3"]},
        "2": {"nl_queries": ["q1"]},
    }
    (tmp_path / "NVBench.json").write_text(json.dumps(payload), encoding="utf-8")
    stats = nvbench_v1_stats(tmp_path)
    assert stats["vis_object_count"] == 2
    assert stats["nl_queries_total"] == 4
    assert stats["published_nl_vis_pairs"] == 25750


def test_nvbench_v1_stats_none_when_missing(tmp_path):
    assert nvbench_v1_stats(tmp_path) is None


# --------------------------------------------------------------------------- #
# manifest assembly + quda + verify
# --------------------------------------------------------------------------- #
def test_build_source_manifest_shape(tmp_path):
    _make_source(tmp_path, readme="MIT license applies.")
    manifest, files = build_source_manifest(
        tmp_path, source_name="demo", repository_url="https://example.com/repo", timestamp=_TS
    )
    assert manifest["source"] == "demo"
    assert manifest["repository_url"] == "https://example.com/repo"
    assert manifest["n_files"] == len(files)
    assert manifest["total_bytes"] == sum(f["size_bytes"] for f in files)
    assert manifest["license"]["status"] == "confirmed"
    assert manifest["registered_at"] == _TS


def test_build_source_manifest_override_to_pending(tmp_path):
    _make_source(tmp_path, readme="# Repo\nNo license here.")
    manifest, _ = build_source_manifest(
        tmp_path,
        source_name="demo2",
        repository_url=None,
        timestamp=_TS,
        license_override_to_pending=True,
    )
    assert manifest["license"]["status"] == "pending"


def test_inspect_quda_empty(tmp_path):
    report = inspect_quda(tmp_path, repository_url=None, timestamp=_TS)
    assert report["corpus_present"] is False
    assert report["n_files"] == 0
    assert report["license"]["status"] == "pending"
    assert report["query_count"] is None


def test_verify_detects_changes(tmp_path):
    _make_source(tmp_path)
    manifest, files = build_source_manifest(
        tmp_path, source_name="demo", repository_url=None, timestamp=_TS
    )
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_sha256sums(files, tmp_path / "SHA256SUMS.txt")

    clean = verify_source(tmp_path)
    assert clean["ok"] is True

    # mutate a raw file -> hash + size change
    (tmp_path / "extracted" / "repo-main" / "data.json").write_text(
        '{"a": 1, "b": 2}', encoding="utf-8"
    )
    # add a new raw file
    (tmp_path / "extracted" / "repo-main" / "new.txt").write_text("new", encoding="utf-8")

    dirty = verify_source(tmp_path)
    assert dirty["ok"] is False
    assert "extracted/repo-main/data.json" in dirty["changed_hashes"]
    assert "extracted/repo-main/data.json" in dirty["changed_sizes"]
    assert "extracted/repo-main/new.txt" in dirty["added_files"]


def test_verify_detects_missing_and_malformed(tmp_path):
    _make_source(tmp_path)
    _, files = build_source_manifest(
        tmp_path, source_name="demo", repository_url=None, timestamp=_TS
    )
    write_sha256sums(files, tmp_path / "SHA256SUMS.txt")
    # append a malformed checksum line and reference a now-missing file
    with (tmp_path / "SHA256SUMS.txt").open("a", encoding="utf-8") as f:
        f.write("garbage-line\n")
        f.write("f" * 64 + "  ghost.txt\n")
    result = verify_source(tmp_path)
    assert result["ok"] is False
    assert "ghost.txt" in result["missing_files"]
    assert "garbage-line" in result["malformed_checksum_entries"]
