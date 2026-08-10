"""Knowledge-base manifest + verification tests (reproducibility of the RAG KB)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pytest

from src.data_pipeline.kb_builder import (
    MANIFEST_FILENAME,
    build_chunks,
    build_manifest,
    compute_kb_version,
    verify_kb,
)
from src.utils.io import write_json, write_jsonl

DOC_A = (
    "# Chart selection\n"
    "## Line charts\nUse a line chart for a numeric trend measured over time.\n"
    "## Bar charts\nUse a bar chart to compare a measure across discrete categories.\n"
)
DOC_B = (
    "# Accessibility\n"
    "## Contrast\nKeep a contrast ratio of at least 4.5 to 1 for body text.\n"
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_kb(kb_dir: Path, docs: Optional[Dict[str, str]] = None) -> dict:
    """Write guideline docs into ``kb_dir`` and build chunks + manifest."""
    docs = docs if docs is not None else {"chart.md": DOC_A, "access.md": DOC_B}
    guidelines = kb_dir / "guidelines"
    guidelines.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        (guidelines / name).write_text(text, encoding="utf-8")

    chunks = build_chunks(guidelines)
    chunks_path = kb_dir / "chunks.jsonl"
    write_jsonl(chunks, chunks_path)
    manifest = build_manifest(guidelines, chunks, chunks_path)
    write_json(manifest, kb_dir / MANIFEST_FILENAME)
    return manifest


def test_kb_version_is_deterministic_across_rebuilds(tmp_path):
    first = _make_kb(tmp_path / "kb1")
    second = _make_kb(tmp_path / "kb2")

    assert first["kb_version"] == second["kb_version"]
    assert first["chunks_sha256"] == second["chunks_sha256"]
    # The timestamp is metadata only and must not feed into the version.
    assert {k: v for k, v in first.items() if k != "created_utc"} == {
        k: v for k, v in second.items() if k != "created_utc"
    }


def test_kb_version_changes_when_a_source_document_changes(tmp_path):
    baseline = _make_kb(tmp_path / "kb1")
    edited = _make_kb(
        tmp_path / "kb2",
        docs={"chart.md": DOC_A + "## Scatter\nUse a scatter plot to show correlation.\n",
              "access.md": DOC_B},
    )

    assert baseline["kb_version"] != edited["kb_version"]


def test_kb_version_ignores_source_ordering():
    pairs = [("b.md", "bb" * 32), ("a.md", "aa" * 32)]
    assert compute_kb_version(pairs) == compute_kb_version(reversed(pairs))


def test_manifest_records_documents_and_totals(tmp_path):
    manifest = _make_kb(tmp_path / "kb")

    assert manifest["n_documents"] == 2
    assert manifest["n_chunks"] == sum(d["n_chunks"] for d in manifest["source_documents"])
    assert [d["filename"] for d in manifest["source_documents"]] == ["access.md", "chart.md"]
    assert all(d["bytes"] > 0 and len(d["sha256"]) == 64 for d in manifest["source_documents"])
    assert manifest["builder"]["min_chunk_words"] == 5


def test_verify_kb_passes_on_a_freshly_built_kb(tmp_path):
    kb_dir = tmp_path / "kb"
    _make_kb(kb_dir)

    ok, problems = verify_kb(kb_dir)
    assert ok and problems == []


def test_verify_kb_detects_a_tampered_source_document(tmp_path):
    kb_dir = tmp_path / "kb"
    _make_kb(kb_dir)
    (kb_dir / "guidelines" / "chart.md").write_text(
        DOC_A + "## Injected\nThis paragraph was added after the build ran.\n",
        encoding="utf-8",
    )

    ok, problems = verify_kb(kb_dir)
    assert not ok
    assert any("chart.md" in p and "changed since build" in p for p in problems)


def test_verify_kb_detects_a_mismatched_chunks_file(tmp_path):
    kb_dir = tmp_path / "kb"
    manifest = _make_kb(kb_dir)
    chunks_path = kb_dir / "chunks.jsonl"
    with chunks_path.open("a", encoding="utf-8") as f:
        f.write('{"id": "rogue_0", "source": "rogue", "heading": "x", "text": "y"}\n')

    ok, problems = verify_kb(kb_dir)
    assert not ok
    assert any("Chunks file changed since build" in p for p in problems)
    assert any(f"manifest records {manifest['n_chunks']}" in p for p in problems)


def test_verify_kb_detects_missing_manifest_and_missing_chunks(tmp_path):
    kb_dir = tmp_path / "kb"
    _make_kb(kb_dir)

    (kb_dir / "chunks.jsonl").unlink()
    ok, problems = verify_kb(kb_dir)
    assert not ok and any("Chunks file not found" in p for p in problems)

    (kb_dir / MANIFEST_FILENAME).unlink()
    ok, problems = verify_kb(kb_dir)
    assert not ok and any("KB manifest not found" in p for p in problems)


def test_verify_kb_detects_an_unrecorded_source_document(tmp_path):
    kb_dir = tmp_path / "kb"
    _make_kb(kb_dir)
    (kb_dir / "guidelines" / "extra.md").write_text(
        "# Extra\nThis document was added without rebuilding the knowledge base.\n",
        encoding="utf-8",
    )

    ok, problems = verify_kb(kb_dir)
    assert not ok
    assert any("not recorded in manifest" in p and "extra.md" in p for p in problems)


def test_verify_kb_passes_on_the_repository_knowledge_base():
    kb_dir = _REPO_ROOT / "data" / "knowledge_base"
    if not (kb_dir / "chunks.jsonl").exists():
        pytest.skip("KB not built in this checkout; run experiments/scripts/build_kb.py")

    ok, problems = verify_kb(kb_dir)
    assert ok, problems
