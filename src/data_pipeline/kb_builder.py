"""Build retrieval chunks from the markdown guideline documents.

Each guideline file is split into chunks at headings (lines starting with '#'),
so one chunk is a heading plus its body text. Chunks get a stable id derived
from their source file and content, so rebuilding the KB does not reshuffle ids.

The build also emits a manifest (``kb_manifest.json``) that records the sha256
of every source document and of the produced ``chunks.jsonl``. Because the
manifest is tracked in Git while the chunks are not, a fresh clone can rebuild
the KB and prove the result is byte-identical to the one used in the
experiments. :func:`verify_kb` performs that check and is safe to call from
preflight scripts (it returns results instead of printing or exiting).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Layout of a built knowledge base directory.
GUIDELINES_DIRNAME = "guidelines"
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "kb_manifest.json"

# Chunking parameters (recorded in the manifest so a rebuild is comparable).
SOURCE_GLOB = "*.md"
MIN_CHUNK_WORDS = 5
CHUNK_ID_HASH = "md5_8"
KB_VERSION_LENGTH = 16

_BUILD_HINT = "Run `python experiments/scripts/build_kb.py`."


def _chunk_markdown(text: str, source: str) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    heading = ""
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if heading or body:
            content = (heading + "\n" + body).strip()
            digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
            chunks.append({
                "id": f"{source}_{digest}",
                "source": source,
                "heading": heading.lstrip("# ").strip(),
                "text": content,
            })

    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            heading = line.strip()
            buffer = []
        else:
            buffer.append(line)
    flush()
    # Drop chunks that are only a heading with no body.
    return [c for c in chunks if len(c["text"].split()) > MIN_CHUNK_WORDS]


def build_chunks(guidelines_dir: str | Path) -> List[Dict[str, str]]:
    """Read every .md file in ``guidelines_dir`` and return retrieval chunks."""
    guidelines_dir = Path(guidelines_dir)
    chunks: List[Dict[str, str]] = []
    for md_path in sorted(guidelines_dir.glob(SOURCE_GLOB)):
        source = md_path.stem
        chunks.extend(_chunk_markdown(md_path.read_text(encoding="utf-8"), source))
    return chunks


# ── Provenance ───────────────────────────────────────────────────────────────


def _sha256_file(path: str | Path) -> str:
    """SHA256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def compute_kb_version(source_hashes: Iterable[Tuple[str, str]]) -> str:
    """Deterministic KB version from the ``(filename, sha256)`` source pairs.

    The pairs are sorted before hashing, so the version depends only on the
    content of the guideline documents -- never on build order, paths or time.
    """
    payload = "\n".join(f"{name}:{digest}" for name, digest in sorted(source_hashes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:KB_VERSION_LENGTH]


def build_manifest(
    guidelines_dir: str | Path,
    chunks: Sequence[Dict[str, str]],
    chunks_path: str | Path,
) -> Dict[str, Any]:
    """Describe a freshly built KB: source hashes, totals and chunk-file hash.

    ``created_utc`` is metadata only and deliberately does not feed into
    ``kb_version``, so two rebuilds of the same sources agree on the version.
    """
    guidelines_dir = Path(guidelines_dir)
    chunks_path = Path(chunks_path)
    per_source = Counter(c.get("source", "") for c in chunks)

    source_documents: List[Dict[str, Any]] = []
    for md_path in sorted(guidelines_dir.glob(SOURCE_GLOB)):
        source_documents.append({
            "filename": md_path.name,
            "sha256": _sha256_file(md_path),
            "bytes": md_path.stat().st_size,
            "n_chunks": int(per_source.get(md_path.stem, 0)),
        })

    return {
        "kb_version": compute_kb_version(
            (d["filename"], d["sha256"]) for d in source_documents
        ),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_documents": len(source_documents),
        "n_chunks": len(chunks),
        "chunks_file": chunks_path.name,
        "chunks_sha256": _sha256_file(chunks_path),
        "builder": {
            "split_on": "markdown_headings",
            "source_glob": SOURCE_GLOB,
            "min_chunk_words": MIN_CHUNK_WORDS,
            "chunk_id_hash": CHUNK_ID_HASH,
            "sorted_sources": True,
        },
        "source_documents": source_documents,
    }


def _count_jsonl_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def verify_kb(kb_dir: str | Path) -> Tuple[bool, List[str]]:
    """Check that a built knowledge base still matches its manifest.

    Verifies that the manifest exists and is readable, that every recorded
    guideline document is present and unchanged (sha256), that no untracked
    guideline document appeared, and that ``chunks.jsonl`` exists with the
    recorded hash and record count.

    Returns ``(ok, problems)``; ``problems`` is empty when ``ok`` is True. This
    function never prints and never exits, so callers (e.g. preflight scripts)
    decide how to report.
    """
    kb_dir = Path(kb_dir)
    problems: List[str] = []

    manifest_path = kb_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return False, [f"KB manifest not found: {manifest_path}. {_BUILD_HINT}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, [f"KB manifest is not valid JSON: {manifest_path} ({exc})"]
    if not isinstance(manifest, dict):
        return False, [f"KB manifest is not a JSON object: {manifest_path}"]

    recorded: List[Dict[str, Any]] = [
        d for d in (manifest.get("source_documents") or []) if isinstance(d, dict)
    ]
    if not recorded:
        problems.append(f"KB manifest lists no source documents: {manifest_path}")

    guidelines_dir = kb_dir / GUIDELINES_DIRNAME
    if not guidelines_dir.is_dir():
        problems.append(f"Guidelines directory not found: {guidelines_dir}")
    else:
        for doc in recorded:
            filename = str(doc.get("filename", ""))
            md_path = guidelines_dir / filename
            if not md_path.exists():
                problems.append(f"Source document missing: {md_path}")
                continue
            expected = str(doc.get("sha256", ""))
            actual = _sha256_file(md_path)
            if actual != expected:
                problems.append(
                    f"Source document changed since build: {filename} "
                    f"(sha256 {actual[:12]} != manifest {expected[:12]})"
                )
        known = {str(d.get("filename", "")) for d in recorded}
        for md_path in sorted(guidelines_dir.glob(SOURCE_GLOB)):
            if md_path.name not in known:
                problems.append(
                    f"Source document not recorded in manifest: {md_path.name}. {_BUILD_HINT}"
                )

    expected_version = compute_kb_version(
        (str(d.get("filename", "")), str(d.get("sha256", ""))) for d in recorded
    )
    if recorded and str(manifest.get("kb_version", "")) != expected_version:
        problems.append(
            f"kb_version does not match the recorded source hashes: "
            f"{manifest.get('kb_version')} != {expected_version}"
        )

    chunks_path = kb_dir / str(manifest.get("chunks_file") or CHUNKS_FILENAME)
    if not chunks_path.exists():
        problems.append(f"Chunks file not found: {chunks_path}. {_BUILD_HINT}")
    else:
        expected = str(manifest.get("chunks_sha256", ""))
        actual = _sha256_file(chunks_path)
        if actual != expected:
            problems.append(
                f"Chunks file changed since build: {chunks_path.name} "
                f"(sha256 {actual[:12]} != manifest {expected[:12]})"
            )
        n_records = _count_jsonl_records(chunks_path)
        if n_records != manifest.get("n_chunks"):
            problems.append(
                f"Chunk count mismatch: {chunks_path.name} holds {n_records} records, "
                f"manifest records {manifest.get('n_chunks')}"
            )

    return not problems, problems
