"""Build the RAG knowledge base from the guideline markdown files.

    python experiments/scripts/build_kb.py

Reads data/knowledge_base/guidelines/*.md, splits them into chunks, and writes
data/knowledge_base/chunks.jsonl (consumed by the TF-IDF retriever) plus
data/knowledge_base/kb_manifest.json, which records the sha256 of every source
document and of the chunks file. The manifest is tracked in Git (the chunks are
not), so a fresh clone can rebuild and verify it reproduced the same KB via
src.data_pipeline.kb_builder.verify_kb.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.kb_builder import MANIFEST_FILENAME, build_chunks, build_manifest
from src.utils.io import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the RAG knowledge base")
    p.add_argument("--guidelines-dir", default="data/knowledge_base/guidelines")
    p.add_argument("--out", default="data/knowledge_base/chunks.jsonl")
    p.add_argument(
        "--manifest",
        default="",
        help=f"Manifest path (default: {MANIFEST_FILENAME} next to --out)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gdir = Path(args.guidelines_dir)
    if not gdir.is_absolute():
        gdir = _PROJECT_ROOT / gdir
    out = Path(args.out)
    if not out.is_absolute():
        out = _PROJECT_ROOT / out
    manifest_path = Path(args.manifest) if args.manifest else out.parent / MANIFEST_FILENAME
    if not manifest_path.is_absolute():
        manifest_path = _PROJECT_ROOT / manifest_path

    chunks = build_chunks(gdir)
    if not chunks:
        raise SystemExit(f"No chunks built from {gdir} (no .md files?)")
    write_jsonl(chunks, out)
    manifest = build_manifest(gdir, chunks, out)
    write_json(manifest, manifest_path)

    print("=" * 56)
    print("KNOWLEDGE BASE BUILT")
    print("=" * 56)
    print(f"  Guidelines : {gdir}")
    print(f"  Output     : {out}")
    print(f"  Manifest   : {manifest_path}")
    print(f"  Documents  : {manifest['n_documents']}")
    print(f"  Chunks     : {len(chunks)}")
    print(f"  Sources    : {sorted({c['source'] for c in chunks})}")
    print(f"  kb_version : {manifest['kb_version']}")
    print(f"  chunks_sha : {manifest['chunks_sha256']}")
    print("=" * 56)


if __name__ == "__main__":
    main()
