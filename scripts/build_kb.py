"""Build the RAG knowledge base from the guideline markdown files.

    python scripts/build_kb.py

Reads data/knowledge_base/guidelines/*.md, splits them into chunks, and writes
data/knowledge_base/chunks.jsonl (consumed by the TF-IDF retriever).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.kb_builder import build_chunks
from src.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the RAG knowledge base")
    p.add_argument("--guidelines-dir", default="data/knowledge_base/guidelines")
    p.add_argument("--out", default="data/knowledge_base/chunks.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gdir = Path(args.guidelines_dir)
    if not gdir.is_absolute():
        gdir = _PROJECT_ROOT / gdir
    out = Path(args.out)
    if not out.is_absolute():
        out = _PROJECT_ROOT / out

    chunks = build_chunks(gdir)
    if not chunks:
        raise SystemExit(f"No chunks built from {gdir} (no .md files?)")
    write_jsonl(chunks, out)

    print("=" * 56)
    print("KNOWLEDGE BASE BUILT")
    print("=" * 56)
    print(f"  Guidelines : {gdir}")
    print(f"  Output     : {out}")
    print(f"  Chunks     : {len(chunks)}")
    print(f"  Sources    : {sorted({c['source'] for c in chunks})}")
    print("=" * 56)


if __name__ == "__main__":
    main()
