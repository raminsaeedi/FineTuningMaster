"""
rag/indexer.py — Build TF-IDF index over dashboard design guidelines.

Reads all Markdown files from rag/guidelines/, splits them into sections
at each ## heading, and builds a TF-IDF index saved to rag/index/.

Run once before using RAGRetriever:
    python -m rag.indexer

Output files:
    rag/index/chunks.json       — section texts with metadata
    rag/index/tfidf_index.pkl   — fitted TfidfVectorizer + sparse matrix

Requires: scikit-learn
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_RAG_DIR       = Path(__file__).resolve().parent
GUIDELINES_DIR = _RAG_DIR / "guidelines"
INDEX_DIR      = _RAG_DIR / "index"
CHUNKS_FILE    = INDEX_DIR / "chunks.json"
TFIDF_FILE     = INDEX_DIR / "tfidf_index.pkl"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_markdown_files(guidelines_dir: Path = GUIDELINES_DIR) -> list[tuple[str, str]]:
    """Return list of (filename, text) for every .md file in guidelines_dir."""
    files = []
    for path in sorted(guidelines_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        files.append((path.name, text))
        logger.debug(f"Loaded {path.name} ({len(text):,} chars)")
    return files


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_into_sections(filename: str, text: str) -> list[dict]:
    """
    Split a Markdown document into chunks at each ## heading.

    Returns a list of dicts with keys:
        source          — source filename (e.g. "chart_selection_guidelines.md")
        section_title   — heading text (stripped of # characters)
        text            — full section text including the heading line
    """
    parts = re.split(r"\n(?=## )", text)
    chunks: list[dict] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        title = lines[0].lstrip("#").strip() if lines else "Untitled"
        chunks.append({
            "source":        filename,
            "section_title": title,
            "text":          part,
        })
    return chunks


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_tfidf_index(chunks: list[dict]):
    """
    Fit a TF-IDF vectorizer over chunk texts.

    Returns:
        vectorizer  — fitted TfidfVectorizer
        matrix      — sparse (n_chunks × n_features) TF-IDF matrix
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for RAG indexing.\n"
            "Install it with: pip install scikit-learn"
        ) from exc

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams for richer matching
        stop_words="english",
        max_features=5_000,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def build_index(
    guidelines_dir: Path = GUIDELINES_DIR,
    index_dir: Path = INDEX_DIR,
    chunks_file: Path = CHUNKS_FILE,
    tfidf_file: Path = TFIDF_FILE,
) -> None:
    """Full pipeline: load guidelines → chunk → build TF-IDF → save."""
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading guidelines from {guidelines_dir} …")
    md_files = load_markdown_files(guidelines_dir)
    if not md_files:
        raise FileNotFoundError(
            f"No .md files found in {guidelines_dir}. "
            "Make sure the guidelines directory is populated."
        )

    all_chunks: list[dict] = []
    for filename, text in md_files:
        sections = split_into_sections(filename, text)
        all_chunks.extend(sections)
        logger.info(f"  {filename}: {len(sections)} sections")

    logger.info(f"Total chunks: {len(all_chunks)}")

    logger.info("Building TF-IDF index …")
    vectorizer, matrix = build_tfidf_index(all_chunks)
    logger.info(
        f"  Vocabulary: {len(vectorizer.vocabulary_):,} terms   "
        f"Matrix: {matrix.shape[0]} × {matrix.shape[1]}"
    )

    # Persist
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    with open(tfidf_file, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f)

    logger.info(f"Index saved to {index_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    print("Building RAG index …")
    build_index()
    print(f"\nDone.")
    print(f"  Chunks : {CHUNKS_FILE}")
    print(f"  Index  : {TFIDF_FILE}")
    print("\nYou can now use RAGRetriever in pipeline/inference.py.")


if __name__ == "__main__":
    main()
