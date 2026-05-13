"""
rag/build_index.py

Reads all Markdown guideline files, splits them into sections (chunks),
and builds a searchable index using TF-IDF (no external vector DB needed).

The index is saved as:
  rag/index/tfidf_index.pkl   — TF-IDF vectorizer + matrix
  rag/index/chunks.json       — the text chunks with metadata

Run this once before using retrieve_context.py:
  python rag/build_index.py

Requirements:
  pip install scikit-learn
"""

import json
import os
import pickle
import re

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(__file__)
GUIDELINES_DIR = os.path.join(BASE_DIR, "guidelines")
INDEX_DIR      = os.path.join(BASE_DIR, "index")
CHUNKS_FILE    = os.path.join(INDEX_DIR, "chunks.json")
TFIDF_FILE     = os.path.join(INDEX_DIR, "tfidf_index.pkl")


# ── Step 1: Load Markdown files ───────────────────────────────────────────────

def load_markdown_files(directory: str) -> list:
    """Return list of (filename, full_text) for every .md file in directory."""
    files = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".md"):
            path = os.path.join(directory, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            files.append((fname, text))
            print(f"  Loaded: {fname} ({len(text)} chars)")
    return files


# ── Step 2: Split into sections ───────────────────────────────────────────────

def split_into_sections(filename: str, text: str) -> list:
    """
    Split a Markdown file into sections at each ## heading.
    Returns list of dicts: {source, section_title, text}.
    """
    # Split on lines that start with ## (level-2 headings)
    parts = re.split(r"\n(?=## )", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Extract the heading as the section title
        lines = part.splitlines()
        title = lines[0].lstrip("#").strip() if lines else "Untitled"
        chunks.append({
            "source":        filename,
            "section_title": title,
            "text":          part,
        })
    return chunks


# ── Step 3: Build TF-IDF index ────────────────────────────────────────────────

def build_tfidf_index(chunks: list):
    """
    Fit a TF-IDF vectorizer on all chunk texts.
    Returns (vectorizer, tfidf_matrix).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        raise ImportError(
            "scikit-learn is required. Install it with:\n"
            "  pip install scikit-learn"
        )

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams for better matching
        stop_words="english",
        max_features=5000,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Loading guideline files ...")
    md_files = load_markdown_files(GUIDELINES_DIR)
    if not md_files:
        print(f"ERROR: No .md files found in {GUIDELINES_DIR}")
        return

    print("\nSplitting into sections ...")
    all_chunks = []
    for filename, text in md_files:
        sections = split_into_sections(filename, text)
        all_chunks.extend(sections)
        print(f"  {filename}: {len(sections)} sections")

    print(f"\nTotal chunks: {len(all_chunks)}")

    print("\nBuilding TF-IDF index ...")
    vectorizer, matrix = build_tfidf_index(all_chunks)
    print(f"  Vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"  Matrix shape    : {matrix.shape}")

    # Save chunks
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Save vectorizer + matrix
    with open(TFIDF_FILE, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f)

    print(f"\nIndex saved.")
    print(f"  Chunks file : {os.path.abspath(CHUNKS_FILE)}")
    print(f"  TF-IDF file : {os.path.abspath(TFIDF_FILE)}")
    print("\nDone. You can now use rag/retrieve_context.py.")


if __name__ == "__main__":
    main()
