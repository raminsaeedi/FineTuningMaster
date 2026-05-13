"""
rag/retrieve_context.py

Retrieves the most relevant guideline sections for a given dashboard brief
using the TF-IDF index built by build_index.py.

Usage:
  # As a module (import into other scripts)
  from rag.retrieve_context import retrieve

  context = retrieve("Sales dashboard with monthly revenue and regional breakdown")
  print(context)

  # As a standalone script (demo mode)
  python rag/retrieve_context.py

Requirements:
  pip install scikit-learn
  Run rag/build_index.py first to create the index.
"""

import json
import os
import pickle

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
INDEX_DIR   = os.path.join(BASE_DIR, "index")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")
TFIDF_FILE  = os.path.join(INDEX_DIR, "tfidf_index.pkl")


# ── Load index (cached after first call) ─────────────────────────────────────
_cache = {}

def _load_index():
    """Load chunks and TF-IDF index from disk. Cached in memory after first load."""
    if _cache:
        return _cache["chunks"], _cache["vectorizer"], _cache["matrix"]

    if not os.path.exists(CHUNKS_FILE) or not os.path.exists(TFIDF_FILE):
        raise FileNotFoundError(
            "Index files not found. Run build_index.py first:\n"
            "  python rag/build_index.py"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(TFIDF_FILE, "rb") as f:
        data = pickle.load(f)

    _cache["chunks"]     = chunks
    _cache["vectorizer"] = data["vectorizer"]
    _cache["matrix"]     = data["matrix"]

    return chunks, data["vectorizer"], data["matrix"]


# ── Core retrieval function ───────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> str:
    """
    Retrieve the top_k most relevant guideline sections for the given query.

    Parameters
    ----------
    query : str
        The dashboard brief or any free-text query.
    top_k : int
        Number of sections to return (default: 3).

    Returns
    -------
    str
        A formatted string containing the retrieved guideline sections,
        ready to be injected into a prompt.
    """
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise ImportError(
            "scikit-learn and numpy are required.\n"
            "Install with: pip install scikit-learn numpy"
        )

    chunks, vectorizer, matrix = _load_index()

    # Vectorize the query
    query_vec = vectorizer.transform([query])

    # Compute cosine similarity between query and all chunks
    scores = cosine_similarity(query_vec, matrix).flatten()

    # Get top_k indices sorted by score (highest first)
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Build the context string
    sections = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk  = chunks[idx]
        score  = scores[idx]
        source = chunk["source"].replace(".md", "").replace("_", " ").title()
        title  = chunk["section_title"]
        text   = chunk["text"].strip()
        sections.append(
            f"[Guideline {rank} | Source: {source} | Section: {title} | Score: {score:.3f}]\n"
            f"{text}"
        )

    if not sections:
        return "No relevant guidelines found."

    return "\n\n---\n\n".join(sections)


def retrieve_as_list(query: str, top_k: int = 3) -> list:
    """
    Same as retrieve() but returns a list of dicts instead of a formatted string.
    Useful when you want to process the results programmatically.

    Each dict has keys: source, section_title, text, score.
    """
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise ImportError("pip install scikit-learn numpy")

    chunks, vectorizer, matrix = _load_index()
    query_vec = vectorizer.transform([query])
    scores    = cosine_similarity(query_vec, matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "source":        chunks[i]["source"],
            "section_title": chunks[i]["section_title"],
            "text":          chunks[i]["text"],
            "score":         float(scores[i]),
        }
        for i in top_indices
    ]


# ── Demo / standalone mode ────────────────────────────────────────────────────

DEMO_QUERIES = [
    "Sales dashboard with monthly revenue trends and regional comparison",
    "I need to show market share by product category",
    "Dashboard for HR team tracking employee turnover and headcount",
    "How should I handle color for users with color blindness?",
    "What filters should I add to a logistics dashboard?",
]

def main():
    print("=" * 65)
    print("RAG Context Retrieval — Demo")
    print("=" * 65)

    for query in DEMO_QUERIES:
        print(f"\nQUERY: {query}")
        print("-" * 65)
        try:
            context = retrieve(query, top_k=2)
            # Print only the first 600 chars of each result to keep output readable
            preview = context[:600] + (" ... [truncated]" if len(context) > 600 else "")
            print(preview)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            break
        print()

    print("=" * 65)
    print("Demo complete. Use retrieve() in your scripts to inject context into prompts.")


if __name__ == "__main__":
    main()
