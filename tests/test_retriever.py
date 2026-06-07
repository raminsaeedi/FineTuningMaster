"""Knowledge-base chunking + TF-IDF retriever tests."""

from src.core.registry import RETRIEVERS
from src.data_pipeline.kb_builder import _chunk_markdown


def test_chunk_markdown_splits_on_headings():
    md = (
        "# Title\n"
        "## Section A\nThis section is about line charts for revenue trends over time.\n"
        "## Section B\nThis section is about accessible color palettes and contrast ratios.\n"
    )
    chunks = _chunk_markdown(md, source="demo")
    headings = {c["heading"] for c in chunks}
    assert "Section A" in headings and "Section B" in headings
    assert all(c["id"].startswith("demo_") for c in chunks)


def test_tfidf_retriever_ranks_relevant_chunk_first():
    import src.retrievers  # noqa: F401  (register)

    retriever = RETRIEVERS.get("tfidf")
    inst = retriever({"top_k": 2})
    # Inject chunks directly to avoid filesystem dependency.
    inst.chunks = [
        {"id": "a", "source": "s", "heading": "Line charts", "text": "use a line chart for trends over time"},
        {"id": "b", "source": "s", "heading": "Color", "text": "ensure sufficient contrast ratio for accessibility"},
    ]
    from sklearn.feature_extraction.text import TfidfVectorizer

    inst.vectorizer = TfidfVectorizer(stop_words="english")
    inst.matrix = inst.vectorizer.fit_transform([c["text"] for c in inst.chunks])

    hits = inst.retrieve("line chart trend over time", k=2)
    assert hits and hits[0]["id"] == "a"


def test_all_four_methods_registered_includes_rag():
    import src.methods  # noqa: F401

    from src.core.registry import METHODS

    assert {"prompt_only", "ft", "rag", "ft_rag"} <= set(METHODS.keys())
