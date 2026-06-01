"""
rag — Retrieval-Augmented Generation module.

Provides lightweight TF-IDF retrieval over dashboard design guidelines,
which can be injected into inference prompts to ground model outputs.

Quick start:
    from rag import RAGRetriever
    retriever = RAGRetriever()          # loads pre-built index
    context   = retriever.retrieve("Sales dashboard with monthly revenue")
    print(context)

Build the index once before first use:
    python -m rag.indexer
"""

from rag.retriever import RAGRetriever

__all__ = ["RAGRetriever"]
