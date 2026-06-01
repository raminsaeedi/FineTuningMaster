"""Source root for the dashboard-design fine-tuning thesis project.

The package is organised around a unified method interface so that the four
study methods (A: prompt-only, B: RAG, C: fine-tuning, D: FT+RAG) expose the
same ``generate(brief) -> GenerationResult`` contract and share one evaluation
stack. See ``src/core`` for the abstractions everything else builds on.
"""

__version__ = "0.2.0"
