"""The four study methods, registered under the METHODS registry.

Importing this package registers every method (A: prompt_only, B: rag,
C: ft, D: ft_rag) so they can be resolved by key. Methods B and D are
registered stubs in v1. None of these imports pull in the training stack.
"""

from src.methods import ft, ft_rag, prompt_only, rag  # noqa: F401  (register on import)

__all__ = ["prompt_only", "ft", "rag", "ft_rag"]
