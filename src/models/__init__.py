"""Model wrappers.

The HuggingFace causal-LM wrapper is the only model backend in v1. It is
import-safe: torch/transformers are imported lazily inside ``load`` and PEFT is
imported only when an adapter is actually requested, so importing this module
never pulls in the training stack.
"""

from src.models.hf_causal import HFCausalModel

__all__ = ["HFCausalModel"]
