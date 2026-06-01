"""Method A — prompt-only.

Runs the base instruction model with no adapter and no retrieval. This is the
baseline the other three methods are measured against.
"""

from __future__ import annotations

from src.core.registry import METHODS
from src.methods.base import HFMethod


@METHODS.register("prompt_only")
class PromptOnlyMethod(HFMethod):
    name = "prompt_only"
