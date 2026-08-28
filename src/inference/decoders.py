"""Constrained JSON decoding via Outlines.

Constrains completed generation to the strict DesignOutput JSON schema.
Truncation or runtime errors can still prevent schema-valid output. This helps
separate *content* quality from *format* validity: report unconstrained results
and constrained results side by side.

Requires the ``[constrained]`` extra (outlines). Imports are lazy, so this module
is safe to import without outlines installed. The schema builder needs no extra.
"""

from __future__ import annotations

import json
from typing import Any


def design_output_json_schema() -> dict:
    """JSON schema for strict constrained generation."""
    from src.core.strict_response_schema import StrictDesignOutput

    return StrictDesignOutput.model_json_schema()


class ConstrainedDecoder:
    """Wraps an Outlines JSON generator around a loaded HF model + tokenizer."""

    def __init__(self, max_new_tokens: int = 1024) -> None:
        self.max_new_tokens = int(max_new_tokens)
        self._generator = None

    def setup(self, model: Any, tokenizer: Any) -> None:
        import outlines  # lazy
        from src.core.strict_response_schema import StrictDesignOutput

        ol_model = outlines.from_transformers(model, tokenizer)
        self._generator = outlines.Generator(ol_model, StrictDesignOutput)

    def generate(self, prompt: str) -> str:
        if self._generator is None:
            raise RuntimeError("ConstrainedDecoder.setup() must be called first.")
        result = self._generator(prompt, max_new_tokens=self.max_new_tokens)
        if isinstance(result, str):
            return result
        if hasattr(result, "model_dump_json"):
            return result.model_dump_json()
        return json.dumps(result, default=str)
