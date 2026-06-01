"""Constrained JSON decoding via Outlines.

Forces generation to conform to the DesignOutput JSON schema, guaranteeing
schema-valid output. This decouples *content* quality from *format* validity:
report unconstrained results (the model's inherent ability) and constrained
results (what perfect formatting would yield) side by side.

Requires the ``[constrained]`` extra (outlines). Imports are lazy, so this module
is safe to import without outlines installed. The schema builder needs no extra.
"""

from __future__ import annotations

import json
from typing import Any


def design_output_json_schema() -> dict:
    """JSON schema for a valid DesignOutput (derived from the Pydantic model)."""
    from src.core.schemas import DesignOutput

    return DesignOutput.model_json_schema()


class ConstrainedDecoder:
    """Wraps an Outlines JSON generator around a loaded HF model + tokenizer."""

    def __init__(self, max_new_tokens: int = 1024) -> None:
        self.max_new_tokens = int(max_new_tokens)
        self._generator = None

    def setup(self, model: Any, tokenizer: Any) -> None:
        import outlines  # lazy

        ol_model = outlines.models.Transformers(model, tokenizer)
        schema = json.dumps(design_output_json_schema())
        self._generator = outlines.generate.json(ol_model, schema)

    def generate(self, prompt: str) -> str:
        if self._generator is None:
            raise RuntimeError("ConstrainedDecoder.setup() must be called first.")
        result = self._generator(prompt, max_tokens=self.max_new_tokens)
        # Outlines returns a dict/obj that already satisfies the schema.
        return result if isinstance(result, str) else json.dumps(result, default=str)
