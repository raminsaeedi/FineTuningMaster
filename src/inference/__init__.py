"""Inference layer: JSON post-processing and the cached batch runner.

Import-safe — depends only on the model wrapper and core schemas, never on the
training stack.
"""

from src.inference.postprocess import parse_json_safe
from src.inference.runner import InferenceRunner

__all__ = ["parse_json_safe", "InferenceRunner"]
