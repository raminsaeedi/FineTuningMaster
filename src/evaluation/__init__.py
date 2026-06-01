"""Evaluation layer: automatic metrics, statistical tests, aggregation.

Importing this package registers all metric classes under the METRICS registry.
Nothing here imports the training stack.
"""

from src.evaluation import metrics  # noqa: F401  (register metrics on import)

__all__ = ["metrics"]
