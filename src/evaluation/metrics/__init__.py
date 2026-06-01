"""Automatic metric classes, registered under the METRICS registry.

Each metric implements ``BaseMetric.compute(results, references) -> dict`` so the
evaluation script can run any subset listed in the eval config without special
casing. Importing this package registers them all.
"""

from src.evaluation.metrics import (  # noqa: F401  (register on import)
    grounding,
    latency,
    llm_judge,
    macro_f1,
    schema_compliance,
    topk_accuracy,
)

__all__ = [
    "schema_compliance", "topk_accuracy", "macro_f1", "latency", "grounding", "llm_judge",
]
