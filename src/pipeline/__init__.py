"""Orchestration layer.

The ExperimentRunner ties methods, the cached inference runner and the metrics
together for the local infer -> eval flow. Training is a separate, standalone
step (``scripts/train.py``) run on the GPU machine, so there is no train stage
here.
"""

from src.pipeline.runner import ExperimentRunner

__all__ = ["ExperimentRunner"]
