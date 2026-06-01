"""Abstract interfaces every pluggable component implements.

Keeping these in one place makes the contracts explicit: a method produces a
``GenerationResult`` from a ``DashboardBrief``; a trainer produces an adapter
folder; a metric reduces predictions + references to a dict of numbers. None of
this imports torch/peft — the heavy work happens in concrete subclasses, which
import their dependencies lazily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.schemas import DashboardBrief, GenerationResult


class BaseMethod(ABC):
    """Unified contract for the four study methods (A, B, C, D)."""

    name: str = "base"

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def setup(self) -> None:
        """Load whatever the method needs (model, adapter, index)."""

    @abstractmethod
    def generate(self, brief: DashboardBrief) -> GenerationResult:
        """Produce a structured recommendation for a single brief."""

    def teardown(self) -> None:
        """Release resources (GPU memory, file handles). Optional override."""
        return None


class BaseRetriever(ABC):
    """Contract for RAG retrievers (used by methods B and D — stubbed in v1)."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[dict]:
        ...


class BaseTrainer(ABC):
    """Contract for fine-tuning algorithms (training side only)."""

    @abstractmethod
    def train(self, train_dataset, eval_dataset, output_dir: str) -> str:
        """Run training and return the path of the saved adapter/model folder."""


class BaseMetric(ABC):
    """Contract for automatic evaluation metrics."""

    name: str = "base"

    def __init__(self, cfg: Any | None = None) -> None:
        self.cfg = cfg

    @abstractmethod
    def compute(
        self,
        results: list[GenerationResult],
        references: list[dict],
    ) -> dict:
        """Reduce predictions (+ optional references) to a dict of numbers."""
