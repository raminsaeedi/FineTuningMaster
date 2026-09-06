"""Abstract interfaces every pluggable component implements.

Keeping these in one place makes the contracts explicit: a method produces a
``GenerationResult`` from a ``DashboardBrief``; a trainer produces an adapter
folder; a metric reduces predictions + references to a dict of numbers. None of
this imports torch/peft — the heavy work happens in concrete subclasses, which
import their dependencies lazily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.core.schemas import DashboardBrief, GenerationResult


class BaseMethod(ABC):
    """Unified contract for the four study methods (A, B, C, D)."""

    name: str = "base"

    #: Items handed to one ``generate_batch`` call. ``1`` — the default — means
    #: the runner keeps its sequential per-item path, which is what every
    #: existing result was produced with. Only a method that implements a real
    #: batched path and whose config explicitly opts in raises this above 1.
    #: See :mod:`src.inference.batching` for why that opt-in is required.
    inference_batch_size: int = 1

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def setup(self) -> None:
        """Load whatever the method needs (model, adapter, index)."""

    @abstractmethod
    def generate(self, brief: DashboardBrief) -> GenerationResult:
        """Produce a structured recommendation for a single brief."""

    def generate_batch(
        self, briefs: list[DashboardBrief]
    ) -> list[GenerationResult | BaseException]:
        """Produce results for several briefs, in the order they were given.

        Returns one entry per brief: a ``GenerationResult``, or the exception
        that item raised. Returning failures instead of raising keeps a bad item
        from taking its whole batch down — the runner logs each one to
        ``errors*.jsonl`` exactly as it does in the sequential path.

        The default is a plain loop, so every method is safe to call this way.
        """
        outcomes: list[GenerationResult | BaseException] = []
        for brief in briefs:
            try:
                outcomes.append(self.generate(brief))
            except Exception as exc:  # per-item failure, recorded by the runner
                outcomes.append(exc)
        return outcomes

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
    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str,
        resume_from_checkpoint: Optional[str] = None,
    ) -> str:
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
