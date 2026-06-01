"""Core abstractions: schemas, registry, interfaces, constants, prompts.

This subpackage holds the contracts the rest of the codebase depends on and
imports nothing heavy (no torch, no peft). It is safe to import from anywhere,
including inference-only environments.
"""

from src.core.registry import METHODS, METRICS, RETRIEVERS, TRAINERS, Registry
from src.core.schemas import (
    DashboardBrief,
    DesignOutput,
    GenerationResult,
    GoldItem,
    KPIChartMapping,
)

__all__ = [
    "Registry",
    "METHODS",
    "RETRIEVERS",
    "METRICS",
    "TRAINERS",
    "DashboardBrief",
    "KPIChartMapping",
    "DesignOutput",
    "GoldItem",
    "GenerationResult",
]
