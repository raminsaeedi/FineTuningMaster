"""Training-data builders: raw source -> project ``GoldItem`` schema.

Augmentation sources are train/val only and never final evaluation gold:
"No dataset artifact, label set, or label-generation lineage is used both for
training/augmentation and final independent evaluation gold."

Builder selection is a simple explicit mapping (no registry); see
``docs/datasets/training_data_mapping.md``.
"""

from src.data_pipeline.builders.base import (
    BaseBuilder,
    DataNotProvisionedError,
    trainval_split,
)
from src.data_pipeline.builders.chartgpt_builder import ChartGPTBuilder
from src.data_pipeline.builders.nvbench_builder import NvBenchBuilder
from src.data_pipeline.builders.quda_builder import QudaBuilder
from src.data_pipeline.builders.synthetic_builder import SyntheticBuilder

#: explicit name -> builder class (only "synthetic" is functional today)
BUILDERS = {
    "synthetic": SyntheticBuilder,
    "chartgpt": ChartGPTBuilder,
    "nvbench": NvBenchBuilder,
    "quda": QudaBuilder,
}


def get_builder(name: str) -> type:
    """Return the builder class registered under ``name``."""
    if name not in BUILDERS:
        raise KeyError(f"unknown builder '{name}'; available: {sorted(BUILDERS)}")
    return BUILDERS[name]


__all__ = [
    "BUILDERS",
    "get_builder",
    "BaseBuilder",
    "DataNotProvisionedError",
    "trainval_split",
    "SyntheticBuilder",
    "ChartGPTBuilder",
    "NvBenchBuilder",
    "QudaBuilder",
]
