"""
Algorithm registry — maps algorithm name strings to fine-tuner classes.

Usage:
    from algorithms import get_algorithm
    cls = get_algorithm("qlora")
    finetuner = cls(config, experiment_dir)
    metrics = finetuner.run(train_dataset)
"""

from algorithms.base import BaseFineTuner
from algorithms.dora import DoRAFineTuner
from algorithms.galore import GaLoreFineTuner
from algorithms.loftq import LoftQFineTuner
from algorithms.orpo import ORPOFineTuner
from algorithms.qlora import QLoRAFineTuner
from algorithms.rslora import RSLoRAFineTuner

ALGORITHM_REGISTRY: dict[str, type[BaseFineTuner]] = {
    "qlora":  QLoRAFineTuner,
    "dora":   DoRAFineTuner,
    "rslora": RSLoRAFineTuner,
    "loftq":  LoftQFineTuner,
    "orpo":   ORPOFineTuner,
    "galore": GaLoreFineTuner,
}

IMPLEMENTED = set(ALGORITHM_REGISTRY)   # all algorithms are now implemented


def get_algorithm(name: str) -> type[BaseFineTuner]:
    """Return the fine-tuner class for the given algorithm name."""
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. "
            f"Available: {sorted(ALGORITHM_REGISTRY.keys())}"
        )
    return ALGORITHM_REGISTRY[name]
