"""
RSLoRA: Rank-Stabilized LoRA.

Paper:  Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning
        with LoRA"  2024 — https://arxiv.org/abs/2312.03732
Requires: peft >= 0.8.0

Key insight
-----------
Standard LoRA uses the scaling factor  alpha / r.  As rank r grows, this
factor shrinks, which destabilizes training at higher ranks (r=64, r=128).
RSLoRA replaces it with  alpha / sqrt(r), which keeps the effective scaling
constant regardless of rank.  This enables stable use of much larger ranks
for better task performance without changing anything else.

The experiment config (configs/experiments/rslora.yaml) sets r=32 (vs 16 for
QLoRA) to take advantage of the stable high-rank training.

Implementation notes
--------------------
RSLoRA is a one-flag change over QLoRA:

    from peft import LoraConfig
    lora_config = LoraConfig(
        ...,
        use_rslora=True,   # ← this is the entire change
    )

Everything else is identical to QLoRAFineTuner.

Status: PLACEHOLDER
"""

from __future__ import annotations

import logging
from pathlib import Path

from algorithms.base import BaseFineTuner

logger = logging.getLogger(__name__)


class RSLoRAFineTuner(BaseFineTuner):
    """
    RSLoRA fine-tuner — rank-stabilized LoRA scaling.

    TODO: Implement by copying QLoRAFineTuner and adding use_rslora=True to LoraConfig.
    """

    def setup(self) -> None:
        raise NotImplementedError(
            "RSLoRA not yet implemented.\n"
            "To implement: copy algorithms/qlora.py::QLoRAFineTuner.setup() and add\n"
            "  use_rslora=True\n"
            "to the LoraConfig constructor call.\n"
            "Reference: https://arxiv.org/abs/2312.03732"
        )

    def train(self, train_dataset, eval_dataset=None) -> dict:
        raise NotImplementedError("RSLoRA not yet implemented — see setup() docstring.")

    def save(self) -> None:
        raise NotImplementedError("RSLoRA not yet implemented.")
