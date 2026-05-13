"""
DoRA: Weight-Decomposed Low-Rank Adaptation.

Paper:  Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation"
        ICML 2024 — https://arxiv.org/abs/2402.09353
Requires: peft >= 0.10.0

Key insight
-----------
Standard LoRA updates weights as  W + A·B  where A,B are low-rank matrices.
DoRA first decomposes the pre-trained weight into magnitude (‖column‖) and
direction (unit column vector), then applies LoRA only to the directional
component.  This separates "how much" from "which direction", giving the
adapter more expressive power at the same rank — with negligible extra cost.

Implementation notes
--------------------
DoRA is a one-flag change over QLoRA:

    from peft import LoraConfig
    lora_config = LoraConfig(
        ...,
        use_dora=True,   # ← this is the entire change
    )

Everything else (BitsAndBytesConfig, SFTTrainer, training args) is
identical to QLoRAFineTuner.  Copy qlora.py, add use_dora=True, done.

Status: PLACEHOLDER
"""

from __future__ import annotations

import logging
from pathlib import Path

from algorithms.base import BaseFineTuner

logger = logging.getLogger(__name__)


class DoRAFineTuner(BaseFineTuner):
    """
    DoRA fine-tuner — weight-decomposed LoRA.

    TODO: Implement by copying QLoRAFineTuner and adding use_dora=True to LoraConfig.
    """

    def setup(self) -> None:
        raise NotImplementedError(
            "DoRA not yet implemented.\n"
            "To implement: copy algorithms/qlora.py::QLoRAFineTuner.setup() and add\n"
            "  use_dora=True\n"
            "to the LoraConfig constructor call.\n"
            "Reference: https://arxiv.org/abs/2402.09353"
        )

    def train(self, train_dataset, eval_dataset=None) -> dict:
        raise NotImplementedError("DoRA not yet implemented — see setup() docstring.")

    def save(self) -> None:
        raise NotImplementedError("DoRA not yet implemented.")
