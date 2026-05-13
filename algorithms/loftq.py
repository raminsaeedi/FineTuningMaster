"""
LoftQ: LoRA-Fine-Tuning-Aware Quantization.

Paper:  Guo et al., "LoftQ: LoRA-Fine-Tuning-Aware Quantization for
        Large Language Models"  NeurIPS 2024 — https://arxiv.org/abs/2310.08659
Requires: peft >= 0.10.0

Key insight
-----------
Standard QLoRA:  quantize first (losing information), then initialize LoRA
to zeros (which means early training only recovers the quantization error).

LoftQ:  jointly optimizes quantization and LoRA initialization via an
alternating procedure:
  1. Quantize the weight W → W_q
  2. Find LoRA matrices (A, B) such that W_q + A·B ≈ W  (minimizes Frobenius norm)
  3. Repeat for N iterations (typically 1–5)

Result: LoRA starts from a better initialisation that already accounts for
quantization error, leading to faster convergence and higher final quality
at the same memory budget as QLoRA.

Implementation notes
--------------------
Replace BitsAndBytesConfig with LoftQConfig from PEFT:

    from peft import LoftQConfig, LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    loftq_config = LoftQConfig(
        loftq_bits=4,       # from config['lora']['loftq_num_bits']
        loftq_iter=1,       # from config['lora']['loftq_num_iter']
    )
    lora_config = LoraConfig(
        ...,
        init_lora_weights="loftq",
        loftq_config=loftq_config,
    )
    # NOTE: model must be loaded in full precision (no bnb_config)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = get_peft_model(model, lora_config)

The SFTTrainer and save() logic are identical to QLoRAFineTuner.

Status: PLACEHOLDER
"""

from __future__ import annotations

import logging

from algorithms.base import BaseFineTuner

logger = logging.getLogger(__name__)


class LoftQFineTuner(BaseFineTuner):
    """
    LoftQ fine-tuner — quantization-aware LoRA initialization.

    TODO: Implement by following the implementation notes in this module docstring.
    Key difference from QLoRA: load model in fp16 first, then apply LoftQConfig
    via LoraConfig(init_lora_weights='loftq', loftq_config=...).
    """

    def setup(self) -> None:
        raise NotImplementedError(
            "LoftQ not yet implemented.\n"
            "To implement: see implementation notes in algorithms/loftq.py module docstring.\n"
            "Reference: https://arxiv.org/abs/2310.08659"
        )

    def train(self, train_dataset, eval_dataset=None) -> dict:
        raise NotImplementedError("LoftQ not yet implemented — see setup() docstring.")

    def save(self) -> None:
        raise NotImplementedError("LoftQ not yet implemented.")
