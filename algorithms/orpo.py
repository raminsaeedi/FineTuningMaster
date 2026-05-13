"""
ORPO: Odds Ratio Preference Optimization.

Paper:  Hong et al., "ORPO: Monolithic Preference Optimization without
        Reference Model"  ACL 2024 — https://arxiv.org/abs/2403.07691
Requires: trl >= 0.8.6

Key insight
-----------
Standard DPO requires a frozen reference model (1× extra GPU memory).
ORPO combines SFT loss and a preference penalty in a SINGLE training step
with NO reference model:

    L_ORPO = L_SFT  +  λ · L_OR

where L_OR is an odds-ratio penalty that:
  - encourages the model to favour "chosen" responses
  - discourages "rejected" responses
  - uses the model's own log-probabilities (no reference needed)

This makes ORPO memory-efficient and simpler to set up than DPO while
still learning from preference signals.

Data format
-----------
ORPO requires preference pairs instead of single demonstrations:
  {"prompt": "...", "chosen": "...", "rejected": "..."}

pipeline/prepare_data.py generates these pairs automatically when
config['algorithm']['name'] == "orpo".  The chosen response is the
ground-truth structured recommendation; the rejected response is
a deliberately incomplete or ill-formatted variant.

Implementation notes
--------------------
    from trl import ORPOConfig, ORPOTrainer

    orpo_args = ORPOConfig(
        output_dir=str(self._checkpoint_dir),
        beta=config['orpo']['beta'],          # odds-ratio penalty weight
        max_prompt_length=config['orpo']['max_prompt_length'],
        max_length=config['orpo']['max_length'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        ...
    )
    trainer = ORPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=orpo_args,
        train_dataset=train_dataset,   # must have 'prompt', 'chosen', 'rejected' columns
    )
    trainer.train()

PEFT (optional): set config['orpo']['use_peft'] = true to wrap model with
LoRA before passing to ORPOTrainer (experimental).

Status: PLACEHOLDER
"""

from __future__ import annotations

import logging

from algorithms.base import BaseFineTuner

logger = logging.getLogger(__name__)


class ORPOFineTuner(BaseFineTuner):
    """
    ORPO fine-tuner — preference optimization without a reference model.

    TODO: Implement by following the implementation notes in this module docstring.
    Key differences from QLoRA:
      1. Dataset must have {prompt, chosen, rejected} columns (see prepare_data.py).
      2. Use ORPOTrainer + ORPOConfig from TRL instead of SFTTrainer.
      3. No reference model required.
    """

    def setup(self) -> None:
        raise NotImplementedError(
            "ORPO not yet implemented.\n"
            "To implement: see implementation notes in algorithms/orpo.py module docstring.\n"
            "IMPORTANT: also update pipeline/prepare_data.py to generate preference pairs\n"
            "(chosen/rejected) when algorithm.name == 'orpo'.\n"
            "Reference: https://arxiv.org/abs/2403.07691"
        )

    def train(self, train_dataset, eval_dataset=None) -> dict:
        raise NotImplementedError("ORPO not yet implemented — see setup() docstring.")

    def save(self) -> None:
        raise NotImplementedError("ORPO not yet implemented.")
