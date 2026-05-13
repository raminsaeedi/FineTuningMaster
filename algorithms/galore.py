"""
GaLore: Gradient Low-Rank Projection.

Paper:  Zhao et al., "GaLore: Memory-Efficient LLM Training by Gradient
        Low-Rank Projection"  ICML 2024 — https://arxiv.org/abs/2403.03507
Requires: pip install galore-torch>=1.0

Key insight
-----------
LoRA reduces memory by adding low-rank adapter matrices (A, B) while
keeping the original weights frozen — but this introduces a rank bottleneck
in every forward pass.

GaLore instead trains ALL weights full-rank, but projects the gradient into
a low-rank subspace during the optimizer step:
  1. Compute full gradient G  (shape: [d_out, d_in])
  2. Project:  G_low = P^T · G · Q   (rank-r subspace)
  3. Apply AdamW update in the low-rank space
  4. Project back to full space

The subspace (P, Q) is re-computed every `update_proj_gap` steps.
Memory savings come from the optimizer state (Adam m, v) being stored in
the low-rank space rather than the full parameter space.

This means GaLore updates full-parameter weights — unlike LoRA there is no
adapter bottleneck — which may be important for complex structured output tasks.

Implementation notes
--------------------
GaLore does NOT use PEFT adapters.  The model is loaded in full precision.
The GaLore optimizer replaces AdamW:

    from galore_torch import GaLoreAdamW
    from transformers import get_scheduler

    # Load full-precision model (no bitsandbytes)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Build parameter groups — GaLore params vs. non-GaLore params
    galore_params = [
        {"params": [...], "rank": config['galore']['rank'],
         "update_proj_gap": config['galore']['update_proj_gap'],
         "scale": config['galore']['scale'],
         "proj_type": config['galore']['proj_type']}
    ]
    optimizer = GaLoreAdamW(galore_params + non_galore_params, lr=lr)

    # Use a standard HuggingFace training loop or Trainer with a custom optimizer.
    # NOTE: GaLore is not yet natively supported by TRL SFTTrainer's optimizer arg;
    # a custom training loop may be needed.

Status: PLACEHOLDER
"""

from __future__ import annotations

import logging

from algorithms.base import BaseFineTuner

logger = logging.getLogger(__name__)


class GaLoreFineTuner(BaseFineTuner):
    """
    GaLore fine-tuner — full-parameter training via gradient projection.

    TODO: Implement by following the implementation notes in this module docstring.
    Key differences from QLoRA:
      1. No PEFT adapters — all weights are trained.
      2. Model loaded in full precision (no bitsandbytes quantization).
      3. AdamW is replaced by GaLoreAdamW from the galore-torch package.
      4. May require a custom training loop (GaLore optimizer + HF Trainer).
    """

    def setup(self) -> None:
        raise NotImplementedError(
            "GaLore not yet implemented.\n"
            "To implement: see implementation notes in algorithms/galore.py module docstring.\n"
            "Install dependency first:  pip install galore-torch>=1.0\n"
            "Reference: https://arxiv.org/abs/2403.03507"
        )

    def train(self, train_dataset, eval_dataset=None) -> dict:
        raise NotImplementedError("GaLore not yet implemented — see setup() docstring.")

    def save(self) -> None:
        raise NotImplementedError("GaLore not yet implemented.")
