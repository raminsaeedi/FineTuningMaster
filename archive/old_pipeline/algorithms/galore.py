"""
GaLore: Gradient Low-Rank Projection.

Paper: Zhao et al., ICML 2024 — https://arxiv.org/abs/2403.03507
Status: IMPLEMENTED

Key technique: Full-parameter fine-tuning that projects gradients into a
low-rank subspace during optimiser steps. Unlike LoRA, there is no rank
bottleneck in the forward pass — every weight in the model is updated —
but memory usage is kept comparable to LoRA via the gradient projection.
The low-rank subspace is periodically re-estimated (every update_proj_gap
steps) using SVD on a sample of gradient matrices.

Implementation notes:
  - No PEFT adapters — all model weights are trainable
  - Model loaded in bfloat16 (no bitsandbytes quantisation)
  - HuggingFace Trainer supports GaLore via optim="galore_adamw" in
    TrainingArguments/SFTConfig (requires galore-torch installed)
  - save() persists the full model weights (~1 GB for 0.5B bfloat16)
  - inference.py detects the absence of adapter_config.json and loads
    the full model directly

Requires: pip install galore-torch>=1.0
"""

from __future__ import annotations

import logging
import os

from algorithms.base import BaseFineTuner

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class GaLoreFineTuner(BaseFineTuner):

    def setup(self) -> None:
        """Load base model in bfloat16 — no adapters, no quantisation."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_cfg = self.config["model"]
        model_name = model_cfg["name"]
        cache_dir = model_cfg.get("cache_dir")

        # ── Tokenizer ────────────────────────────────────────────────────────
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = model_cfg.get("max_seq_length", 2048)

        # ── Base model (full precision, all parameters) ───────────────────────
        # GaLore trains all weights — no adapter config needed.
        logger.info(f"Loading full model in bfloat16 (GaLore): {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        self.model.config.use_cache = False
        n_params = self.model.num_parameters()
        logger.info(
            f"Model loaded — {n_params:,} parameters "
            f"({n_params * 2 / 1e9:.2f} GB in bfloat16)"
        )

    def train(self, train_dataset, eval_dataset=None) -> dict:
        """Train with GaLore-patched AdamW via SFTTrainer and return metrics."""
        from trl import SFTConfig, SFTTrainer

        train_cfg = self.config["training"]
        galore_cfg = self.config.get("galore", {})

        rank = galore_cfg.get("rank", 128)
        update_proj_gap = galore_cfg.get("update_proj_gap", 200)
        scale = galore_cfg.get("scale", 0.25)
        proj_type = galore_cfg.get("proj_type", "std")
        target_modules = galore_cfg.get("target_modules_list", ["attn", "mlp"])

        # HuggingFace Trainer dispatches to GaLoreAdamW when optim="galore_adamw".
        # The galore-torch package registers this optimizer name on import.
        # optim_args passes algorithm hyperparameters as a comma-separated string.
        optim_args = (
            f"rank={rank}, update_proj_gap={update_proj_gap}, "
            f"scale={scale}, proj_type={proj_type}"
        )

        logger.info(
            f"GaLore config — rank={rank}, update_proj_gap={update_proj_gap}, "
            f"scale={scale}, proj_type={proj_type}, targets={target_modules}"
        )

        training_args = SFTConfig(
            output_dir=str(self._checkpoint_dir),
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
            bf16=True,     # Model loaded in bfloat16
            fp16=False,
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            save_total_limit=train_cfg["save_total_limit"],
            eval_strategy=train_cfg.get("evaluation_strategy", "no"),
            report_to=train_cfg.get("report_to", "none"),
            seed=self.config.get("meta", {}).get("seed", 42),
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", False),
            remove_unused_columns=True,
            dataset_text_field="text",
            packing=train_cfg.get("packing", False),
            # GaLore optimizer settings
            optim="galore_adamw",
            optim_target_modules=target_modules,
            optim_args=optim_args,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        logger.info("GaLore training started…")
        result = self.trainer.train()
        metrics = result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        logger.info(f"Training complete. Loss: {metrics.get('train_loss', 'N/A'):.4f}")
        return metrics

    def save(self) -> None:
        """Save full model weights and tokenizer (no adapter — full model)."""
        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving full GaLore model to {self._adapter_dir}")
        # Full model save — no adapter_config.json will be written, which is
        # how pipeline/inference.py detects that this is a GaLore experiment.
        self.model.save_pretrained(str(self._adapter_dir))
        self.tokenizer.save_pretrained(str(self._adapter_dir))
        galore_cfg = self.config.get("galore", {})
        self._write_metadata({
            "quantization": "none",
            "full_model": True,
            "galore_rank": galore_cfg.get("rank", 128),
            "galore_update_proj_gap": galore_cfg.get("update_proj_gap", 200),
            "galore_scale": galore_cfg.get("scale", 0.25),
        })
