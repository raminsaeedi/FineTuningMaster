"""
ORPO: Odds Ratio Preference Optimization.

Paper: Hong et al., ACL 2024 — https://arxiv.org/abs/2403.07691
Status: IMPLEMENTED

Key technique: Combines supervised fine-tuning loss with a preference
optimisation penalty in a single training step, without requiring a reference
model. The odds-ratio term penalises rejected responses while rewarding chosen
ones, making it simpler and more memory-efficient than DPO or PPO.

Dataset requirement: Each example must have three fields:
  - "prompt"   : instruction string (formatted with chat template)
  - "chosen"   : ground-truth structured JSON recommendation
  - "rejected" : deliberately incomplete / degraded response

pipeline/train.py generates this format automatically via
_format_orpo_examples() when algorithm.name == "orpo".

Implementation notes:
  - Uses ORPOTrainer + ORPOConfig from TRL >= 0.9
  - No quantisation by default (model loaded in bfloat16)
  - Optional LoRA via orpo.use_peft: true in experiment config
  - Saves full model weights (no PEFT adapter) unless use_peft is enabled

Requires: trl >= 0.8.6
"""

from __future__ import annotations

import logging
import os

from algorithms.base import BaseFineTuner

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ORPOFineTuner(BaseFineTuner):

    def setup(self) -> None:
        """Load base model in bfloat16; optionally apply LoRA if use_peft=True."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_cfg = self.config["model"]
        model_name = model_cfg["name"]
        cache_dir = model_cfg.get("cache_dir")
        orpo_cfg = self.config.get("orpo", {})
        self._use_peft = orpo_cfg.get("use_peft", False)

        # ── Tokenizer ────────────────────────────────────────────────────────
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",    # ORPO / causal generation prefers left-padding
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = model_cfg.get("max_seq_length", 2048)

        # ── Base model ───────────────────────────────────────────────────────
        # ORPO trains in full precision (bfloat16) — no quantisation required.
        logger.info(f"Loading model in bfloat16 (ORPO): {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        self.model.config.use_cache = False
        logger.info(f"Model loaded — {self.model.num_parameters():,} parameters")

        # ── Optional LoRA adapters ────────────────────────────────────────────
        if self._use_peft:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_cfg = self.config["lora"]
            logger.info("Applying LoRA adapters for ORPO+PEFT training")
            lora_config = LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["lora_alpha"],
                lora_dropout=lora_cfg["lora_dropout"],
                bias=lora_cfg.get("bias", "none"),
                task_type=TaskType.CAUSAL_LM,
                target_modules=lora_cfg["target_modules"],
            )
            self.model = get_peft_model(self.model, lora_config)
            trainable, total = self.model.get_nb_trainable_parameters()
            pct = 100.0 * trainable / total if total > 0 else 0.0
            logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")

    def train(self, train_dataset, eval_dataset=None) -> dict:
        """Train with ORPOTrainer and return metrics."""
        from trl import ORPOConfig, ORPOTrainer

        train_cfg = self.config["training"]
        orpo_cfg = self.config.get("orpo", {})

        # ORPOConfig extends TrainingArguments with ORPO-specific params.
        orpo_config = ORPOConfig(
            output_dir=str(self._checkpoint_dir),
            # ORPO-specific
            beta=orpo_cfg.get("beta", 0.1),
            max_length=orpo_cfg.get("max_length", 1024),
            max_prompt_length=orpo_cfg.get("max_prompt_length", 512),
            # Standard training
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg["weight_decay"],
            bf16=True,   # Model loaded in bfloat16
            fp16=False,
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            save_total_limit=train_cfg["save_total_limit"],
            eval_strategy=train_cfg.get("evaluation_strategy", "no"),
            report_to=train_cfg.get("report_to", "none"),
            seed=self.config.get("meta", {}).get("seed", 42),
            dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", False),
            remove_unused_columns=False,   # ORPO needs prompt/chosen/rejected columns
        )

        self.trainer = ORPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=orpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        logger.info("ORPO training started…")
        result = self.trainer.train()
        metrics = result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        logger.info(f"Training complete. Loss: {metrics.get('train_loss', 'N/A'):.4f}")
        return metrics

    def save(self) -> None:
        """Save model weights and tokenizer."""
        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving ORPO model to {self._adapter_dir}")
        self.model.save_pretrained(str(self._adapter_dir))
        self.tokenizer.save_pretrained(str(self._adapter_dir))
        orpo_cfg = self.config.get("orpo", {})
        self._write_metadata({
            "quantization": "none",
            "orpo_beta": orpo_cfg.get("beta", 0.1),
            "use_peft": self._use_peft,
        })
