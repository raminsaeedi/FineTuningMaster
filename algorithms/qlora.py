"""
QLoRA: 4-bit quantized base model + LoRA adapters via PEFT + SFTTrainer.

Paper: Dettmers et al., NeurIPS 2023 — https://arxiv.org/abs/2305.14314
Status: IMPLEMENTED (baseline algorithm)

Key technique: bitsandbytes 4-bit NF4 quantization reduces VRAM by ~60%
while LoRA (rank-16 by default) limits trainable params to ~2-5M out of 500M+.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from algorithms.base import BaseFineTuner

logger = logging.getLogger(__name__)

# Suppress tokenizer parallelism warnings from HuggingFace
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class QLoRAFineTuner(BaseFineTuner):

    def setup(self) -> None:
        """Load base model with 4-bit quantization and apply LoRA adapters."""
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_cfg = self.config["model"]
        lora_cfg = self.config["lora"]
        model_name = model_cfg["name"]
        cache_dir = model_cfg.get("cache_dir")

        # ── Tokenizer ────────────────────────────────────────────────────────
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",   # Required for SFT causal LM
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = model_cfg.get("max_seq_length", 2048)

        # ── Quantization config ──────────────────────────────────────────────
        bnb_config = None
        if model_cfg.get("load_in_4bit", True):
            logger.info("Configuring 4-bit NF4 quantization (QLoRA)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,     # Saves ~0.4 GB extra
            )
        elif model_cfg.get("load_in_8bit", False):
            logger.info("Configuring 8-bit quantization")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # ── Base model ───────────────────────────────────────────────────────
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16 if not bnb_config else None,
            cache_dir=cache_dir,
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        logger.info(f"Model loaded — {self.model.num_parameters():,} parameters")

        # ── Prepare for k-bit training ───────────────────────────────────────
        if bnb_config:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True
            )

        # ── LoRA adapters ────────────────────────────────────────────────────
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
        """Train with SFTTrainer and return metrics."""
        from trl import SFTConfig, SFTTrainer

        train_cfg = self.config["training"]
        model_cfg = self.config["model"]

        # SFTConfig extends TrainingArguments with SFT-specific params (TRL >= 1.0)
        training_args = SFTConfig(
            output_dir=str(self._checkpoint_dir),
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg["weight_decay"],
            fp16=train_cfg.get("fp16", False),
            bf16=train_cfg.get("bf16", False),
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
        )

        # processing_class replaces tokenizer in TRL >= 0.9
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        logger.info("Training started…")
        result = self.trainer.train()
        metrics = result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        logger.info(f"Training complete. Loss: {metrics.get('train_loss', 'N/A'):.4f}")
        return metrics

    def save(self) -> None:
        """Save LoRA adapter weights and tokenizer."""
        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving LoRA adapter to {self._adapter_dir}")
        self.model.save_pretrained(str(self._adapter_dir))
        self.tokenizer.save_pretrained(str(self._adapter_dir))
        self._write_metadata({"quantization": "4bit_nf4"})
