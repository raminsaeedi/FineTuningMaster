"""
LoftQ: LoRA-Fine-Tuning-Aware Quantization.

Paper: Guo et al., NeurIPS 2024 — https://arxiv.org/abs/2310.08659
Status: IMPLEMENTED

Key technique: Jointly optimises the quantization and LoRA initialisation so
that the quantisation error is minimised at the start of fine-tuning. This gives
a better starting point than QLoRA (which quantises independently) at the same
4-bit memory budget.

Implementation differences from QLoRA:
  - Model loaded in full FP16 precision (no bitsandbytes runtime quantisation)
  - LoftQConfig drives iterative SVD-based quantisation inside PEFT
  - init_lora_weights="loftq" tells PEFT to run the LoftQ initialisation
  - prepare_model_for_kbit_training is NOT called (model is not k-bit at load time)

Requires: peft >= 0.10.0  (LoftQConfig was added in 0.7.0, stabilised in 0.10)
"""

from __future__ import annotations

import logging
import os

from algorithms.base import BaseFineTuner

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class LoftQFineTuner(BaseFineTuner):

    def setup(self) -> None:
        """Load base model in FP16 and apply LoftQ quantisation-aware LoRA."""
        import torch
        from peft import LoftQConfig, LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_cfg = self.config["model"]
        lora_cfg = self.config["lora"]
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

        # ── Base model (full precision — LoftQ handles quantisation via PEFT) ─
        # LoftQ loads the model in float16 then iteratively quantises weight
        # matrices and computes SVD-initialised LoRA weights to compensate.
        logger.info(f"Loading model in FP16 for LoftQ initialisation: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16,
            cache_dir=cache_dir,
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        logger.info(f"Model loaded — {self.model.num_parameters():,} parameters")

        # ── LoftQ configuration ───────────────────────────────────────────────
        loftq_num_bits = lora_cfg.get("loftq_num_bits", 4)
        loftq_num_iter = lora_cfg.get("loftq_num_iter", 1)
        logger.info(
            f"Applying LoftQ: bits={loftq_num_bits}, iterations={loftq_num_iter}"
        )
        loftq_config = LoftQConfig(
            loftq_bits=loftq_num_bits,
            loftq_iter=loftq_num_iter,
        )

        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_cfg["target_modules"],
            init_lora_weights="loftq",
            loftq_config=loftq_config,
        )
        self.model = get_peft_model(self.model, lora_config)

        trainable, total = self.model.get_nb_trainable_parameters()
        pct = 100.0 * trainable / total if total > 0 else 0.0
        logger.info(f"LoftQ applied: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")

    def train(self, train_dataset, eval_dataset=None) -> dict:
        """Train with SFTTrainer and return metrics."""
        from trl import SFTConfig, SFTTrainer

        train_cfg = self.config["training"]

        training_args = SFTConfig(
            output_dir=str(self._checkpoint_dir),
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg["weight_decay"],
            fp16=train_cfg.get("fp16", True),   # FP16 training (model already in FP16)
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
        """Save LoftQ adapter weights and tokenizer."""
        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving LoftQ adapter to {self._adapter_dir}")
        self.model.save_pretrained(str(self._adapter_dir))
        self.tokenizer.save_pretrained(str(self._adapter_dir))
        lora_cfg = self.config.get("lora", {})
        self._write_metadata({
            "quantization": f"loftq_{lora_cfg.get('loftq_num_bits', 4)}bit",
            "loftq_num_bits": lora_cfg.get("loftq_num_bits", 4),
            "loftq_num_iter": lora_cfg.get("loftq_num_iter", 1),
        })
