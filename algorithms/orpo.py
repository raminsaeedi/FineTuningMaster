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
  - TRL 1.3.0 dropped ORPOConfig/ORPOTrainer; this file implements ORPO
    from scratch using transformers.Trainer with a custom compute_loss().
  - No quantisation by default (model loaded in bfloat16)
  - Optional LoRA via orpo.use_peft: true in experiment config
  - Saves full model weights (no PEFT adapter) unless use_peft is enabled

Loss formula (Hong et al., 2024):
  L_ORPO = L_SFT + beta * L_OR
  L_SFT  = -E[mean_token_log_p(y_chosen | x)]
  L_OR   = -E[log sigmoid(log_odds_ratio(y_chosen, y_rejected | x))]
  where log_odds(y|x) = log_p(y|x) - log(1 - p(y|x))

Requires: transformers, peft (optional, for use_peft mode)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from algorithms.base import BaseFineTuner

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class _ORPODataCollator:
    """
    Pad a list of tokenised ORPO examples into batch tensors.

    Each example is expected to have keys:
        chosen_input_ids, chosen_attention_mask, chosen_labels
        rejected_input_ids, rejected_attention_mask, rejected_labels
    """
    pad_token_id: int
    max_length: int = 1024

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        import torch
        batch: dict[str, Any] = {}
        for key in (
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
        ):
            seqs = [f[key][: self.max_length] for f in features]
            max_len = max(len(s) for s in seqs)

            if "labels" in key:
                pad_val = -100
            elif "attention_mask" in key:
                pad_val = 0
            else:
                pad_val = self.pad_token_id

            padded = [s + [pad_val] * (max_len - len(s)) for s in seqs]
            batch[key] = torch.tensor(padded, dtype=torch.long)

        return batch


# ---------------------------------------------------------------------------
# Custom ORPO Trainer
# ---------------------------------------------------------------------------

class _ORPOTrainer:
    """
    Thin wrapper that monkey-patches compute_loss onto transformers.Trainer.

    We use inheritance-via-mixin so that all standard Trainer features
    (logging, checkpointing, gradient accumulation, etc.) work unchanged.
    """

    def __init__(self, orpo_beta: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.orpo_beta = orpo_beta

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_token_logps(logits, labels):
        """
        Compute the mean per-token log probability for each example in a batch.

        Only response tokens (labels != -100) contribute; prompt/pad positions
        are masked out.

        Returns a (batch_size,) tensor.
        """
        import torch
        import torch.nn.functional as F
        # Shift by one: predict token[t] from token[t-1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather the log prob of each target token
        target_ids = shift_labels.clamp(min=0)   # guard against -100 index
        token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # Mask prompt/pad positions
        response_mask = (shift_labels != -100).float()
        denom         = response_mask.sum(-1).clamp(min=1.0)
        return (token_logps * response_mask).sum(-1) / denom   # (B,)

    # ------------------------------------------------------------------
    # ORPO loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        """
        ORPO loss = SFT loss on chosen + beta * odds-ratio loss.

        Forward pass is run separately for chosen and rejected to allow
        gradient flow through both paths.
        """
        import torch
        import torch.nn.functional as F
        # --- Forward passes -------------------------------------------
        chosen_out = model(
            input_ids      = inputs["chosen_input_ids"],
            attention_mask = inputs["chosen_attention_mask"],
        )
        rejected_out = model(
            input_ids      = inputs["rejected_input_ids"],
            attention_mask = inputs["rejected_attention_mask"],
        )

        chosen_logps   = self._mean_token_logps(chosen_out.logits,   inputs["chosen_labels"])
        rejected_logps = self._mean_token_logps(rejected_out.logits, inputs["rejected_labels"])

        # --- SFT loss (NLL on chosen responses) -----------------------
        sft_loss = -chosen_logps.mean()

        # --- Odds-ratio loss ------------------------------------------
        # log_odds(y|x) = log p(y|x) - log(1 - p(y|x))
        # Numerically stable form using log1p(-exp(logp))
        eps = 1e-7
        log_odds_chosen   = chosen_logps   - torch.log1p(-chosen_logps.exp().clamp(max=1 - eps))
        log_odds_rejected = rejected_logps - torch.log1p(-rejected_logps.exp().clamp(max=1 - eps))
        log_odds_ratio    = log_odds_chosen - log_odds_rejected
        or_loss = -F.logsigmoid(log_odds_ratio).mean()

        loss = sft_loss + self.orpo_beta * or_loss

        logger.debug(
            f"ORPO loss — sft={sft_loss.item():.4f}  or={or_loss.item():.4f}  "
            f"total={loss.item():.4f}"
        )

        return (loss, chosen_out) if return_outputs else loss


# ---------------------------------------------------------------------------
# Fine-tuner class
# ---------------------------------------------------------------------------

class ORPOFineTuner(BaseFineTuner):

    def setup(self) -> None:
        """Load base model in bfloat16; optionally apply LoRA if use_peft=True."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_cfg = self.config["model"]
        model_name = model_cfg["name"]
        cache_dir  = model_cfg.get("cache_dir")
        orpo_cfg   = self.config.get("orpo", {})
        self._use_peft = orpo_cfg.get("use_peft", False)

        # ── Tokenizer ────────────────────────────────────────────────────────
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",    # left-pad for causal generation
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = model_cfg.get("max_seq_length", 2048)

        # ── Base model ───────────────────────────────────────────────────────
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
            logger.info(
                f"LoRA applied: {trainable:,} trainable / {total:,} total ({pct:.2f}%)"
            )

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def _tokenize_dataset(self, dataset, max_length: int, max_prompt_length: int):
        """Map {prompt, chosen, rejected} → tokenised tensors."""
        tokenizer = self.tokenizer

        def tokenize_fn(example: dict) -> dict:
            prompt   = example["prompt"]
            chosen   = example["chosen"]
            rejected = example["rejected"]

            # Prompt token count (needed to mask labels)
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=True,
                max_length=max_prompt_length,
            )["input_ids"]
            n_prompt = len(prompt_ids)

            def encode_pair(response: str) -> dict:
                enc = tokenizer(
                    prompt + response,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    add_special_tokens=True,
                )
                ids    = enc["input_ids"]
                labels = [-100] * min(n_prompt, len(ids)) + ids[min(n_prompt, len(ids)):]
                return {
                    "input_ids":      ids,
                    "attention_mask": enc["attention_mask"],
                    "labels":         labels,
                }

            c = encode_pair(chosen)
            r = encode_pair(rejected)
            return {
                "chosen_input_ids":       c["input_ids"],
                "chosen_attention_mask":  c["attention_mask"],
                "chosen_labels":          c["labels"],
                "rejected_input_ids":     r["input_ids"],
                "rejected_attention_mask": r["attention_mask"],
                "rejected_labels":        r["labels"],
            }

        return dataset.map(
            tokenize_fn,
            batched=False,
            remove_columns=dataset.column_names,
            desc="Tokenising ORPO pairs",
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_dataset, eval_dataset=None) -> dict:
        """Train with the custom ORPO loss and return metrics."""
        from transformers import Trainer, TrainingArguments

        train_cfg = self.config["training"]
        orpo_cfg  = self.config.get("orpo", {})
        beta             = orpo_cfg.get("beta", 0.1)
        max_length       = orpo_cfg.get("max_length", 1024)
        max_prompt_length = orpo_cfg.get("max_prompt_length", 512)

        logger.info(
            f"ORPO config — beta={beta}, max_length={max_length}, "
            f"max_prompt_length={max_prompt_length}"
        )

        # Tokenise
        logger.info("Tokenising training dataset for ORPO …")
        tok_train = self._tokenize_dataset(train_dataset, max_length, max_prompt_length)
        tok_eval  = (
            self._tokenize_dataset(eval_dataset, max_length, max_prompt_length)
            if eval_dataset else None
        )

        data_collator = _ORPODataCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=max_length,
        )

        # Build a concrete Trainer class that mixes in the ORPO loss
        class ORPOTrainerConcrete(_ORPOTrainer, Trainer):
            pass

        training_args = TrainingArguments(
            output_dir=str(self._checkpoint_dir),
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg["weight_decay"],
            bf16=True,
            fp16=False,
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            save_total_limit=train_cfg["save_total_limit"],
            eval_strategy=train_cfg.get("evaluation_strategy", "no"),
            report_to=train_cfg.get("report_to", "none"),
            seed=self.config.get("meta", {}).get("seed", 42),
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", False),
            remove_unused_columns=False,   # collator handles columns
        )

        self.trainer = ORPOTrainerConcrete(
            orpo_beta=beta,
            model=self.model,
            args=training_args,
            train_dataset=tok_train,
            eval_dataset=tok_eval,
            data_collator=data_collator,
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
