"""
train.py
========
Fine-tunes Qwen2.5-0.5B-Instruct with LoRA/QLoRA using the SFTTrainer
from the TRL library. Reads all settings from config.yaml.

Usage:
    python train.py
    python train.py --config config.yaml
    python train.py --config config.yaml --debug   # runs on 10 samples only

What this script does (step by step):
    1. Load configuration from config.yaml
    2. Detect compute device (GPU / CPU)
    3. Load the base model with optional 4-bit quantization (QLoRA)
    4. Apply LoRA adapters via PEFT
    5. Load and format the training dataset
    6. Run SFTTrainer (Supervised Fine-Tuning)
    7. Save the final LoRA adapter weights
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Suppress tokenizer parallelism warning ──────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.helpers import (
    ensure_dirs,
    format_training_example,
    load_config,
    load_jsonl,
    print_device_info,
    setup_logging,
)

logger = setup_logging(log_file="./outputs/logs/train.log")


# ============================================================
# Step 1 – Load Model & Tokenizer
# ============================================================

def load_model_and_tokenizer(config: dict):
    """
    Load the base model and tokenizer.

    For QLoRA (4-bit):
        - The model is loaded in 4-bit NF4 quantization via bitsandbytes
        - This reduces VRAM from ~1 GB to ~0.4 GB for the 0.5B model
        - LoRA adapters are trained in fp32/bf16 on top of the frozen 4-bit weights

    Args:
        config: Full configuration dictionary

    Returns:
        (model, tokenizer) tuple
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    model_name = model_cfg["name"]
    load_in_4bit = model_cfg.get("load_in_4bit", True)
    load_in_8bit = model_cfg.get("load_in_8bit", False)
    max_seq_length = model_cfg.get("max_seq_length", 2048)

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",   # Required for SFT with causal LM
    )

    # Qwen models may not have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # ── Quantization config (for QLoRA) ─────────────────────────────────────
    bnb_config = None
    if load_in_4bit:
        logger.info("Configuring 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4 is best for LLMs
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,     # Nested quantization saves ~0.4 GB
        )
    elif load_in_8bit:
        logger.info("Configuring 8-bit quantization")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # ── Load Model ───────────────────────────────────────────────────────────
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically place layers on available devices
        trust_remote_code=True,
        torch_dtype=torch.float16 if not load_in_4bit else None,
    )

    # Required for gradient checkpointing with quantized models
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer


# ============================================================
# Step 2 – Apply LoRA Adapters
# ============================================================

def apply_lora(model, config: dict):
    """
    Wrap the model with LoRA adapters using PEFT.

    LoRA (Low-Rank Adaptation) freezes the original model weights and
    adds small trainable matrices (rank r) to selected layers.
    This reduces trainable parameters from ~500M to ~2-5M.

    Args:
        model:  Loaded base model
        config: Full configuration dictionary

    Returns:
        PEFT model with LoRA adapters
    """
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    lora_cfg = config["lora"]

    # Prepare quantized model for training (important for QLoRA!)
    if config["model"].get("load_in_4bit") or config["model"].get("load_in_8bit"):
        logger.info("Preparing quantized model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameter count
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total if total > 0 else 0
    logger.info(
        f"LoRA applied: {trainable:,} trainable params "
        f"/ {total:,} total ({pct:.2f}%)"
    )
    model.print_trainable_parameters()

    return model


# ============================================================
# Step 3 – Prepare Dataset
# ============================================================

def prepare_dataset(config: dict, tokenizer, debug: bool = False):
    """
    Load JSONL data and format it into HuggingFace Dataset objects.

    Each example is formatted as a full instruction-response text using
    the model's chat template. The SFTTrainer handles tokenization.

    Args:
        config:    Full configuration dictionary
        tokenizer: Loaded tokenizer
        debug:     If True, use only 10 samples (for quick testing)

    Returns:
        HuggingFace Dataset object (train split)
    """
    from datasets import Dataset

    data_cfg = config["data"]
    train_file = data_cfg["train_file"]

    # Check if data file exists
    if not Path(train_file).exists():
        logger.error(
            f"Training data not found: {train_file}\n"
            f"Please run: python generate_dataset.py"
        )
        sys.exit(1)

    raw_data = load_jsonl(train_file)

    if debug:
        raw_data = raw_data[:10]
        logger.warning(f"DEBUG MODE: Using only {len(raw_data)} samples")

    max_samples = config["data"].get("max_samples")
    if max_samples:
        raw_data = raw_data[:max_samples]

    # Format each example into a single training text string
    formatted_texts = []
    for item in raw_data:
        text = format_training_example(
            brief=item["brief"],
            recommendation=item["recommendation"],
            tokenizer=tokenizer,
        )
        formatted_texts.append({"text": text})

    dataset = Dataset.from_list(formatted_texts)
    logger.info(f"Dataset prepared: {len(dataset)} examples")

    # Show a sample
    logger.debug(f"Sample text (first 300 chars):\n{dataset[0]['text'][:300]}...")

    return dataset


# ============================================================
# Step 4 – Configure Trainer
# ============================================================

def build_trainer(model, tokenizer, dataset, config: dict):
    """
    Build the SFTTrainer with training arguments.

    SFTTrainer (Supervised Fine-Tuning Trainer) from TRL handles:
    - Tokenization and padding
    - Loss computation (next-token prediction on the response only)
    - Gradient accumulation
    - Logging and checkpointing

    Args:
        model:     PEFT model with LoRA
        tokenizer: Loaded tokenizer
        dataset:   Formatted HuggingFace Dataset
        config:    Full configuration dictionary

    Returns:
        Configured SFTTrainer instance
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    train_cfg = config["training"]
    model_cfg = config["model"]

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
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
        evaluation_strategy=train_cfg.get("evaluation_strategy", "no"),
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg.get("seed", 42),
        # Gradient checkpointing saves VRAM at the cost of ~20% slower training
        gradient_checkpointing=True,
        dataloader_pin_memory=False,    # Set False to avoid issues on some systems
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",      # Column name in our dataset
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        args=training_args,
        packing=False,                  # Packing combines short examples; keep False for clarity
    )

    return trainer


# ============================================================
# Step 5 – Save Final Model
# ============================================================

def save_model(model, tokenizer, config: dict) -> None:
    """
    Save the trained LoRA adapter weights and tokenizer.

    Note: We save ONLY the LoRA adapter weights (~10-50 MB),
    NOT the full model (~1 GB). During inference, we load the
    base model and merge the adapter on top.

    Args:
        model:     Trained PEFT model
        tokenizer: Tokenizer
        config:    Full configuration dictionary
    """
    save_path = config["paths"]["final_model_dir"]
    Path(save_path).mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving LoRA adapter to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save training metadata
    metadata = {
        "base_model": config["model"]["name"],
        "lora_r": config["lora"]["r"],
        "lora_alpha": config["lora"]["lora_alpha"],
        "num_epochs": config["training"]["num_train_epochs"],
        "learning_rate": config["training"]["learning_rate"],
    }
    with open(Path(save_path) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved successfully!")
    logger.info(f"   Adapter path: {Path(save_path).resolve()}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune dashboard design recommendation model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: use only 10 samples, 1 epoch")
    args = parser.parse_args()

    # ── Load Config ──────────────────────────────────────────────────────────
    config = load_config(args.config)
    ensure_dirs(config)

    if args.debug:
        logger.warning("DEBUG MODE ACTIVE: Using minimal settings")
        config["training"]["num_train_epochs"] = 1
        config["training"]["save_steps"] = 5
        config["training"]["logging_steps"] = 1

    # ── Print Device Info ────────────────────────────────────────────────────
    print_device_info()

    # ── Step 1: Load Model ───────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 1: Loading model and tokenizer")
    logger.info("=" * 50)
    model, tokenizer = load_model_and_tokenizer(config)

    # ── Step 2: Apply LoRA ───────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 2: Applying LoRA adapters")
    logger.info("=" * 50)
    model = apply_lora(model, config)

    # ── Step 3: Prepare Dataset ──────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 3: Preparing dataset")
    logger.info("=" * 50)
    dataset = prepare_dataset(config, tokenizer, debug=args.debug)

    # ── Step 4: Train ────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 4: Starting training")
    logger.info("=" * 50)
    trainer = build_trainer(model, tokenizer, dataset, config)

    logger.info("Training started...")
    train_result = trainer.train()

    # Log training metrics
    metrics = train_result.metrics
    logger.info(f"Training complete! Metrics: {metrics}")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ── Step 5: Save Model ───────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STEP 5: Saving model")
    logger.info("=" * 50)
    save_model(model, tokenizer, config)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"   Loss:          {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Runtime:       {metrics.get('train_runtime', 0):.0f}s")
    print(f"   Samples/sec:   {metrics.get('train_samples_per_second', 0):.2f}")
    print(f"\n   Next step: python inference.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
