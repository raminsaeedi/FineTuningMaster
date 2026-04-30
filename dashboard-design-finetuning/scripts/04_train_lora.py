"""
scripts/04_train_lora.py
=========================
Fine-tunes Qwen2.5-0.5B-Instruct with LoRA (or QLoRA) using the
SFTTrainer from the TRL library.

What this script does step by step:
    1. Load config from config/train_config.yaml
    2. Load the base model (with optional 4-bit quantization for QLoRA)
    3. Apply LoRA adapters to the model (only ~2-5M params become trainable)
    4. Load the formatted training dataset from data/processed/train.jsonl
    5. Train with SFTTrainer (handles tokenization, loss, checkpointing)
    6. Save the LoRA adapter weights to outputs/models/final/

Usage:
    # Normal training
    python scripts/04_train_lora.py

    # Quick test with 10 samples and 1 epoch (to check everything works)
    python scripts/04_train_lora.py --debug

Requirements:
    - GPU strongly recommended (training on CPU takes hours)
    - For Google Colab: use notebooks/colab_finetuning_qwen_0_5b.ipynb
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Suppress tokenizer parallelism warning (safe to ignore)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


# ============================================================
# Config Loader
# ============================================================

def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "train_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Step 1: Load Model and Tokenizer
# ============================================================

def load_model_and_tokenizer(config: dict):
    """
    Load the base model and tokenizer from HuggingFace.

    If load_in_4bit is True (QLoRA):
        - Model weights are quantized to 4-bit NF4 format
        - This reduces VRAM from ~1 GB to ~0.4 GB for the 0.5B model
        - LoRA adapters are trained in fp16/fp32 on top of frozen 4-bit weights

    Args:
        config: Full configuration dictionary

    Returns:
        (model, tokenizer) tuple
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    model_name    = model_cfg["name"]
    load_in_4bit  = model_cfg.get("load_in_4bit", True)
    load_in_8bit  = model_cfg.get("load_in_8bit", False)
    max_seq_len   = model_cfg.get("max_seq_length", 2048)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",   # Required for causal LM training
    )
    # Qwen models may not have a pad token – use eos_token as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  Set pad_token = eos_token")

    # Build quantization config for QLoRA
    bnb_config = None
    if load_in_4bit:
        print("  Quantization: 4-bit NF4 (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NF4 is best for LLMs
            bnb_4bit_compute_dtype=torch.float16, # Compute in fp16
            bnb_4bit_use_double_quant=True,       # Nested quantization saves ~0.4 GB extra
        )
    elif load_in_8bit:
        print("  Quantization: 8-bit")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        print("  Quantization: none (full precision)")

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically place on GPU/CPU
        trust_remote_code=True,
        torch_dtype=torch.float16 if not load_in_4bit else None,
    )

    # Required settings for training
    model.config.use_cache = False          # Disable KV cache during training
    model.config.pretraining_tp = 1         # Tensor parallelism = 1 (single GPU)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded. Total parameters: {total_params:,}")
    return model, tokenizer


# ============================================================
# Step 2: Apply LoRA Adapters
# ============================================================

def apply_lora(model, config: dict):
    """
    Wrap the model with LoRA adapters using PEFT.

    LoRA adds small trainable matrices (rank r) to selected layers.
    The original model weights are FROZEN – only the LoRA matrices train.

    Example with r=16:
        Original weight matrix: 4096 x 4096 = 16.7M parameters (frozen)
        LoRA matrices: 4096x16 + 16x4096 = 131K parameters (trainable)
        Reduction: 99.2% fewer trainable parameters!

    Args:
        model:  Loaded base model
        config: Full configuration dictionary

    Returns:
        PEFT model with LoRA adapters applied
    """
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    lora_cfg = config["lora"]
    model_cfg = config["model"]

    # For quantized models (QLoRA), we need this preparation step
    if model_cfg.get("load_in_4bit") or model_cfg.get("load_in_8bit"):
        print("Preparing quantized model for k-bit training...")
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(f"LoRA applied:")
    print(f"  Trainable parameters: {trainable:,} ({pct:.2f}%)")
    print(f"  Frozen parameters:    {total - trainable:,}")

    return model


# ============================================================
# Step 3: Load Dataset
# ============================================================

def load_training_dataset(config: dict, debug: bool = False):
    """
    Load the formatted training data from data/processed/train.jsonl.

    Each record has a "text" field containing the full ChatML-formatted
    training example (system + user + assistant turns).

    Args:
        config: Full configuration dictionary
        debug:  If True, use only 10 samples (for quick testing)

    Returns:
        HuggingFace Dataset object
    """
    from datasets import Dataset

    data_cfg = config["data"]
    train_path = PROJECT_ROOT / data_cfg["train_file"].lstrip("./")

    if not train_path.exists():
        print(f"[ERROR] Training data not found: {train_path}")
        print("        Run scripts 02 and 03 first.")
        sys.exit(1)

    # Load JSONL
    records = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if debug:
        records = records[:10]
        print(f"DEBUG MODE: Using only {len(records)} samples")

    print(f"Loaded {len(records)} training examples from {train_path}")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(records)
    return dataset


# ============================================================
# Step 4: Build Trainer
# ============================================================

def build_trainer(model, tokenizer, dataset, config: dict):
    """
    Configure and return the SFTTrainer.

    SFTTrainer (Supervised Fine-Tuning Trainer) from TRL:
    - Tokenizes the "text" field automatically
    - Computes next-token prediction loss on the full sequence
    - Handles gradient accumulation, logging, and checkpointing

    Args:
        model:     PEFT model with LoRA
        tokenizer: Loaded tokenizer
        dataset:   HuggingFace Dataset with "text" column
        config:    Full configuration dictionary

    Returns:
        Configured SFTTrainer instance
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    train_cfg = config["training"]
    model_cfg = config["model"]

    # Create output directory
    output_dir = PROJECT_ROOT / train_cfg["output_dir"].lstrip("./")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        evaluation_strategy="no",       # No validation during training (keep it simple)
        report_to="none",               # Disable wandb/tensorboard
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        dataloader_pin_memory=False,    # Avoids issues on some Windows setups
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",                          # Column containing training text
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        args=training_args,
        packing=False,                                      # Don't pack multiple examples into one sequence
    )

    return trainer


# ============================================================
# Step 5: Save Model
# ============================================================

def save_model(model, tokenizer, config: dict) -> None:
    """
    Save the trained LoRA adapter weights and tokenizer.

    IMPORTANT: We save ONLY the LoRA adapter (~10-50 MB),
    NOT the full base model (~1 GB). During inference, we load
    the base model and attach the adapter on top.

    Args:
        model:     Trained PEFT model
        tokenizer: Tokenizer
        config:    Full configuration dictionary
    """
    save_path = PROJECT_ROOT / config["paths"]["final_model_dir"].lstrip("./")
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving LoRA adapter to: {save_path}")
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    # Save metadata for reference
    metadata = {
        "base_model":    config["model"]["name"],
        "lora_r":        config["lora"]["r"],
        "lora_alpha":    config["lora"]["lora_alpha"],
        "num_epochs":    config["training"]["num_epochs"],
        "learning_rate": config["training"]["learning_rate"],
        "batch_size":    config["training"]["batch_size"],
    }
    with open(save_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model saved successfully!")
    print(f"  Adapter path: {save_path.resolve()}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune with LoRA/QLoRA")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 10 samples, 1 epoch (quick test)")
    args = parser.parse_args()

    config = load_config()

    if args.debug:
        print("=" * 50)
        print("DEBUG MODE: minimal settings for quick test")
        print("=" * 50)
        config["training"]["num_epochs"] = 1
        config["training"]["save_steps"] = 5
        config["training"]["logging_steps"] = 1

    # Print device info
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {name}  (VRAM: {vram:.1f} GB)")
        else:
            print("No GPU detected. Training on CPU (this will be slow).")
    except ImportError:
        pass

    print()

    # Step 1: Load model
    print("=" * 50)
    print("STEP 1: Loading model and tokenizer")
    print("=" * 50)
    model, tokenizer = load_model_and_tokenizer(config)

    # Step 2: Apply LoRA
    print()
    print("=" * 50)
    print("STEP 2: Applying LoRA adapters")
    print("=" * 50)
    model = apply_lora(model, config)

    # Step 3: Load dataset
    print()
    print("=" * 50)
    print("STEP 3: Loading training dataset")
    print("=" * 50)
    dataset = load_training_dataset(config, debug=args.debug)

    # Step 4: Train
    print()
    print("=" * 50)
    print("STEP 4: Training")
    print("=" * 50)
    print(f"  Epochs:          {config['training']['num_epochs']}")
    print(f"  Batch size:      {config['training']['batch_size']}")
    print(f"  Grad accum:      {config['training']['gradient_accumulation_steps']}")
    eff_batch = config['training']['batch_size'] * config['training']['gradient_accumulation_steps']
    print(f"  Effective batch: {eff_batch}")
    print(f"  Learning rate:   {config['training']['learning_rate']}")
    print()

    trainer = build_trainer(model, tokenizer, dataset, config)
    print("Training started...")
    train_result = trainer.train()

    # Log metrics
    metrics = train_result.metrics
    print()
    print("Training complete!")
    print(f"  Final loss:    {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Runtime:       {metrics.get('train_runtime', 0):.0f}s")
    print(f"  Samples/sec:   {metrics.get('train_samples_per_second', 0):.2f}")

    # Step 5: Save
    print()
    print("=" * 50)
    print("STEP 5: Saving model")
    print("=" * 50)
    save_model(model, tokenizer, config)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print()
    print("Next steps:")
    print("  python scripts/05_inference_base_model.py   (compare base model)")
    print("  python scripts/06_inference_finetuned_model.py  (test fine-tuned)")
    print("=" * 60)


if __name__ == "__main__":
    main()
