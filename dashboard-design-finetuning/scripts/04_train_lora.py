"""
04_train_lora.py

Fine-tunes Qwen/Qwen2.5-0.5B-Instruct with LoRA on the synthetic
dashboard dataset using Hugging Face TRL SFTTrainer.

Input:
  data/processed/train.jsonl
  data/processed/validation.jsonl

Output:
  outputs/models/qwen-dashboard-lora   (LoRA adapter + tokenizer)
  outputs/logs/                        (training logs)

Requirements:
  pip install transformers datasets peft trl accelerate torch
"""

import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
TRAIN_FILE  = os.path.join(BASE_DIR, "..", "data", "processed", "train.jsonl")
VAL_FILE    = os.path.join(BASE_DIR, "..", "data", "processed", "validation.jsonl")
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "outputs", "models", "qwen-dashboard-lora")
LOG_DIR     = os.path.join(BASE_DIR, "..", "outputs", "logs")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ── LoRA hyperparameters ──────────────────────────────────────────────────────
LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ── Training hyperparameters ──────────────────────────────────────────────────
NUM_EPOCHS              = 1
BATCH_SIZE              = 1
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE           = 2e-4
MAX_SEQ_LENGTH          = 1024
LOGGING_STEPS           = 5
SAVE_STEPS              = 25


def check_imports():
    """Verify all required packages are installed before proceeding."""
    missing = []
    for pkg in ["transformers", "datasets", "peft", "trl", "accelerate", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("ERROR: The following packages are missing:")
        for m in missing:
            print(f"  - {m}")
        print("\nInstall them with:")
        print("  pip install transformers datasets peft trl accelerate torch")
        sys.exit(1)


def get_device():
    """Return 'cuda' if a GPU is available, else 'cpu' with a warning."""
    import torch
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print(
            "\nWARNING: No CUDA GPU detected. Training will run on CPU.\n"
            "  This will be very slow for a language model.\n"
            "  Consider using Google Colab (free GPU) or a machine with a GPU.\n"
        )
        return "cpu"


def load_datasets():
    """Load train and validation JSONL files with Hugging Face datasets."""
    from datasets import load_dataset

    for path in [TRAIN_FILE, VAL_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                "Run 03_prepare_dataset.py first."
            )

    print("Loading datasets ...")
    dataset = load_dataset(
        "json",
        data_files={
            "train":      os.path.abspath(TRAIN_FILE),
            "validation": os.path.abspath(VAL_FILE),
        },
    )
    print(f"  Train samples      : {len(dataset['train'])}")
    print(f"  Validation samples : {len(dataset['validation'])}")
    return dataset


def load_model_and_tokenizer(device: str):
    """Load the base model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    # Qwen tokenizer may not have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_NAME}  (this may take a moment ...)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.config.use_cache = False   # required for gradient checkpointing

    return tokenizer, model


def apply_lora(model):
    """Wrap the model with a LoRA adapter using PEFT."""
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(model, tokenizer, dataset, device: str):
    """Set up SFTTrainer and run training."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_dir=LOG_DIR,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="epoch",     # replaces deprecated evaluation_strategy
        fp16=(device == "cuda"),   # mixed precision only on GPU
        bf16=False,
        report_to="none",          # disable wandb / other trackers
        load_best_model_at_end=False,
        dataloader_num_workers=0,  # safer on Windows
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",   # the field created by 03_prepare_dataset.py
        packing=False,
    )

    print("\nStarting training ...")
    trainer.train()
    return trainer


def save_model(trainer, tokenizer):
    """Save the LoRA adapter weights and tokenizer."""
    print(f"\nSaving LoRA adapter to: {os.path.abspath(OUTPUT_DIR)}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model and tokenizer saved.")


def main():
    # 1. Check dependencies
    check_imports()

    # 2. Detect device
    device = get_device()

    # 3. Load data
    dataset = load_datasets()

    # 4. Load base model
    tokenizer, model = load_model_and_tokenizer(device)

    # 5. Apply LoRA
    model = apply_lora(model)

    # 6. Train
    trainer = train(model, tokenizer, dataset, device)

    # 7. Save
    save_model(trainer, tokenizer)

    print("\nTraining complete.")
    print(f"  LoRA adapter : {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Training logs: {os.path.abspath(LOG_DIR)}")


if __name__ == "__main__":
    main()
