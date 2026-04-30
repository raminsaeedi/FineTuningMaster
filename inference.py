"""
inference.py
============
Load the fine-tuned LoRA adapter and generate dashboard design
recommendations for new briefs. Also includes a simple evaluation
loop that runs on the test set and computes basic metrics.

Usage:
    # Interactive mode (enter a brief manually):
    python inference.py

    # Evaluate on test set:
    python inference.py --evaluate

    # Single prediction from a JSON file:
    python inference.py --brief-file examples/my_brief.json

    # Use a specific adapter path:
    python inference.py --adapter-path ./outputs/final_model
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.helpers import (
    extract_json_from_text,
    format_instruction_prompt,
    load_config,
    load_jsonl,
    print_device_info,
    setup_logging,
    validate_recommendation,
)

logger = setup_logging(log_file="./outputs/logs/inference.log")


# ============================================================
# Model Loading
# ============================================================

def load_finetuned_model(adapter_path: str, config: dict):
    """
    Load the base model and merge the LoRA adapter for inference.

    Two strategies:
    A) Load base model + adapter separately (memory efficient, slower)
    B) Merge adapter into base model (faster inference, more memory)

    We use strategy A here (load separately) which is simpler and
    works well for evaluation purposes.

    Args:
        adapter_path: Path to saved LoRA adapter directory
        config:       Full configuration dictionary

    Returns:
        (model, tokenizer) tuple ready for inference
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        logger.error(
            f"Adapter not found at: {adapter_path}\n"
            f"Please run: python train.py"
        )
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    # Read base model name from saved metadata (if available)
    metadata_file = adapter_path / "training_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        base_model_name = metadata.get("base_model", config["model"]["name"])
        logger.info(f"Loaded metadata: base model = {base_model_name}")
    else:
        base_model_name = config["model"]["name"]

    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,           # Tokenizer was saved alongside the adapter
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (without quantization for inference)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Load and attach LoRA adapter
    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()    # Set to evaluation mode (disables dropout)

    logger.info("Fine-tuned model loaded successfully")
    return model, tokenizer


def load_base_model_only(config: dict):
    """
    Load the base model WITHOUT any fine-tuning adapter.
    Used for baseline comparison.

    Args:
        config: Full configuration dictionary

    Returns:
        (model, tokenizer) tuple
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model"]["name"]
    logger.info(f"Loading BASE model (no adapter): {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


# ============================================================
# Inference
# ============================================================

def generate_recommendation(
    brief: Dict[str, Any],
    model,
    tokenizer,
    config: dict,
) -> Dict[str, Any]:
    """
    Generate a structured dashboard design recommendation for a brief.

    Args:
        brief:     Dashboard brief dictionary
        model:     Loaded (fine-tuned) model
        tokenizer: Loaded tokenizer
        config:    Full configuration dictionary

    Returns:
        Dictionary with:
            - 'raw_output': raw model text
            - 'parsed':     parsed JSON recommendation (or None)
            - 'valid':      whether all required keys are present
            - 'missing_keys': list of missing keys
            - 'latency_s':  generation time in seconds
    """
    import torch

    inf_cfg = config["inference"]

    # Format the prompt using the model's chat template
    prompt = format_instruction_prompt(brief, tokenizer)

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config["model"].get("max_seq_length", 2048) - inf_cfg.get("max_new_tokens", 1024),
    )

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=inf_cfg.get("max_new_tokens", 1024),
            temperature=inf_cfg.get("temperature", 0.1),
            top_p=inf_cfg.get("top_p", 0.9),
            do_sample=inf_cfg.get("do_sample", True),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - start_time

    # Decode only the newly generated tokens (not the prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_length:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Parse JSON from output
    parsed = extract_json_from_text(raw_output)
    is_valid, missing_keys = validate_recommendation(parsed) if parsed else (False, list())

    return {
        "raw_output": raw_output,
        "parsed": parsed,
        "valid": is_valid,
        "missing_keys": missing_keys,
        "latency_s": round(latency, 2),
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_on_test_set(model, tokenizer, config: dict) -> Dict[str, Any]:
    """
    Run inference on the test set and compute evaluation metrics.

    Metrics computed:
    - JSON parse rate: % of outputs that are valid JSON
    - Schema validity rate: % of outputs with all 6 required keys
    - Average latency per example

    Args:
        model:     Loaded model
        tokenizer: Loaded tokenizer
        config:    Full configuration dictionary

    Returns:
        Dictionary with aggregated metrics
    """
    test_file = config["data"].get("test_file", "./data/test.jsonl")

    if not Path(test_file).exists():
        logger.error(f"Test file not found: {test_file}")
        return {}

    test_data = load_jsonl(test_file)
    logger.info(f"Evaluating on {len(test_data)} test examples...")

    results = []
    for i, item in enumerate(test_data):
        logger.info(f"  [{i+1}/{len(test_data)}] Generating recommendation...")
        result = generate_recommendation(item["brief"], model, tokenizer, config)
        result["brief_title"] = item["brief"].get("title", f"Example {i+1}")
        results.append(result)

        # Print progress
        status = "OK - valid" if result["valid"] else f"MISSING: {result['missing_keys']}"
        print(f"  [{i+1:2d}] {result['brief_title'][:40]:<40} | {status} | {result['latency_s']}s")

    # Compute aggregate metrics
    n = len(results)
    json_parseable = sum(1 for r in results if r["parsed"] is not None)
    schema_valid   = sum(1 for r in results if r["valid"])
    avg_latency    = sum(r["latency_s"] for r in results) / n if n > 0 else 0

    metrics = {
        "total_examples":      n,
        "json_parse_rate":     round(json_parseable / n * 100, 1) if n > 0 else 0,
        "schema_validity_rate": round(schema_valid / n * 100, 1) if n > 0 else 0,
        "avg_latency_s":       round(avg_latency, 2),
    }

    # Save results
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {results_file}")
    return metrics


# ============================================================
# Interactive Demo
# ============================================================

DEMO_BRIEF = {
    "title": "E-Commerce Sales Dashboard",
    "target_audience": "Sales Managers and Regional Directors",
    "business_goals": "Monitor daily revenue and conversion rates",
    "kpis": ["Revenue", "Conversion Rate", "Average Order Value", "Cart Abandonment Rate"],
    "data_context": "Data from Shopify and Google Analytics. Updated daily. Covers last 12 months.",
    "update_frequency": "Daily",
    "user_expertise": "Intermediate",
    "industry": "E-Commerce",
}


def interactive_demo(model, tokenizer, config: dict) -> None:
    """
    Run an interactive demo where the user can enter a brief or use the default.

    Args:
        model:     Loaded model
        tokenizer: Loaded tokenizer
        config:    Full configuration dictionary
    """
    print("\n" + "=" * 60)
    print("  DASHBOARD DESIGN RECOMMENDATION SYSTEM")
    print("  Fine-tuned with LoRA on Qwen2.5-0.5B-Instruct")
    print("=" * 60)
    print("\nUsing demo brief:")
    print(json.dumps(DEMO_BRIEF, indent=2))
    print("\nGenerating recommendation... (this may take 30-60s on CPU)")
    print("-" * 60)

    result = generate_recommendation(DEMO_BRIEF, model, tokenizer, config)

    print(f"\nGeneration time: {result['latency_s']}s")
    print("\n--- RAW MODEL OUTPUT ---")
    print(result["raw_output"][:2000])  # Truncate for display

    if result["parsed"]:
        print("\n--- PARSED RECOMMENDATION ---")
        print(json.dumps(result["parsed"], indent=2, ensure_ascii=False)[:3000])
        if result["valid"]:
            print("\nAll 6 required keys present!")
        else:
            print(f"\nMissing keys: {result['missing_keys']}")
    else:
        print("\nCould not parse JSON from model output.")
        print("   This is normal for early training stages or CPU-only inference.")
        print("   Try increasing num_train_epochs in config.yaml")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned dashboard recommendation model"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--adapter-path", default=None,
                        help="Override adapter path from config")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation on test set")
    parser.add_argument("--base-only", action="store_true",
                        help="Use base model without fine-tuning (for comparison)")
    parser.add_argument("--brief-file", default=None,
                        help="Path to a JSON file containing a dashboard brief")
    args = parser.parse_args()

    # ── Load Config ──────────────────────────────────────────────────────────
    config = load_config(args.config)
    print_device_info()

    # ── Load Model ───────────────────────────────────────────────────────────
    if args.base_only:
        logger.info("Loading BASE model (no fine-tuning) for comparison")
        model, tokenizer = load_base_model_only(config)
    else:
        adapter_path = args.adapter_path or config["inference"]["adapter_path"]
        model, tokenizer = load_finetuned_model(adapter_path, config)

    # ── Run Mode ─────────────────────────────────────────────────────────────
    if args.evaluate:
        # Evaluation mode
        logger.info("Running evaluation on test set...")
        metrics = evaluate_on_test_set(model, tokenizer, config)

        print("\n" + "=" * 60)
        print("  EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Total examples:       {metrics.get('total_examples', 0)}")
        print(f"  JSON parse rate:      {metrics.get('json_parse_rate', 0)}%")
        print(f"  Schema validity rate: {metrics.get('schema_validity_rate', 0)}%")
        print(f"  Avg latency:          {metrics.get('avg_latency_s', 0)}s")
        print("=" * 60)

    elif args.brief_file:
        # Single brief from file
        brief_path = Path(args.brief_file)
        if not brief_path.exists():
            logger.error(f"Brief file not found: {brief_path}")
            return
        with open(brief_path, encoding="utf-8") as f:
            brief = json.load(f)
        logger.info(f"Loaded brief from: {brief_path}")
        result = generate_recommendation(brief, model, tokenizer, config)
        print(json.dumps(result["parsed"] or {"error": "parse_failed", "raw": result["raw_output"]},
                         indent=2, ensure_ascii=False))

    else:
        # Interactive demo
        interactive_demo(model, tokenizer, config)


if __name__ == "__main__":
    main()
