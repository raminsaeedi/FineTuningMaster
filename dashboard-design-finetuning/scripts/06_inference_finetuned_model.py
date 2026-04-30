"""
scripts/06_inference_finetuned_model.py
=========================================
Run inference with the FINE-TUNED model (base model + LoRA adapter).

Purpose:
    Compare this output against script 05 (base model) to see the
    improvement from fine-tuning. The fine-tuned model should:
    - Consistently output valid JSON
    - Include all 6 required keys
    - Provide more relevant and structured recommendations

Usage:
    # Use the built-in demo brief
    python scripts/06_inference_finetuned_model.py

    # Use a custom brief from a JSON file
    python scripts/06_inference_finetuned_model.py --brief data/examples/example_brief.json

    # Use a different adapter path
    python scripts/06_inference_finetuned_model.py --adapter outputs/models/checkpoints/checkpoint-50

Output:
    - Recommendation printed to console
    - Result saved to outputs/predictions/finetuned_model_output.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


# ============================================================
# Config
# ============================================================

def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "train_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Prompt Builder (same as script 05)
# ============================================================

SYSTEM_PROMPT = (
    "You are an expert dashboard design consultant. "
    "Given a dashboard brief, you generate structured, professional design recommendations. "
    "Always respond with valid JSON following the exact schema provided."
)

REQUIRED_KEYS = [
    "context_summary",
    "kpi_task_chart_mapping",
    "layout_hierarchy",
    "labels_scales_colors",
    "interactions",
    "design_rationales",
]


def build_prompt(brief: dict, tokenizer) -> str:
    """Build the inference prompt using the model's chat template."""
    kpis_str = ", ".join(brief.get("kpis", []))
    user_message = (
        "Please generate a structured dashboard design recommendation for the following brief:\n\n"
        f"Dashboard Title: {brief.get('title', 'N/A')}\n"
        f"Target Audience: {brief.get('target_audience', 'N/A')}\n"
        f"Business Goals: {brief.get('business_goals', 'N/A')}\n"
        f"KPIs: {kpis_str}\n"
        f"Data Context: {brief.get('data_context', 'N/A')}\n"
        f"Update Frequency: {brief.get('update_frequency', 'N/A')}\n"
        f"User Expertise: {brief.get('user_expertise', 'N/A')}\n\n"
        "Respond ONLY with a valid JSON object containing these exact keys:\n"
        + "\n".join(f"  {i+1}. {k}" for i, k in enumerate(REQUIRED_KEYS))
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ============================================================
# Model Loading (with LoRA adapter)
# ============================================================

def load_finetuned_model(adapter_path: Path, config: dict):
    """
    Load the base model and attach the LoRA adapter.

    How it works:
        1. Load the base model (frozen weights)
        2. Load the LoRA adapter from adapter_path
        3. PeftModel merges the adapter on top of the base model
        4. The result behaves like a single model during inference

    Args:
        adapter_path: Path to the saved LoRA adapter directory
        config:       Full configuration dictionary

    Returns:
        (model, tokenizer) tuple
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not adapter_path.exists():
        print(f"[ERROR] Adapter not found: {adapter_path}")
        print("        Run script 04 first: python scripts/04_train_lora.py")
        sys.exit(1)

    # Read base model name from saved metadata
    metadata_file = adapter_path / "training_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        base_model_name = metadata.get("base_model", config["model"]["name"])
        print(f"Adapter metadata: base model = {base_model_name}")
    else:
        base_model_name = config["model"]["name"]

    # Load tokenizer from adapter directory (it was saved there)
    print(f"Loading tokenizer from adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Attach LoRA adapter
    print(f"Attaching LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    print("Fine-tuned model loaded successfully.")
    return model, tokenizer


# ============================================================
# Inference (same logic as script 05)
# ============================================================

def try_parse_json(text: str):
    """Try to extract and parse JSON from model output."""
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def validate_schema(parsed) -> tuple:
    """Check if parsed output has all required keys."""
    if parsed is None:
        return False, REQUIRED_KEYS[:]
    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    return len(missing) == 0, missing


def run_inference(brief: dict, model, tokenizer, config: dict) -> dict:
    """Generate a recommendation for a given brief."""
    import torch

    inf_cfg = config["inference"]
    prompt = build_prompt(brief, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config["model"].get("max_seq_length", 2048) - inf_cfg.get("max_new_tokens", 1024),
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()
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
    latency = round(time.time() - start, 2)

    input_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0][input_len:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)

    parsed = try_parse_json(raw_output)
    is_valid, missing = validate_schema(parsed)

    return {
        "raw_output":   raw_output,
        "parsed":       parsed,
        "valid":        is_valid,
        "missing_keys": missing,
        "latency_s":    latency,
    }


# ============================================================
# Demo Brief
# ============================================================

DEMO_BRIEF = {
    "title": "E-Commerce Sales Dashboard",
    "target_audience": "Sales Managers and Regional Directors",
    "business_goals": "Monitor daily revenue and conversion rates",
    "kpis": ["Revenue", "Conversion Rate", "Average Order Value", "Cart Abandonment Rate"],
    "data_context": "Data from Shopify and Google Analytics. Updated daily.",
    "update_frequency": "Daily",
    "user_expertise": "Intermediate",
    "industry": "E-Commerce",
}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned model (base + LoRA adapter)"
    )
    parser.add_argument("--brief", default=None,
                        help="Path to a JSON file with a dashboard brief")
    parser.add_argument("--adapter", default=None,
                        help="Override adapter path from config")
    args = parser.parse_args()

    config = load_config()

    # Determine adapter path
    if args.adapter:
        adapter_path = Path(args.adapter)
    else:
        adapter_path = PROJECT_ROOT / config["paths"]["final_model_dir"].lstrip("./")

    # Load brief
    if args.brief:
        brief_path = Path(args.brief)
        if not brief_path.exists():
            print(f"[ERROR] Brief file not found: {brief_path}")
            sys.exit(1)
        with open(brief_path, encoding="utf-8") as f:
            brief = json.load(f)
        print(f"Loaded brief from: {brief_path}")
    else:
        brief = DEMO_BRIEF
        print("Using built-in demo brief.")

    print()
    print("Brief:")
    print(json.dumps(brief, indent=2, ensure_ascii=False))
    print()

    # Load fine-tuned model
    model, tokenizer = load_finetuned_model(adapter_path, config)

    # Run inference
    print()
    print("Generating recommendation...")
    print("-" * 60)
    result = run_inference(brief, model, tokenizer, config)

    # Print results
    print(f"Generation time: {result['latency_s']}s")
    print()
    print("--- RAW MODEL OUTPUT ---")
    print(result["raw_output"][:2000])

    print()
    if result["parsed"]:
        print("--- PARSED RECOMMENDATION ---")
        print(json.dumps(result["parsed"], indent=2, ensure_ascii=False)[:3000])
        print()
        if result["valid"]:
            print("All 6 required keys present! Fine-tuning is working.")
        else:
            print(f"Missing keys: {result['missing_keys']}")
            print("Try increasing num_epochs in config/train_config.yaml")
    else:
        print("Could not parse JSON from model output.")
        print("Try increasing num_epochs to 5 in config/train_config.yaml")

    # Save result
    out_dir = PROJECT_ROOT / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "finetuned_model_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": f"fine-tuned (adapter: {adapter_path})",
            "brief": brief,
            "result": {
                "raw_output":   result["raw_output"],
                "parsed":       result["parsed"],
                "valid":        result["valid"],
                "missing_keys": result["missing_keys"],
                "latency_s":    result["latency_s"],
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResult saved to: {out_file}")
    print()
    print("Next step: python scripts/07_evaluate_schema_compliance.py")


if __name__ == "__main__":
    main()
