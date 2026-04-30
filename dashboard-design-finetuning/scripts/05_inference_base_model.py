"""
scripts/05_inference_base_model.py
====================================
Run inference with the BASE model (no fine-tuning).

Purpose:
    This is your BASELINE. Run this BEFORE fine-tuning to see how the
    base model responds to dashboard briefs without any training.
    Then compare with script 06 (fine-tuned model) to measure improvement.

What to expect from the base model:
    - May generate text instead of JSON
    - May generate JSON but with wrong keys
    - May hallucinate or give generic answers
    - This is NORMAL – fine-tuning fixes this

Usage:
    # Use the built-in demo brief
    python scripts/05_inference_base_model.py

    # Use a custom brief from a JSON file
    python scripts/05_inference_base_model.py --brief data/examples/example_brief.json

Output:
    - Raw model output printed to console
    - Result saved to outputs/predictions/base_model_output.json
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
# Prompt Builder
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
    """
    Build the inference prompt using the model's chat template.

    Args:
        brief:     Dashboard brief dictionary
        tokenizer: Loaded tokenizer (needed for apply_chat_template)

    Returns:
        Formatted prompt string ready for tokenization
    """
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

    # apply_chat_template adds the model-specific special tokens
    # add_generation_prompt=True adds the assistant turn opener
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# ============================================================
# Model Loading
# ============================================================

def load_base_model(config: dict):
    """
    Load the base model WITHOUT any fine-tuning adapter.

    Args:
        config: Full configuration dictionary

    Returns:
        (model, tokenizer) tuple
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model"]["name"]
    print(f"Loading base model: {model_name}")
    print("(This downloads ~1 GB on first run – subsequent runs use cache)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()  # Set to evaluation mode (disables dropout)

    print("Base model loaded.")
    return model, tokenizer


# ============================================================
# Inference
# ============================================================

def run_inference(brief: dict, model, tokenizer, config: dict) -> dict:
    """
    Generate a recommendation for a given brief.

    Args:
        brief:     Dashboard brief dictionary
        model:     Loaded model
        tokenizer: Loaded tokenizer
        config:    Full configuration dictionary

    Returns:
        Dictionary with raw_output, parsed JSON (or None), validity info, latency
    """
    import torch

    inf_cfg = config["inference"]
    prompt = build_prompt(brief, tokenizer)

    # Tokenize the prompt
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

    # Decode only the newly generated tokens (skip the prompt)
    input_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0][input_len:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Try to parse JSON from the output
    parsed = try_parse_json(raw_output)
    is_valid, missing = validate_schema(parsed)

    return {
        "raw_output":    raw_output,
        "parsed":        parsed,
        "valid":         is_valid,
        "missing_keys":  missing,
        "latency_s":     latency,
    }


def try_parse_json(text: str):
    """
    Try to extract and parse a JSON object from model output text.
    Handles cases where the model wraps JSON in markdown code blocks.

    Args:
        text: Raw model output string

    Returns:
        Parsed dict or None
    """
    import re

    # Try to find JSON inside ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def validate_schema(parsed: dict) -> tuple:
    """
    Check if parsed output contains all required keys.

    Args:
        parsed: Parsed dictionary (or None)

    Returns:
        (is_valid, list_of_missing_keys)
    """
    if parsed is None:
        return False, REQUIRED_KEYS[:]
    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    return len(missing) == 0, missing


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
    parser = argparse.ArgumentParser(description="Run inference with base model (no fine-tuning)")
    parser.add_argument("--brief", default=None,
                        help="Path to a JSON file with a dashboard brief")
    args = parser.parse_args()

    config = load_config()

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

    # Load model
    model, tokenizer = load_base_model(config)

    # Run inference
    print("Generating recommendation (this may take 30-120s on CPU)...")
    print("-" * 60)
    result = run_inference(brief, model, tokenizer, config)

    # Print results
    print(f"Generation time: {result['latency_s']}s")
    print()
    print("--- RAW MODEL OUTPUT ---")
    print(result["raw_output"][:2000])  # Truncate for readability

    print()
    if result["parsed"]:
        print("--- PARSED JSON ---")
        print(json.dumps(result["parsed"], indent=2, ensure_ascii=False)[:2000])
        if result["valid"]:
            print("\nAll 6 required keys present!")
        else:
            print(f"\nMissing keys: {result['missing_keys']}")
    else:
        print("Could not parse JSON from model output.")
        print("This is expected for the base model without fine-tuning.")

    # Save result
    out_dir = PROJECT_ROOT / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "base_model_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": "base (no fine-tuning)",
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
    print("Next step: python scripts/04_train_lora.py  (if not done yet)")
    print("Then:       python scripts/06_inference_finetuned_model.py")


if __name__ == "__main__":
    main()
