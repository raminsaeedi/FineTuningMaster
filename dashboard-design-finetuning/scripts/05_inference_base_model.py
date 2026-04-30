"""
05_inference_base_model.py

Tests the base model Qwen/Qwen2.5-0.5B-Instruct (without fine-tuning)
on a fixed dashboard brief and saves the prediction to:
  outputs/predictions/base_model_prediction.json

Requirements:
  pip install transformers torch accelerate
"""

import json
import os

# ── Output path ───────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
OUTPUT_DIR   = os.path.join(BASE_DIR, "..", "outputs", "predictions")
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "base_model_prediction.json")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Fixed test brief ──────────────────────────────────────────────────────────
TEST_BRIEF = (
    "Create a dashboard for a sales manager who wants to monitor monthly revenue, "
    "sales by region, top products, conversion rate, and customer churn. "
    "The data contains date, region, product category, revenue, number of leads, "
    "conversions, and churn status."
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a dashboard design assistant. "
    "You generate structured, practical, and heuristic-aligned "
    "dashboard design recommendations."
)

# ── JSON schema the model should fill in ─────────────────────────────────────
SCHEMA_HINT = json.dumps(
    {
        "context_summary": "",
        "kpi_task_chart_mapping": [],
        "layout_hierarchy": "",
        "labels_scales_colors": "",
        "interactions": [],
        "design_rationales": [],
    },
    indent=2,
)

# ── User message sent to the model ───────────────────────────────────────────
USER_MESSAGE = (
    f"Dashboard brief:\n{TEST_BRIEF}\n\n"
    "Please fill in the following JSON schema with a complete dashboard design "
    "recommendation. Return only valid JSON, no extra text.\n\n"
    f"{SCHEMA_HINT}"
)


def load_model_and_tokenizer(model_name: str):
    """Load the tokenizer and model. Raises a clear error on failure."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        raise ImportError(
            "Required packages are missing. Install them with:\n"
            "  pip install transformers torch accelerate"
        )

    print(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer for '{model_name}'.\n"
            f"Check your internet connection or Hugging Face access.\n"
            f"Error: {e}"
        )

    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {model_name}  (this may take a moment ...)")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",        # fp16 on GPU, fp32 on CPU
            device_map=device,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}'.\n"
            f"If you are on CPU and run out of memory, try a smaller model.\n"
            f"Error: {e}"
        )

    model.eval()
    return tokenizer, model, device


def run_inference(tokenizer, model, device: str) -> str:
    """Build the chat prompt and generate a response."""
    import torch

    # Qwen chat template: list of role/content dicts
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_MESSAGE},
    ]

    # Apply the model's built-in chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    print("Running inference ...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,          # greedy decoding for reproducibility
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def try_parse_json(text: str):
    """Try to parse the model output as JSON. Return dict or raw string."""
    # Sometimes the model wraps JSON in markdown code fences — strip them
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first and last fence lines
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Return raw text so we can still save something useful
        return {"raw_output": text, "parse_error": "Model output was not valid JSON."}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load model
    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)

    # 2. Run inference
    raw_response = run_inference(tokenizer, model, device)

    # 3. Parse response
    parsed = try_parse_json(raw_response)

    # 4. Build result document
    result = {
        "model": MODEL_NAME,
        "device": device,
        "test_brief": TEST_BRIEF,
        "prediction": parsed,
    }

    # 5. Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nInference complete.")
    print(f"  Output saved to : {os.path.abspath(OUTPUT_FILE)}")
    print("\n--- Model response (raw) ---")
    print(raw_response[:1000], "..." if len(raw_response) > 1000 else "")
    print("--- End ---")


if __name__ == "__main__":
    main()
