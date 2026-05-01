"""
06_inference_finetuned_model.py

Loads Qwen/Qwen2.5-0.5B-Instruct + the saved LoRA adapter from
  outputs/models/qwen-dashboard-lora

then runs inference on the same fixed dashboard brief used in
05_inference_base_model.py and saves the result to:
  outputs/predictions/finetuned_model_prediction.json

Requirements:
  pip install transformers peft torch accelerate
"""

import json
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
ADAPTER_DIR   = os.path.join(BASE_DIR, "..", "outputs", "models", "qwen-dashboard-lora")
OUTPUT_DIR    = os.path.join(BASE_DIR, "..", "outputs", "predictions")
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "finetuned_model_prediction.json")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Same test brief as 05_inference_base_model.py ────────────────────────────
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

# ── User message ──────────────────────────────────────────────────────────────
USER_MESSAGE = (
    f"Dashboard brief:\n{TEST_BRIEF}\n\n"
    "Please fill in the following JSON schema with a complete dashboard design "
    "recommendation. Return only valid JSON, no extra text.\n\n"
    f"{SCHEMA_HINT}"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    import torch
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("No GPU detected — running on CPU.")
    return "cpu"


def load_tokenizer():
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from adapter directory: {os.path.abspath(ADAPTER_DIR)}")
    # The tokenizer was saved alongside the adapter by 04_train_lora.py
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_with_adapter(device: str):
    """Load the base model and merge the LoRA adapter on top."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    import torch

    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError(
            f"LoRA adapter not found at: {os.path.abspath(ADAPTER_DIR)}\n"
            "Run 04_train_lora.py first to train and save the adapter."
        )

    print(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,  # torch_dtype renamed to dtype in transformers >= 4.50
        device_map=device,
    )

    print(f"Loading LoRA adapter from: {os.path.abspath(ADAPTER_DIR)}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    return model


def run_inference(tokenizer, model, device: str) -> str:
    """Build the chat prompt and generate a response."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_MESSAGE},
    ]

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
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def try_parse_json(text: str):
    """
    Try to parse the model output as JSON.
    Strips markdown code fences if present.
    Returns (parsed_dict_or_None, error_message_or_None).
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as e:
        return None, str(e)


def pretty_print_prediction(parsed: dict):
    """Print the prediction in a readable format."""
    print("\n" + "=" * 60)
    print("FINE-TUNED MODEL PREDICTION")
    print("=" * 60)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Device
    device = get_device()

    # 2. Tokenizer
    tokenizer = load_tokenizer()

    # 3. Base model + LoRA adapter
    model = load_model_with_adapter(device)

    # 4. Inference
    raw_response = run_inference(tokenizer, model, device)

    # 5. Parse JSON
    parsed, parse_error = try_parse_json(raw_response)

    if parse_error:
        print(f"\nWARNING: Model output is not valid JSON.\n  Error: {parse_error}")
        print("  Saving raw output instead.")
        prediction = {
            "raw_output": raw_response,
            "parse_error": parse_error,
        }
    else:
        prediction = parsed
        pretty_print_prediction(parsed)

    # 6. Save result
    result = {
        "model":       MODEL_NAME,
        "adapter":     os.path.abspath(ADAPTER_DIR),
        "device":      device,
        "test_brief":  TEST_BRIEF,
        "prediction":  prediction,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResult saved to: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
