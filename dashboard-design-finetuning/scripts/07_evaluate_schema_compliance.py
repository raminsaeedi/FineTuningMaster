"""
scripts/07_evaluate_schema_compliance.py
==========================================
Evaluates the fine-tuned model on the test set and reports metrics.

Metrics computed:
    1. JSON Parse Rate       – % of outputs that are valid JSON
    2. Schema Validity Rate  – % of outputs with all 6 required keys
    3. Key Coverage          – Which keys are most often missing
    4. Average Latency       – Seconds per example
    5. Per-example details   – Valid/invalid for each test example

This script also compares the fine-tuned model against the base model
if base model predictions are available in outputs/predictions/.

Usage:
    # Evaluate fine-tuned model on test set
    python scripts/07_evaluate_schema_compliance.py

    # Also compare against base model
    python scripts/07_evaluate_schema_compliance.py --compare-base

    # Use a specific adapter
    python scripts/07_evaluate_schema_compliance.py --adapter outputs/models/checkpoints/checkpoint-50

Output:
    outputs/predictions/evaluation_report.json
    Console: formatted metrics table
"""

import argparse
import json
import re
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
# Constants
# ============================================================

REQUIRED_KEYS = [
    "context_summary",
    "kpi_task_chart_mapping",
    "layout_hierarchy",
    "labels_scales_colors",
    "interactions",
    "design_rationales",
]

SYSTEM_PROMPT = (
    "You are an expert dashboard design consultant. "
    "Given a dashboard brief, you generate structured, professional design recommendations. "
    "Always respond with valid JSON following the exact schema provided."
)


# ============================================================
# Helpers (same as scripts 05/06)
# ============================================================

def build_prompt(brief: dict, tokenizer) -> str:
    """Build the inference prompt."""
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


def try_parse_json(text: str):
    """Try to extract and parse JSON from model output."""
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


# ============================================================
# Model Loading
# ============================================================

def load_finetuned_model(adapter_path: Path, config: dict):
    """Load base model + LoRA adapter for inference."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not adapter_path.exists():
        print(f"[ERROR] Adapter not found: {adapter_path}")
        print("        Run script 04 first: python scripts/04_train_lora.py")
        sys.exit(1)

    metadata_file = adapter_path / "training_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        base_model_name = metadata.get("base_model", config["model"]["name"])
    else:
        base_model_name = config["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def load_base_model(config: dict):
    """Load base model without adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config["model"]["name"]
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
# Single Example Inference
# ============================================================

def run_single(brief: dict, model, tokenizer, config: dict) -> dict:
    """Run inference for one brief and return structured result."""
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
    raw_output = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

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
# Evaluation Loop
# ============================================================

def evaluate_model(model, tokenizer, config: dict, model_label: str) -> dict:
    """
    Run inference on all test examples and compute metrics.

    Args:
        model:       Loaded model
        tokenizer:   Loaded tokenizer
        config:      Full configuration dictionary
        model_label: Label for this model (e.g., "fine-tuned" or "base")

    Returns:
        Dictionary with metrics and per-example details
    """
    test_path = PROJECT_ROOT / config["data"]["test_file"].lstrip("./")

    if not test_path.exists():
        print(f"[ERROR] Test file not found: {test_path}")
        print("        Run scripts 02 and 03 first.")
        sys.exit(1)

    # Load test examples (raw format with brief + recommendation)
    # The test file in data/processed/ has "text" field, but we need the brief.
    # So we load from data/raw/test.jsonl instead.
    raw_test_path = PROJECT_ROOT / config["data"]["raw_dir"].lstrip("./") / "test.jsonl"
    if not raw_test_path.exists():
        print(f"[ERROR] Raw test file not found: {raw_test_path}")
        print("        Run script 02 first.")
        sys.exit(1)

    test_records = []
    with open(raw_test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_records.append(json.loads(line))

    print(f"\nEvaluating {model_label} on {len(test_records)} test examples...")
    print("-" * 70)
    print(f"  {'#':>3}  {'Title':<35}  {'JSON':>5}  {'Valid':>5}  {'Latency':>8}")
    print("-" * 70)

    details = []
    for i, record in enumerate(test_records):
        brief = record["brief"]
        title = brief.get("title", f"Example {i+1}")[:33]

        result = run_single(brief, model, tokenizer, config)

        json_ok  = "OK" if result["parsed"] is not None else "FAIL"
        valid_ok = "OK" if result["valid"] else "FAIL"
        print(f"  {i+1:>3}  {title:<35}  {json_ok:>5}  {valid_ok:>5}  {result['latency_s']:>7.1f}s")

        details.append({
            "index":        i + 1,
            "brief_title":  brief.get("title", "Unknown"),
            "industry":     brief.get("industry", "Unknown"),
            "json_parseable": result["parsed"] is not None,
            "schema_valid": result["valid"],
            "missing_keys": result["missing_keys"],
            "latency_s":    result["latency_s"],
        })

    print("-" * 70)

    # Compute aggregate metrics
    n = len(details)
    json_ok_count  = sum(1 for d in details if d["json_parseable"])
    valid_count    = sum(1 for d in details if d["schema_valid"])
    avg_latency    = sum(d["latency_s"] for d in details) / n if n > 0 else 0

    # Count missing keys across all examples
    key_missing_counts = {k: 0 for k in REQUIRED_KEYS}
    for d in details:
        for k in d["missing_keys"]:
            if k in key_missing_counts:
                key_missing_counts[k] += 1

    metrics = {
        "model_label":         model_label,
        "total_examples":      n,
        "json_parse_rate":     round(json_ok_count / n * 100, 1) if n > 0 else 0,
        "schema_validity_rate": round(valid_count / n * 100, 1) if n > 0 else 0,
        "avg_latency_s":       round(avg_latency, 2),
        "key_missing_counts":  key_missing_counts,
    }

    return {"metrics": metrics, "details": details}


# ============================================================
# Report Printer
# ============================================================

def print_report(results: list) -> None:
    """Print a formatted comparison report."""
    print()
    print("=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)

    for r in results:
        m = r["metrics"]
        print(f"\n  Model: {m['model_label']}")
        print(f"  {'Total examples:':<30} {m['total_examples']}")
        print(f"  {'JSON parse rate:':<30} {m['json_parse_rate']}%")
        print(f"  {'Schema validity rate:':<30} {m['schema_validity_rate']}%")
        print(f"  {'Avg latency per example:':<30} {m['avg_latency_s']}s")

        if any(v > 0 for v in m["key_missing_counts"].values()):
            print(f"\n  Most frequently missing keys:")
            for key, count in sorted(m["key_missing_counts"].items(),
                                     key=lambda x: -x[1]):
                if count > 0:
                    pct = round(count / m["total_examples"] * 100, 1)
                    print(f"    {key:<35} missing in {count}/{m['total_examples']} ({pct}%)")

    if len(results) == 2:
        m1 = results[0]["metrics"]
        m2 = results[1]["metrics"]
        print()
        print("  IMPROVEMENT (fine-tuned vs. base):")
        json_diff  = m1["json_parse_rate"] - m2["json_parse_rate"]
        valid_diff = m1["schema_validity_rate"] - m2["schema_validity_rate"]
        print(f"    JSON parse rate:      {json_diff:+.1f}%")
        print(f"    Schema validity rate: {valid_diff:+.1f}%")

    print()
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate schema compliance on test set")
    parser.add_argument("--adapter", default=None,
                        help="Override adapter path from config")
    parser.add_argument("--compare-base", action="store_true",
                        help="Also evaluate base model for comparison")
    args = parser.parse_args()

    config = load_config()

    adapter_path = (
        Path(args.adapter) if args.adapter
        else PROJECT_ROOT / config["paths"]["final_model_dir"].lstrip("./")
    )

    all_results = []

    # Evaluate fine-tuned model
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_finetuned_model(adapter_path, config)
    ft_results = evaluate_model(ft_model, ft_tokenizer, config, "fine-tuned (LoRA)")
    all_results.append(ft_results)

    # Optionally evaluate base model
    if args.compare_base:
        print("\nLoading base model for comparison...")
        # Free fine-tuned model memory first
        import torch
        del ft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base_model, base_tokenizer = load_base_model(config)
        base_results = evaluate_model(base_model, base_tokenizer, config, "base (no fine-tuning)")
        all_results.append(base_results)

    # Print report
    print_report(all_results)

    # Save report
    out_dir = PROJECT_ROOT / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Full report saved to: {report_file}")


if __name__ == "__main__":
    main()
