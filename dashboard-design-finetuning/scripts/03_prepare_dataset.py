"""
scripts/03_prepare_dataset.py
==============================
Reads the raw JSONL files from data/raw/ and formats each example
into an instruction-response text that the model can be trained on.

What this script does:
    1. Loads raw examples from data/raw/train.jsonl (and val/test)
    2. Formats each example as a chat-style prompt + JSON response
    3. Saves formatted examples to data/processed/

The formatted text looks like this:
    <|im_start|>system
    You are an expert dashboard design consultant...
    <|im_end|>
    <|im_start|>user
    Please generate a structured recommendation for:
    Title: E-Commerce Sales Dashboard
    ...
    <|im_end|>
    <|im_start|>assistant
    {
      "context_summary": "...",
      ...
    }
    <|im_end|>

Usage:
    python scripts/03_prepare_dataset.py

Output:
    data/processed/train.jsonl
    data/processed/val.jsonl
    data/processed/test.jsonl
"""

import json
import sys
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
# Prompt Templates
# ============================================================

# System prompt: tells the model its role
SYSTEM_PROMPT = (
    "You are an expert dashboard design consultant. "
    "Given a dashboard brief, you generate structured, professional design recommendations. "
    "Always respond with valid JSON following the exact schema provided."
)

# The 6 required output keys
REQUIRED_KEYS = [
    "context_summary",
    "kpi_task_chart_mapping",
    "layout_hierarchy",
    "labels_scales_colors",
    "interactions",
    "design_rationales",
]


def build_user_message(brief: dict) -> str:
    """
    Convert a brief dictionary into a readable user message.

    Args:
        brief: Dictionary with dashboard brief fields

    Returns:
        Formatted user message string
    """
    kpis_str = ", ".join(brief.get("kpis", []))
    lines = [
        "Please generate a structured dashboard design recommendation for the following brief:",
        "",
        f"Dashboard Title: {brief.get('title', 'N/A')}",
        f"Target Audience: {brief.get('target_audience', 'N/A')}",
        f"Business Goals: {brief.get('business_goals', 'N/A')}",
        f"KPIs: {kpis_str}",
        f"Data Context: {brief.get('data_context', 'N/A')}",
        f"Update Frequency: {brief.get('update_frequency', 'N/A')}",
        f"User Expertise: {brief.get('user_expertise', 'N/A')}",
        "",
        "Respond ONLY with a valid JSON object containing these exact keys:",
    ]
    for i, key in enumerate(REQUIRED_KEYS, 1):
        lines.append(f"  {i}. {key}")
    return "\n".join(lines)


def format_as_chatml(brief: dict, recommendation: dict) -> str:
    """
    Format a brief + recommendation as a ChatML-style training text.

    ChatML is the format used by Qwen2.5 and many other models:
        <|im_start|>role
        content
        <|im_end|>

    The model learns to generate the assistant turn given the
    system + user turns as context.

    Args:
        brief:          Dashboard brief dictionary
        recommendation: Expected recommendation dictionary

    Returns:
        Full training text string
    """
    user_message = build_user_message(brief)
    response_json = json.dumps(recommendation, ensure_ascii=False, indent=2)

    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n{response_json}<|im_end|>"
    )
    return text


def format_as_alpaca(brief: dict, recommendation: dict) -> str:
    """
    Alternative: Alpaca-style format (simpler, no special tokens).
    Use this if the model does not support ChatML.

    Args:
        brief:          Dashboard brief dictionary
        recommendation: Expected recommendation dictionary

    Returns:
        Full training text string
    """
    user_message = build_user_message(brief)
    response_json = json.dumps(recommendation, ensure_ascii=False, indent=2)

    text = (
        f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
        f"### Input:\n{user_message}\n\n"
        f"### Response:\n{response_json}"
    )
    return text


# ============================================================
# File I/O
# ============================================================

def load_jsonl(filepath: Path) -> list:
    """Load a JSONL file into a list of dicts."""
    if not filepath.exists():
        print(f"  [WARN] File not found: {filepath}")
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed JSON at line {line_num}: {e}")
    return records


def save_jsonl(records: list, filepath: Path) -> None:
    """Save a list of dicts to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records -> {filepath}")


# ============================================================
# Main Processing
# ============================================================

def process_split(raw_file: Path, processed_file: Path, fmt: str = "chatml") -> int:
    """
    Load a raw JSONL split, format each example, and save to processed/.

    Args:
        raw_file:       Path to raw JSONL file
        processed_file: Path to output processed JSONL file
        fmt:            Format style: "chatml" or "alpaca"

    Returns:
        Number of examples processed
    """
    raw_records = load_jsonl(raw_file)
    if not raw_records:
        print(f"  [SKIP] No records in {raw_file}")
        return 0

    processed = []
    skipped = 0
    for record in raw_records:
        brief = record.get("brief")
        recommendation = record.get("recommendation")

        # Validate that both fields exist
        if not brief or not recommendation:
            skipped += 1
            continue

        # Validate that recommendation has all required keys
        missing = [k for k in REQUIRED_KEYS if k not in recommendation]
        if missing:
            print(f"  [WARN] Skipping example missing keys: {missing}")
            skipped += 1
            continue

        # Format the text
        if fmt == "chatml":
            text = format_as_chatml(brief, recommendation)
        else:
            text = format_as_alpaca(brief, recommendation)

        processed.append({
            "text": text,
            "brief_title": brief.get("title", "Unknown"),
            "industry": brief.get("industry", "Unknown"),
        })

    if skipped > 0:
        print(f"  [INFO] Skipped {skipped} invalid records")

    save_jsonl(processed, processed_file)
    return len(processed)


def main():
    config = load_config()
    data_cfg = config["data"]

    raw_dir       = PROJECT_ROOT / data_cfg.get("raw_dir", "./data/raw").lstrip("./")
    train_file    = PROJECT_ROOT / data_cfg.get("train_file", "./data/processed/train.jsonl").lstrip("./")
    val_file      = PROJECT_ROOT / data_cfg.get("val_file",   "./data/processed/val.jsonl").lstrip("./")
    test_file     = PROJECT_ROOT / data_cfg.get("test_file",  "./data/processed/test.jsonl").lstrip("./")

    # Check that raw data exists
    if not raw_dir.exists():
        print(f"[ERROR] Raw data directory not found: {raw_dir}")
        print("        Run script 02 first: python scripts/02_generate_synthetic_dataset.py")
        sys.exit(1)

    print("Preparing dataset (format: ChatML)...")
    print()

    total = 0
    for split_name, raw_name, out_path in [
        ("train", "train.jsonl", train_file),
        ("val",   "val.jsonl",   val_file),
        ("test",  "test.jsonl",  test_file),
    ]:
        print(f"Processing {split_name} split...")
        n = process_split(raw_dir / raw_name, out_path, fmt="chatml")
        total += n

    print()
    print(f"Dataset preparation complete! Total: {total} formatted examples.")
    print()

    # Show a sample of the formatted text
    if train_file.exists():
        with open(train_file, "r", encoding="utf-8") as f:
            sample = json.loads(f.readline())
        print("--- SAMPLE FORMATTED TEXT (first 600 chars) ---")
        print(sample["text"][:600])
        print("...")

    print()
    print("Next step: python scripts/04_train_lora.py")


if __name__ == "__main__":
    main()
