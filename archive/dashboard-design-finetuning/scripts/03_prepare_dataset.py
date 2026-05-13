"""
03_prepare_dataset.py

Converts data/raw/synthetic_dashboard_recommendations.jsonl
into Hugging Face TRL SFTTrainer-compatible JSONL files:
  data/processed/train.jsonl
  data/processed/validation.jsonl

Each output line contains a single "text" field formatted for
instruction tuning.
"""

import json
import os
import random

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
INPUT_FILE  = os.path.join(BASE_DIR, "..", "data", "raw",
                           "synthetic_dashboard_recommendations.jsonl")
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "data", "processed")
TRAIN_FILE  = os.path.join(OUTPUT_DIR, "train.jsonl")
VAL_FILE    = os.path.join(OUTPUT_DIR, "validation.jsonl")

SYSTEM_PROMPT = (
    "You are a dashboard design assistant. "
    "You generate structured, practical, and heuristic-aligned "
    "dashboard design recommendations."
)

TRAIN_RATIO = 0.90
RANDOM_SEED = 42


# ── Format one example into the instruction-tuning text ───────────────────────
def format_example(example: dict) -> str:
    instruction = example["instruction"]
    inp         = example["input"]
    output      = example["output"]

    # Pretty-print the output JSON so the model learns structured formatting
    output_str = json.dumps(output, indent=2, ensure_ascii=False)

    text = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n"
        f"Instruction:\n{instruction}\n\n"
        f"Dashboard brief:\n{inp}\n"
        f"<|assistant|>\n{output_str}"
    )
    return text


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load raw JSONL
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run 02_generate_synthetic_dataset.py first."
        )

    examples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {line_num} is not valid JSON: {e}")
                continue

            # 2. Validate that 'output' is a dict (already parsed JSON)
            if not isinstance(record.get("output"), dict):
                print(f"  [WARN] Line {line_num}: 'output' is not a JSON object, skipping.")
                continue

            examples.append(record)

    print(f"Loaded {len(examples)} valid examples from input file.")

    # 3. Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(examples)

    split_idx   = int(len(examples) * TRAIN_RATIO)
    train_data  = examples[:split_idx]
    val_data    = examples[split_idx:]

    # 4. Write output files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def write_jsonl(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                text = format_example(rec)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    write_jsonl(train_data, TRAIN_FILE)
    write_jsonl(val_data,   VAL_FILE)

    # 5. Summary
    print(f"\nDataset preparation complete.")
    print(f"  Train examples      : {len(train_data)}")
    print(f"  Validation examples : {len(val_data)}")
    print(f"  Train file          : {os.path.abspath(TRAIN_FILE)}")
    print(f"  Validation file     : {os.path.abspath(VAL_FILE)}")

    # 6. Print one formatted sample
    print("\n--- Example formatted training sample (first record) ---\n")
    print(format_example(train_data[0]))
    print("\n--- End of sample ---")


if __name__ == "__main__":
    main()
