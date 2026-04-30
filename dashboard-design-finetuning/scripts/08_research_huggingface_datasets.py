"""
08_research_huggingface_datasets.py

Investigates Hugging Face datasets relevant to chart/dashboard/visualization
research for the master thesis.

For each dataset it tries to:
  1. Load the dataset (or a small split of it)
  2. Print available splits
  3. Print column names
  4. Show the first example
  5. Catch and document any errors gracefully

Output:
  outputs/logs/dataset_research_report.json

Requirements:
  pip install datasets
"""

import json
import os
import traceback

# ── Output path ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
LOG_DIR     = os.path.join(BASE_DIR, "..", "outputs", "logs")
REPORT_FILE = os.path.join(LOG_DIR, "dataset_research_report.json")

# ── Dataset candidates ────────────────────────────────────────────────────────
# Format: (dataset_id, config_name_or_None, split_to_try)
# Add more entries here as you discover new datasets.
DATASETS_TO_CHECK = [
    # Chart Question Answering
    ("HuggingFaceM4/ChartQA",           None,       "test"),
    ("ahmed-masry/ChartQA",             None,       "test"),
    ("lmms-lab/ChartQA",                None,       "test"),

    # Chart-to-Text / Chart Summarization
    ("saadob12/chart-to-text",          None,       "train"),
    ("ahmed-masry/Chart-to-Text",       None,       "train"),
    ("shreyasharma/chart-to-text",      None,       "train"),

    # Data Visualization / Table-to-Text
    ("wikitablequestions",              None,       "train"),
    ("Stanford/wikitablequestions",     None,       "train"),
    ("GEM/dart",                        None,       "train"),
    ("GEM/totto",                       None,       "train"),
    ("kasnerz/scigen",                  None,       "train"),

    # Dashboard / BI related (placeholders — add real IDs when found)
    # ("your-org/dashboard-dataset",    None,       "train"),
    # ("your-org/bi-report-dataset",    None,       "train"),
]

# Maximum number of bytes to store from the first example (to keep report readable)
MAX_EXAMPLE_CHARS = 1000


def inspect_dataset(dataset_id: str, config: str, split: str) -> dict:
    """
    Try to load one split of a dataset and collect metadata.
    Returns a result dict — never raises.
    """
    result = {
        "dataset_id":   dataset_id,
        "config":       config,
        "split_tried":  split,
        "status":       "unknown",
        "error":        None,
        "splits":       None,
        "columns":      None,
        "num_rows":     None,
        "first_example": None,
    }

    try:
        from datasets import load_dataset, get_dataset_split_names

        # Try to get available splits first (lightweight)
        try:
            available_splits = get_dataset_split_names(dataset_id, config_name=config)
            result["splits"] = available_splits
        except Exception:
            result["splits"] = "could not retrieve"

        # Load just the requested split (streaming=True avoids downloading everything)
        print(f"  Loading {dataset_id} / split={split} ...")
        ds = load_dataset(
            dataset_id,
            config,
            split=split,
            streaming=True,
            # trust_remote_code removed: deprecated in datasets >= 2.20
        )

        # Get column names from the first batch
        first = next(iter(ds))
        result["columns"]       = list(first.keys())
        result["num_rows"]      = "streaming (unknown)"

        # Truncate the first example so the report stays readable
        example_str = json.dumps(first, ensure_ascii=False, default=str)
        if len(example_str) > MAX_EXAMPLE_CHARS:
            example_str = example_str[:MAX_EXAMPLE_CHARS] + "  ... [truncated]"
        result["first_example"] = example_str
        result["status"]        = "success"

    except ImportError:
        result["status"] = "error"
        result["error"]  = "datasets package not installed. Run: pip install datasets"
    except Exception as e:
        result["status"] = "error"
        result["error"]  = f"{type(e).__name__}: {str(e)}"
        # Uncomment the next line for full tracebacks during debugging:
        # result["traceback"] = traceback.format_exc()

    return result


def print_result(r: dict):
    """Print a compact summary of one dataset inspection."""
    status_tag = "[OK]" if r["status"] == "success" else "[FAIL]"
    print(f"  {status_tag} {r['dataset_id']}")
    if r["status"] == "success":
        print(f"         splits  : {r['splits']}")
        print(f"         columns : {r['columns']}")
    else:
        print(f"         error   : {r['error']}")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("Hugging Face Dataset Research")
    print(f"Checking {len(DATASETS_TO_CHECK)} datasets ...")
    print("=" * 60)

    results = []
    success_count = 0
    fail_count    = 0

    for dataset_id, config, split in DATASETS_TO_CHECK:
        print(f"\n[{len(results)+1}/{len(DATASETS_TO_CHECK)}] {dataset_id}")
        r = inspect_dataset(dataset_id, config, split)
        results.append(r)
        print_result(r)
        if r["status"] == "success":
            success_count += 1
        else:
            fail_count += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Results: {success_count} loaded successfully, {fail_count} failed.")
    print("=" * 60)

    report = {
        "total_checked":   len(results),
        "success_count":   success_count,
        "fail_count":      fail_count,
        "datasets":        results,
        "notes": [
            "Datasets marked 'error' may still exist but require manual download,",
            "a Hugging Face token, or a different config/split name.",
            "Use streaming=True to avoid downloading large datasets fully.",
            "Add more dataset IDs to DATASETS_TO_CHECK at the top of this script.",
        ],
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {os.path.abspath(REPORT_FILE)}")


if __name__ == "__main__":
    main()
