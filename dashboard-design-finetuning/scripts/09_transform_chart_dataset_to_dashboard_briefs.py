"""
09_transform_chart_dataset_to_dashboard_briefs.py

Prototype: Transforms chart-metadata records into dashboard design
training examples (dashboard brief -> structured JSON recommendation).

Input:
  Either an external JSONL file (pass path as first CLI argument)
  or the 5 built-in demo examples (default, no argument needed).

Output:
  data/processed/transformed_chart_dataset_examples.jsonl

Each output record follows the same schema used throughout this project:
  {
    "instruction": "...",
    "input":       "Dashboard brief ...",
    "output": {
      "context_summary":      "...",
      "kpi_task_chart_mapping": [...],
      "layout_hierarchy":     "...",
      "labels_scales_colors": "...",
      "interactions":         [...],
      "design_rationales":    [...]
    }
  }

Usage:
  # Use built-in demo examples
  python scripts/09_transform_chart_dataset_to_dashboard_briefs.py

  # Use an external JSONL file
  python scripts/09_transform_chart_dataset_to_dashboard_briefs.py path/to/charts.jsonl
"""

import json
import os
import sys

# ── Output path ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "transformed_chart_dataset_examples.jsonl")

INSTRUCTION = (
    "Generate a structured dashboard design recommendation based on the given dashboard brief."
)

# ── Chart-type mapping rules ──────────────────────────────────────────────────
# Each rule is a (condition_fn, chart_type, rationale) tuple.
# Rules are evaluated in order; the first match wins.

def _x_is_time(record: dict) -> bool:
    x = (record.get("x_axis") or "").lower()
    return any(kw in x for kw in ["date", "time", "month", "year", "week", "quarter", "day"])

def _is_composition(record: dict) -> bool:
    title = (record.get("title") or "").lower()
    chart = (record.get("chart_type") or "").lower()
    return any(kw in title or kw in chart for kw in ["share", "composition", "proportion", "breakdown", "distribution", "pie", "donut"])

def _is_correlation(record: dict) -> bool:
    title = (record.get("title") or "").lower()
    chart = (record.get("chart_type") or "").lower()
    return any(kw in title or kw in chart for kw in ["correlation", "scatter", "relationship", "vs.", "versus"])

def _is_category_comparison(record: dict) -> bool:
    x = (record.get("x_axis") or "").lower()
    chart = (record.get("chart_type") or "").lower()
    return any(kw in x or kw in chart for kw in ["category", "region", "product", "department", "group", "bar"])

CHART_RULES = [
    (
        _x_is_time,
        "line chart",
        "Line charts effectively communicate trends over continuous time periods.",
    ),
    (
        _is_composition,
        "stacked bar chart or donut chart",
        "Donut/stacked bar charts show part-to-whole relationships clearly.",
    ),
    (
        _is_correlation,
        "scatter plot",
        "Scatter plots expose correlations and outliers between two variables.",
    ),
    (
        _is_category_comparison,
        "bar chart",
        "Bar charts allow quick comparison across discrete categories.",
    ),
]

DEFAULT_CHART      = "bar chart"
DEFAULT_RATIONALE  = "Bar charts are a versatile default for comparing values across groups."


def infer_chart_type(record: dict) -> tuple:
    """Return (chart_type, rationale) based on the mapping rules."""
    for condition, chart, rationale in CHART_RULES:
        if condition(record):
            return chart, rationale
    return DEFAULT_CHART, DEFAULT_RATIONALE


# ── Brief generator ───────────────────────────────────────────────────────────

def build_brief(record: dict) -> str:
    """
    Construct a natural-language dashboard brief from chart metadata.
    Fields used (all optional): title, chart_type, x_axis, y_axis, summary, question, answer.
    """
    parts = []

    title = record.get("title", "").strip()
    if title:
        parts.append(f"The chart is titled '{title}'.")

    chart_type = record.get("chart_type", "").strip()
    if chart_type:
        parts.append(f"It is a {chart_type}.")

    x = record.get("x_axis", "").strip()
    y = record.get("y_axis", "").strip()
    if x and y:
        parts.append(f"The x-axis shows {x} and the y-axis shows {y}.")
    elif x:
        parts.append(f"The x-axis shows {x}.")
    elif y:
        parts.append(f"The y-axis shows {y}.")

    summary = record.get("summary", "").strip()
    if summary:
        parts.append(f"Summary: {summary}")

    question = record.get("question", "").strip()
    answer   = record.get("answer", "").strip()
    if question and answer:
        parts.append(f"A key question this chart answers: '{question}' — Answer: {answer}.")
    elif question:
        parts.append(f"A key question: '{question}'.")

    if not parts:
        parts.append("A chart with no additional metadata provided.")

    return " ".join(parts)


# ── Recommendation builder ────────────────────────────────────────────────────

def build_recommendation(record: dict) -> dict:
    """Build the structured JSON recommendation from a chart record."""
    title      = record.get("title", "Untitled Chart")
    x          = record.get("x_axis", "unknown dimension")
    y          = record.get("y_axis", "unknown metric")
    summary    = record.get("summary", "")
    chart_type_raw = record.get("chart_type", "chart")

    chart_type, rationale = infer_chart_type(record)

    context_summary = (
        f"This dashboard visualizes '{title}'. "
        f"It tracks {y} across {x}. "
        + (f"{summary}" if summary else "")
    ).strip()

    kpi_task_chart_mapping = [
        {
            "kpi":               y if y != "unknown metric" else title,
            "user_task":         "trend over time" if _x_is_time(record) else "category comparison",
            "recommended_chart": chart_type,
            "rationale":         rationale,
        },
        {
            "kpi":               f"Summary KPI for {title}",
            "user_task":         "main KPIs",
            "recommended_chart": "KPI cards",
            "rationale":         "KPI cards provide immediate visibility into the most critical metrics.",
        },
    ]

    layout_hierarchy = (
        "Top row: KPI cards showing headline numbers. "
        f"Center: {chart_type} for {y} over {x}. "
        "Bottom: data table for detailed record inspection."
    )

    labels_scales_colors = (
        "Use a clear axis label with units. "
        "Apply a sequential color palette for continuous data or distinct colors per category. "
        "Ensure sufficient contrast for accessibility."
    )

    interactions = [
        "Date range or category filter",
        "Tooltip on hover showing exact values",
        "Export to CSV button",
    ]

    design_rationales = [
        rationale,
        "KPI cards give users an immediate summary before they explore the detailed chart.",
    ]

    return {
        "context_summary":        context_summary,
        "kpi_task_chart_mapping": kpi_task_chart_mapping,
        "layout_hierarchy":       layout_hierarchy,
        "labels_scales_colors":   labels_scales_colors,
        "interactions":           interactions,
        "design_rationales":      design_rationales,
    }


# ── Transform one record ──────────────────────────────────────────────────────

def transform(record: dict) -> dict:
    """Convert one chart-metadata record into a training example."""
    return {
        "instruction": INSTRUCTION,
        "input":       build_brief(record),
        "output":      build_recommendation(record),
    }


# ── Built-in demo examples ────────────────────────────────────────────────────

DEMO_RECORDS = [
    {
        "chart_type": "line chart",
        "title":      "Monthly Revenue 2023",
        "x_axis":     "Month (date)",
        "y_axis":     "Revenue (USD)",
        "summary":    "Revenue grew steadily from January to December with a peak in Q4.",
        "question":   "Which month had the highest revenue?",
        "answer":     "December",
    },
    {
        "chart_type": "bar chart",
        "title":      "Sales by Region",
        "x_axis":     "Region (category)",
        "y_axis":     "Total Sales (USD)",
        "summary":    "The North region outperformed all others in total sales.",
        "question":   "Which region had the lowest sales?",
        "answer":     "South",
    },
    {
        "chart_type": "pie chart",
        "title":      "Market Share by Product Category",
        "x_axis":     "Product Category",
        "y_axis":     "Share (%)",
        "summary":    "Electronics accounts for 45% of total market share.",
        "question":   "What is the share of Electronics?",
        "answer":     "45%",
    },
    {
        "chart_type": "scatter plot",
        "title":      "Correlation between Ad Spend and Conversions",
        "x_axis":     "Ad Spend (USD)",
        "y_axis":     "Conversions",
        "summary":    "Higher ad spend is positively correlated with more conversions.",
        "question":   "Is there a correlation between ad spend and conversions?",
        "answer":     "Yes, positive correlation",
    },
    {
        "chart_type": "bar chart",
        "title":      "Employee Headcount by Department",
        "x_axis":     "Department (category)",
        "y_axis":     "Headcount",
        "summary":    "Engineering has the largest headcount, followed by Sales.",
        "question":   "Which department has the most employees?",
        "answer":     "Engineering",
    },
]


# ── Main ──────────────────────────────────────────────────────────────────────

def load_records(source_path: str) -> list:
    """Load records from an external JSONL file."""
    records = []
    with open(source_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {line_num} skipped (invalid JSON): {e}")
    return records


def main():
    # Determine input source
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
        if not os.path.exists(source_path):
            print(f"ERROR: File not found: {source_path}")
            sys.exit(1)
        print(f"Loading records from: {source_path}")
        records = load_records(source_path)
    else:
        print("No input file provided — using 5 built-in demo examples.")
        records = DEMO_RECORDS

    print(f"Records to transform: {len(records)}")

    # Transform
    examples = []
    for i, record in enumerate(records):
        try:
            example = transform(record)
            examples.append(example)
        except Exception as e:
            print(f"  [WARN] Record {i+1} skipped due to error: {e}")

    # Write output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Summary
    print(f"\nTransformation complete.")
    print(f"  Examples written : {len(examples)}")
    print(f"  Output file      : {os.path.abspath(OUTPUT_FILE)}")

    # Show first example
    if examples:
        print("\n--- First transformed example ---")
        print(json.dumps(examples[0], indent=2, ensure_ascii=False)[:1200])
        if len(json.dumps(examples[0])) > 1200:
            print("  ... [truncated]")
        print("--- End ---")


if __name__ == "__main__":
    main()
