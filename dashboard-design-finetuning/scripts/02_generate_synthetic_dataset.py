"""
scripts/02_generate_synthetic_dataset.py
=========================================
Generates a synthetic dataset of dashboard design briefs and their
structured recommendations. Saves raw JSONL files to data/raw/.

This script does NOT require a GPU or any ML libraries.
It only uses Python standard library + pyyaml.

Usage:
    python scripts/02_generate_synthetic_dataset.py

Output:
    data/raw/train.jsonl   (80 examples)
    data/raw/val.jsonl     (10 examples)
    data/raw/test.jsonl    (10 examples)
"""

import json
import os
import random
import sys
from pathlib import Path

# Add project root to path so we can import config loader
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


# ============================================================
# Load Config
# ============================================================

def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "train_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Domain Vocabulary
# ============================================================

INDUSTRIES = [
    "E-Commerce", "Healthcare", "Finance & Banking", "Manufacturing",
    "Logistics & Supply Chain", "Marketing & Advertising",
    "HR & People Analytics", "Retail", "SaaS / Software", "Energy & Utilities",
]

AUDIENCES = [
    "C-Level Executives (CEO, CFO, COO)",
    "Sales Managers and Regional Directors",
    "Data Analysts and BI Developers",
    "Operations Team Leads",
    "Marketing Campaign Managers",
    "HR Business Partners",
    "Customer Success Managers",
    "Supply Chain Coordinators",
    "Product Managers",
    "Financial Controllers",
]

EXPERTISE_LEVELS = ["Beginner", "Intermediate", "Advanced / Expert"]
UPDATE_FREQUENCIES = ["Real-time", "Hourly", "Daily", "Weekly", "Monthly"]

# Business goals per industry
GOALS = {
    "E-Commerce": [
        "Monitor daily revenue and conversion rates",
        "Track cart abandonment and checkout funnel performance",
        "Analyze customer acquisition cost vs. lifetime value",
    ],
    "Healthcare": [
        "Monitor patient wait times and bed occupancy",
        "Track medication adherence and treatment outcomes",
        "Analyze staff utilization and shift coverage",
    ],
    "Finance & Banking": [
        "Monitor portfolio performance and risk exposure",
        "Track loan default rates and credit scores",
        "Analyze transaction volumes and fraud detection rates",
    ],
    "Manufacturing": [
        "Monitor production throughput and machine utilization",
        "Track defect rates and quality control metrics",
        "Analyze downtime causes and maintenance schedules",
    ],
    "Logistics & Supply Chain": [
        "Monitor on-time delivery rates and shipment status",
        "Track inventory levels and reorder points",
        "Analyze supplier performance and lead times",
    ],
    "Marketing & Advertising": [
        "Monitor campaign ROI and cost per acquisition",
        "Track click-through rates and conversion funnels",
        "Analyze audience segmentation and engagement metrics",
    ],
    "HR & People Analytics": [
        "Monitor employee turnover and retention rates",
        "Track recruitment pipeline and time-to-hire",
        "Analyze training completion and skill gap metrics",
    ],
    "Retail": [
        "Monitor store sales performance by region",
        "Track inventory shrinkage and stock availability",
        "Analyze customer foot traffic and basket size",
    ],
    "SaaS / Software": [
        "Monitor monthly recurring revenue (MRR) and churn",
        "Track feature adoption and user engagement",
        "Analyze support ticket volume and resolution time",
    ],
    "Energy & Utilities": [
        "Monitor energy consumption and peak demand",
        "Track renewable energy output vs. grid demand",
        "Analyze outage frequency and restoration times",
    ],
}

# KPI sets per industry (multiple options per industry for variety)
KPIS = {
    "E-Commerce": [
        ["Revenue", "Conversion Rate", "Average Order Value", "Cart Abandonment Rate"],
        ["Customer Acquisition Cost", "Customer Lifetime Value", "Return Rate", "Sessions"],
    ],
    "Healthcare": [
        ["Patient Wait Time", "Bed Occupancy Rate", "Readmission Rate", "Staff Utilization"],
        ["Medication Adherence", "Treatment Success Rate", "Average Length of Stay", "ER Visits"],
    ],
    "Finance & Banking": [
        ["Portfolio Return", "Risk-Adjusted Return", "Loan Default Rate", "Net Interest Margin"],
        ["Transaction Volume", "Fraud Detection Rate", "Cost-to-Income Ratio", "Capital Adequacy Ratio"],
    ],
    "Manufacturing": [
        ["Overall Equipment Effectiveness (OEE)", "Defect Rate", "Production Throughput", "Downtime Hours"],
        ["Cycle Time", "First Pass Yield", "Scrap Rate", "Machine Utilization"],
    ],
    "Logistics & Supply Chain": [
        ["On-Time Delivery Rate", "Order Fulfillment Cycle Time", "Inventory Turnover", "Freight Cost per Unit"],
        ["Perfect Order Rate", "Supplier Lead Time", "Warehouse Utilization", "Return Rate"],
    ],
    "Marketing & Advertising": [
        ["Campaign ROI", "Cost per Acquisition", "Click-Through Rate", "Conversion Rate"],
        ["Impressions", "Engagement Rate", "Cost per Click", "Revenue Attributed"],
    ],
    "HR & People Analytics": [
        ["Employee Turnover Rate", "Time-to-Hire", "Employee Satisfaction Score", "Training Completion Rate"],
        ["Absenteeism Rate", "Internal Promotion Rate", "Cost per Hire", "Headcount by Department"],
    ],
    "Retail": [
        ["Sales per Square Meter", "Inventory Shrinkage", "Gross Margin", "Customer Foot Traffic"],
        ["Average Transaction Value", "Stock Availability", "Sell-Through Rate", "Return Rate"],
    ],
    "SaaS / Software": [
        ["Monthly Recurring Revenue (MRR)", "Churn Rate", "Customer Acquisition Cost", "Net Promoter Score"],
        ["Daily Active Users", "Feature Adoption Rate", "Support Ticket Volume", "Average Resolution Time"],
    ],
    "Energy & Utilities": [
        ["Energy Consumption (MWh)", "Peak Demand", "Renewable Energy Share", "Grid Reliability Index"],
        ["Outage Frequency", "Mean Time to Restore", "Cost per kWh", "Carbon Emissions"],
    ],
}

CHART_TYPES = [
    "Line Chart", "Bar Chart", "Stacked Bar Chart", "KPI Card / Scorecard",
    "Gauge Chart", "Pie Chart", "Donut Chart", "Scatter Plot",
    "Heatmap", "Waterfall Chart", "Bullet Chart", "Area Chart",
    "Treemap", "Funnel Chart", "Table / Data Grid",
]

COLOR_PALETTES = [
    "Corporate Blue (#003366, #0066CC) with red alerts (#CC0000)",
    "Neutral Grey background with teal accents (#008080) and amber warnings (#FFA500)",
    "Dark theme: Charcoal (#1E1E1E) with green (#00FF88) for positive, red (#FF4444) for negative",
    "Healthcare White with calming blue (#4A90D9) and green (#27AE60) for good metrics",
    "Financial Navy (#1A237E) with gold (#FFD700) highlights and traffic-light status indicators",
]

LAYOUT_PATTERNS = [
    "Top row: 3-4 KPI scorecards. Middle: 2 main charts side-by-side. Bottom: detail table.",
    "Left sidebar: navigation/filters. Main area: hero chart at top, 2x2 grid of supporting charts.",
    "Full-width header with KPI cards. Below: tabbed sections for different analysis views.",
    "Single-page scrollable: Summary -> Trend -> Breakdown -> Details.",
    "Grid layout: 12-column responsive. Hero metric top-left, supporting charts fill remaining space.",
]

INTERACTION_SETS = [
    ["Date range picker (last 7/30/90 days)", "Drill-down from summary to detail", "Hover tooltips with exact values"],
    ["Cross-filtering: clicking one chart filters all others", "Export to PDF/Excel", "Bookmark current view"],
    ["Dropdown filters for region, product, segment", "Toggle between absolute and percentage values", "Comparison mode: current vs. previous period"],
]


# ============================================================
# Example Generator
# ============================================================

def build_brief(industry: str, rng: random.Random) -> dict:
    """Build a dashboard brief dictionary for a given industry."""
    audience = rng.choice(AUDIENCES)
    expertise = rng.choice(EXPERTISE_LEVELS)
    freq = rng.choice(UPDATE_FREQUENCIES)
    goal = rng.choice(GOALS[industry])
    kpi_set = rng.choice(KPIS[industry])
    # Use 3 or 4 KPIs
    num_kpis = rng.randint(3, len(kpi_set))
    kpis = kpi_set[:num_kpis]

    return {
        "title": f"{industry} Performance Dashboard",
        "target_audience": audience,
        "business_goals": goal,
        "kpis": kpis,
        "data_context": (
            f"Data sourced from internal {industry.lower()} systems. "
            f"Updated {freq.lower()}. "
            f"Covers the last 12 months with historical comparison."
        ),
        "update_frequency": freq,
        "user_expertise": expertise,
        "industry": industry,
    }


def build_recommendation(brief: dict, rng: random.Random) -> dict:
    """Build a structured recommendation for a given brief."""
    industry = brief["industry"]
    kpis = brief["kpis"]
    audience = brief["target_audience"]
    expertise = brief["user_expertise"]
    freq = brief["update_frequency"]

    # 1. Context Summary
    context_summary = (
        f"This dashboard serves {audience} in the {industry} sector. "
        f"The primary goal is to {brief['business_goals'].lower()}. "
        f"Data is refreshed {freq.lower()}, and users have {expertise.lower()} "
        f"data literacy. The design prioritizes clarity and actionability."
    )

    # 2. KPI -> Task -> Chart Mapping
    tasks = [
        "Monitor trend over time",
        "Compare across categories",
        "Track against target",
        "Identify outliers",
        "Show composition",
    ]
    chart_pool = rng.sample(CHART_TYPES, k=min(len(kpis) + 3, len(CHART_TYPES)))
    kpi_task_chart_mapping = []
    for i, kpi in enumerate(kpis):
        task = tasks[i % len(tasks)]
        chart = chart_pool[i % len(chart_pool)]
        kpi_task_chart_mapping.append({
            "kpi": kpi,
            "user_task": task,
            "recommended_chart": chart,
            "rationale": f"{chart} is well-suited for '{task}' with {kpi}.",
        })

    # 3. Layout Hierarchy
    layout = rng.choice(LAYOUT_PATTERNS)
    layout_hierarchy = {
        "pattern": layout,
        "primary_section": f"KPI scorecards for {', '.join(kpis[:2])}",
        "secondary_section": f"Trend analysis for {kpis[0]} over time",
        "tertiary_section": "Breakdown by segment or region",
        "responsive_design": expertise != "Beginner",
    }

    # 4. Labels, Scales, Colors
    palette = rng.choice(COLOR_PALETTES)
    labels_scales_colors = {
        "color_palette": palette,
        "axis_labels": "Always label axes with units (e.g., Revenue in EUR, Time in Days)",
        "number_format": "K/M suffix for large numbers; 1 decimal for percentages",
        "scale_type": "Linear for most KPIs; logarithmic if data spans multiple orders of magnitude",
        "legend": "Include legend only when comparing 3+ series; use direct labeling otherwise",
        "accessibility": "Ensure 4.5:1 contrast ratio; do not rely on color alone",
    }

    # 5. Interactions
    interactions_list = rng.choice(INTERACTION_SETS)
    interactions = {
        "primary_interactions": interactions_list,
        "filter_defaults": "Default view: last 30 days, all segments",
        "drill_down_available": len(kpis) > 2,
        "mobile_optimized": expertise == "Beginner",
    }

    # 6. Design Rationales
    design_rationales = {
        "audience_adaptation": (
            f"For {expertise.lower()} users: "
            + ("use simple KPI cards and avoid complex charts." if expertise == "Beginner"
               else "provide drill-down and advanced filters." if "Expert" in expertise
               else "balance simplicity with analytical depth.")
        ),
        "update_frequency_impact": (
            f"With {freq.lower()} updates, "
            + ("use auto-refresh and highlight recent changes." if freq in ["Real-time", "Hourly"]
               else "include a prominent 'last updated' timestamp.")
        ),
        "chart_selection_principle": (
            "Charts selected based on primary user task: "
            "trend -> line chart; comparison -> bar chart; "
            "part-to-whole -> pie/donut; single value -> KPI card."
        ),
        "cognitive_load": (
            "Limit to 5-7 visual elements per screen. "
            "Group related KPIs with whitespace and section headers."
        ),
    }

    return {
        "context_summary": context_summary,
        "kpi_task_chart_mapping": kpi_task_chart_mapping,
        "layout_hierarchy": layout_hierarchy,
        "labels_scales_colors": labels_scales_colors,
        "interactions": interactions,
        "design_rationales": design_rationales,
    }


def generate_all(num_train: int, num_val: int, num_test: int, seed: int):
    """Generate all examples and return (train, val, test) lists."""
    rng = random.Random(seed)
    total = num_train + num_val + num_test
    print(f"Generating {total} examples ({num_train} train / {num_val} val / {num_test} test)...")

    all_examples = []
    for i in range(total):
        industry = INDUSTRIES[i % len(INDUSTRIES)]
        brief = build_brief(industry, rng)
        recommendation = build_recommendation(brief, rng)
        all_examples.append({"brief": brief, "recommendation": recommendation})

    # Shuffle before splitting
    rng.shuffle(all_examples)

    train = all_examples[:num_train]
    val   = all_examples[num_train:num_train + num_val]
    test  = all_examples[num_train + num_val:]
    return train, val, test


# ============================================================
# Save JSONL
# ============================================================

def save_jsonl(records: list, filepath: Path) -> None:
    """Save a list of dicts to a JSONL file (one JSON object per line)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records -> {filepath}")


# ============================================================
# Main
# ============================================================

def main():
    config = load_config()
    data_cfg = config["data"]

    num_train = data_cfg.get("num_train", 80)
    num_val   = data_cfg.get("num_val",   10)
    num_test  = data_cfg.get("num_test",  10)
    seed      = data_cfg.get("seed",      42)
    raw_dir   = PROJECT_ROOT / data_cfg.get("raw_dir", "./data/raw").lstrip("./")

    train, val, test = generate_all(num_train, num_val, num_test, seed)

    save_jsonl(train, raw_dir / "train.jsonl")
    save_jsonl(val,   raw_dir / "val.jsonl")
    save_jsonl(test,  raw_dir / "test.jsonl")

    # Print a sample so you can verify the output
    print("\n--- SAMPLE BRIEF ---")
    print(json.dumps(train[0]["brief"], indent=2, ensure_ascii=False))
    print("\n--- SAMPLE RECOMMENDATION (context_summary only) ---")
    print(train[0]["recommendation"]["context_summary"])
    print("\nDataset generation complete!")
    print("Next step: python scripts/03_prepare_dataset.py")


if __name__ == "__main__":
    main()
