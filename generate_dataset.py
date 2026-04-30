"""
generate_dataset.py
===================
Generates a synthetic dataset of dashboard design briefs and their
structured recommendations. Produces train/val/test JSONL splits.

Usage:
    python generate_dataset.py
    python generate_dataset.py --config config.yaml --output-dir ./data
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.helpers import load_config, save_jsonl, setup_logging

logger = setup_logging()


# ============================================================
# Domain Vocabulary
# ============================================================
# These lists provide variety so each generated example is unique.

INDUSTRIES = [
    "E-Commerce", "Healthcare", "Finance & Banking", "Manufacturing",
    "Logistics & Supply Chain", "Marketing & Advertising", "HR & People Analytics",
    "Retail", "SaaS / Software", "Energy & Utilities",
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

GOALS_BY_INDUSTRY = {
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

KPIS_BY_INDUSTRY = {
    "E-Commerce": [
        ["Revenue", "Conversion Rate", "Average Order Value", "Cart Abandonment Rate"],
        ["Customer Acquisition Cost", "Customer Lifetime Value", "Return Rate", "Sessions"],
        ["Gross Margin", "Revenue by Channel", "New vs. Returning Customers", "Bounce Rate"],
    ],
    "Healthcare": [
        ["Patient Wait Time", "Bed Occupancy Rate", "Readmission Rate", "Staff Utilization"],
        ["Medication Adherence", "Treatment Success Rate", "Average Length of Stay", "ER Visits"],
        ["Patient Satisfaction Score", "Appointment No-Show Rate", "Cost per Patient", "Discharge Rate"],
    ],
    "Finance & Banking": [
        ["Portfolio Return", "Risk-Adjusted Return", "Loan Default Rate", "Net Interest Margin"],
        ["Transaction Volume", "Fraud Detection Rate", "Cost-to-Income Ratio", "Capital Adequacy Ratio"],
        ["Customer Deposits", "Credit Score Distribution", "Fee Revenue", "Operational Losses"],
    ],
    "Manufacturing": [
        ["Overall Equipment Effectiveness (OEE)", "Defect Rate", "Production Throughput", "Downtime Hours"],
        ["Cycle Time", "First Pass Yield", "Scrap Rate", "Machine Utilization"],
        ["On-Time Production Rate", "Energy Consumption per Unit", "Maintenance Cost", "Safety Incidents"],
    ],
    "Logistics & Supply Chain": [
        ["On-Time Delivery Rate", "Order Fulfillment Cycle Time", "Inventory Turnover", "Freight Cost per Unit"],
        ["Perfect Order Rate", "Supplier Lead Time", "Warehouse Utilization", "Return Rate"],
        ["Backorder Rate", "Demand Forecast Accuracy", "Cost per Shipment", "Carbon Footprint per Delivery"],
    ],
    "Marketing & Advertising": [
        ["Campaign ROI", "Cost per Acquisition", "Click-Through Rate", "Conversion Rate"],
        ["Impressions", "Engagement Rate", "Cost per Click", "Revenue Attributed"],
        ["Email Open Rate", "Social Media Reach", "Lead Quality Score", "Marketing Qualified Leads"],
    ],
    "HR & People Analytics": [
        ["Employee Turnover Rate", "Time-to-Hire", "Employee Satisfaction Score", "Training Completion Rate"],
        ["Absenteeism Rate", "Internal Promotion Rate", "Cost per Hire", "Headcount by Department"],
        ["Performance Rating Distribution", "Diversity Index", "Overtime Hours", "Retention Rate"],
    ],
    "Retail": [
        ["Sales per Square Meter", "Inventory Shrinkage", "Gross Margin", "Customer Foot Traffic"],
        ["Average Transaction Value", "Stock Availability", "Sell-Through Rate", "Return Rate"],
        ["Same-Store Sales Growth", "Basket Size", "Loyalty Program Participation", "Markdowns"],
    ],
    "SaaS / Software": [
        ["Monthly Recurring Revenue (MRR)", "Churn Rate", "Customer Acquisition Cost", "Net Promoter Score"],
        ["Daily Active Users", "Feature Adoption Rate", "Support Ticket Volume", "Average Resolution Time"],
        ["Annual Recurring Revenue (ARR)", "Expansion Revenue", "Trial-to-Paid Conversion", "Uptime %"],
    ],
    "Energy & Utilities": [
        ["Energy Consumption (MWh)", "Peak Demand", "Renewable Energy Share", "Grid Reliability Index"],
        ["Outage Frequency", "Mean Time to Restore", "Cost per kWh", "Carbon Emissions"],
        ["Customer Satisfaction Score", "Meter Reading Accuracy", "Revenue per Customer", "Energy Loss Rate"],
    ],
}

CHART_TYPES = [
    "Line Chart", "Bar Chart", "Stacked Bar Chart", "Grouped Bar Chart",
    "KPI Card / Scorecard", "Gauge Chart", "Pie Chart", "Donut Chart",
    "Scatter Plot", "Heatmap", "Waterfall Chart", "Bullet Chart",
    "Area Chart", "Treemap", "Funnel Chart", "Table / Data Grid",
]

COLOR_PALETTES = [
    "Corporate Blue (#003366, #0066CC, #66B2FF) with red alerts (#CC0000)",
    "Neutral Grey (#F5F5F5 background) with teal accents (#008080) and amber warnings (#FFA500)",
    "Dark theme: Charcoal (#1E1E1E) with neon green (#00FF88) for positive, red (#FF4444) for negative",
    "Healthcare White (#FFFFFF) with calming blue (#4A90D9) and green (#27AE60) for good metrics",
    "Financial Navy (#1A237E) with gold (#FFD700) highlights and traffic-light status indicators",
    "Minimal: White background, single accent color (#E74C3C), grey for secondary data",
]

LAYOUT_PATTERNS = [
    "Top row: 3-4 KPI scorecards. Middle: 2 main charts side-by-side. Bottom: detail table.",
    "Left sidebar: navigation/filters. Main area: hero chart at top, 2x2 grid of supporting charts.",
    "Full-width header with KPI cards. Below: tabbed sections for different analysis views.",
    "Single-page scrollable: Summary section → Trend section → Breakdown section → Details.",
    "Grid layout: 12-column responsive grid. Hero metric top-left, supporting charts fill remaining space.",
    "Executive summary strip at top. Two-column layout: charts left, KPI cards and commentary right.",
]

INTERACTION_PATTERNS = [
    ["Date range picker (last 7/30/90 days, custom)", "Drill-down from summary to detail", "Hover tooltips with exact values"],
    ["Cross-filtering: clicking one chart filters all others", "Export to PDF/Excel button", "Bookmark/save current view"],
    ["Dropdown filters for region, product, segment", "Toggle between absolute values and percentages", "Comparison mode: current vs. previous period"],
    ["Search/filter within data tables", "Zoom and pan on time-series charts", "Alert thresholds with visual indicators"],
    ["Mobile-responsive touch interactions", "Annotation mode for adding comments", "Scheduled email report subscription"],
]


# ============================================================
# Example Generator
# ============================================================

def generate_example(industry: str, seed_offset: int = 0) -> Dict[str, Any]:
    """
    Generate one complete training example (brief + recommendation).

    Args:
        industry:    Industry domain for this example
        seed_offset: Offset to vary random choices within same industry

    Returns:
        Dictionary with 'brief' and 'recommendation' keys
    """
    rng = random.Random(hash(industry) + seed_offset)

    # --- Build the Brief ---
    audience = rng.choice(AUDIENCES)
    expertise = rng.choice(EXPERTISE_LEVELS)
    update_freq = rng.choice(UPDATE_FREQUENCIES)
    goals_list = GOALS_BY_INDUSTRY[industry]
    goal = rng.choice(goals_list)
    kpi_options = KPIS_BY_INDUSTRY[industry]
    kpis = rng.choice(kpi_options)
    num_kpis = rng.randint(3, len(kpis))
    selected_kpis = kpis[:num_kpis]

    brief = {
        "title": f"{industry} Performance Dashboard",
        "target_audience": audience,
        "business_goals": goal,
        "kpis": selected_kpis,
        "data_context": (
            f"Data sourced from internal {industry.lower()} systems. "
            f"Updated {update_freq.lower()}. "
            f"Covers the last 12 months with historical comparison."
        ),
        "update_frequency": update_freq,
        "user_expertise": expertise,
        "industry": industry,
    }

    # --- Build the Recommendation ---
    recommendation = _generate_recommendation(brief, rng)

    return {"brief": brief, "recommendation": recommendation}


def _generate_recommendation(brief: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Generate a structured recommendation for a given brief."""

    industry = brief["industry"]
    kpis = brief["kpis"]
    audience = brief["target_audience"]
    expertise = brief["user_expertise"]
    update_freq = brief["update_frequency"]

    # 1. Context Summary
    context_summary = (
        f"This dashboard serves {audience} in the {industry} sector. "
        f"The primary goal is to {brief['business_goals'].lower()}. "
        f"Data is refreshed {update_freq.lower()}, and the target users have "
        f"{expertise.lower()} data literacy. "
        f"The design should prioritize clarity and actionability over analytical depth."
    )

    # 2. KPI → Task → Chart Mapping
    chart_choices = rng.sample(CHART_TYPES, k=min(len(kpis) + 2, len(CHART_TYPES)))
    kpi_task_chart_mapping = []
    tasks = ["Monitor trend over time", "Compare across categories",
             "Track against target", "Identify outliers", "Show composition"]
    for i, kpi in enumerate(kpis):
        task = tasks[i % len(tasks)]
        chart = chart_choices[i % len(chart_choices)]
        kpi_task_chart_mapping.append({
            "kpi": kpi,
            "user_task": task,
            "recommended_chart": chart,
            "rationale": _chart_rationale(chart, task, kpi),
        })

    # 3. Layout Hierarchy
    layout = rng.choice(LAYOUT_PATTERNS)
    layout_hierarchy = {
        "pattern": layout,
        "primary_section": f"KPI scorecards for {', '.join(kpis[:2])}",
        "secondary_section": f"Trend analysis for {kpis[0]} over time",
        "tertiary_section": "Breakdown by segment or region" if len(kpis) > 2 else "Detail table",
        "responsive": expertise != "Beginner",
    }

    # 4. Labels, Scales, Colors
    palette = rng.choice(COLOR_PALETTES)
    labels_scales_colors = {
        "color_palette": palette,
        "axis_labels": "Always label axes with units (e.g., Revenue in EUR, Time in Days)",
        "number_format": _number_format_for_kpis(kpis),
        "scale_type": "Linear for most KPIs; logarithmic if data spans multiple orders of magnitude",
        "legend": "Include legend only when comparing 3+ series; use direct labeling otherwise",
        "font_size": "Minimum 12px for body text, 16px for KPI values, 20px for dashboard title",
        "accessibility": "Ensure 4.5:1 contrast ratio; do not rely on color alone to convey meaning",
    }

    # 5. Interactions
    interactions = rng.choice(INTERACTION_PATTERNS)
    interaction_dict = {
        "primary_interactions": interactions,
        "filter_defaults": f"Default view: last 30 days, all segments",
        "drill_down_available": len(kpis) > 2,
        "mobile_optimized": expertise == "Beginner",
    }

    # 6. Design Rationales
    design_rationales = {
        "audience_adaptation": (
            f"For {expertise.lower()} users: "
            + ("use simple KPI cards and avoid complex charts." if expertise == "Beginner"
               else "provide drill-down capabilities and advanced filters." if expertise == "Advanced / Expert"
               else "balance simplicity with analytical depth.")
        ),
        "update_frequency_impact": (
            f"With {update_freq.lower()} updates, "
            + ("use auto-refresh and highlight recent changes." if update_freq in ["Real-time", "Hourly"]
               else "include a 'last updated' timestamp prominently.")
        ),
        "chart_selection_principle": (
            "Charts were selected based on the primary user task: "
            "trend analysis → line charts; comparison → bar charts; "
            "part-to-whole → pie/donut; single value → KPI card."
        ),
        "cognitive_load": (
            "Limit to 5-7 visual elements per screen to avoid cognitive overload. "
            "Group related KPIs visually using whitespace and section headers."
        ),
        "data_ink_ratio": (
            "Remove chart borders, gridlines, and decorative elements. "
            "Every pixel should encode data or aid comprehension."
        ),
    }

    return {
        "context_summary": context_summary,
        "kpi_task_chart_mapping": kpi_task_chart_mapping,
        "layout_hierarchy": layout_hierarchy,
        "labels_scales_colors": labels_scales_colors,
        "interactions": interaction_dict,
        "design_rationales": design_rationales,
    }


def _chart_rationale(chart: str, task: str, kpi: str) -> str:
    """Generate a brief rationale for a chart choice."""
    rationales = {
        "Line Chart": f"Line charts are ideal for showing {kpi} trends over time, making patterns and seasonality visible.",
        "Bar Chart": f"Bar charts enable easy comparison of {kpi} across discrete categories or time periods.",
        "KPI Card / Scorecard": f"A KPI card gives immediate visibility into the current {kpi} value against its target.",
        "Gauge Chart": f"A gauge chart shows {kpi} performance relative to a defined threshold or goal.",
        "Heatmap": f"A heatmap reveals patterns in {kpi} across two dimensions simultaneously.",
        "Scatter Plot": f"A scatter plot exposes correlations between {kpi} and other variables.",
        "Funnel Chart": f"A funnel chart visualizes the conversion steps contributing to {kpi}.",
        "Waterfall Chart": f"A waterfall chart shows how individual components build up to the total {kpi}.",
        "Treemap": f"A treemap shows the proportional contribution of sub-categories to total {kpi}.",
        "Table / Data Grid": f"A table provides precise {kpi} values for users who need exact numbers.",
    }
    return rationales.get(chart, f"{chart} is appropriate for the task of '{task}' with {kpi}.")


def _number_format_for_kpis(kpis: List[str]) -> str:
    """Suggest number formatting based on KPI names."""
    kpi_text = " ".join(kpis).lower()
    if any(word in kpi_text for word in ["rate", "ratio", "%", "percentage", "share"]):
        return "Percentages: 1 decimal place (e.g., 23.4%). Large numbers: K/M suffix (e.g., 1.2M)."
    elif any(word in kpi_text for word in ["revenue", "cost", "margin", "value", "price"]):
        return "Currency: 2 decimal places for small values, K/M suffix for large (e.g., €1.2M). Use locale-appropriate separators."
    elif any(word in kpi_text for word in ["time", "duration", "wait", "cycle"]):
        return "Time values: use appropriate unit (minutes, hours, days). Round to 1 decimal place."
    else:
        return "Integers for counts; 1-2 decimal places for averages; K/M suffix for large numbers."


# ============================================================
# Dataset Split Generator
# ============================================================

def generate_dataset(
    num_train: int = 80,
    num_val: int = 10,
    num_test: int = 10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate train/val/test splits of the synthetic dataset.

    Args:
        num_train: Number of training examples
        num_val:   Number of validation examples
        num_test:  Number of test examples
        seed:      Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    random.seed(seed)
    total = num_train + num_val + num_test
    logger.info(f"Generating {total} examples ({num_train} train, {num_val} val, {num_test} test)...")

    all_examples = []
    industries = INDUSTRIES.copy()

    for i in range(total):
        industry = industries[i % len(industries)]
        example = generate_example(industry, seed_offset=i * 137 + seed)
        all_examples.append(example)

    # Shuffle before splitting
    random.shuffle(all_examples)

    train = all_examples[:num_train]
    val   = all_examples[num_train:num_train + num_val]
    test  = all_examples[num_train + num_val:]

    logger.info(f"Split: {len(train)} train | {len(val)} val | {len(test)} test")
    return train, val, test


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dashboard design dataset"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", default="./data", help="Output directory for JSONL files")
    parser.add_argument("--num-train", type=int, default=None, help="Override number of train samples")
    parser.add_argument("--num-val",   type=int, default=None, help="Override number of val samples")
    parser.add_argument("--num-test",  type=int, default=None, help="Override number of test samples")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    gen_cfg = config.get("dataset_generation", {})

    num_train = args.num_train or gen_cfg.get("num_train_samples", 80)
    num_val   = args.num_val   or gen_cfg.get("num_val_samples",   10)
    num_test  = args.num_test  or gen_cfg.get("num_test_samples",  10)
    seed      = gen_cfg.get("seed", 42)

    # Generate
    train_data, val_data, test_data = generate_dataset(num_train, num_val, num_test, seed)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data,   output_dir / "val.jsonl")
    save_jsonl(test_data,  output_dir / "test.jsonl")

    # Print a sample
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING EXAMPLE")
    print("=" * 60)
    sample = train_data[0]
    print("\n--- BRIEF ---")
    print(json.dumps(sample["brief"], indent=2, ensure_ascii=False))
    print("\n--- RECOMMENDATION (first 2 keys) ---")
    rec_preview = {k: v for i, (k, v) in enumerate(sample["recommendation"].items()) if i < 2}
    print(json.dumps(rec_preview, indent=2, ensure_ascii=False))
    print("\nDataset generation complete!")
    print(f"   Files saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
