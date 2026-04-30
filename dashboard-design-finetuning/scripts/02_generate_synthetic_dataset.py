"""
02_generate_synthetic_dataset.py

Generates 80 synthetic dashboard design recommendation examples
(10 domains × 8 variants each) and writes them to:
  data/raw/synthetic_dashboard_recommendations.jsonl

Uses only Python standard library. No external APIs.
"""

import json
import os
import random

# ── Output path ──────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_dashboard_recommendations.jsonl")

# ── Domain definitions ────────────────────────────────────────────────────────
DOMAINS = [
    "Sales",
    "Finance",
    "HR",
    "Marketing",
    "Logistics",
    "Healthcare",
    "Energy Consumption",
    "Customer Support",
    "E-Commerce",
    "Project Management",
]

KPI_BY_DOMAIN = {
    "Sales": [
        "Total Revenue", "Monthly Sales Growth", "Win Rate", "Average Deal Size",
        "Sales by Region", "Pipeline Value", "Quota Attainment", "Churn Rate",
    ],
    "Finance": [
        "Net Profit Margin", "Operating Expenses", "Cash Flow", "Budget Variance",
        "Revenue vs. Cost", "EBITDA", "Accounts Receivable", "Debt-to-Equity Ratio",
    ],
    "HR": [
        "Headcount", "Employee Turnover Rate", "Time to Hire", "Absenteeism Rate",
        "Training Completion Rate", "Employee Satisfaction Score", "Diversity Ratio", "Overtime Hours",
    ],
    "Marketing": [
        "Campaign ROI", "Lead Conversion Rate", "Cost per Lead", "Website Traffic",
        "Email Open Rate", "Social Media Engagement", "Customer Acquisition Cost", "Brand Awareness Score",
    ],
    "Logistics": [
        "On-Time Delivery Rate", "Shipment Volume", "Warehouse Utilization", "Order Fulfillment Time",
        "Return Rate", "Freight Cost per Unit", "Inventory Turnover", "Carrier Performance",
    ],
    "Healthcare": [
        "Patient Admission Rate", "Average Length of Stay", "Bed Occupancy Rate", "Readmission Rate",
        "Treatment Success Rate", "Staff-to-Patient Ratio", "Appointment No-Show Rate", "Medication Error Rate",
    ],
    "Energy Consumption": [
        "Total Energy Usage (kWh)", "Peak Demand", "Renewable Energy Share", "Energy Cost per Unit",
        "Carbon Emissions", "Energy Intensity", "Equipment Efficiency", "Downtime Hours",
    ],
    "Customer Support": [
        "First Response Time", "Ticket Resolution Rate", "Customer Satisfaction (CSAT)", "Average Handle Time",
        "Escalation Rate", "Open Tickets", "Agent Utilization", "Net Promoter Score (NPS)",
    ],
    "E-Commerce": [
        "Gross Merchandise Value (GMV)", "Conversion Rate", "Cart Abandonment Rate", "Average Order Value",
        "Return Rate", "Customer Lifetime Value", "Revenue by Category", "Traffic by Channel",
    ],
    "Project Management": [
        "Task Completion Rate", "Budget Utilization", "Schedule Variance", "Resource Allocation",
        "Risk Count", "Milestone Achievement Rate", "Team Velocity", "Defect Rate",
    ],
}

USER_GOALS_BY_DOMAIN = {
    "Sales": [
        "track revenue trends", "compare regional performance", "monitor pipeline health",
        "identify top-performing products", "forecast quarterly targets",
        "analyze win/loss ratios", "evaluate rep performance", "detect churn risk",
    ],
    "Finance": [
        "monitor cash flow", "compare budget vs. actuals", "track profit margins",
        "analyze expense categories", "forecast revenue", "evaluate financial health",
        "identify cost overruns", "review debt levels",
    ],
    "HR": [
        "track headcount changes", "monitor turnover trends", "evaluate hiring efficiency",
        "analyze absenteeism patterns", "review training progress", "assess workforce diversity",
        "identify overtime hotspots", "measure employee satisfaction",
    ],
    "Marketing": [
        "evaluate campaign performance", "track lead funnel", "compare channel ROI",
        "monitor website traffic", "analyze email engagement", "measure brand reach",
        "optimize ad spend", "identify top-converting campaigns",
    ],
    "Logistics": [
        "monitor delivery performance", "track shipment volumes", "optimize warehouse usage",
        "analyze fulfillment times", "review return rates", "compare carrier performance",
        "manage inventory levels", "reduce freight costs",
    ],
    "Healthcare": [
        "monitor patient admissions", "track bed occupancy", "analyze readmission patterns",
        "evaluate treatment outcomes", "manage staff ratios", "reduce no-show rates",
        "track medication errors", "optimize length of stay",
    ],
    "Energy Consumption": [
        "monitor total energy usage", "identify peak demand periods", "track renewable share",
        "analyze energy costs", "reduce carbon emissions", "improve equipment efficiency",
        "detect downtime patterns", "benchmark energy intensity",
    ],
    "Customer Support": [
        "track response times", "monitor ticket resolution", "measure customer satisfaction",
        "analyze handle time trends", "reduce escalation rates", "manage open ticket backlog",
        "optimize agent workload", "improve NPS scores",
    ],
    "E-Commerce": [
        "track sales performance", "analyze conversion funnel", "reduce cart abandonment",
        "monitor order values", "manage return rates", "evaluate customer lifetime value",
        "compare category revenue", "analyze traffic sources",
    ],
    "Project Management": [
        "track task completion", "monitor budget utilization", "identify schedule delays",
        "manage resource allocation", "track risk exposure", "review milestone progress",
        "measure team velocity", "analyze defect trends",
    ],
}

DATA_FIELDS_BY_DOMAIN = {
    "Sales": ["date", "region", "product", "rep_name", "revenue", "deals_closed", "pipeline_stage"],
    "Finance": ["date", "department", "account", "budget", "actual_spend", "profit", "expense_category"],
    "HR": ["date", "department", "employee_id", "hire_date", "exit_date", "training_status", "satisfaction_score"],
    "Marketing": ["date", "channel", "campaign_name", "impressions", "clicks", "leads", "conversions", "spend"],
    "Logistics": ["date", "carrier", "region", "shipment_id", "delivery_status", "fulfillment_time", "freight_cost"],
    "Healthcare": ["date", "ward", "patient_id", "admission_type", "length_of_stay", "readmitted", "staff_count"],
    "Energy Consumption": ["date", "facility", "meter_id", "kwh_used", "peak_demand", "renewable_kwh", "cost"],
    "Customer Support": ["date", "agent_id", "ticket_id", "channel", "resolution_time", "csat_score", "escalated"],
    "E-Commerce": ["date", "product_category", "sku", "orders", "revenue", "returns", "traffic_source", "cart_events"],
    "Project Management": ["date", "project_id", "task_id", "assignee", "status", "planned_hours", "actual_hours", "risk_level"],
}

# ── Chart rules ───────────────────────────────────────────────────────────────
CHART_RULES = {
    "trend over time": "line chart",
    "category comparison": "bar chart",
    "composition/share": "stacked bar chart or donut chart",
    "relationship/correlation": "scatter plot",
    "detailed records": "table",
    "main KPIs": "KPI cards",
    "patterns over time/categories": "heatmap",
}

TASK_TO_CHART = [
    ("trend over time",          "line chart"),
    ("category comparison",      "bar chart"),
    ("composition/share",        "stacked bar chart or donut chart"),
    ("relationship/correlation", "scatter plot"),
    ("detailed records",         "table"),
    ("main KPIs",                "KPI cards"),
    ("patterns over time/categories", "heatmap"),
]

LAYOUT_OPTIONS = [
    "Top row: KPI cards. Middle: main chart (full width). Bottom: supporting charts side by side.",
    "Left panel: KPI cards (vertical). Right panel: main chart on top, detail table below.",
    "Top: KPI cards row. Center: two charts side by side. Bottom: trend line chart full width.",
    "Header: KPI summary cards. Body: tabbed views per category. Footer: data table.",
    "Single-page layout: KPI cards at top, heatmap in center, bar chart and line chart at bottom.",
]

INTERACTION_OPTIONS = [
    "Date range filter", "Region/department dropdown filter", "Drill-down on chart click",
    "Tooltip on hover", "Export to CSV button", "Search/filter on table",
    "Toggle between chart types", "Highlight on selection", "Cross-filter between charts",
    "Zoom on time axis",
]

COLOR_OPTIONS = [
    "Blue-green palette for positive trends; red for negative deviations. Clear axis labels with units.",
    "Neutral grey base with accent colors per category. Consistent legend placement.",
    "Traffic-light colors (green/yellow/red) for KPI status indicators. Accessible contrast ratios.",
    "Sequential color scale for heatmap. Diverging palette for variance charts.",
    "Brand-aligned primary color for main metrics; muted tones for secondary data.",
]

RATIONALE_SNIPPETS = [
    "Line charts effectively communicate trends over continuous time periods.",
    "Bar charts allow quick comparison across discrete categories.",
    "KPI cards provide immediate visibility into the most critical metrics.",
    "Heatmaps reveal patterns across two dimensions simultaneously.",
    "Donut/stacked bar charts show part-to-whole relationships clearly.",
    "Tables support detailed record inspection and sorting.",
    "Scatter plots expose correlations and outliers between two variables.",
    "Drill-down interactions let users move from summary to detail without leaving the dashboard.",
    "Consistent color coding reduces cognitive load and speeds interpretation.",
    "Filtering controls empower users to focus on relevant subsets of data.",
]

INSTRUCTION = (
    "Generate a structured dashboard design recommendation based on the given dashboard brief."
)

# ── Helper: pick N unique items from a list ───────────────────────────────────
def pick(lst, n):
    return random.sample(lst, min(n, len(lst)))


# ── Build one example ─────────────────────────────────────────────────────────
def build_example(domain: str, variant_index: int) -> dict:
    kpis = KPI_BY_DOMAIN[domain]
    goals = USER_GOALS_BY_DOMAIN[domain]
    fields = DATA_FIELDS_BY_DOMAIN[domain]

    # Rotate selections deterministically per variant so variants differ
    rng = random.Random(hash(domain) + variant_index * 31)

    selected_kpis = rng.sample(kpis, k=min(3, len(kpis)))
    selected_goals = rng.sample(goals, k=min(3, len(goals)))
    selected_fields = rng.sample(fields, k=min(5, len(fields)))

    # Build input brief
    input_text = (
        f"Domain: {domain}\n"
        f"Available data fields: {', '.join(selected_fields)}\n"
        f"Key metrics to track: {', '.join(selected_kpis)}\n"
        f"User goals: {', '.join(selected_goals)}"
    )

    # Build KPI-task-chart mapping (one entry per selected KPI)
    task_chart_pairs = rng.sample(TASK_TO_CHART, k=min(len(selected_kpis), len(TASK_TO_CHART)))
    kpi_task_chart = []
    for i, kpi in enumerate(selected_kpis):
        user_task, chart = task_chart_pairs[i % len(task_chart_pairs)]
        rationale = rng.choice(RATIONALE_SNIPPETS)
        kpi_task_chart.append({
            "kpi": kpi,
            "user_task": user_task,
            "recommended_chart": chart,
            "rationale": rationale,
        })

    # Layout, colors, interactions, rationales
    layout = rng.choice(LAYOUT_OPTIONS)
    colors = rng.choice(COLOR_OPTIONS)
    interactions = rng.sample(INTERACTION_OPTIONS, k=3)
    design_rationales = rng.sample(RATIONALE_SNIPPETS, k=2)

    context_summary = (
        f"This {domain} dashboard is designed to help users {selected_goals[0]} "
        f"and {selected_goals[1]}. It focuses on {', '.join(selected_kpis)} "
        f"using data from fields such as {', '.join(selected_fields[:3])}."
    )

    output = {
        "context_summary": context_summary,
        "kpi_task_chart_mapping": kpi_task_chart,
        "layout_hierarchy": layout,
        "labels_scales_colors": colors,
        "interactions": interactions,
        "design_rationales": design_rationales,
    }

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": output,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    examples = []
    for domain in DOMAINS:
        for variant in range(8):
            example = build_example(domain, variant)
            examples.append(example)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Collect used chart types
    used_charts = set()
    for ex in examples:
        for mapping in ex["output"]["kpi_task_chart_mapping"]:
            used_charts.add(mapping["recommended_chart"])

    print("\nDataset generation complete.")
    print(f"  Examples generated : {len(examples)}")
    print(f"  Output file        : {os.path.abspath(OUTPUT_FILE)}")
    print(f"  Domains used       : {', '.join(DOMAINS)}")
    print(f"  Chart types used   : {', '.join(sorted(used_charts))}")


if __name__ == "__main__":
    main()
