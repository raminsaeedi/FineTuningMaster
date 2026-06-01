"""Principled synthetic gold-data generator (masterplan schema).

Maps each analytical *task_type* to its *correct* *chart_type* using established
visualization principles (Cleveland & McGill 1984; Few 2006; Munzner 2014), so
training teaches correct chart selection and top-k accuracy is meaningful.

Emits the masterplan Teil-3 shape: a brief with users/goals/kpis/columns, and a
DesignOutput with context_summary, typed kpi_chart_mapping, layout, styling,
interactions and rationales. Chart/task values come from the ChartType/TaskType
vocabularies (so e.g. deviation maps to bar and flow to sankey — waterfall/funnel
are not in the enum). Deterministic per item index.
"""

from __future__ import annotations

import random
from typing import Dict, List

# task_type -> (primary chart_type, [alternative chart_types]) within the enums.
TASK_CHART: Dict[str, tuple] = {
    "trend": ("line", ["area"]),
    "comparison": ("bar", ["grouped_bar", "table"]),
    "composition": ("stacked_bar", ["area"]),
    "part_to_whole": ("donut", ["pie", "treemap"]),
    "distribution": ("histogram", ["box"]),
    "correlation": ("scatter", ["heatmap"]),
    "ranking": ("bar", ["table"]),
    "deviation": ("bar", ["table"]),       # waterfall not in ChartType enum
    "flow": ("sankey", ["table"]),         # funnel not in ChartType enum
}

PRINCIPLE: Dict[str, str] = {
    "trend": "Position along a common time axis is the most accurate encoding for temporal trends.",
    "comparison": "Bar length supports precise magnitude comparison across discrete categories.",
    "composition": "Stacked bars show how a total divides into parts within each category.",
    "part_to_whole": "Angular/area encodings communicate proportions for a small number of categories.",
    "distribution": "Histograms reveal how values spread across ranges.",
    "correlation": "Scatter plots expose the relationship between two quantitative metrics.",
    "ranking": "Sorted bars make rank order and gaps easy to read.",
    "deviation": "Bars baselined at zero make positive/negative deviations explicit.",
    "flow": "Sankey diagrams encode volume of flow between stages by link width.",
}

# KPI keyword -> task_type (chart stays correct regardless).
KEYWORD_TASK = [
    (("funnel", "pipeline", "stage", "onboarding", "acquisition"), "flow"),
    (("by region", "by category", "by segment", "by product", "by channel", "mix", "share", "breakdown"), "part_to_whole"),
    (("distribution", "spread", "range"), "distribution"),
    (("loss", "driver", "deviation", "variance", "change"), "deviation"),
    (("vs", "correlation", "relationship", "impact"), "correlation"),
    (("rate", "ratio", "conversion", "churn", "margin", "score", "utilization"), "comparison"),
    (("revenue", "sales", "traffic", "volume", "demand", "usage", "visits", "users", "load"), "trend"),
]

INDUSTRIES: Dict[str, Dict[str, List[str]]] = {
    "E-Commerce": {"audiences": ["Sales Managers", "Marketing Leads", "Category Managers"],
                   "goals": ["grow online revenue and conversion", "reduce cart abandonment", "optimize product mix"],
                   "kpis": ["Revenue", "Conversion Rate", "Average Order Value", "Cart Abandonment Rate",
                            "Traffic by Channel", "Revenue by Category", "Onboarding Funnel"],
                   "palette": "Blue/Orange accessible palette"},
    "Healthcare": {"audiences": ["Hospital Administrators", "Clinical Leads", "Operations Managers"],
                   "goals": ["improve patient throughput", "reduce readmission rates", "monitor bed utilization"],
                   "kpis": ["Patient Volume", "Average Length of Stay", "Readmission Rate", "Bed Utilization",
                            "Admissions by Department", "Wait Time Distribution", "Cost Drivers"],
                   "palette": "Teal/Grey clinical palette"},
    "Finance & Banking": {"audiences": ["Risk Officers", "Branch Managers", "Portfolio Analysts"],
                          "goals": ["monitor loan performance and risk", "track deposits and fee revenue", "control operational losses"],
                          "kpis": ["Net Interest Margin", "Loan Default Rate", "Customer Deposits", "Fee Revenue",
                                   "Operational Losses", "Revenue by Region", "Credit Score Distribution"],
                          "palette": "Navy/Gold financial palette"},
    "SaaS / Software": {"audiences": ["Product Managers", "Customer Success Leads", "Revenue Operations"],
                        "goals": ["grow MRR and reduce churn", "improve trial-to-paid conversion", "track feature adoption"],
                        "kpis": ["Monthly Recurring Revenue", "Churn Rate", "Trial-to-Paid Conversion", "Expansion Revenue",
                                 "Daily Active Users", "Revenue by Plan", "Onboarding Funnel"],
                        "palette": "Indigo/Cyan product palette"},
    "Manufacturing": {"audiences": ["Plant Managers", "Quality Engineers", "Supply Planners"],
                      "goals": ["maximize throughput and quality", "reduce defect rate", "monitor equipment efficiency"],
                      "kpis": ["Overall Equipment Effectiveness", "Defect Rate", "Output Volume", "Downtime Hours",
                               "Yield by Line", "Scrap Cost Drivers", "Cycle Time Distribution"],
                      "palette": "Steel/Amber industrial palette"},
    "Logistics & Supply Chain": {"audiences": ["Operations Directors", "Fleet Managers", "Warehouse Leads"],
                                 "goals": ["improve on-time delivery", "reduce shipping cost", "balance inventory"],
                                 "kpis": ["On-Time Delivery Rate", "Shipping Cost per Order", "Inventory Turnover", "Order Volume",
                                          "Deliveries by Region", "Warehouse Utilization", "Lead Time Distribution"],
                                 "palette": "Green/Slate logistics palette"},
    "Marketing & Advertising": {"audiences": ["CMOs", "Campaign Managers", "Growth Leads"],
                                "goals": ["improve campaign ROI", "grow qualified leads", "optimize channel spend"],
                                "kpis": ["Return on Ad Spend", "Cost per Lead", "Lead Volume", "Conversion Rate",
                                         "Spend by Channel", "Engagement Rate", "Lead Funnel"],
                                "palette": "Magenta/Blue marketing palette"},
    "HR & People Analytics": {"audiences": ["HR Directors", "Talent Leads", "People Operations"],
                              "goals": ["reduce attrition", "improve time-to-hire", "monitor engagement"],
                              "kpis": ["Attrition Rate", "Time-to-Hire", "Headcount", "Engagement Score",
                                       "Headcount by Department", "Offer Acceptance Rate", "Tenure Distribution"],
                              "palette": "Purple/Green people palette"},
    "Retail": {"audiences": ["Store Operations", "Merchandising Leads", "Regional Directors"],
               "goals": ["grow same-store sales", "optimize inventory", "improve basket size"],
               "kpis": ["Same-Store Sales", "Basket Size", "Stock-Out Rate", "Footfall",
                        "Sales by Region", "Margin by Category", "Sales Distribution"],
               "palette": "Red/Charcoal retail palette"},
    "Energy & Utilities": {"audiences": ["Grid Operators", "Sustainability Leads", "Asset Managers"],
                           "goals": ["balance load and capacity", "reduce outages", "track renewable mix"],
                           "kpis": ["Peak Load", "Outage Duration", "Energy Mix", "Capacity Utilization",
                                    "Consumption by Region", "Renewable Share", "Load Distribution"],
                           "palette": "Green/Blue utility palette"},
}

UPDATE_FREQ = ["real-time", "hourly", "daily", "weekly", "monthly"]
EXPERTISE = ["beginner", "intermediate", "advanced"]


def _task_for_kpi(kpi: str, rng: random.Random) -> str:
    low = kpi.lower()
    for keywords, task in KEYWORD_TASK:
        if any(k in low for k in keywords):
            return task
    return rng.choice(list(TASK_CHART))


def _columns_for(kpis: List[str]) -> List[Dict[str, str]]:
    cols: List[Dict[str, str]] = [{"name": "date", "dtype": "datetime"},
                                  {"name": "segment", "dtype": "categorical"},
                                  {"name": "region", "dtype": "categorical"}]
    for k in kpis:
        cols.append({"name": k.lower().replace(" ", "_"), "dtype": "numeric"})
    return cols


def _mapping_for_kpi(kpi: str, rng: random.Random) -> dict:
    task = _task_for_kpi(kpi, rng)
    primary, alts = TASK_CHART[task]
    return {
        "kpi": kpi,
        "task_type": task,
        "chart_type": primary,
        "alternatives": alts,
        "encoding": {"x": "date" if task == "trend" else "segment", "y": kpi.lower().replace(" ", "_")},
    }


def _build_brief(index: int, rng: random.Random) -> dict:
    industry = list(INDUSTRIES)[index % len(INDUSTRIES)]
    spec = INDUSTRIES[industry]
    audience = rng.choice(spec["audiences"])
    n_goals = rng.randint(1, 2)
    goals = rng.sample(spec["goals"], k=min(n_goals, len(spec["goals"])))
    n_kpis = rng.randint(3, 5)
    kpis = rng.sample(spec["kpis"], k=min(n_kpis, len(spec["kpis"])))
    expertise = rng.choice(EXPERTISE)
    freq = rng.choice(UPDATE_FREQ)
    return {
        "_industry": industry,
        "_expertise": expertise,
        "_frequency": freq,
        "users": f"{audience} in the {industry} sector ({expertise} data literacy)",
        "goals": goals,
        "kpis": kpis,
        "columns": _columns_for(kpis),
        "constraints": f"Data refreshes {freq}; respect WCAG AA accessibility.",
    }


def _build_recommendation(brief: dict, rng: random.Random) -> dict:
    kpis = brief["kpis"]
    mappings = [_mapping_for_kpi(k, rng) for k in kpis]
    industry = brief["_industry"]
    expertise = brief["_expertise"]
    freq = brief["_frequency"]

    rationales = [{
        "claim": f"Use a {m['chart_type']} chart for {m['kpi']}.",
        "principle": PRINCIPLE[m["task_type"]],
    } for m in mappings[:4]]
    rationales.append({
        "claim": "Limit each view to 5-7 elements and group related KPIs.",
        "principle": "Managing cognitive load keeps the dashboard scannable (Few 2006).",
    })

    interactions = (["cross-filtering across charts", "drill-down to detail", "export to PDF/Excel"]
                    if expertise != "beginner" else ["date-range filter", "export to PDF"])

    return {
        "context_summary": {
            "audience": brief["users"],
            "domain": industry,
            "primary_goal": brief["goals"][0] if brief["goals"] else "",
            "data_literacy": expertise,
            "update_frequency": freq,
        },
        "kpi_chart_mapping": mappings,
        "layout": {
            "pattern": "Headline KPI strip on top; supporting charts ordered by importance.",
            "primary_section": f"Headline metrics for {kpis[0]}" + (f" and {kpis[1]}" if len(kpis) > 1 else ""),
            "secondary_section": "Trend and comparison charts for the remaining KPIs",
            "responsive": expertise != "advanced",
        },
        "styling": {
            "color_palette": INDUSTRIES[industry]["palette"],
            "number_format": "percentages to 1 decimal; large values with K/M suffix",
            "accessibility": "maintain >= 4.5:1 contrast; never rely on color alone",
        },
        "interactions": interactions,
        "rationales": rationales,
    }


def generate_dataset(n: int = 600, base_seed: int = 42) -> List[dict]:
    """Generate ``n`` {brief, recommendation} items. Deterministic per index."""
    items: List[dict] = []
    for i in range(n):
        rng = random.Random(base_seed * 100_003 + i)
        brief = _build_brief(i, rng)
        recommendation = _build_recommendation(brief, rng)
        clean_brief = {k: v for k, v in brief.items() if not k.startswith("_")}
        items.append({"brief": clean_brief, "recommendation": recommendation})
    return items
