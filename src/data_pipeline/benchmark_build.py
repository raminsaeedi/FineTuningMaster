"""Build the independent evaluation benchmark `benchmark_v1`.

Combines verified public briefs (`real_public`, strong evidence) with author-drafted
realistic briefs (`realistic_manual`, weaker evidence). Labels are assigned with a
lineage **disjoint from the synthetic generator**:
  * `task_type` via the independent crosswalk `data/eval/task_crosswalk.yaml`;
  * `acceptable_chart_types` (set) from the independent L1 literature table where the
    task is covered, else a documented expert set — never `TASK_CHART`.

See `docs/evaluation/benchmark_dataset_construction.md`. Output is EVALUATION-ONLY
(benchmark lock).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from src.evaluation.l1_independent import DEFAULT_L1_CSV, load_effective_sets
from src.utils.io import read_jsonl, read_yaml

CROSSWALK_PATH = "data/eval/task_crosswalk.yaml"
REAL_BRIEFS_PATH = "data/eval/real_briefs/items.jsonl"

# Provenance for the verified public briefs (from docs/datasets/real_briefs_provenance.md).
REAL_PUBLIC_META: Dict[str, Dict[str, str]] = {
    "rb_001": {"domain": "Retail", "source_name": "Power BI: Retail Analysis Sample", "usage": "Microsoft sample dataset (documented)"},
    "rb_002": {"domain": "Supply Chain", "source_name": "Power BI: Supplier Quality Analysis Sample", "usage": "Microsoft sample dataset (documented)"},
    "rb_003": {"domain": "HR", "source_name": "Power BI: Human Resources Sample", "usage": "Microsoft sample dataset (documented)"},
    "rb_004": {"domain": "IT Finance", "source_name": "Power BI: IT Spend Analysis Sample", "usage": "Microsoft sample dataset (documented)"},
    "rb_005": {"domain": "Finance", "source_name": "Power BI: Customer Profitability Sample", "usage": "Microsoft sample dataset (documented)"},
    "rb_006": {"domain": "Procurement", "source_name": "Power BI: Procurement Analysis Sample", "usage": "Microsoft sample dataset (documented)"},
    "rb_007": {"domain": "Marketing", "source_name": "Power BI: Sales & Marketing (VanArsdel) Sample", "usage": "Microsoft sample dataset (documented)"},
    "rb_008": {"domain": "Finance", "source_name": "Power BI: Financial Sample", "usage": "Microsoft sample dataset (documented)"},
    "rb_009": {"domain": "Retail", "source_name": "Tableau: Sample - Superstore", "usage": "Tableau sample (cite-and-ask)"},
    "rb_010": {"domain": "IT Ops", "source_name": "Grafana Play", "usage": "Grafana Play snapshot (cite-and-ask)"},
}

# Cosmetic domain-label normalization: merge variant spellings to one canonical
# label (from the shared INDUSTRIES vocabulary). Only collapses true duplicates;
# does NOT touch task_type or chart labels.
DOMAIN_CANON: Dict[str, str] = {
    "HR": "HR & People Analytics",
    "Supply Chain": "Logistics & Supply Chain",
}


def _canon_domain(domain: str) -> str:
    return DOMAIN_CANON.get(domain, domain)


# Author-drafted realistic briefs. Deliberately span covered and L1-uncovered task
# types so the benchmark exercises coverage honestly. No external source.
REALISTIC_SEEDS: List[dict] = [
    {"domain": "SaaS / Software", "users": "Growth team tracking self-serve signups",
     "goals": ["Analyze the signup-to-paid conversion funnel across stages"],
     "kpis": ["Signups", "Activated Users", "Trials Started", "Paid Conversions"]},
    {"domain": "E-Commerce", "users": "Merchandising leads reviewing category mix",
     "goals": ["Understand the share of revenue contributed by each category"],
     "kpis": ["Revenue", "Revenue Share by Category", "Units Sold"]},
    {"domain": "Finance", "users": "FP&A analysts reviewing cost structure",
     "goals": ["See the breakdown by department composed of cost components"],
     "kpis": ["Total Cost", "Cost by Department", "Cost by Type"]},
    {"domain": "Marketing", "users": "Performance marketers optimizing spend",
     "goals": ["Assess the relationship between ad spend and conversions"],
     "kpis": ["Ad Spend", "Conversions", "Cost per Acquisition"]},
    {"domain": "Logistics & Supply Chain", "users": "Ops analysts monitoring delivery times",
     "goals": ["Examine the distribution of delivery lead times"],
     "kpis": ["Lead Time", "Orders", "On-Time Rate"]},
    {"domain": "Healthcare", "users": "Operations managers tracking budget adherence",
     "goals": ["Track variance of actual spend versus plan by unit"],
     "kpis": ["Plan Spend", "Actual Spend", "Variance to Plan"]},
    {"domain": "Retail", "users": "Category managers ranking product performance",
     "goals": ["Identify the best and worst performing products"],
     "kpis": ["Sales", "Margin", "Units Sold"]},
    {"domain": "SaaS / Software", "users": "Revenue operations tracking recurring revenue",
     "goals": ["Track monthly recurring revenue trend over time"],
     "kpis": ["MRR", "New MRR", "Churned MRR"]},
    {"domain": "Manufacturing", "users": "Plant managers comparing lines",
     "goals": ["Compare output this quarter against the prior quarter"],
     "kpis": ["Output Volume", "Defect Rate", "Downtime Hours"]},
    {"domain": "Energy & Utilities", "users": "Grid operators analyzing consumption",
     "goals": ["Examine the distribution of hourly load across the day"],
     "kpis": ["Hourly Load", "Peak Load", "Capacity Utilization"]},
    {"domain": "HR & People Analytics", "users": "HR leaders reviewing workforce mix",
     "goals": ["Understand the share of headcount by department"],
     "kpis": ["Headcount", "Headcount Share by Department", "Attrition Rate"]},
    {"domain": "Marketing", "users": "Growth leads reviewing the acquisition funnel",
     "goals": ["Analyze the lead funnel from awareness to closed deals"],
     "kpis": ["Leads", "Qualified Leads", "Opportunities", "Closed Deals"]},
    {"domain": "Finance", "users": "Treasury analysts monitoring cash trend",
     "goals": ["Track the cash balance trend over the fiscal year"],
     "kpis": ["Cash Balance", "Inflows", "Outflows"]},
    {"domain": "E-Commerce", "users": "Analysts studying pricing effects",
     "goals": ["Assess the relationship between discount depth and units sold"],
     "kpis": ["Discount Depth", "Units Sold", "Revenue"]},
    {"domain": "Logistics & Supply Chain", "users": "Planners ranking carrier performance",
     "goals": ["Rank carriers by on-time delivery performance"],
     "kpis": ["On-Time Rate", "Shipments", "Cost per Shipment"]},
    {"domain": "Healthcare", "users": "Clinic administrators comparing departments",
     "goals": ["Compare patient volume against the same period last year"],
     "kpis": ["Patient Volume", "Average Wait Time", "Readmission Rate"]},
    {"domain": "SaaS / Software", "users": "Product managers reviewing usage composition",
     "goals": ["See the breakdown of active usage composed of feature areas"],
     "kpis": ["Active Users", "Usage by Feature", "Sessions"]},
    {"domain": "Manufacturing", "users": "Quality engineers monitoring defect variance",
     "goals": ["Track deviation of defect counts from the control baseline"],
     "kpis": ["Defect Count", "Baseline Defects", "Variance to Baseline"]},
    {"domain": "Retail", "users": "Store analysts studying basket relationships",
     "goals": ["Examine the relationship between footfall and basket size"],
     "kpis": ["Footfall", "Basket Size", "Conversion Rate"]},
    {"domain": "Energy & Utilities", "users": "Sustainability leads tracking renewable mix",
     "goals": ["Understand the share of generation from each energy source"],
     "kpis": ["Total Generation", "Generation Share by Source", "Emissions"]},
]


def _load_crosswalk(root: Path) -> dict:
    return read_yaml(root / CROSSWALK_PATH)


def assign_task_type(text: str, crosswalk: dict) -> str:
    """Assign a TaskType from analytical-intent text via the independent crosswalk."""
    low = f" {text.lower()} "
    for rule in crosswalk.get("intent_to_task", []):
        for kw in rule.get("keywords", []):
            if kw.lower() in low:
                return rule["task_type"]
    return crosswalk.get("default_task", "comparison")


def _columns(kpis: List[str]) -> List[Dict[str, str]]:
    cols = [
        {"name": "date", "dtype": "datetime"},
        {"name": "category", "dtype": "categorical"},
        {"name": "region", "dtype": "categorical"},
    ]
    for k in kpis:
        name = re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")
        cols.append({"name": name, "dtype": "numeric"})
    return cols


def _acceptable(task: str, effective_sets: Dict[str, set], crosswalk: dict):
    """Return (sorted acceptable chart list, label_source, auto_scorable)."""
    if task in effective_sets:
        return sorted(effective_sets[task]), "literature_L1", True
    expert = crosswalk.get("expert_acceptable_charts", {}).get(task, [])
    return list(expert), "manual_expert", False


def _confidence(source_type: str, label_source: str) -> str:
    if source_type == "real_public" and label_source == "literature_L1":
        return "high"
    if label_source == "manual_expert":
        return "low"
    return "medium"


def _make_item(bid: str, domain: str, users: str, goals: List[str], kpis: List[str],
               columns: List[dict], constraints, source_type: str, source_name: str,
               source_reference: str, usage: str, effective_sets, crosswalk) -> dict:
    task = assign_task_type(" ".join(goals + kpis), crosswalk)
    acceptable, label_source, auto = _acceptable(task, effective_sets, crosswalk)
    return {
        "benchmark_id": bid,
        "domain": _canon_domain(domain),
        "users": users,
        "goals": goals,
        "kpis": kpis,
        "columns": columns,
        "constraints": constraints,
        "task_type": task,
        "acceptable_chart_types": acceptable,
        "rationale": f"Primary analytical task '{task}' assigned from documented intent; "
                     f"acceptable charts from {label_source}.",
        "source_name": source_name,
        "source_type": source_type,
        "source_reference": source_reference,
        "license_or_usage_note": usage,
        "label_source": label_source,
        "label_confidence": _confidence(source_type, label_source),
        "suitable_for_auto_scoring": auto,
        "suitable_for_human_eval": True,
        "notes": "evaluation-only (benchmark lock): never use for training/tuning/selection",
    }


def build_items(project_root: str | Path) -> List[dict]:
    root = Path(project_root)
    crosswalk = _load_crosswalk(root)
    effective_sets = load_effective_sets(root / DEFAULT_L1_CSV)
    items: List[dict] = []
    idx = 1

    # 1) Verified public briefs — strong evidence.
    real_path = root / REAL_BRIEFS_PATH
    if real_path.exists():
        for rec in read_jsonl(real_path):
            pid = (rec.get("extra") or {}).get("provenance_id") or rec.get("item_id", "")
            meta = REAL_PUBLIC_META.get(pid)
            if not meta:
                continue
            items.append(_make_item(
                f"bm_v1_{idx:03d}", meta["domain"], rec.get("users", ""),
                list(rec.get("goals", [])), list(rec.get("kpis", [])),
                list(rec.get("columns", [])), rec.get("constraints"),
                "real_public", meta["source_name"],
                f"{meta['source_name']} (provenance: docs/datasets/real_briefs_provenance.md#{pid})",
                meta["usage"], effective_sets, crosswalk,
            ))
            idx += 1

    # 2) Author-drafted realistic briefs — weaker evidence.
    for seed in REALISTIC_SEEDS:
        items.append(_make_item(
            f"bm_v1_{idx:03d}", seed["domain"], seed["users"],
            list(seed["goals"]), list(seed["kpis"]),
            _columns(seed["kpis"]), seed.get("constraints"),
            "realistic_manual", "author-drafted realistic scenario",
            "author-drafted realistic scenario (no external source)",
            "author-drafted; free to use for evaluation only", effective_sets, crosswalk,
        ))
        idx += 1

    return items
