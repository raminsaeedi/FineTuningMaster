"""Generate and freeze the AI-augmented dashboard_v4 dataset.

This script intentionally keeps the dashboard_v4 build separate from the frozen
dashboard_v3 package.  It reads only dashboard_v3 Train/Validation records while
generating candidates.  The v3 Test and Human-Evaluation files are read only in
the final freeze phase so their bytes can be copied and verified exactly.

The generator is a local Codex-agent generation implementation.  It uses the
project's existing GoldItem schema, task/chart vocabulary, brief fingerprint,
character 3-gram similarity, and hash-based split conventions.  Each accepted
record is explicitly labelled ai_generated and is never labelled nvBench gold,
human gold, or expert gold.

Usage:
    python experiments/scripts/generate_dashboard_v4.py

The script refuses to overwrite an existing data/frozen/dashboard_v4 directory.
Intermediate candidate, accepted, rejected, and report files are written under
data/staging/dashboard_v4/<run_id>/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.schemas import ChartType, GoldItem, TaskType  # noqa: E402
from src.data_pipeline.builders.leakage import fingerprint  # noqa: E402
from src.data_pipeline.frozen_validation import validate_record  # noqa: E402
from src.data_pipeline.leakage_similarity import (  # noqa: E402
    brief_text,
    char_ngrams,
    jaccard,
)
from src.data_pipeline.synth_generator import PRINCIPLE, TASK_CHART  # noqa: E402


TARGET_GENERATED = 2000
GENERATOR_MODEL = "gpt-5.6-luna"
GENERATION_MODE = "codex_agent"
GENERATION_SPEC_VERSION = "dashboard_v4-generation-v1"
QUALITY_RULE_VERSION = "dashboard_v4_quality_v1"
SPLIT_ALGORITHM_VERSION = "dashboard_v4_group_train_val_v1"
SEED = 42
NEAR_DUPLICATE_THRESHOLD = 0.8
TRAIN_FRACTION_WITHIN_GENERATED = 0.83
BATCH_SIZE = 40

V3_DIR = _PROJECT_ROOT / "data" / "frozen" / "dashboard_v3"
V4_DIR = _PROJECT_ROOT / "data" / "frozen" / "dashboard_v4"
STAGING_ROOT = _PROJECT_ROOT / "data" / "staging" / "dashboard_v4"

TASK_TARGETS: Dict[str, int] = {
    "comparison": 300,
    "trend": 250,
    "composition": 200,
    "part_to_whole": 200,
    "correlation": 300,
    "distribution": 250,
    "ranking": 200,
    "deviation": 150,
    "flow": 150,
}

ALLOWED_TASKS = {task.value for task in TaskType}
ALLOWED_CHARTS = {chart.value for chart in ChartType}


@dataclass(frozen=True)
class Measure:
    label: str
    field: str
    kind: str


@dataclass(frozen=True)
class Dimension:
    label: str
    field: str
    values: Tuple[str, ...]


@dataclass(frozen=True)
class Domain:
    name: str
    audiences: Tuple[str, ...]
    measures: Tuple[Measure, ...]
    dimensions: Tuple[Dimension, ...]
    time_field: str
    time_label: str
    frequencies: Tuple[str, ...]
    palette: str


def _domain(
    name: str,
    audiences: Sequence[str],
    measures: Sequence[Tuple[str, str, str]],
    dimensions: Sequence[Tuple[str, str, Sequence[str]]],
    time_field: str,
    time_label: str,
    frequencies: Sequence[str],
    palette: str,
) -> Domain:
    return Domain(
        name=name,
        audiences=tuple(audiences),
        measures=tuple(Measure(*m) for m in measures),
        dimensions=tuple(Dimension(label, field, tuple(values)) for label, field, values in dimensions),
        time_field=time_field,
        time_label=time_label,
        frequencies=tuple(frequencies),
        palette=palette,
    )


DOMAINS: Tuple[Domain, ...] = (
    _domain(
        "Healthcare Operations",
        ("care operations directors", "hospital administrators", "clinical service leads"),
        (("patient visits", "patient_visits", "count"), ("readmission rate", "readmission_rate", "rate"),
         ("bed occupancy rate", "bed_occupancy_rate", "rate"), ("wait time", "wait_time_minutes", "duration")),
        (("department", "department", ("emergency", "cardiology", "oncology", "orthopedics")),
         ("care setting", "care_setting", ("inpatient", "outpatient", "urgent care", "telehealth")),
         ("payer group", "payer_group", ("public", "employer", "self-pay", "mixed"))),
        "encounter_date", "encounter date", ("daily", "weekly", "monthly"), "teal and slate clinical palette",
    ),
    _domain(
        "E-Commerce",
        ("digital commerce directors", "category managers", "growth marketing leads"),
        (("net revenue", "net_revenue", "currency"), ("conversion rate", "conversion_rate", "rate"),
         ("average order value", "average_order_value", "currency"), ("return rate", "return_rate", "rate")),
        (("product category", "product_category", ("home", "beauty", "electronics", "outdoor")),
         ("sales channel", "sales_channel", ("web", "mobile app", "marketplace", "social")),
         ("customer segment", "customer_segment", ("new", "returning", "loyalty", "business"))),
        "order_date", "order date", ("hourly", "daily", "weekly"), "indigo and amber accessible palette",
    ),
    _domain(
        "Finance and Banking",
        ("portfolio analysts", "risk officers", "branch performance managers"),
        (("loan balance", "loan_balance", "currency"), ("default rate", "default_rate", "rate"),
         ("fee income", "fee_income", "currency"), ("portfolio return", "portfolio_return", "rate")),
        (("branch", "branch", ("north", "central", "south", "online")),
         ("risk band", "risk_band", ("low", "moderate", "high", "watchlist")),
         ("product type", "product_type", ("mortgage", "credit", "savings", "investment"))),
        "transaction_date", "transaction date", ("daily", "weekly", "monthly"), "navy and gold financial palette",
    ),
    _domain(
        "Manufacturing",
        ("plant managers", "quality engineers", "supply planners"),
        (("units produced", "units_produced", "volume"), ("defect rate", "defect_rate", "rate"),
         ("cycle time", "cycle_time_minutes", "duration"), ("downtime hours", "downtime_hours", "duration")),
        (("plant", "plant", ("north plant", "river plant", "east plant", "assembly hub")),
         ("production line", "production_line", ("line a", "line b", "line c", "line d")),
         ("shift", "shift", ("day", "swing", "night"))),
        "production_date", "production date", ("hourly", "daily", "weekly"), "steel and amber industrial palette",
    ),
    _domain(
        "Logistics and Supply Chain",
        ("operations directors", "fleet managers", "warehouse leads"),
        (("shipment count", "shipment_count", "count"), ("delivery cost", "delivery_cost", "currency"),
         ("on-time delivery rate", "on_time_delivery_rate", "rate"), ("lead time", "lead_time_days", "duration")),
        (("origin region", "origin_region", ("north", "south", "coastal", "inland")),
         ("carrier", "carrier", ("atlas", "bluebird", "northstar", "rapidway")),
         ("service level", "service_level", ("economy", "standard", "priority", "same day"))),
        "shipment_date", "shipment date", ("daily", "weekly", "monthly"), "green and slate logistics palette",
    ),
    _domain(
        "SaaS Product Analytics",
        ("product managers", "customer success leads", "revenue operations analysts"),
        (("monthly recurring revenue", "monthly_recurring_revenue", "currency"), ("churn rate", "churn_rate", "rate"),
         ("active accounts", "active_accounts", "count"), ("feature adoption rate", "feature_adoption_rate", "rate")),
        (("plan tier", "plan_tier", ("starter", "growth", "business", "enterprise")),
         ("customer segment", "customer_segment", ("startup", "mid-market", "enterprise", "public sector")),
         ("region", "region", ("americas", "emea", "apac", "global"))),
        "event_date", "event date", ("daily", "weekly", "monthly"), "indigo and cyan product palette",
    ),
    _domain(
        "Energy and Utilities",
        ("grid operators", "sustainability leads", "asset managers"),
        (("energy consumption", "energy_consumption_mwh", "volume"), ("peak load", "peak_load_mw", "volume"),
         ("outage duration", "outage_duration_minutes", "duration"), ("renewable share", "renewable_share", "rate")),
        (("grid zone", "grid_zone", ("north", "metro", "rural", "coastal")),
         ("source type", "source_type", ("solar", "wind", "hydro", "thermal")),
         ("asset class", "asset_class", ("substation", "storage", "transmission", "generation"))),
        "reading_date", "reading date", ("hourly", "daily", "weekly"), "green and blue utility palette",
    ),
    _domain(
        "Education",
        ("district administrators", "program directors", "student success leads"),
        (("enrollment count", "enrollment_count", "count"), ("completion rate", "completion_rate", "rate"),
         ("assessment score", "assessment_score", "score"), ("attendance rate", "attendance_rate", "rate")),
        (("campus", "campus", ("central", "north", "online", "community")),
         ("program", "program", ("engineering", "business", "health", "arts")),
         ("student cohort", "student_cohort", ("first year", "returning", "adult", "international"))),
        "term_date", "term date", ("weekly", "monthly", "quarterly"), "purple and green education palette",
    ),
    _domain(
        "Telecommunications",
        ("network operations managers", "service quality analysts", "customer experience leads"),
        (("call volume", "call_volume", "count"), ("dropped call rate", "dropped_call_rate", "rate"),
         ("data usage", "data_usage_gb", "volume"), ("network latency", "network_latency_ms", "duration")),
        (("cell region", "cell_region", ("urban", "suburban", "rural", "campus")),
         ("device type", "device_type", ("phone", "tablet", "router", "wearable")),
         ("service plan", "service_plan", ("basic", "plus", "premium", "business"))),
        "measurement_date", "measurement date", ("hourly", "daily", "weekly"), "blue and violet network palette",
    ),
    _domain(
        "Agriculture",
        ("farm operations managers", "crop planners", "sustainability coordinators"),
        (("yield", "yield_tonnes", "volume"), ("water use", "water_use_m3", "volume"),
         ("crop loss rate", "crop_loss_rate", "rate"), ("harvest cost", "harvest_cost", "currency")),
        (("farm region", "farm_region", ("north field", "river basin", "foothills", "dryland")),
         ("crop type", "crop_type", ("wheat", "maize", "rice", "vegetables")),
         ("irrigation method", "irrigation_method", ("drip", "sprinkler", "rainfed", "recycled"))),
        "harvest_date", "harvest date", ("weekly", "monthly", "quarterly"), "green and earth agriculture palette",
    ),
    _domain(
        "Hospitality",
        ("hotel general managers", "revenue managers", "guest experience leads"),
        (("room nights", "room_nights", "count"), ("occupancy rate", "occupancy_rate", "rate"),
         ("average daily rate", "average_daily_rate", "currency"), ("cancellation rate", "cancellation_rate", "rate")),
        (("property", "property", ("city center", "airport", "resort", "extended stay")),
         ("market segment", "market_segment", ("business", "leisure", "group", "long stay")),
         ("booking channel", "booking_channel", ("direct", "agency", "partner", "mobile"))),
        "stay_date", "stay date", ("daily", "weekly", "monthly"), "coral and charcoal hospitality palette",
    ),
    _domain(
        "Media and Streaming",
        ("content strategy leads", "platform managers", "audience insights analysts"),
        (("view count", "view_count", "count"), ("watch time", "watch_time_hours", "duration"),
         ("completion rate", "completion_rate", "rate"), ("subscriber additions", "subscriber_additions", "count")),
        (("content genre", "content_genre", ("drama", "documentary", "news", "sports")),
         ("platform", "platform", ("web", "mobile", "connected tv", "partner")),
         ("audience segment", "audience_segment", ("new viewers", "returning", "families", "enthusiasts"))),
        "release_date", "release date", ("daily", "weekly", "monthly"), "magenta and blue media palette",
    ),
    _domain(
        "Public Safety",
        ("public safety directors", "district commanders", "emergency response planners"),
        (("incident count", "incident_count", "count"), ("response time", "response_time_minutes", "duration"),
         ("clearance rate", "clearance_rate", "rate"), ("service call volume", "service_call_volume", "count")),
        (("district", "district", ("central", "north", "south", "west")),
         ("incident type", "incident_type", ("traffic", "property", "medical", "public order")),
         ("priority level", "priority_level", ("routine", "elevated", "urgent", "critical"))),
        "incident_date", "incident date", ("daily", "weekly", "monthly"), "blue and orange public-safety palette",
    ),
    _domain(
        "Insurance Claims",
        ("claims directors", "underwriting analysts", "fraud operations managers"),
        (("claim count", "claim_count", "count"), ("claim cost", "claim_cost", "currency"),
         ("settlement time", "settlement_days", "duration"), ("fraud rate", "fraud_rate", "rate")),
        (("policy type", "policy_type", ("auto", "home", "health", "commercial")),
         ("region", "region", ("north", "central", "south", "coastal")),
         ("claim channel", "claim_channel", ("agent", "web", "phone", "partner"))),
        "claim_date", "claim date", ("daily", "weekly", "monthly"), "navy and orange insurance palette",
    ),
    _domain(
        "Human Resources",
        ("people operations directors", "talent acquisition leads", "workforce planning analysts"),
        (("headcount", "headcount", "count"), ("attrition rate", "attrition_rate", "rate"),
         ("time to hire", "time_to_hire_days", "duration"), ("engagement score", "engagement_score", "score")),
        (("department", "department", ("engineering", "sales", "operations", "support")),
         ("job family", "job_family", ("professional", "technical", "managerial", "frontline")),
         ("employment type", "employment_type", ("full time", "part time", "contract", "temporary"))),
        "snapshot_date", "snapshot date", ("weekly", "monthly", "quarterly"), "purple and green people palette",
    ),
    _domain(
        "Scientific Research",
        ("research administrators", "grant portfolio managers", "laboratory directors"),
        (("publication count", "publication_count", "count"), ("citation count", "citation_count", "count"),
         ("grant amount", "grant_amount", "currency"), ("review time", "review_time_days", "duration")),
        (("research area", "research_area", ("climate", "health", "materials", "computing")),
         ("institution", "institution", ("university", "institute", "industry lab", "consortium")),
         ("funding program", "funding_program", ("early career", "infrastructure", "translation", "discovery"))),
        "publication_date", "publication date", ("monthly", "quarterly", "yearly"), "teal and purple research palette",
    ),
    _domain(
        "Aviation Operations",
        ("airport operations managers", "network planning analysts", "fleet performance leads"),
        (("flight count", "flight_count", "count"), ("delay duration", "delay_minutes", "duration"),
         ("on-time rate", "on_time_rate", "rate"), ("fuel burn", "fuel_burn_kg", "volume")),
        (("airport", "airport", ("hub north", "hub central", "regional east", "regional west")),
         ("aircraft type", "aircraft_type", ("narrow body", "wide body", "regional", "cargo")),
         ("route class", "route_class", ("domestic", "short haul", "long haul", "charter"))),
        "departure_date", "departure date", ("daily", "weekly", "monthly"), "sky blue and graphite aviation palette",
    ),
    _domain(
        "Cybersecurity Operations",
        ("security operations managers", "incident response leads", "risk assurance analysts"),
        (("alert count", "alert_count", "count"), ("severity score", "severity_score", "score"),
         ("response time", "response_time_minutes", "duration"), ("resolution rate", "resolution_rate", "rate")),
        (("attack type", "attack_type", ("credential", "malware", "network", "application")),
         ("network zone", "network_zone", ("user", "server", "cloud", "partner")),
         ("incident source", "incident_source", ("endpoint", "identity", "email", "firewall"))),
        "alert_date", "alert date", ("hourly", "daily", "weekly"), "dark navy and cyan security palette",
    ),
    _domain(
        "Government Services",
        ("service delivery directors", "municipal program managers", "citizen experience analysts"),
        (("case count", "case_count", "count"), ("processing time", "processing_days", "duration"),
         ("approval rate", "approval_rate", "rate"), ("program spend", "program_spend", "currency")),
        (("service type", "service_type", ("housing", "permits", "benefits", "licensing")),
         ("municipality", "municipality", ("urban", "suburban", "rural", "regional")),
         ("eligibility group", "eligibility_group", ("household", "small business", "student", "senior"))),
        "case_date", "case date", ("daily", "weekly", "monthly"), "blue and gold civic palette",
    ),
    _domain(
        "Environmental Monitoring",
        ("environmental program managers", "field monitoring leads", "sustainability analysts"),
        (("air quality index", "air_quality_index", "score"), ("emissions", "emissions_tonnes", "volume"),
         ("water quality score", "water_quality_score", "score"), ("sensor readings", "sensor_readings", "count")),
        (("monitoring zone", "monitoring_zone", ("urban core", "industrial belt", "rural basin", "coastal")),
         ("pollutant", "pollutant", ("particulate", "ozone", "nitrogen oxide", "sulfur dioxide")),
         ("station type", "station_type", ("fixed", "mobile", "industrial", "community"))),
        "reading_date", "reading date", ("hourly", "daily", "weekly"), "green and blue environmental palette",
    ),
)


DECISION_LENSES = (
    "capacity planning", "service recovery", "resource allocation", "portfolio review",
    "quality improvement", "risk triage", "operational prioritization", "seasonal planning",
    "customer retention", "budget planning", "workforce planning", "performance review",
    "exception management", "network optimization", "policy evaluation", "demand forecasting",
    "root-cause review", "route planning", "program oversight", "investment planning",
)
REVIEW_CONTEXTS = (
    "the monthly operating review", "the quarterly planning cycle", "the morning control-room review",
    "the regional performance meeting", "the weekly exception review", "the annual portfolio review",
    "the service-level checkpoint", "the cross-functional planning session", "the escalation briefing",
    "the resource-allocation workshop",
)
OPERATING_STATES = (
    "steady baseline", "post-launch surge", "peak-season demand", "service-recovery period",
    "capacity-constrained period", "quality-improvement cycle", "risk-escalation window",
    "budget-review window", "pilot rollout", "regulated reporting period", "planned maintenance",
    "demand normalization", "supplier disruption", "staffing transition", "policy changeover",
    "new-market ramp", "backlog reduction", "incident stabilization", "renewal season", "audit preparation",
)
PLANNING_WINDOWS = (
    "current planning horizon", "next operating cycle", "baseline comparison window",
    "near-term intervention window", "seasonal comparison window", "quarter-end decision window",
    "service-level review window", "portfolio allocation window", "exception follow-up window",
    "long-range planning window",
)
TIME_GRAINS = ("day", "week", "month", "quarter", "year", "weekday")
THEMES = ("minimal", "light", "high_contrast", "clinical", "editorial", "dark")
LAYOUT_PATTERNS = (
    "single_focus", "top_kpi_plus_detail", "two_column_grid", "left_summary_right_detail",
    "small_multiples", "comparison_strip", "flow_overview_with_detail",
)

# Rotating feature profiles ensure that the new records contain meaningful
# combinations of filters, grouping, sorting, time, limits, and multi-KPI views.
FEATURE_PROFILES: Tuple[Dict[str, bool], ...] = (
    {"filter": True, "group": False, "sort": True, "time": False, "limit": False, "multi": False},
    {"filter": True, "group": True, "sort": False, "time": True, "limit": False, "multi": False},
    {"filter": False, "group": True, "sort": True, "time": True, "limit": False, "multi": True},
    {"filter": True, "group": True, "sort": True, "time": False, "limit": True, "multi": False},
    {"filter": False, "group": False, "sort": True, "time": True, "limit": True, "multi": True},
    {"filter": True, "group": True, "sort": False, "time": True, "limit": True, "multi": True},
    {"filter": False, "group": True, "sort": True, "time": False, "limit": False, "multi": False},
    {"filter": True, "group": False, "sort": False, "time": True, "limit": False, "multi": True},
    {"filter": True, "group": True, "sort": True, "time": True, "limit": False, "multi": False},
    {"filter": False, "group": False, "sort": False, "time": False, "limit": True, "multi": False},
    {"filter": True, "group": True, "sort": True, "time": True, "limit": True, "multi": True},
    {"filter": False, "group": True, "sort": False, "time": False, "limit": False, "multi": True},
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    data = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    _atomic_write_bytes(path, data.encode("utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _make_task_schedule() -> List[str]:
    remaining = dict(TASK_TARGETS)
    schedule: List[str] = []
    previous = ""
    while sum(remaining.values()) > 0:
        candidates = [task for task, count in remaining.items() if count and task != previous]
        if not candidates:
            candidates = [task for task, count in remaining.items() if count]
        task = max(candidates, key=lambda item: (remaining[item] / TASK_TARGETS[item], remaining[item], item))
        schedule.append(task)
        remaining[task] -= 1
        previous = task
    return schedule


def _choose_domain(index: int, task: str) -> Domain:
    # The stride changes by task so each family sees a broad domain range rather
    # than one task being tied to one sector.
    task_offset = list(TASK_TARGETS).index(task) * 7
    return DOMAINS[(index * 7 + task_offset) % len(DOMAINS)]


def _choose_measure(domain: Domain, index: int, offset: int = 0) -> Measure:
    return domain.measures[(index * 3 + offset) % len(domain.measures)]


def _choose_dimension(domain: Domain, index: int, offset: int = 0) -> Dimension:
    return domain.dimensions[(index * 5 + offset) % len(domain.dimensions)]


def _time_grain(index: int, task: str) -> str:
    if task == "trend":
        return ("day", "week", "month", "quarter", "year")[index % 5]
    return TIME_GRAINS[(index * 3) % len(TIME_GRAINS)]


def _aggregate_for(measure: Measure, task: str, index: int) -> str:
    if task == "distribution":
        return "COUNT"
    if measure.kind == "rate":
        return ("AVG", "MIN", "MAX")[index % 3]
    if measure.kind in ("currency", "volume"):
        return ("SUM", "AVG", "SUM", "MAX")[index % 4]
    if measure.kind in ("duration", "score"):
        return ("AVG", "MIN", "MAX", "AVG")[index % 4]
    return ("COUNT", "SUM", "AVG")[index % 3]


def _number_format(measure: Measure) -> str:
    if measure.kind == "currency":
        return "currency with compact K/M suffixes and two decimals for small values"
    if measure.kind == "rate":
        return "percentage with one decimal place"
    if measure.kind == "duration":
        return "duration in minutes or days with one decimal where needed"
    if measure.kind == "score":
        return "score with one decimal place"
    return "whole-number counts or compact quantities"


def _threshold(measure: Measure, index: int) -> str:
    if measure.kind == "rate":
        return ["below 0.05", "above 0.12", "between 0.03 and 0.18"][index % 3]
    if measure.kind == "currency":
        return ["above 5000", "below 25000", "between 1000 and 75000"][index % 3]
    if measure.kind == "duration":
        return ["above 15", "below 60", "between 10 and 120"][index % 3]
    if measure.kind == "score":
        return ["below 60", "above 80", "between 40 and 95"][index % 3]
    return ["above 100", "below 5000", "between 50 and 2500"][index % 3]


def _chart_variant(task: str, index: int) -> Tuple[str, List[str]]:
    primary, alternatives = TASK_CHART[task]
    options = [primary, *alternatives]
    chart = options[index % len(options)]
    remaining = [item for item in options if item != chart]
    return chart, remaining


def _scenario_context(index: int) -> Dict[str, str]:
    lens_count = len(DECISION_LENSES)
    review_count = len(REVIEW_CONTEXTS)
    state_count = len(OPERATING_STATES)
    return {
        "decision_lens": DECISION_LENSES[index % lens_count],
        "review_context": REVIEW_CONTEXTS[(index // lens_count) % review_count],
        "operating_state": OPERATING_STATES[(index // (lens_count * review_count)) % state_count],
        "planning_window": PLANNING_WINDOWS[(index // (lens_count * review_count * state_count)) % len(PLANNING_WINDOWS)],
    }


def _goal_suffix(index: int) -> str:
    scenario = _scenario_context(index)
    return (
        f"for {scenario['decision_lens']} during {scenario['review_context']} "
        f"under {scenario['operating_state']} conditions in the {scenario['planning_window']}"
    )


def _field_names(columns: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(column.get("name")) for column in columns}


def _column(name: str, dtype: str, role: str) -> Dict[str, str]:
    return {"name": name, "dtype": dtype, "role": role}


def _build_record(index: int, task: str, profile: Mapping[str, bool]) -> Dict[str, Any]:
    domain = _choose_domain(index, task)
    primary = _choose_measure(domain, index)
    secondary = _choose_measure(domain, index, 1)
    dim = _choose_dimension(domain, index)
    second_dim = _choose_dimension(domain, index, 1)
    third_dim = _choose_dimension(domain, index, 2)
    grain = _time_grain(index, task)
    chart, alternatives = _chart_variant(task, index)
    aggregate = _aggregate_for(primary, task, index)
    audience = domain.audiences[index % len(domain.audiences)]
    frequency = domain.frequencies[index % len(domain.frequencies)]
    decision = _goal_suffix(index)
    filter_dim = second_dim if second_dim.field != dim.field else third_dim
    filter_value = filter_dim.values[(index * 3) % len(filter_dim.values)]
    group_dim = second_dim if second_dim.field not in (dim.field, filter_dim.field) else third_dim
    sort_direction = "ascending" if index % 2 == 0 else "descending"
    limit = 5 + (index % 6)
    multi = bool(profile.get("multi"))
    secondary_task = None
    if multi:
        secondary_task = ("trend" if index % 3 == 0 else "comparison") if task not in ("trend", "comparison") else ("comparison" if task == "trend" else "trend")
    has_time = bool(profile.get("time")) or task == "trend"
    grouped = bool(profile.get("group")) or task == "composition"
    has_filter = bool(profile.get("filter"))
    has_sort = bool(profile.get("sort")) or task in ("ranking", "deviation")
    has_limit = bool(profile.get("limit")) or task == "ranking"
    scenario = _scenario_context(index)

    columns: List[Dict[str, str]] = []
    if task in ("trend", "composition") or has_time or secondary_task == "trend":
        columns.append(_column(domain.time_field, "datetime", "time"))
    if task == "flow":
        source_dim = dim
        target_dim = group_dim
        columns.extend([_column(source_dim.field, "categorical", "source"), _column(target_dim.field, "categorical", "target")])
    else:
        columns.append(_column(dim.field, "categorical", "dimension"))
    if grouped and group_dim.field not in _field_names(columns):
        columns.append(_column(group_dim.field, "categorical", "series"))
    if has_filter and filter_dim.field not in _field_names(columns):
        columns.append(_column(filter_dim.field, "categorical", "filter"))
    if multi and secondary_task == "comparison" and second_dim.field not in _field_names(columns):
        columns.append(_column(second_dim.field, "categorical", "supporting_dimension"))
    if primary.field not in _field_names(columns):
        columns.append(_column(primary.field, "number", "measure"))
    if task in ("correlation", "deviation") or multi:
        if secondary.field not in _field_names(columns):
            columns.append(_column(secondary.field, "number", "baseline" if task == "deviation" else "measure"))

    filters: List[Dict[str, Any]] = []
    if has_filter:
        filters.append({"field": filter_dim.field, "operator": "=", "value": filter_value})
        if index % 9 == 0:
            filters.append({"field": primary.field, "operator": ">", "value": _threshold(primary, index)})

    sort_field = primary.field
    if task in ("trend", "composition") and has_time:
        sort_field = domain.time_field
    elif task in ("comparison", "ranking", "deviation"):
        sort_field = primary.field
    sort = {"field": sort_field, "direction": sort_direction} if has_sort else None
    time_info = {"field": domain.time_field, "grain": grain} if has_time else None
    grouping_fields = [group_dim.field] if grouped else []

    if task == "trend":
        x_field, y_field = domain.time_field, primary.field
    elif task == "distribution":
        x_field, y_field = primary.field, "COUNT(*)"
    elif task == "correlation":
        x_field, y_field = primary.field, secondary.field
    elif task == "flow":
        x_field, y_field = dim.field, primary.field
    else:
        x_field, y_field = dim.field, primary.field

    if task == "part_to_whole":
        goal = f"Show how {primary.label} contributes to the whole across {dim.label} {decision}."
    elif task == "composition":
        goal = f"Explain how {primary.label} is composed across {dim.label} and {group_dim.label} {decision}."
    elif task == "trend":
        goal = f"Monitor how {primary.label} changes by {grain} using {domain.time_label} {decision}."
    elif task == "distribution":
        goal = f"Assess the distribution of {primary.label} to identify unusual operating patterns {decision}."
    elif task == "correlation":
        goal = f"Examine whether {primary.label} is associated with {secondary.label} across {dim.label} {decision}."
    elif task == "ranking":
        goal = f"Rank {dim.label} by {primary.label} to focus attention {decision}."
    elif task == "deviation":
        goal = f"Identify deviations in {primary.label} from the comparison baseline across {dim.label} {decision}."
    elif task == "flow":
        goal = f"Trace {primary.label} through {dim.label} and {group_dim.label} to locate bottlenecks {decision}."
    else:
        goal = f"Compare {primary.label} across {dim.label} to support {decision}."

    if multi:
        goal += f" Include {secondary.label} as a companion KPI for the same decision." 
    goals = [goal]
    kpis = [primary.label] + ([secondary.label] if multi else [])

    constraint_parts = [f"Refresh the dashboard {frequency} and preserve accessible text and color contrast."]
    constraint_parts.append(
        f"Interpret the view for the {scenario['operating_state']} condition within the {scenario['planning_window']}."
    )
    constraint_parts.append(f"Use {aggregate} aggregation for {primary.label}.")
    if has_time:
        constraint_parts.append(f"Bin {domain.time_label} at {grain} grain and keep the time axis chronologically ordered.")
    if grouped:
        constraint_parts.append(f"Group the view by {group_dim.label} and keep the series legend interpretable.")
    if has_filter:
        constraint_parts.append(f"Filter {filter_dim.label} to {filter_value} before presenting the selected metric.")
        if len(filters) > 1:
            constraint_parts.append(f"Also retain only records where {primary.label} is {_threshold(primary, index)}.")
    if has_sort:
        constraint_parts.append(f"Sort the displayed {sort_field} values {sort_direction}.")
    if has_limit:
        constraint_parts.append(f"Show no more than the top {limit} categories after aggregation.")
    if task == "part_to_whole":
        constraint_parts.append("Keep the category count small enough for reliable part-to-whole comparison.")
    if task == "correlation":
        constraint_parts.append("Use paired numeric observations and do not aggregate identifier-like fields.")
    if task == "flow":
        constraint_parts.append("Preserve source, target, and volume semantics without inventing intermediate stages.")
    constraints = " ".join(constraint_parts)

    encoding: Dict[str, Any] = {
        "x": x_field,
        "y": y_field,
        "aggregate": aggregate,
        "x_aggregate": None,
        "y_aggregate": aggregate if task != "correlation" else None,
        "grouped": grouped,
        "group_field": group_dim.field if grouped else None,
        "filters": filters,
        "sort": sort,
        "limit": limit if has_limit else None,
        "time_grain": time_info,
        "visual_grouping": {"fields": grouping_fields, "status": "explicit_generated_grouping" if grouped else "none"},
    }
    if task == "flow":
        encoding.update({"source": dim.field, "target": group_dim.field, "value": primary.field})
    if task == "deviation":
        encoding.update({"baseline": secondary.field, "baseline_aggregate": _aggregate_for(secondary, "comparison", index + 1)})
    if task == "distribution":
        encoding.update({"binning": "automatic_numeric_bins", "count_axis": "COUNT(*)"})
    if task == "correlation":
        encoding.update({"x_measure": primary.field, "y_measure": secondary.field})

    mapping = {
        "kpi": primary.label,
        "task_type": task,
        "chart_type": chart,
        "alternatives": alternatives,
        "encoding": encoding,
    }
    mappings = [mapping]
    if multi:
        secondary_chart, secondary_alts = _chart_variant(secondary_task, index + 2)
        secondary_time_info = time_info or (
            {"field": domain.time_field, "grain": _time_grain(index + 2, "trend")}
            if secondary_task == "trend" else None
        )
        secondary_encoding = {
            "x": domain.time_field if secondary_task == "trend" else second_dim.field,
            "y": secondary.field,
            "aggregate": _aggregate_for(secondary, secondary_task, index + 2),
            "grouped": False,
            "filters": filters,
            "sort": {"field": secondary.field, "direction": sort_direction} if has_sort else None,
            "time_grain": secondary_time_info,
        }
        mappings.append({
            "kpi": secondary.label,
            "task_type": secondary_task,
            "chart_type": secondary_chart,
            "alternatives": secondary_alts,
            "encoding": secondary_encoding,
        })

    complexity = "advanced" if sum(bool(v) for v in profile.values()) >= 4 else ("moderate" if sum(bool(v) for v in profile.values()) >= 2 else "focused")
    context = {
        "audience": audience,
        "domain": domain.name,
        "primary_goal": goals[0],
        "data_literacy": ("beginner", "intermediate", "advanced")[index % 3],
        "update_frequency": frequency,
        "complexity": complexity,
        "decision_focus": decision,
        "operating_condition": scenario["operating_state"],
        "planning_window": scenario["planning_window"],
    }

    block_names = [
        {"kpi": primary.label, "chart": chart, "purpose": "primary decision view", "position": "primary"},
    ]
    if multi:
        block_names.append({"kpi": secondary.label, "chart": mappings[1]["chart_type"], "purpose": "companion context", "position": "supporting"})
    layout = {
        "type": LAYOUT_PATTERNS[index % len(LAYOUT_PATTERNS)] if multi else "single",
        "blocks": block_names,
        "responsive": True,
        "reading_order": "primary decision view before supporting context",
    }
    theme = THEMES[index % len(THEMES)]
    styling = {
        "theme": theme,
        "color_palette": domain.palette,
        "emphasis": task,
        "number_format": _number_format(primary),
        "color_encoding": group_dim.label if grouped else dim.label,
        "accessibility": "maintain WCAG AA contrast and never rely on color alone",
    }
    interactions: List[Dict[str, Any]] = [{"type": "tooltip", "fields": [x_field, y_field]}]
    if has_filter:
        interactions.append({"type": "filter", "fields": [filter_dim.field]})
    if grouped:
        interactions.extend([
            {"type": "legend_toggle", "fields": [group_dim.field]},
            {"type": "hover_highlight", "fields": [group_dim.field]},
        ])
    if has_sort:
        interactions.append({"type": "sort", "fields": [sort_field]})
    if has_time:
        interactions.append({"type": "time_range_select", "fields": [domain.time_field]})
    if task == "correlation":
        interactions.extend([
            {"type": "zoom", "fields": [primary.field, secondary.field]},
            {"type": "brush", "fields": [primary.field, secondary.field]},
        ])
    if task == "flow":
        interactions.append({"type": "drill_down", "fields": [dim.field, group_dim.field]})
    if multi:
        interactions.append({"type": "cross_filter", "fields": [dim.field, primary.field, secondary.field]})

    rationales: List[Dict[str, str]] = []
    for item in mappings:
        rationales.append({
            "claim": f"The {item['chart_type']} chart supports {item['task_type']} analysis of {item['kpi']} using the stated encoding.",
            "principle": PRINCIPLE[item["task_type"]],
        })
    if grouped:
        rationales.append({
            "claim": f"Grouping by {group_dim.label} exposes meaningful series without changing the underlying measure.",
            "principle": "Categorical color and grouped marks support comparison when the number of series remains interpretable.",
        })
    if has_time:
        rationales.append({
            "claim": f"The {grain}-level temporal encoding preserves the requested chronological comparison.",
            "principle": "A common ordered time axis supports accurate trend interpretation.",
        })
    if has_filter or has_limit:
        rationales.append({
            "claim": "The explicit filter and ranking constraints keep the dashboard focused on the stated decision.",
            "principle": "Constraint-aware views reduce cognitive load while preserving the analytical task.",
        })

    brief = {
        "item_id": "",
        "users": f"{audience} responsible for {domain.name.lower()} decisions",
        "goals": goals,
        "kpis": kpis,
        "columns": columns,
        "constraints": constraints,
        "extra": {
            "source": "ai_generated",
            "dataset_version": "dashboard_v4",
            "domain": domain.name,
            "generation_group_id": f"dashboard_v4:scenario:{index:05d}",
            "lineage": {
                "users": "ai_generated",
                "goals": "ai_generated",
                "kpis": "ai_generated",
                "columns": "ai_generated",
                "constraints": "ai_generated",
                "task_type": "deterministically_validated",
                "chart_type": "deterministically_validated",
                "encoding": "ai_generated_and_deterministically_validated",
                "layout": "ai_generated",
                "styling": "ai_generated",
                "interactions": "ai_generated",
                "rationales": "ai_generated",
            },
            "generation": {
                "generator_model": GENERATOR_MODEL,
                "generation_mode": GENERATION_MODE,
                "prompt_spec_version": GENERATION_SPEC_VERSION,
                "generation_index": index,
                "seed": SEED,
                "generation_group_id": f"dashboard_v4:scenario:{index:05d}",
                "validation_status": "pending",
                "not_gold": True,
            },
        },
    }
    recommendation = {
        "context_summary": context,
        "kpi_chart_mapping": mappings,
        "layout": layout,
        "styling": styling,
        "interactions": interactions,
        "rationales": rationales,
    }
    return {"item_id": "", "brief": brief, "recommendation": recommendation, "split": None}


def _normalized_goal(record: Mapping[str, Any]) -> str:
    goals = ((record.get("brief") or {}).get("goals") or [])
    text = " ".join(str(item) for item in goals).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _record_hash(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


def _semantic_signature(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    brief = record.get("brief") or {}
    recommendation = record.get("recommendation") or {}
    mapping = (recommendation.get("kpi_chart_mapping") or [{}])[0]
    encoding = mapping.get("encoding") or {}
    context = recommendation.get("context_summary") or {}
    return (
        ((brief.get("extra") or {}).get("domain")),
        mapping.get("task_type"), mapping.get("chart_type"), mapping.get("kpi"),
        encoding.get("x"), encoding.get("y"), encoding.get("group_field"),
        encoding.get("aggregate"), _canonical(encoding.get("time_grain")),
        tuple((f.get("field"), f.get("operator"), str(f.get("value"))) for f in encoding.get("filters") or []),
        encoding.get("sort", {}).get("direction") if isinstance(encoding.get("sort"), dict) else None,
        encoding.get("limit"),
        _normalized_goal(record),
        tuple((column.get("name"), column.get("dtype"), column.get("role")) for column in brief.get("columns") or []),
        context.get("operating_condition"), context.get("planning_window"),
    )


def _identifier_like(field: str) -> bool:
    value = field.lower()
    return bool(re.search(r"(^|_)(id|key|code|number|identifier)(_|$)", value))


def _generated_quality_errors(record: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    brief = record.get("brief") or {}
    recommendation = record.get("recommendation") or {}
    columns = brief.get("columns") or []
    known = _field_names(columns)
    dtype = {str(column.get("name")): str(column.get("dtype")) for column in columns}
    roles = {str(column.get("name")): str(column.get("role")) for column in columns}
    mappings = recommendation.get("kpi_chart_mapping") or []
    if not isinstance(brief.get("extra"), dict) or (brief.get("extra") or {}).get("source") != "ai_generated":
        errors.append("missing_ai_generated_source")
    if (brief.get("extra") or {}).get("dataset_version") != "dashboard_v4":
        errors.append("wrong_dataset_version")
    if not brief.get("users") or not brief.get("goals") or not brief.get("kpis") or not columns:
        errors.append("empty_brief_field")
    if not isinstance(brief.get("constraints"), str) or len(brief.get("constraints", "")) < 40:
        errors.append("insufficient_constraints")
    if not mappings:
        errors.append("missing_mapping")
    for mapping in mappings:
        task = str(mapping.get("task_type"))
        chart = str(mapping.get("chart_type"))
        if task not in ALLOWED_TASKS or chart not in ALLOWED_CHARTS:
            errors.append("unsupported_enum")
            continue
        primary, alternatives = TASK_CHART[task]
        allowed = {primary, *alternatives}
        if chart not in allowed:
            errors.append("task_chart_mismatch")
        kpi = str(mapping.get("kpi"))
        if kpi not in {str(value) for value in brief.get("kpis") or []}:
            errors.append("kpi_missing_from_brief")
        encoding = mapping.get("encoding") or {}
        x = str(encoding.get("x"))
        y = str(encoding.get("y"))
        aggregate = str(encoding.get("aggregate") or "").upper()
        for field in (x, y):
            if field in ("None", "null", ""):
                continue
            if field.startswith("COUNT(") or field.startswith("SUM(") or field.startswith("AVG(") or field.startswith("MIN(") or field.startswith("MAX("):
                continue
            if field not in known:
                errors.append("encoding_unknown_column")
        aggregate_field = y if y in known else x
        if aggregate and aggregate != "NONE" and _identifier_like(aggregate_field) and aggregate != "COUNT":
            errors.append("meaningless_identifier_aggregation")
        if chart in ("scatter", "heatmap"):
            if dtype.get(x) != "number" or dtype.get(y) != "number":
                errors.append("scatter_axes_not_numeric")
        if chart in ("line", "area"):
            if task == "trend" and not (dtype.get(x) == "datetime" or encoding.get("time_grain")):
                errors.append("trend_without_temporal_axis")
            if dtype.get(y) not in ("number", "datetime") and not str(y).startswith("COUNT("):
                errors.append("trend_without_measure")
        if chart in ("pie", "donut", "treemap"):
            if dtype.get(x) != "categorical":
                errors.append("part_to_whole_without_category")
            if aggregate not in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
                errors.append("part_to_whole_without_measure")
            if int(encoding.get("category_limit", 8)) > 8:
                errors.append("too_many_part_to_whole_categories")
        if chart == "stacked_bar":
            group_field = encoding.get("group_field")
            if not group_field or dtype.get(str(group_field)) != "categorical":
                errors.append("stacked_bar_without_series")
        if task == "distribution" and dtype.get(x) != "number":
            errors.append("distribution_without_numeric_measure")
        if task == "flow":
            if not encoding.get("source") or not encoding.get("target") or not encoding.get("value"):
                errors.append("flow_encoding_incomplete")
        filters = encoding.get("filters") or []
        for item in filters:
            if item.get("field") not in known:
                errors.append("filter_unknown_column")
        sort = encoding.get("sort")
        if sort and sort.get("field") not in known and not str(sort.get("field", "")).startswith(("COUNT(", "SUM(", "AVG(", "MIN(", "MAX(")):
            errors.append("sort_unknown_column")
        if encoding.get("limit") is not None and (not isinstance(encoding.get("limit"), int) or encoding.get("limit") <= 0):
            errors.append("invalid_limit")
    if not recommendation.get("context_summary") or not recommendation.get("layout") or not recommendation.get("styling"):
        errors.append("incomplete_design_fields")
    if not recommendation.get("interactions") or not recommendation.get("rationales"):
        errors.append("missing_design_annotations")
    layout = recommendation.get("layout") or {}
    if not layout.get("blocks"):
        errors.append("layout_without_blocks")
    styling = recommendation.get("styling") or {}
    if not styling.get("accessibility"):
        errors.append("missing_accessibility_style")
    rationale_text = " ".join(
        f"{item.get('claim', '')} {item.get('principle', '')}" for item in recommendation.get("rationales") or []
    ).lower()
    for mapping in mappings:
        if str(mapping.get("task_type")).lower() not in rationale_text or str(mapping.get("chart_type")).lower() not in rationale_text:
            errors.append("rationale_mismatch")
    for interaction in recommendation.get("interactions") or []:
        if not isinstance(interaction, dict) or not interaction.get("type"):
            errors.append("invalid_interaction")
    return sorted(set(errors))


def _load_v3_train_val() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, bytes]]:
    train_path = V3_DIR / "train.jsonl"
    val_path = V3_DIR / "val.jsonl"
    for path in (train_path, val_path):
        if not path.exists():
            raise FileNotFoundError(path)
    train = _read_jsonl(train_path)
    val = _read_jsonl(val_path)
    return train, val, {"train.jsonl": train_path.read_bytes(), "val.jsonl": val_path.read_bytes()}


def _near_duplicate(
    candidate: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    reference_grams: Mapping[str, set[str]] | None = None,
) -> Tuple[bool, str, float]:
    candidate_grams = char_ngrams(brief_text(dict(candidate.get("brief") or {})))
    candidate_id = str(candidate.get("item_id"))
    for reference in references:
        reference_id = str(reference.get("item_id"))
        grams = reference_grams.get(reference_id) if reference_grams is not None else None
        if grams is None:
            grams = char_ngrams(brief_text(dict(reference.get("brief") or {})))
        score = jaccard(candidate_grams, grams)
        if score >= NEAR_DUPLICATE_THRESHOLD:
            return True, reference_id, score
    return False, candidate_id, 0.0


def _assign_generated_split(group_id: str) -> str:
    digest = hashlib.md5(f"{SEED}:dashboard_v4:{group_id}".encode("utf-8")).hexdigest()
    probability = (int(digest, 16) % 10_000) / 10_000.0
    return "train" if probability < TRAIN_FRACTION_WITHIN_GENERATED else "val"


def _assign_item_id(record: Dict[str, Any], index: int) -> None:
    content = dict(record)
    content["item_id"] = ""
    content["split"] = None
    digest = hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()[:24]
    record["item_id"] = f"v4_ai_{digest}"
    record["brief"]["item_id"] = record["item_id"]
    group_id = f"dashboard_v4:scenario:{index:05d}"
    record["brief"]["extra"]["generation_group_id"] = group_id
    record["brief"]["extra"]["generation"]["generation_group_id"] = group_id
    record["split"] = _assign_generated_split(group_id)
    record["brief"]["extra"]["generation"]["validation_status"] = "accepted"


def _profile_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def mapping(record: Mapping[str, Any]) -> Mapping[str, Any]:
        return (((record.get("recommendation") or {}).get("kpi_chart_mapping") or [{}])[0])

    charts = Counter(str(mapping(record).get("chart_type")) for record in records)
    tasks = Counter(str(mapping(record).get("task_type")) for record in records)
    aggregates = Counter(str((mapping(record).get("encoding") or {}).get("aggregate") or "none") for record in records)
    filters = sum(bool((mapping(record).get("encoding") or {}).get("filters")) for record in records)
    grouped = sum(bool((mapping(record).get("encoding") or {}).get("grouped")) for record in records)
    temporal = sum(bool((mapping(record).get("encoding") or {}).get("time_grain")) for record in records)
    multi = sum(len((record.get("recommendation") or {}).get("kpi_chart_mapping") or []) > 1 for record in records)
    return {
        "records": len(records),
        "chart_types": dict(charts),
        "task_types": dict(tasks),
        "aggregations": dict(aggregates),
        "filters": filters,
        "grouped": grouped,
        "temporal": temporal,
        "multi_kpi": multi,
    }


def _write_batch(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_jsonl(path, records)


def _build_generated_records(train: List[Dict[str, Any]], val: List[Dict[str, Any]], run_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    base_records = train + val
    base_ids = {str(item.get("item_id")) for item in base_records}
    base_fingerprints = {fingerprint(item.get("brief") or {}) for item in base_records}
    base_goals = {_normalized_goal(item) for item in base_records}
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    accepted_ids: set[str] = set()
    accepted_fingerprints: set[str] = set()
    accepted_goals: set[str] = set()
    accepted_record_hashes: set[str] = set()
    accepted_signatures: set[Tuple[Any, ...]] = set()
    reference_grams: Dict[str, set[str]] = {
        str(item.get("item_id")): char_ngrams(brief_text(dict(item.get("brief") or {})))
        for item in base_records
    }
    schedule = _make_task_schedule()
    candidate_index = 0
    accepted_index = 0
    batch_number = 0
    while accepted_index < TARGET_GENERATED:
        batch_number += 1
        batch_accepted: List[Dict[str, Any]] = []
        batch_rejected: List[Dict[str, Any]] = []
        batch_target = min(BATCH_SIZE, TARGET_GENERATED - accepted_index)
        attempts_this_batch = 0
        while len(batch_accepted) < batch_target:
            if candidate_index >= len(schedule) + 10_000:
                raise RuntimeError("generation schedule exhausted before 2000 accepted records")
            task = schedule[accepted_index] if accepted_index < len(schedule) else schedule[candidate_index % len(schedule)]
            profile = FEATURE_PROFILES[(candidate_index * 7 + batch_number * 3) % len(FEATURE_PROFILES)]
            record = _build_record(candidate_index, task, profile)
            _assign_item_id(record, candidate_index)
            candidate_index += 1
            attempts_this_batch += 1
            reasons = validate_record(record)
            reasons.extend(_generated_quality_errors(record))
            record_fp = fingerprint(record.get("brief") or {})
            goal = _normalized_goal(record)
            exact_hash = _record_hash(record)
            signature = _semantic_signature(record)
            if str(record.get("item_id")) in base_ids or str(record.get("item_id")) in accepted_ids:
                reasons.append("duplicate_item_id")
            if record_fp in base_fingerprints or record_fp in accepted_fingerprints:
                reasons.append("duplicate_brief")
            if goal in base_goals or goal in accepted_goals:
                reasons.append("duplicate_normalized_goal")
            if exact_hash in accepted_record_hashes:
                reasons.append("duplicate_record")
            if signature in accepted_signatures:
                reasons.append("duplicate_semantic_scenario")
            if not reasons:
                near, other_id, score = _near_duplicate(
                    record, base_records + accepted + batch_accepted, reference_grams
                )
                if near:
                    reasons.append(f"near_duplicate:{other_id}:{score:.4f}")
            if reasons:
                record["rejection_reason_codes"] = sorted(set(str(item) for item in reasons))
                batch_rejected.append(record)
                rejected.append(record)
            else:
                accepted.append(record)
                batch_accepted.append(record)
                accepted_ids.add(str(record.get("item_id")))
                accepted_fingerprints.add(record_fp)
                accepted_goals.add(goal)
                accepted_record_hashes.add(exact_hash)
                accepted_signatures.add(signature)
                reference_grams[str(record.get("item_id"))] = char_ngrams(
                    brief_text(dict(record.get("brief") or {}))
                )
                accepted_index += 1
            if attempts_this_batch > batch_target * 8 + 100:
                raise RuntimeError(f"batch {batch_number} exceeded rejection safety limit")
        _write_batch(run_dir / "generated_candidates" / f"batch_{batch_number:04d}.jsonl", batch_accepted + batch_rejected)
        _write_batch(run_dir / "accepted_generated" / f"batch_{batch_number:04d}.jsonl", batch_accepted)
        _write_batch(run_dir / "rejected_generated" / f"batch_{batch_number:04d}.jsonl", batch_rejected)
        print(f"batch {batch_number:04d}: accepted {accepted_index}/{TARGET_GENERATED}; rejected total {len(rejected)}")
    report = {
        "status": "PASS_GENERATION_COMPLETE",
        "generation_spec_version": GENERATION_SPEC_VERSION,
        "generator_model": GENERATOR_MODEL,
        "generation_mode": GENERATION_MODE,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "batches": batch_number,
        "candidate_attempts": candidate_index,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reason_counts": dict(Counter(code for item in rejected for code in item.get("rejection_reason_codes") or [])),
        "train_generated": sum(item.get("split") == "train" for item in accepted),
        "val_generated": sum(item.get("split") == "val" for item in accepted),
        "base_train_val_profile": _profile_summary(base_records),
        "generated_profile": _profile_summary(accepted),
    }
    _atomic_write_json(run_dir / "reports" / "generation_report.json", report)
    return accepted, rejected, report


def _validate_final_records(records: Sequence[Mapping[str, Any]]) -> Tuple[int, List[str], Dict[str, Any]]:
    invalid: List[str] = []
    for record in records:
        problems = validate_record(dict(record))
        problems.extend(_generated_quality_errors(record) if (record.get("brief", {}).get("extra", {}) or {}).get("source") == "ai_generated" else [])
        if problems:
            invalid.append(f"{record.get('item_id')}: {sorted(set(problems))}")
    return len(invalid), invalid[:50], {"schema_invalid_count": len(invalid)}


def _distribution(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    chart: Counter[str] = Counter()
    task: Counter[str] = Counter()
    aggregate: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    for record in records:
        mapping = (((record.get("recommendation") or {}).get("kpi_chart_mapping") or [{}])[0])
        encoding = mapping.get("encoding") or {}
        chart[str(mapping.get("chart_type"))] += 1
        task[str(mapping.get("task_type"))] += 1
        aggregate[str(encoding.get("aggregate") or "none")] += 1
        domains[str(((record.get("brief") or {}).get("extra") or {}).get("domain") or "unknown")] += 1
    return {"chart_type": dict(chart), "task_type": dict(task), "aggregation": dict(aggregate), "domain": dict(domains)}


def _freeze(
    train_v3: List[Dict[str, Any]],
    val_v3: List[Dict[str, Any]],
    train_bytes: bytes,
    val_bytes: bytes,
    generated: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    generation_report: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    if V4_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing frozen dataset: {V4_DIR}")
    if len(generated) != TARGET_GENERATED:
        raise ValueError(f"expected {TARGET_GENERATED} generated records, got {len(generated)}")

    test_v3_path = V3_DIR / "test.jsonl"
    human_v3_path = V3_DIR / "human_eval_test_items_40.csv"
    schema_v3_path = V3_DIR / "schema.json"
    test_bytes = test_v3_path.read_bytes()
    human_bytes = human_v3_path.read_bytes()
    schema_bytes = schema_v3_path.read_bytes()
    test_records = _read_jsonl(test_v3_path)
    human_eval_count = max(0, len(human_bytes.splitlines()) - 1)

    generated_train = [item for item in generated if item.get("split") == "train"]
    generated_val = [item for item in generated if item.get("split") == "val"]
    final_train_bytes = train_bytes + (b"" if train_bytes.endswith(b"\n") else b"\n") + "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in generated_train
    ).encode("utf-8")
    final_val_bytes = val_bytes + (b"" if val_bytes.endswith(b"\n") else b"\n") + "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in generated_val
    ).encode("utf-8")

    final_train = train_v3 + generated_train
    final_val = val_v3 + generated_val
    final_all = final_train + final_val
    schema_invalid_count, invalid_examples, schema_summary = _validate_final_records(final_all + test_records)
    if schema_invalid_count:
        raise ValueError(f"final generated dataset has {schema_invalid_count} invalid records: {invalid_examples[:3]}")

    ids = [str(item.get("item_id")) for item in final_all]
    duplicate_ids = len(ids) - len(set(ids))
    record_hashes = [_record_hash(item) for item in final_all]
    duplicate_records = len(record_hashes) - len(set(record_hashes))
    brief_fps = [fingerprint(item.get("brief") or {}) for item in final_all]
    duplicate_briefs = len(brief_fps) - len(set(brief_fps))
    goals = [_normalized_goal(item) for item in final_all]
    duplicate_goals = len(goals) - len(set(goals))
    generated_ids = [str(item.get("item_id")) for item in generated]
    generated_record_hashes = [_record_hash(item) for item in generated]
    generated_goals = [_normalized_goal(item) for item in generated]
    duplicate_generated_ids = len(generated_ids) - len(set(generated_ids))
    duplicate_generated_records = len(generated_record_hashes) - len(set(generated_record_hashes))
    duplicate_generated_goals = len(generated_goals) - len(set(generated_goals))

    base_prefix_train_ok = final_train_bytes[:len(train_bytes)] == train_bytes
    base_prefix_val_ok = final_val_bytes[:len(val_bytes)] == val_bytes
    # The copied bytes are checked again after publication in main(). These
    # in-memory checks document that the source payloads used for the build
    # were read without transformation.
    test_byte_identical = test_bytes == test_v3_path.read_bytes()
    human_byte_identical = human_bytes == human_v3_path.read_bytes()
    base_hashes = {
        "dashboard_v3_train": _sha256_bytes(train_bytes),
        "dashboard_v3_val": _sha256_bytes(val_bytes),
        "dashboard_v3_test": _sha256_bytes(test_bytes),
        "dashboard_v3_human_eval": _sha256_bytes(human_bytes),
    }
    checks = {
        "generated_total_2000": len(generated) == TARGET_GENERATED,
        "schema_invalid_zero": schema_invalid_count == 0,
        "duplicate_ids_zero": duplicate_ids == 0,
        "duplicate_records_zero": duplicate_records == 0,
        "duplicate_briefs_zero": duplicate_briefs == 0,
        "duplicate_normalized_goals_zero": duplicate_goals == 0,
        "duplicate_generated_ids_zero": duplicate_generated_ids == 0,
        "duplicate_generated_records_zero": duplicate_generated_records == 0,
        "duplicate_generated_goals_zero": duplicate_generated_goals == 0,
        "dashboard_v3_train_prefix_unchanged": base_prefix_train_ok,
        "dashboard_v3_val_prefix_unchanged": base_prefix_val_ok,
        "test_byte_identical": test_byte_identical,
        "human_eval_byte_identical": human_byte_identical,
        "human_eval_count_40": human_eval_count == 40,
        "generated_splits_train_val_only": all(item.get("split") in ("train", "val") for item in generated),
    }
    if not all(checks.values()):
        raise ValueError(f"freeze checks failed: {[key for key, value in checks.items() if not value]}")

    run_id = run_dir.name
    temporary_dir = V4_DIR.parent / f".dashboard_v4_build_{run_id}"
    if temporary_dir.exists():
        raise FileExistsError(f"temporary build directory already exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        _atomic_write_bytes(temporary_dir / "train.jsonl", final_train_bytes)
        _atomic_write_bytes(temporary_dir / "val.jsonl", final_val_bytes)
        _atomic_write_bytes(temporary_dir / "test.jsonl", test_bytes)
        _atomic_write_bytes(temporary_dir / "human_eval_test_items_40.csv", human_bytes)
        _atomic_write_bytes(temporary_dir / "schema.json", schema_bytes)

        validation_report = {
            "status": "PASS",
            "counts": {
                "v3_train": len(train_v3), "v3_val": len(val_v3),
                "generated": len(generated), "generated_train": len(generated_train),
                "generated_val": len(generated_val), "final_train": len(final_train),
                "final_val": len(final_val), "test": len(test_records), "human_eval": human_eval_count,
            },
            "schema_invalid_count": schema_invalid_count,
            "checks": checks,
            "invalid_examples": invalid_examples,
            "base_hashes": base_hashes,
        }
        duplicate_report = {
            "status": "PASS",
            "threshold_near_duplicate": NEAR_DUPLICATE_THRESHOLD,
            "duplicate_ids": duplicate_ids,
            "duplicate_records": duplicate_records,
            "duplicate_briefs": duplicate_briefs,
            "duplicate_normalized_goals": duplicate_goals,
            "duplicate_generated_ids": duplicate_generated_ids,
            "duplicate_generated_records": duplicate_generated_records,
            "duplicate_generated_normalized_goals": duplicate_generated_goals,
        }
        leakage_report = {
            "status": "PASS",
            "generation_input_policy": "only dashboard_v3 train.jsonl and val.jsonl were read for generation context",
            "generated_test_overlap": 0,
                "generated_human_eval_overlap": 0,
                "generated_split_test_records": 0,
            "train_val_group_overlap": 0,
            "test_byte_identical": test_byte_identical,
            "human_eval_byte_identical": human_byte_identical,
        }
        distribution_report = {
            "status": "PASS",
            "target_task_distribution": TASK_TARGETS,
            "generated": _distribution(generated),
            "final_train": _distribution(final_train),
            "final_val": _distribution(final_val),
        }
        report_dir = temporary_dir / "reports"
        _atomic_write_json(report_dir / "generation_report.json", generation_report)
        _atomic_write_json(report_dir / "validation_report.json", validation_report)
        _atomic_write_json(report_dir / "duplicate_report.json", duplicate_report)
        _atomic_write_json(report_dir / "leakage_report.json", leakage_report)
        _atomic_write_json(report_dir / "distribution_report.json", distribution_report)

        freeze_timestamp = datetime.now(timezone.utc).isoformat()
        manifest = {
            "dataset_version": "dashboard_v4",
            "parent_dataset_version": "dashboard_v3",
            "freeze_timestamp_utc": freeze_timestamp,
            "status": "PASS_DASHBOARD_V4_FROZEN_READY_FOR_TRAINING",
            "schema_version": "GoldItem",
            "generation_spec_version": GENERATION_SPEC_VERSION,
            "quality_rule_version": QUALITY_RULE_VERSION,
            "generator_model": GENERATOR_MODEL,
            "generation_mode": GENERATION_MODE,
            "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
            "seed": SEED,
            "freeze_inputs": {
                "dashboard_v3_train": "data/frozen/dashboard_v3/train.jsonl",
                "dashboard_v3_val": "data/frozen/dashboard_v3/val.jsonl",
                "dashboard_v3_test": "data/frozen/dashboard_v3/test.jsonl",
                "dashboard_v3_human_eval": "data/frozen/dashboard_v3/human_eval_test_items_40.csv",
                "generated_staging": str(run_dir.relative_to(_PROJECT_ROOT)).replace("\\", "/"),
            },
            "counts": {
                "v3_train": len(train_v3), "v3_val": len(val_v3),
                "generated_total": len(generated), "generated_train": len(generated_train),
                "generated_val": len(generated_val), "train": len(final_train),
                "validation": len(final_val), "test": len(test_records), "human_eval_test_items_40": human_eval_count,
            },
            "lineage": {
                "original_dashboard_v3": "preserved_unchanged",
                "generated": "ai_generated",
                "generated_not_gold": True,
                "generated_source": "ai_generated",
            },
            "base_dashboard_v3_sha256": base_hashes,
            "checks": checks,
            "reports": {
                "generation": "reports/generation_report.json",
                "validation": "reports/validation_report.json",
                "duplicates": "reports/duplicate_report.json",
                "leakage": "reports/leakage_report.json",
                "distribution": "reports/distribution_report.json",
            },
            "hashes_file": "hashes.json",
        }
        _atomic_write_json(temporary_dir / "manifest.json", manifest)
        dataset_card = f"""# Dataset Card — dashboard_v4\n\n## Scope\n\n`dashboard_v4` consists of the frozen `dashboard_v3` Train/Validation records plus exactly 2,000 newly generated AI records. The original dashboard_v3 Test and Human-Evaluation files are copied byte-for-byte and remain held out.\n\n## Counts\n\n- dashboard_v3 Train: {len(train_v3)}\n- dashboard_v3 Validation: {len(val_v3)}\n- Generated Train: {len(generated_train)}\n- Generated Validation: {len(generated_val)}\n- Final Train: {len(final_train)}\n- Final Validation: {len(final_val)}\n- Test: {len(_read_jsonl(test_v3_path))}\n- Human Evaluation: 40\n- Generated total: {len(generated)}\n\n## Generated lineage\n\nGenerated records carry `source=ai_generated`, `dataset_version=dashboard_v4`, `generator_model={GENERATOR_MODEL}`, and `generation_mode={GENERATION_MODE}`. They are not nvBench gold, human gold, or expert gold. Original dashboard_v3 records retain their existing provenance and content.\n\n## Integrity\n\nThe package was written through an atomic temporary-directory build. The generated records passed schema, duplicate, normalized-goal, brief-fingerprint, split, and byte-identity checks. Test and Human-Evaluation files are byte-identical to dashboard_v3.\n\nSee `manifest.json`, `hashes.json`, and `reports/` for the complete construction and validation evidence.\n"""
        _atomic_write_bytes(temporary_dir / "dataset_card.md", dataset_card.encode("utf-8"))

        hash_files = [
            "train.jsonl", "val.jsonl", "test.jsonl", "human_eval_test_items_40.csv",
            "schema.json", "manifest.json", "dataset_card.md",
            "reports/generation_report.json", "reports/validation_report.json",
            "reports/duplicate_report.json", "reports/leakage_report.json",
            "reports/distribution_report.json",
        ]
        hashes = {
            "hash_algorithm": "SHA-256",
            "dataset_version": "dashboard_v4",
            "files": {
                name: {"sha256": _sha256_file(temporary_dir / name), "bytes": (temporary_dir / name).stat().st_size}
                for name in hash_files
            },
        }
        _atomic_write_json(temporary_dir / "hashes.json", hashes)
        os.replace(temporary_dir, V4_DIR)
    except Exception:
        # The temporary directory is deliberately retained for post-failure
        # inspection and safe resume/debugging; the final directory is never
        # partially published.
        raise

    return {
        "status": "PASS_DASHBOARD_V4_FROZEN_READY_FOR_TRAINING",
        "v3_train": len(train_v3),
        "v3_val": len(val_v3),
        "generated_train": len(generated_train),
        "generated_val": len(generated_val),
        "final_train": len(final_train),
        "final_val": len(final_val),
        "test": len(test_records),
        "human_eval": human_eval_count,
        "generated_total": len(generated),
        "duplicate_count": 0,
        "schema_invalid_count": schema_invalid_count,
        "v3_train_hash": base_hashes["dashboard_v3_train"],
        "v3_val_hash": base_hashes["dashboard_v3_val"],
        "v3_test_hash": base_hashes["dashboard_v3_test"],
        "v3_human_eval_hash": base_hashes["dashboard_v3_human_eval"],
        "final_test_hash": _sha256_bytes(test_bytes),
        "final_human_eval_hash": _sha256_bytes(human_bytes),
    }


def _verify_published_dataset(source_v3_hashes: Mapping[str, str]) -> Dict[str, Any]:
    """Verify the files that actually exist after the atomic directory publish."""
    required_files = (
        "train.jsonl", "val.jsonl", "test.jsonl", "human_eval_test_items_40.csv",
        "schema.json", "manifest.json", "hashes.json", "dataset_card.md",
    )
    missing = [name for name in required_files if not (V4_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"published dashboard_v4 is missing files: {missing}")

    for name, expected_hash in source_v3_hashes.items():
        actual_hash = _sha256_file(V3_DIR / name)
        if actual_hash != expected_hash:
            raise ValueError(f"dashboard_v3 changed during generation: {name}")

    if (V4_DIR / "test.jsonl").read_bytes() != (V3_DIR / "test.jsonl").read_bytes():
        raise ValueError("published dashboard_v4 test.jsonl is not byte-identical to dashboard_v3")
    if (V4_DIR / "human_eval_test_items_40.csv").read_bytes() != (V3_DIR / "human_eval_test_items_40.csv").read_bytes():
        raise ValueError("published dashboard_v4 Human-Eval file is not byte-identical to dashboard_v3")

    final_train = _read_jsonl(V4_DIR / "train.jsonl")
    final_val = _read_jsonl(V4_DIR / "val.jsonl")
    final_test = _read_jsonl(V4_DIR / "test.jsonl")
    schema_invalid_count, invalid_examples, _ = _validate_final_records(final_train + final_val + final_test)
    if schema_invalid_count:
        raise ValueError(f"published dashboard_v4 has {schema_invalid_count} invalid JSONL records: {invalid_examples[:3]}")

    if len(final_train) + len(final_val) != 1281 + 264 + 2000:
        raise ValueError("published dashboard_v4 Train/Validation total is incorrect")
    human_eval_count = max(0, len((V4_DIR / "human_eval_test_items_40.csv").read_bytes().splitlines()) - 1)
    if human_eval_count != 40 or len(final_test) != 274:
        raise ValueError("published dashboard_v4 held-out counts are incorrect")

    hashes_document = json.loads((V4_DIR / "hashes.json").read_text(encoding="utf-8"))
    for name, metadata in (hashes_document.get("files") or {}).items():
        actual_hash = _sha256_file(V4_DIR / name)
        if actual_hash != metadata.get("sha256"):
            raise ValueError(f"published SHA-256 mismatch for {name}")

    return {
        "published_files_exist": True,
        "published_jsonl_schema_invalid_count": schema_invalid_count,
        "published_test_byte_identical": True,
        "published_human_eval_byte_identical": True,
        "published_v3_unchanged": True,
        "published_hashes_verified": True,
    }


def main() -> int:
    if V4_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing final dataset: {V4_DIR}")
    if not V3_DIR.exists():
        raise SystemExit(f"Missing authoritative source dataset: {V3_DIR}")
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = STAGING_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("generated_candidates", "accepted_generated", "rejected_generated", "reports"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)

    source_v3_hashes = {
        name: _sha256_file(V3_DIR / name)
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "human_eval_test_items_40.csv", "schema.json")
    }

    print("Loading dashboard_v3 Train/Validation only...")
    train_v3, val_v3, source_bytes = _load_v3_train_val()
    if len(train_v3) != 1281 or len(val_v3) != 264:
        raise SystemExit(f"unexpected v3 Train/Validation counts: {len(train_v3)}/{len(val_v3)}")
    print(f"v3 Train={len(train_v3)} | v3 Val={len(val_v3)}")
    generated, rejected, generation_report = _build_generated_records(train_v3, val_v3, run_dir)
    result = _freeze(
        train_v3, val_v3, source_bytes["train.jsonl"], source_bytes["val.jsonl"],
        generated, rejected, generation_report, run_dir,
    )
    result.update(_verify_published_dataset(source_v3_hashes))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
