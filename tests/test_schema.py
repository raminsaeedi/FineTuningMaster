"""Schema round-trip and tolerance tests (masterplan schema)."""

from src.core.schemas import (
    ChartType,
    DashboardBrief,
    DesignOutput,
    GenerationResult,
    GoldItem,
    TaskType,
)


def test_brief_round_trip():
    brief = DashboardBrief(
        item_id="item_1",
        users="Sales Managers (intermediate)",
        goals=["grow revenue"],
        kpis=["Revenue", "Churn Rate"],
        columns=[{"name": "date", "dtype": "datetime"}],
    )
    dumped = brief.model_dump(mode="json")
    assert DashboardBrief(**dumped).kpis == ["Revenue", "Churn Rate"]


def test_design_output_enums_and_defaults():
    out = DesignOutput(
        context_summary={"audience": "execs"},
        kpi_chart_mapping=[{"kpi": "Revenue", "task_type": "trend", "chart_type": "line"}],
        interactions=["cross-filtering"],
        rationales=[{"claim": "use a line chart", "principle": "trends use position over time"}],
    )
    assert out.kpi_chart_mapping[0].chart_type == ChartType.LINE
    assert out.kpi_chart_mapping[0].task_type == TaskType.TREND
    assert out.layout == {} and out.styling == {}  # default-filled


def test_generation_result_minimal():
    res = GenerationResult(item_id="i", method_name="prompt_only", model_name="qwen")
    assert res.parsed is None and res.variant == "original"


def test_gold_item():
    item = GoldItem(
        item_id="item_1",
        brief={"users": "X", "kpis": ["Revenue"]},
        recommendation={"kpi_chart_mapping": [{"kpi": "Revenue", "task_type": "trend", "chart_type": "line"}]},
    )
    assert item.brief.users == "X"
    assert item.recommendation.kpi_chart_mapping[0].chart_type == ChartType.LINE
