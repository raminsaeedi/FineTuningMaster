"""Rating rubric for dashboard design recommendations.

Each dimension is scored on a 1-5 Likert scale. Anchors for 1/3/5 are shown to
raters during calibration and in the app to keep scoring consistent. Keep the
rubric stable once rating starts — changing it mid-study breaks comparability.
"""

from __future__ import annotations

from typing import Dict, List

LIKERT_MIN = 1
LIKERT_MAX = 5

RUBRIC: List[Dict[str, object]] = [
    {
        "key": "chart_appropriateness",
        "label": "Chart appropriateness",
        "description": "Do the recommended chart types fit the KPIs and analytical tasks?",
        "anchors": {1: "Mostly wrong chart choices", 3: "Mixed / defensible but not ideal", 5: "Consistently correct choices"},
    },
    {
        "key": "layout_quality",
        "label": "Layout & hierarchy",
        "description": "Is the proposed layout clear, prioritized and easy to scan?",
        "anchors": {1: "Confusing / no clear hierarchy", 3: "Reasonable but generic", 5: "Clear, well-prioritized"},
    },
    {
        "key": "styling_accessibility",
        "label": "Styling & accessibility",
        "description": "Are color, contrast and formatting choices sound and accessible?",
        "anchors": {1: "Ignores accessibility", 3: "Partly addressed", 5: "Thorough and accessible"},
    },
    {
        "key": "interaction_design",
        "label": "Interaction design",
        "description": "Are the proposed interactions useful and appropriate for the audience?",
        "anchors": {1: "Missing or inappropriate", 3: "Adequate", 5: "Well-matched to users"},
    },
    {
        "key": "rationale_quality",
        "label": "Rationale quality",
        "description": "Are the justifications correct, specific and grounded in design principles?",
        "anchors": {1: "Vague or wrong", 3: "Plausible but shallow", 5: "Specific and principled"},
    },
    {
        "key": "overall_usefulness",
        "label": "Overall usefulness",
        "description": "Overall, how useful is this recommendation to the target users?",
        "anchors": {1: "Not usable", 3: "Somewhat useful", 5: "Highly useful"},
    },
]

RUBRIC_KEYS: List[str] = [d["key"] for d in RUBRIC]  # type: ignore[misc]
