"""Blind human-rating app for dashboard design recommendations.

Run via ``python experiments/scripts/run_human_eval.py --study-dir ...``. The
items, assignment and ratings directory are resolved from that study directory.

Raters never see which method produced an output — rating is blind. Progress is
saved after every submission and resumes automatically.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make 'src' importable when Streamlit runs this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.evaluation.human.render import render_brief, render_output
from src.evaluation.human.rubric import LIKERT_MAX, LIKERT_MIN, RUBRIC
from src.evaluation.human.storage import append_rating, load_done_units

_DEFAULT_STUDY_DIR = _PROJECT_ROOT / "experiments" / "results" / "human_eval"
EVAL_DIR = Path(os.environ.get("HUMAN_EVAL_STUDY_DIR", os.environ.get("HUMAN_EVAL_DIR", _DEFAULT_STUDY_DIR)))
RATINGS_DIR = Path(os.environ.get("HUMAN_RATINGS_DIR", EVAL_DIR / "ratings"))


@st.cache_data
def _load_eval():
    items_path = EVAL_DIR / "items.jsonl"
    assign_path = EVAL_DIR / "assignment.json"
    if not items_path.exists() or not assign_path.exists():
        return None, None
    items = {}
    with items_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                items[row["item_id"]] = row
    assignment = json.loads(assign_path.read_text(encoding="utf-8"))
    return items, assignment


def main() -> None:
    st.set_page_config(page_title="Dashboard Recommendation Rating", layout="wide")
    st.title("Dashboard Design Recommendation — Blind Rating")

    items, assignment = _load_eval()
    if items is None:
        st.error(
            f"No eval set found in {EVAL_DIR}. Run `python scripts/build_human_eval.py` first."
        )
        return

    raters = list(assignment["raters"].keys())
    with st.sidebar:
        st.header("Rater")
        rater_id = st.selectbox("Select your rater ID", raters)
        st.caption(
            "Your ratings are saved automatically and you can stop and resume any time."
        )

    tasks = assignment["raters"][rater_id]
    done = load_done_units(RATINGS_DIR, rater_id)
    remaining = [t for t in tasks if t["unit_id"] not in done]

    total = len(tasks)
    completed = total - len(remaining)
    st.progress(completed / total if total else 1.0)
    st.write(f"**Progress:** {completed} / {total} rated")

    if not remaining:
        st.success("All your assigned items are rated. Thank you!")
        return

    task = remaining[0]
    item = items[task["item_id"]]
    # Assignment stores method for later unblinding, but no method/model/seed
    # identity is rendered to the rater.
    output_key = task.get("output_id", task.get("method"))
    output = item["outputs"][output_key]

    left, right = st.columns(2)
    with left:
        st.subheader("Dashboard brief")
        st.markdown(render_brief(item["brief"]))
    with right:
        st.subheader("Recommendation to rate")
        st.markdown(render_output(output))

    st.divider()
    st.subheader("Your ratings (1 = poor, 5 = excellent)")

    with st.form(key=f"form_{task['unit_id']}", clear_on_submit=True):
        scores = {}
        for dim in RUBRIC:
            anchors = dim["anchors"]
            st.markdown(f"**{dim['label']}** — {dim['description']}")
            st.caption(f"1: {anchors[1]}  •  3: {anchors[3]}  •  5: {anchors[5]}")
            scores[dim["key"]] = st.radio(
                dim["label"],
                options=list(range(LIKERT_MIN, LIKERT_MAX + 1)),
                index=2,
                horizontal=True,
                label_visibility="collapsed",
                key=f"{task['unit_id']}_{dim['key']}",
            )
        comment = st.text_area("Optional comment", key=f"{task['unit_id']}_comment")
        submitted = st.form_submit_button("Submit and next")

    if submitted:
        append_rating(
            RATINGS_DIR,
            rater_id=rater_id,
            unit_id=task["unit_id"],
            item_id=task["item_id"],
            method=task["method"],
            scores=scores,
            comment=comment,
        )
        st.rerun()


if __name__ == "__main__":
    main()
