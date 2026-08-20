# Tiny Dataset Human-Evaluation Guide

Use one CSV per evaluation condition. Each row is one fixed input brief.

Score these fields after reviewing model output:

1. `goal_fidelity` — recommendation addresses stated user goal.
2. `chart_appropriate` — selected chart fits task and variables.
3. `encoding_correct` — x/y, aggregation, grouping, filters, sorting match source evidence.
4. `source_fidelity` — no unsupported data claims or invented source facts.
5. `overall_rating` — overall usefulness.

Use reviewer IDs, record an error category, and explain disagreements in `review_comment`. Do not use either human-evaluation CSV for training.
