# NOT FOR TRAINING

The files in this directory are **independent evaluation gold** and must **never**
be used for training or augmentation:

- `l1_chart_effectiveness_v1.csv` — literature-derived chart effectiveness
  (Saket 2019; Kim & Heer 2018). Lineage is disjoint from the synthetic generator,
  so it can support independent chart-selection claims (L1). Coverage is partial;
  uncovered `(task_type, data_shape)` cells are excluded from L1 accuracy and the
  coverage rate is always reported.
- `real_briefs_v1.jsonl` — external `DashboardBrief` inputs with **no** chart
  labels. Used for format/schema/grounding checks and the human-rated sample.
  Each item carries `extra.not_for_training = true`.
- `real_briefs/items.jsonl` — the original external brief seed set.

Training must read only `data/frozen/dashboard_v2/train.jsonl` and `val.jsonl`
(see `src/config/data/dashboard_v2.yaml` → `not_for_training`).

`real_briefs_v1.jsonl` currently reuses the 10 verified external briefs as a seed
and should be expanded to the planned 30–40 items before final evaluation.
