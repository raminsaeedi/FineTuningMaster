# Human Evaluation — Scientific Protocol

Defensible protocol for the human evaluation that is the **validity anchor** for claims
about usefulness, actionability, and perceived dashboard-design quality. Consolidates
`human_eval_plan.md` and the implemented infrastructure in `src/evaluation/human/*`.

> **Claim gate.** No claim about usefulness, actionability, or real dashboard-design
> quality may be made until ratings are collected with acceptable inter-rater reliability.
> As of this writing **no ratings are on disk** — such claims are therefore not yet supported.

## Design
- **Blind rating.** Raters see the brief and a generated output with **no method label**
  and no identifying order (`src/evaluation/human/streamlit_app.py`,
  `assignment.py`). Method identity is never shown.
- **Same items across all methods.** Each evaluation item is rated for every method
  (A/B/C/D) so comparisons are **paired**. Assignment is balanced/blind
  (`assignment.py`, BIBD-style).
- **Replication.** Target **≥ 3 ratings per output**. Minimum fallback 2/output only if
  rater supply is constrained (report which was used).
- **Items.** Draw from the independent `benchmark_v1` (and, if needed, a stratified
  synthetic sample), so quality is judged on realistic, non-circular briefs. Record which
  items are `real_public` vs `realistic_manual` (evidence strength).

## Rubric (1–5 Likert, anchored)
Six dimensions (`src/evaluation/human/rubric.py`): `chart_appropriateness`,
`layout_quality`, `styling_accessibility`, `interaction_design`, `rationale_quality`,
`overall_usefulness`. Each dimension carries explicit 1/3/5 anchors shown in the UI. A short
**pilot** (2–3 raters, ~5 items) is run first to calibrate anchors and surface ambiguity;
pilot ratings are excluded from the final analysis.

## Statistics
- **Inter-rater reliability:** Krippendorff's α, ordinal, per dimension
  (`src/evaluation/human/irr.py`). Report α with the sample size.
- **Omnibus + pairwise:** Friedman across the four methods (`stats/friedman.py`), then
  Wilcoxon signed-rank with **Holm** correction (`stats/wilcoxon_holm.py`) per dimension.
- **Effect sizes:** paired rank-biserial / Cohen's d_z (`stats/effect_size.py`) — appropriate
  for ordinal/paired Likert data.
- **Confidence intervals:** bootstrap (`stats/bootstrap_ci.py`) on per-dimension means.
- **Optional LLM-judge** may be reported **only** as supportive evidence and **only** if it
  correlates with human ratings (report Spearman ρ); it never substitutes for humans.

## Reporting rules & limitations
- Report per-dimension means with CIs, the Friedman/Wilcoxon+Holm outcomes, and α.
- **If IRR is low** (e.g. α below a pre-registered threshold), state it prominently and
  **downgrade** the strength of any human-based claim accordingly — do not suppress it.
- Flag small n; do not over-generalize from a small item/rater pool.
- Keep human-evaluation claims in their own evidence tier (see the claim policy in
  `scientific_dataset_validity_implementation_plan.md`); never merge them with synthetic
  diagnostics or independent-L1 numbers.

## Current status
Infrastructure implemented (rubric, blind assignment, Streamlit app, storage, Krippendorff
α); assignment built for 50 items; **0 ratings collected**. Next step is a pilot, then the
full/recommended collection (e.g. 40 items × 4 methods × 3 raters), before any usefulness
claim is written.
