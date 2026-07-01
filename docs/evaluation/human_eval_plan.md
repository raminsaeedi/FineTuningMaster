# Human Evaluation Plan

How the human evaluation (RQ2) is run on the four methods (A prompt-only, B RAG, C QLoRA, D FT+RAG).
This is the **plan/protocol**; ratings are not yet collected. It reuses the existing infrastructure in
`src/evaluation/human/` and `experiments/scripts/{build_human_eval,run_human_eval,compute_irr}.py`.

## 1. Role: the validity anchor
**Human evaluation is the main validity anchor for RQ2.** Claims about usefulness, actionability, and
overall dashboard recommendation quality require human ratings. Automatic metrics (L1–L3) and the
LLM-judge are supporting evidence only and cannot substitute for human ratings here.

## 2. Rubric (6 Likert dimensions, 1–5)
The existing rubric (`human/rubric.py`) is used unchanged:
- `chart_appropriateness` — do the recommended chart types fit the KPIs/tasks?
- `layout_quality` — clear, prioritized, scannable layout?
- `styling_accessibility` — sound, accessible color/contrast/formatting?
- `interaction_design` — useful, appropriate interactions for the audience?
- `rationale_quality` — correct, specific, principle-grounded justifications?
- `overall_usefulness` — overall usefulness to the target users?

Anchors for 1/3/5 are shown during calibration and in the app. The rubric must stay fixed once rating
starts (changing it mid-study breaks comparability).

## 3. Assignment (blind, balanced, paired)
Using `human/assignment.py`:
- **Method hidden** from raters (blind); only the brief + the system output are shown.
- **Shuffled task order** per rater so method order is unpredictable.
- **Same items across A/B/C/D** wherever possible (common-item intersection) → a **paired design** for
  all statistical tests.
- **Balanced load:** each output receives the configured number of independent ratings; each rater gets
  a fair share (greedy least-loaded assignment, fixed seed).

## 4. Study scale (three tiers)
| Tier | items | methods | ratings/output | total outputs | total ratings |
| --- | --- | --- | --- | --- | --- |
| Full | 100 | 4 | 3 | 400 | 1200 |
| **Recommended (target)** | **40** | **4** | **3** | **160** | **480** |
| Minimal fallback | 30 | 4 | 2 | 120 | 240 |

**The recommended target is 40 items with 3 ratings/output**, unless enough raters are available to run
the full 100-item study. The minimal tier (30×4×2) is the documented floor; below it, report results as
qualitative-leaning with explicit small-n / wide-CI caveats. Reduce `n_items` before reducing
ratings/output (3 ratings enable majority voting and more reliable α).

## 5. Pilot phase (before the main study)
- **10–15 items**, **2–3 raters** (≈80–180 ratings).
- Purpose: check **rubric clarity**, **time per rating**, **display quality** (brief + output
  rendering), and **rating variance** (do raters diverge wildly → rubric/calibration issue?).
- Outcome: refine anchors/instructions and confirm the per-rating time estimate **before** committing to
  the main tier. Pilot ratings are not pooled into the main analysis unless the rubric is unchanged.

## 6. Sampling & strata
The sample **must include the external real-brief set from Task 4**
([`data/eval/real_briefs/items.jsonl`](../../data/eval/real_briefs/items.jsonl)) as the
**highest-value, non-circular subset**. Synthetic test items may be included only as
supplementary/comparison items, clearly labeled. Document the strata for the sampled items:
- **domain** (e.g. retail, finance, HR, supply chain, marketing, operations),
- **implicit task coverage** (which `TaskType`s the briefs collectively exercise),
- **number of KPIs** (brief complexity),
- **source type:** real brief vs synthetic.
Stratify so the sample spans domains/tasks rather than a random draw dominated by one area.

## 7. Derived binary human-acceptability signal
A binary signal is derived (not a separate rubric item) named **`human_chart_acceptability`**
(equivalently `human_acceptable_chart_fit`):
- **Definition:** `human_chart_acceptability = (chart_appropriateness >= 4)` per (item, method, rater);
  aggregate to per (item, method) by **majority** of its raters (requires ≥3 ratings).
- **This is NOT objective chart correctness.** It is a supportive **human-acceptability** signal only;
  objective chart correctness is the (separate, non-circular) L1 layer. Report it as human-acceptability,
  never as gold correctness.

## 8. Analysis plan
- **Inter-rater reliability:** Krippendorff's α **per dimension** (ordinal) — `compute_irr.py`
  (`human/irr.py`).
- **Likert quality:** **Friedman** test across the four systems + **pairwise Wilcoxon signed-rank with
  Holm correction**, reported with paired effect sizes (rank-biserial, Cohen's d_z) and **bootstrap
  confidence intervals**. (Existing `compute_irr.py` runs these on the across-dimension aggregate;
  per-dimension reporting may need a later code extension — see status.)
- **Derived binary human-acceptability:** **Cochran-Q** omnibus + **pairwise McNemar with Holm** on the
  `human_chart_acceptability` vectors — **if implemented later** (Task 8b; reuses
  `stats/cochran_mcnemar.py`).
- **LLM-judge ↔ human:** **Spearman** correlation per dimension and overall — **only if per-item judge
  scores are later produced** (Task 8b). 
- Always report CIs, flag small n, and Holm-correct each pairwise family.

## 9. LLM-judge
The LLM-judge (`metrics/llm_judge.py`) is **supportive only**. It cannot replace human ratings. It may
be reported **only if it correlates sufficiently with human ratings** (Spearman ρ); otherwise it is
omitted from quality claims. It currently stores aggregate means only, so per-item scores must be
produced before any correlation is computed.

## 10. Workload estimate
Assuming **≈3 minutes per rating** (read brief + structured output, score 6 dimensions). **This is an
estimate**; the pilot will refine it (plausible range 2–4 min → ±33%).

| Tier | total ratings | total time | per rater (3 raters) | per rater (4) | per rater (6) |
| --- | --- | --- | --- | --- | --- |
| Full (100×4×3) | 1200 | ~60 h | ~20 h | ~15 h | ~10 h |
| Recommended (40×4×3) | 480 | ~24 h | ~8 h | ~6 h | ~4 h |
| Minimal (30×4×2) | 240 | ~12 h | ~4 h | ~3 h | — |
| Pilot (≈12×4×2) | ~96 | ~5 h | — | — | — |

## 11. Ethics & consent
- Raters are **informed** about the study purpose before participating.
- Ratings are **anonymized or pseudonymized** (rater IDs, not personal identities, in the data).
- **No sensitive personal data** is collected (ratings + optional free-text comments only).
- **Participation is voluntary** and raters may withdraw.

## 12. Implementation status (honest)
- ✅ Human-evaluation **infrastructure exists** (rubric, balanced blind assignment, Streamlit app,
  storage, Krippendorff α, Friedman + Wilcoxon/Holm, bootstrap).
- ❌ **Ratings not yet collected.**
- ❌ **Binary `human_chart_acceptability` analysis (Cochran-Q + McNemar/Holm) not yet implemented.**
- ❌ **Spearman LLM-judge↔human correlation not yet implemented** (needs per-item judge scores).
- ⚠️ **Per-dimension** Friedman/Wilcoxon reporting may require a later code extension (current tests run
  on the across-dimension aggregate; per-dimension α is already produced).

## 13. Remaining code gaps (Task 8b, when approved)
1. `stats/spearman.py` (+ export + unit test) for judge↔human correlation.
2. Extend `compute_irr.py`: derive `human_chart_acceptability` and run Cochran-Q + McNemar/Holm;
   add per-dimension Friedman/Wilcoxon.
3. Produce per-item per-dimension LLM-judge scores (extend `metrics/llm_judge.py` output) before any
   Spearman computation.
None of these are required to *start* collecting ratings (which capture the 6 Likert dimensions today).
