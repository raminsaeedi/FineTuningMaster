# Evaluation Protocol

How the four methods (A prompt-only, B RAG, C QLoRA, D FT+RAG) are evaluated. The evaluation is an
**assembled, non-circular benchmark**: no single public benchmark maps
`{users, goals, KPIs, columns, constraints} → multi-chart dashboard` in this schema, so evidence is
assembled across four layers, each scoped to what it can and cannot support.

## Assembled-benchmark rationale
- Chart/encoding correctness is judged against **independent human-effectiveness** results, not the
  project's own generator (whose `task_type→chart_type` rule is shared by train and test).
- Format/robustness is measured on held-out synthetic + perturbation sets.
- Realism is compared to a real-world dashboard corpus (descriptive only).
- Usefulness/quality is judged by **humans** — the validity anchor.

## Strict non-circularity / separation rule
> **No dataset artifact, label set, or label-generation lineage is used both for
> training/augmentation and final independent evaluation gold.**

Training/augmentation = synthetic generator + (future) ChartGPT/nvBench/Quda. Independent eval gold =
L1 human-effectiveness table + the external real-brief set + human ratings. The synthetic **test** split
is *internal* (circular for chart choice) and is used only for L2 format/robustness, never as a primary
chart-quality claim.

## Research-question → layer map
| RQ | Question | Layer |
| --- | --- | --- |
| RQ1a | Better chart selection? | **L1** |
| RQ1b | Better schema/format & robustness? | **L2** |
| RQ1c | Better grounding (RAG)? | **L2** (grounding sub-metric) |
| RQ1d | More realistic dashboards? | **L3** |
| RQ2 | More useful dashboards (human)? | **L4** |

## Layer × dataset matrix
| Layer | Synthetic held-out test | Perturbation sets (paraphrase / missing-info) | L1 human-effectiveness gold | Real-brief external set | Tableau Census | Human ratings |
| --- | :--: | :--: | :--: | :--: | :--: | :--: |
| L1 chart correctness | covered items only | – | **gold** | (where covered) | – | – |
| L2 format/robustness/grounding | ✓ | ✓ | – | ✓ (format/grounding) | – | – |
| L3 realism | ✓ (generated structure) | – | – | ✓ (generated structure) | **reference** | – |
| L4 quality/usefulness | sample | – | – | ✓ (sample) | – | **ratings** |

---

## L1 — Chart-type correctness vs human-effectiveness gold (RQ1a)
- **What is tested:** Is the model's *primary* chart per KPI within the **human-effective set** for the
  KPI's gold `(task_type, data_shape)`?
- **Data:** [`data/eval/human_effectiveness_gold.csv`](../../data/eval/human_effectiveness_gold.csv)
  (Saket 2019 + Kim & Heer 2018), applied to **covered** items; lookup keyed by the gold `task_type`
  (+ derived `data_shape`). Independent of the generator.
- **Metric:** set-valued **Top-1 (membership) accuracy** on covered items + `L1_coverage_rate` +
  per-`task_type` accuracy. Parse-failure on a covered item = wrong; uncovered KPIs are **excluded and
  reported**, not counted wrong.
- **Interpretation:** higher = more often picks a human-effective chart. **Only covered cells are
  evidence**; report coverage alongside accuracy.
- **Limitation:** narrow coverage (≈5 chart types; `correlation/ranking/deviation/distribution/comparison`
  only — no `trend/composition/part_to_whole/flow` or exotic charts); `data_shape` is inferred; both
  sources are cite-and-ask; small n → wide CIs.
- **Statistical test (binary, paired):** Cochran's Q omnibus across the 4 methods
  (`stats/cochran_mcnemar.py::cochran_q`); exact **McNemar + Holm** pairwise
  (`pairwise_mcnemar`); **paired accuracy difference with bootstrap CI**
  (`stats/bootstrap_ci.py`); optional **McNemar odds ratio** (from the discordant counts `b`, `c`).
  *Do not use Cohen's d_z / rank-biserial here* — those are for ordinal/continuous scores (L4).
- **Implementation status:** **designed, not implemented.** (The existing
  `metrics/topk_accuracy.py::top_1_accuracy` compares to the *single synthetic* gold — that is the
  internal/circular check below, **not** this L1 scorer.)

## L2 — Schema / format / robustness (+ grounding) (RQ1b, RQ1c)
- **What is tested:** Does the output parse, validate against the full schema, and carry non-empty
  required content? Is it stable and still correct under paraphrase? Does it ask for clarification when
  the brief is under-specified? (RAG) Are rationale claims supported by retrieved context?
- **Data:** synthetic held-out **test** split + `test_paraphrased` + `test_missing_info`; retrieved KB
  passages for grounding. Also applicable to the real-brief set for parse/schema/grounding (no chart gold needed).
- **Metric (exact keys):**
  - `metrics/schema_compliance.py`: `json_parse_rate`, `schema_validity_rate` (full Pydantic, strict,
    on the raw extracted object — no lenient enum repair), `required_keys_rate` (lenient, presence-only),
    `completeness_score` (mean fraction of required keys present **and non-empty**), `field_coverage`.
  - `metrics/robustness.py` (`compute_robustness`): `paraphrase_consistency`, `paraphrase_accuracy`,
    `paraphrase_accuracy_delta`, `missing_info_clarification_rate`, `missing_info_schema_rate`.
  - `metrics/grounding.py`: `supported_claim_rate` / `unsupported_claim_rate` with a `mode` field
    (`"semantic"` or `"lexical_proxy"`).
- **Interpretation:** format reliability + robustness. Read grounding **only** with its `mode`: a
  `lexical_proxy` number is a word-overlap proxy, not a faithfulness judge. `missing_info_schema_rate`
  high is **not** good (confident output on under-specified input).
- **Limitation:** synthetic content is circular for chart *choice* (so `paraphrase_accuracy` uses the
  single synthetic gold and inherits that circularity); `missing_info_clarification_rate` is a **regex
  heuristic** for clarification language; consistency ≠ correctness; grounding defaults to lexical proxy.
- **Statistical test:** binary rates → Cochran's Q + McNemar/Holm (`stats/cochran_mcnemar.py`);
  continuous (`completeness_score`, consistency) → Friedman (`stats/friedman.py`) + Wilcoxon signed-rank
  + Holm (`stats/wilcoxon_holm.py`), paired effect sizes (`stats/effect_size.py`); bootstrap CIs
  (`stats/bootstrap_ci.py`).
- **Implementation status (nuanced):**
  - parse / schema-validity / completeness: **implemented** (`schema_compliance.py`, inspect-confirmed).
  - robustness (paraphrase accuracy/consistency/delta, missing-info clarification/schema):
    **implemented** (`robustness.py`) — with the caveats above (regex clarification heuristic;
    paraphrase accuracy uses single synthetic gold).
  - grounding: **lexical proxy implemented**; **semantic implemented but opt-in**
    (`GROUNDING_SEMANTIC=1` + `sentence-transformers`); **Draco `hard.lp` legality check NOT implemented
    — pending.**

## L3 — Dashboard realism vs Tableau Census (RQ1d)
- **What is tested:** Do generated dashboards' **structural distributions** (chart-type mix, view/KPI
  count, interaction-type mix) resemble real dashboards?
- **Data:** generated outputs (all methods, synthetic + real briefs) vs the **Tableau Public Census**
  (arXiv:2306.16513; paper CC-BY-4.0, **OSF corpus terms to confirm**). See
  [`sources_table.md`](../datasets/sources_table.md).
- **Metric:** distribution distance — **Jensen–Shannon divergence** and **total variation** over
  chart-type and interaction-type mixes; view/KPI-count distribution comparison; (optional χ²
  goodness-of-fit). Report with bootstrap CIs.
- **Interpretation:** **Dashboard realism is descriptive structural evidence, not evidence of design
  optimality.** Lower divergence = structurally closer to real dashboards.
- **Limitation:** the census reflects **popularity, not effectiveness**; mapping census node-link graphs
  → this project's `DesignOutput` structure is approximate; corpus/licensing pending.
- **Statistical test:** bootstrap CIs on the divergences (`stats/bootstrap_ci.py`); optional permutation
  test for distribution difference. Treated as descriptive, not a significance-driven claim.
- **Implementation status:** **not implemented; Tableau Census not acquired/mapped.** L3 is
  **pending-data**.

## L4 — Human quality / usefulness (RQ2) — the validity anchor
- **What is tested:** Across six rubric dimensions (1–5 Likert): `chart_appropriateness`,
  `layout_quality`, `styling_accessibility`, `interaction_design`, `rationale_quality`,
  `overall_usefulness` (`human/rubric.py`).
- **Data:** a stratified sample of generated outputs including the **real-brief external set** (Task 4)
  plus synthetic items, each rated by ≥2 raters via `human/` (assignment, app, storage).
- **Metric:** per-dimension rubric **means**; **Krippendorff's α** for inter-rater reliability
  (`human/irr.py::krippendorff_alpha`, `level="ordinal"` for Likert); optional LLM-judge score
  (`metrics/llm_judge.py`).
- **Interpretation:** **Human ratings are the main validity anchor for claims about usefulness,
  actionability, and overall dashboard recommendation quality.** The LLM-judge may be used **only as
  supportive evidence if it correlates with human ratings** (report Spearman ρ; do not substitute it for
  humans).
- **Limitation:** **no ratings collected yet**; small sample; rater subjectivity; LLM-judge bias.
- **Statistical test:** Friedman omnibus (`stats/friedman.py`) + Wilcoxon signed-rank + Holm
  (`stats/wilcoxon_holm.py`) on rubric scores; paired effect sizes — **rank-biserial / Cohen's d_z**
  (`stats/effect_size.py`) — appropriate here (ordinal/continuous paired); Krippendorff α for IRR;
  Spearman for judge↔human.
- **Implementation status:** **infrastructure implemented** (rubric, assignment, IRR, storage,
  Streamlit app, LLM-judge); **no human ratings collected yet.**

---

## Statistics catalogue (tests → project files)
| Test / quantity | File | Function(s) | Used for |
| --- | --- | --- | --- |
| Cochran's Q (omnibus, binary, k≥3) | `stats/cochran_mcnemar.py` | `cochran_q` | L1 Top-1; L2 binary rates |
| McNemar (exact) + Holm (pairwise binary) | `stats/cochran_mcnemar.py` | `mcnemar_test`, `pairwise_mcnemar` | L1; L2 binary rates |
| Wilcoxon signed-rank + Holm | `stats/wilcoxon_holm.py` | `holm_correction` (+ Wilcoxon) | L2 continuous; L4 rubric pairwise |
| Friedman (omnibus, ordinal/continuous, k≥3) | `stats/friedman.py` | (see file) | L4 rubric; L2 continuous |
| Paired effect size (rank-biserial, Cohen's d_z) | `stats/effect_size.py` | `paired_rank_biserial`, `cohen_dz` | L4 (ordinal/continuous paired); L2 continuous |
| Cliff's delta (independent-sample effect size) | `stats/cliff_delta.py` | `cliffs_delta` | unpaired comparisons only |
| Bootstrap CI | `stats/bootstrap_ci.py` | (see file) | CIs on rates/diffs; L1 paired-acc diff; L3 divergences |
| Krippendorff's α (IRR, ordinal) | `human/irr.py` | `krippendorff_alpha` | L4 inter-rater reliability |

## Global reporting rules
- Always report **confidence intervals** (bootstrap) alongside point estimates.
- **Flag small n** (notably L1 covered subset and the L4 sample); do not over-generalize.
- Apply **Holm correction** across each family of pairwise tests.
- Report **multi-seed** results (seeds 42/43/44; 43 and 44 still pending) — means ± spread across seeds.
- Report **coverage gaps** explicitly (L1 uncovered task/chart types; L3 census mapping gaps).
- State **implementation status** for every reported metric (implemented / designed / pending-data).

## Honest limitations & pending items
- **L1 set-valued scorer not implemented** (designed in Task 2); the existing `top_1_accuracy` is
  synthetic-gold and circular — usable only as an internal L2-style check, never a primary chart-quality claim.
- **Draco `hard.lp` legality check not implemented** (planned validity filter, not present in
  `grounding.py` or elsewhere).
- **Semantic grounding is opt-in**; default runs report a **lexical proxy** — read the `mode` field.
- **L3 not implemented and Tableau Census not acquired** (license/OSF terms to confirm).
- **L4 has no ratings yet** — all usefulness/quality claims await human evaluation.
- `paraphrase_accuracy` and `missing_info_clarification_rate` carry method caveats (single synthetic
  gold; regex heuristic) — candidates for the Task 7 metric review.
