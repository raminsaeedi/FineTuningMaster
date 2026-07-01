# Thesis Writing Outline — Scientific Defensibility

Chapter structure and **claim-control** scaffolding for the thesis. This is an outline + guardrails
document, **not** thesis prose. Its purpose is to keep every written claim bounded to the evidence that
actually exists in the repository, to fence synthetic/internal diagnostics off from validity claims, and
to mark pending evidence honestly. (Document language: English, for project consistency; the final
thesis language can be decided separately.)

## How to use this document

1. Write each chapter from its **purpose / contents / supporting evidence / claims-allowed /
   claims-forbidden** block.
2. Before stating any quantitative result, locate it in the **Claim ↔ Evidence ↔ Status matrix** and use
   only the **permitted language** for that evidence level.
3. Fill numeric cells **only** from generated result artifacts; never from memory or hard-coded values.
   Where artifacts do not yet exist, use `[fill from <artifact> when generated]` and mark pending.

## Evidence ledger (verified by repository inspection)

**Available, inspectable artifacts (design/provenance/metrics code):**

- Sources & licenses: `docs/datasets/sources_table.md`, `docs/datasets/sources.bib`
  (also `docs/SOURCES.md`, `docs/sources.bib`).
- L1 human-effectiveness gold: `data/eval/human_effectiveness_gold.csv` (+ `data/eval/README.md`);
  task crosswalk: `data/eval/task_crosswalk.yaml`.
- External real briefs (first tranche, 10 items): `data/eval/real_briefs/items.jsonl`
  (+ `docs/datasets/real_briefs_provenance.md`).
- Training adapters mapping (designed, data not ingested): `docs/datasets/training_data_mapping.md`,
  `src/data_pipeline/builders/`.
- Evaluation protocol & human-eval plan: `docs/evaluation/evaluation_protocol.md`,
  `docs/evaluation/human_eval_plan.md`.
- Metrics/stats code: `src/evaluation/metrics/*`, `src/evaluation/stats/*`,
  `src/evaluation/reporting.py`.

**Pending / not present in the repository (must NOT be stated as results):**

- **No aggregated result files** in `experiments/results/` (empty) → numeric results are not yet
  inspectable; treat all result numbers as pending until `metrics.json` / `comparison_table.csv` /
  `final_report.md` are generated.
- L1 set-valued human-effectiveness **scorer** not implemented (only the gold table + design exist).
- L3 realism not implemented; Tableau Census not acquired.
- L4 human ratings not collected (infrastructure exists).
- Multi-seed: seeds 43/44 not run.
- External real-brief set is a 10-item first tranche (not yet 20–40).
- ChartGPT/nvBench/Quda not ingested (builders are stubs); Draco `hard.lp` legality check not implemented.

## Evidence layers (keep separated throughout the thesis)

- **Internal synthetic diagnostic** — accuracy against the synthetic generator's own gold (circular).
- **Independent L1** — chart/encoding correctness vs human-effectiveness gold (covered items + coverage).
- **L2** — schema / format / robustness.
- **L3** — dashboard realism (descriptive).
- **L4** — human evaluation (the usefulness/quality anchor).

## Chapters

### 1. Introduction & gap

- **Purpose:** motivate the task and state the gap + research questions + contributions.
- **Contents:** task (brief `{users, goals, KPIs, columns, constraints}` → structured multi-chart
  dashboard JSON); the gap (no public benchmark maps this input to this schema); RQ1a chart selection,
  RQ1b format/robustness, RQ1c grounding, RQ1d realism, RQ2 usefulness; contributions framed as
  _assembling/proposing/building_.
- **Evidence:** `evaluation_protocol.md`, `sources_table.md`.
- **Claims allowed:** "no existing benchmark covers this exact mapping"; "we assemble a non-circular
  evaluation".
- **Claims forbidden:** SOTA / proven superiority / proven usefulness.

### 2. Related work

- **Purpose:** position the work and justify the assembled-benchmark + non-circularity choice.
- **Contents:** chart/encoding effectiveness (Saket 2019, Kim & Heer 2018); NL→vis (nvBench/2.0, ChartGPT,
  Quda, NLV, VisEval); dashboard design (Bach patterns, Tableau Census); constraint models (Draco); LLM
  fine-tuning + RAG.
- **Evidence:** `docs/datasets/sources.bib`, `sources_table.md` (roles + licenses + rejected sources).
- **Claims allowed:** factual positioning; the gap argument.
- **Claims forbidden:** claiming reuse of sources whose license is `cite-and-ask` as if already vendored.

### 3. Dataset & provenance

- **Purpose:** document data origin, separation, and limits.
- **Contents:** synthetic generator **with explicit circularity disclosure** (fixed shared task→chart
  rule); the assembled benchmark; licenses/`cite-and-ask`; L1 gold + **coverage gaps**; task crosswalk;
  the **10-item** external real-brief set (extending); training builders (designed, not ingested).
- **Evidence:** `sources_table.md`, `human_effectiveness_gold.csv` (+README), `task_crosswalk.yaml`,
  `real_briefs/items.jsonl` (+ provenance), `training_data_mapping.md`.
- **Claims allowed:** train vs independent-eval separation; provenance is reproducible.
- **Claims forbidden:** synthetic gold = independent ground truth; full external set (it is 10 items).

### 4. Method

- **Purpose:** describe the system and its reproducibility.
- **Contents:** schema/data contract (`src/core/schemas.py`); methods A/B/C/D; QLoRA training; RAG
  retriever; train/infer decoupling; config-driven pipeline; reproducibility (deterministic splits,
  seeds, config/git hashes, `manifest.json`).
- **Evidence:** `src/` code; `src/utils/artifacts.py` (manifest), `src/data_pipeline/splits.py`.
- **Claims allowed:** design + reproducibility properties.
- **Claims forbidden:** performance claims (belong in Results).

### 5. Evaluation protocol

- **Purpose:** define what each layer can and cannot support.
- **Contents:** L1–L4 from `evaluation_protocol.md`; exact (corrected) metric definitions; statistics
  (Cochran-Q/McNemar+Holm; Friedman/Wilcoxon+Holm; paired effect sizes; bootstrap CIs); human-eval design
  (`human_eval_plan.md`).
- **Evidence:** `evaluation_protocol.md`, `human_eval_plan.md`, `src/evaluation/{metrics,stats}/*`.
- **Claims allowed:** scope of each layer; that synthetic = diagnostic.
- **Claims forbidden:** asserting metrics not implemented (L1 scorer, L3, Draco) are operational.

### 6. Results

- **Purpose:** report measured outcomes per layer, separating diagnostic from independent.
- **Contents:** sub-sections **(a) internal synthetic diagnostic**, **(b) independent L1**, **(c) L2**,
  **(d) L3**, **(e) L4** — each reporting numbers **only** from generated artifacts with CIs, else
  pending. Use placeholders, e.g. `[fill from experiments/results/final_report.md /
comparison_table.csv]`; mark L1 scorer, L3, L4, and seeds 43/44 as pending.
- **Evidence:** `experiments/results/*` (currently **absent** → all numeric cells are pending),
  `metrics.json` / `eval_per_item.jsonl` (produced per run by `reporting.py`).
- **Claims allowed:** L2 format/robustness differences **where stats + CIs support them**; report L1 as
  **covered accuracy AND coverage rate** once available.
- **Claims forbidden:** any numeric result not read from an artifact; usefulness claims (need L4);
  synthetic chart-accuracy as a validity claim; generalization from a single seed.

### 7. Error analysis

- **Purpose:** characterize failure modes.
- **Contents:** parse failures; schema-invalid cases; chart-selection error patterns (confusion matrix,
  once figures exist); robustness failures (paraphrase/missing-info); RAG grounding failures.
- **Evidence:** `eval_per_item.jsonl` (per-item flags), `metrics/{schema_compliance,robustness,
grounding,macro_f1}.py`.
- **Claims allowed:** descriptive patterns; frequency of failure types (from artifacts).
- **Claims forbidden:** causal/over-generalized explanations beyond the observed sample.

### 8. Threats to validity (defensibility core)

- **Purpose:** state every limitation explicitly.
- **Contents:** **Internal** — circularity (rule reproduction ≠ design quality, esp. method C).
  **External** — small n, single seed, narrow L1 coverage, 10-item external set, single small model
  (Qwen2.5-0.5B), domain skew. **Construct** — Likert subjectivity, lexical-proxy grounding, regex
  clarification heuristic, `human_chart_acceptability` ≠ objective correctness. **Statistical** —
  multiple comparisons (Holm), wide CIs at small n. **Reproducibility** — gitignored data rebuild,
  `cite-and-ask` licenses.
- **Evidence:** the pending-evidence list; protocol limitations; provenance caveats.
- **Claims allowed:** honest acknowledgment of each threat.
- **Claims forbidden:** dismissing a threat without justification.

### 9. Conclusion

- **Purpose:** summarize bounded contributions + future work.
- **Contents:** restate contributions **as supported by available evidence** (what is shown vs pending);
  future work (implement L1 scorer; acquire L3 census; run L4 human eval; seeds 43/44; extend external
  set; larger models; ingest external training data).
- **Claims allowed:** contributions phrased within the evidence ledger.
- **Claims forbidden:** concluding usefulness/superiority not backed by L4 + stats + adequate n.

## Claim ↔ Evidence ↔ Status matrix

| RQ / claim                         | Layer    | Evidence artifact                                           | Strength                                            | Status                                                            |
| ---------------------------------- | -------- | ----------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------- |
| RQ1a chart selection (independent) | L1       | `human_effectiveness_gold.csv` + (pending) scorer           | strong if covered                                   | **pending** (scorer unimplemented; report covered acc + coverage) |
| Chart accuracy vs synthetic gold   | internal | `metrics/topk_accuracy.py`, `metrics.json`                  | diagnostic only                                     | available-as-diagnostic; **numbers pending** (no result files)    |
| RQ1b format/robustness             | L2       | `metrics/{schema_compliance,robustness}.py`, `metrics.json` | moderate (synthetic)                                | code available; **numbers pending**                               |
| RQ1c grounding (RAG)               | L2       | `metrics/grounding.py`                                      | weak (lexical proxy; semantic opt-in; Draco absent) | partial; **numbers pending**                                      |
| RQ1d realism                       | L3       | Tableau Census (not acquired)                               | descriptive only                                    | **pending-data**                                                  |
| RQ2 usefulness/actionability       | L4       | `human/*`, `human_eval_plan.md`                             | the validity anchor                                 | **pending-ratings**                                               |
| Multi-seed generalization          | all      | seeds 42/43/44                                              | —                                                   | seed 42 only on host; 43/44 **pending**                           |

## Overclaim guardrails (apply everywhere)

- Synthetic chart accuracy is **diagnostic only**, never the main validity claim.
- Usefulness/actionability claims require **L4 human ratings** (with IRR + tests); none collected yet.
- L1 must report **both covered accuracy and coverage rate** (never accuracy alone).
- L3 realism is **descriptive, not proof of optimality**.
- **Single-seed results cannot support strong generalization**; label as preliminary.
- Always report **confidence intervals** where available; flag small n.
- **Mark pending evidence honestly**; do not imply unimplemented layers were run.

## Hedging-language conventions

| Evidence level                              | Permitted language                                      | Avoid                                    |
| ------------------------------------------- | ------------------------------------------------------- | ---------------------------------------- |
| Pending / not collected                     | "not yet evaluated", "planned", "we will"               | any result verb                          |
| Internal synthetic diagnostic               | "reproduces the generator rule", "diagnostic indicator" | "selects better charts", "more accurate" |
| L3 descriptive                              | "resembles", "distributionally close to"                | "optimal", "more correct"                |
| Single seed / small n                       | "preliminary", "suggests", "indicative"                 | "demonstrates", "generalizes", "proves"  |
| Stats + CIs + adequate n + independent gold | "significantly", "shows", "demonstrates"                | unqualified universal claims             |
| L4 human + IRR + tests                      | "more useful (human-rated)", "rated higher"             | usefulness claims without L4             |

## Pending-evidence list (mark each in the text)

- L1 human-effectiveness scorer (designed, unimplemented).
- L3 realism + Tableau Census acquisition/mapping.
- L4 human ratings (infrastructure exists; none collected).
- Aggregated result artifacts in `experiments/results/` (currently absent).
- Seeds 43/44 (single-seed only otherwise).
- External real-brief set beyond the 10-item first tranche.
- ChartGPT/nvBench/Quda ingestion; Draco `hard.lp` legality check; semantic grounding by default.

## Thesis-writing checklist (pre-submission)

- [ ] Every quantitative claim cites a generated artifact (no memory / no hard-coded numbers).
- [ ] Internal-synthetic results are fenced and labeled diagnostic, never a validity claim.
- [ ] L1 reports covered accuracy **and** coverage rate.
- [ ] L3 is phrased descriptively (no optimality claims).
- [ ] No usefulness/actionability claim without L4 human ratings + IRR + tests.
- [ ] CIs reported where available; small-n / single-seed caveats stated.
- [ ] Holm correction noted for each family of pairwise tests.
- [ ] Circularity of the synthetic gold disclosed (esp. for method C).
- [ ] `cite-and-ask` licenses and coverage gaps acknowledged.
- [ ] Reproducibility (deterministic splits, seeds, config/git hashes, manifest) described.
- [ ] Every pending item from the list above is marked pending in the relevant chapter.
