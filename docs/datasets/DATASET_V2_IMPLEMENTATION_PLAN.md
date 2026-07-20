# Frozen Dataset v2 — Implementation Plan

> **Scope:** this document is the implementation plan only. **No code, generator, or data
> is built yet.**
>
> **`dashboard_v2` is source-conditioned *synthetic* data — not a real, human-labeled
> dataset.** Briefs and reference recommendations are machine-generated, conditioned on
> documented source templates and published chart–task mappings. This constrains what it
> can support scientifically (see §7).

## Context

The current thesis dataset is synthetic `data/gold.jsonl` (600 items), split
deterministically into `data/processed/` as **491 train / 59 val / 50 test** via a
content-hash rule (`src/data_pipeline/splits.py::assign_split`, 0.8/0.1/0.1). The
labels (`task_type → chart_type`) come from a fixed rule in
`src/data_pipeline/synth_generator.py` (`TASK_CHART` dict). Because **train and test
share the same label-generation lineage**, any chart-choice accuracy measured on the
synthetic test split is **circular** and cannot support an independent quality claim
(this is already codified in `docs/evaluation/evaluation_protocol.md` — the
non-circularity rule).

**Goal of v2:** freeze a cleanly-separated dataset where (a) an enlarged, documented,
**source-conditioned** synthetic set is used for training + internal diagnostics, and
(b) genuinely **independent** evaluation gold (literature-derived chart effectiveness +
external real briefs with no chart labels) lives apart and is never trained on. The old
600 items are retained in a **legacy/smoke-test role only** and are not the thesis
training set.

**Decisions locked with supervisor:**
- Train source = **new source-conditioned generation** from documented source templates,
  public/scientific chart–task mappings, and realistic domain/KPI/column seeds. Do **not**
  append to the old generator; archive the old 600.
- L1 gold = **literature values** (Saket 2019 + Kim & Heer 2018), independent of the generator.

---

## 1. Target files

```
data/frozen/dashboard_v2/
├── train.jsonl                 # GoldItem records — TRAINABLE (gradient updates)
├── val.jsonl                   # GoldItem records — VALIDATION-ONLY (no gradient updates)
├── internal_test.jsonl         # GoldItem records — DIAGNOSTIC ONLY, never train
├── test_paraphrased.jsonl      # perturbation of internal_test — never train
├── test_missing_info.jsonl     # perturbation of internal_test — never train
├── dataset_card.md             # provenance, composition, intended use, limitations
├── generation_spec.yaml        # frozen, reproducible generation config
├── validation_report.md        # output of the validator (checks §5)
└── hashes.json                 # SHA256 + record counts per frozen file

data/eval/
├── l1_chart_effectiveness_v1.csv   # literature gold (Saket+Kim&Heer) — never train
└── real_briefs_v1.jsonl            # external briefs, NO chart labels — never train
```

Everything under `data/frozen/dashboard_v2/` and the two `data/eval/` files is
**write-once / frozen**: regeneration is allowed only by bumping the generator version
in `generation_spec.yaml` and re-emitting `hashes.json`.

## 2. Recommended sizes

| File | Target | Notes |
| --- | --- | --- |
| `train.jsonl` | **1200–1500** | main SFT corpus (gradient updates) |
| `val.jsonl` | **150–200** | validation-only — early stopping / loss curve, **no gradient updates** |
| `internal_test.jsonl` | **150–200** | circular diagnostic (L2 format/robustness) |
| `test_paraphrased.jsonl` | = internal_test | 1:1 derived, paired by `base_item_id` |
| `test_missing_info.jsonl` | = internal_test | 1:1 derived, paired by `base_item_id` |
| `l1_chart_effectiveness_v1.csv` | **30–50 rows** | (task_type, data_shape) → effective set |
| `real_briefs_v1.jsonl` | **30–40** | external; create fresh, reuse only if a verified file is found |

Split assignment reuses the deterministic content-hash rule (`assign_split`) so adding
items never reshuffles existing ones; pool size is tuned so counts land in the ranges
above. Distribution targets are enforced by the validator (§5), not by luck.

**Split naming vs stored value.** `assign_split` returns `"train" | "val" | "test"`.
To stay compatible with the current schema and loaders, the **stored `split` value keeps
those literals** — the third split is stored as `split="test"` even though its file is
named `internal_test.jsonl`. The file-name → stored-value mapping is fixed and documented
here:

| File | Stored `split` value |
| --- | --- |
| `train.jsonl` | `"train"` |
| `val.jsonl` | `"val"` |
| `internal_test.jsonl` | `"test"` |

(If a distinct `"internal_test"` literal is later preferred, it requires a code change to
`splits.py`/loaders and is called out as such — not assumed here.)

## 3. Exact schema per file

All models are the **existing** Pydantic contract in `src/core/schemas.py`
(`GoldItem`, `DashboardBrief`, `DesignOutput`, `KPIChartMapping`, `Rationale`).
`TaskType` (9 values) and `ChartType` (17 values) are the enums in the same file.

### 3.1 `train.jsonl` / `val.jsonl` / `internal_test.jsonl` — one `GoldItem` per line

```jsonc
{
  "item_id": "v2_<md5-8>",              // stable, content-derived
  "split": "train" | "val" | "test",    // stored literal stays "test" (assign_split output);
                                         //   internal_test.jsonl holds the "test" records
  "brief": {                             // DashboardBrief
    "item_id": "v2_<md5-8>",
    "users": "string",
    "goals": ["string", ...],
    "kpis": ["string", ...],
    "columns": [{"name": "string", "dtype": "datetime|numeric|categorical"}, ...],
    "constraints": "string | null",
    "extra": {                           // provenance lives here (extra=allow)
      "source_id": "string",             // documented source template id
      "source_ref": "string",            // citation / URL
      "generator_version": "v2.x"
    }
  },
  "recommendation": {                    // DesignOutput
    "context_summary": {"domain": "...", "audience": "...", "primary_goal": "...",
                         "data_literacy": "beginner|intermediate|advanced",
                         "update_frequency": "..."},
    "kpi_chart_mapping": [
      {"kpi": "string",
       "task_type": "<TaskType>",        // trend|comparison|composition|distribution|
                                         //   correlation|ranking|deviation|part_to_whole|flow
       "chart_type": "<ChartType>",      // line|bar|stacked_bar|grouped_bar|area|pie|donut|
                                         //   scatter|heatmap|histogram|box|kpi_card|table|
                                         //   gauge|sankey|treemap|map
       "alternatives": ["<ChartType>", ...],
       "encoding": {"x": "col", "y": "col", ...}}
    ],
    "layout": { ... }, "styling": { ... },
    "interactions": ["string", ...],
    "rationales": [{"claim": "string", "principle": "string"}, ...]
  }
}
```

> Note: `GoldItem` currently drops unknown top-level keys, so `source_id` is stored in
> `brief.extra` (which is `extra="allow"`). Optionally, a small backward-compatible code
> change adds a top-level `meta` field to `GoldItem` — deferred to implementation.

### 3.2 `test_paraphrased.jsonl` / `test_missing_info.jsonl` — one `GoldItem` per line

Same shape as §3.1, derived 1:1 from `internal_test.jsonl`, with two added keys in
`brief.extra` for pairing/robustness bookkeeping:

```jsonc
"extra": { ..., "variant": "paraphrased" | "missing_info", "base_item_id": "v2_<md5-8>" }
```

- `brief` is perturbed (`src/data_pipeline/perturbations.py`): paraphrase = deterministic
  meaning-preserving synonym swaps on framing words (domain nouns untouched); missing_info =
  drop `constraints` + last KPI + last column.
- `recommendation` (gold) is **unchanged** — it is the reference the perturbed input is
  scored against.

### 3.3 `data/eval/l1_chart_effectiveness_v1.csv`

Literature-derived, generator-independent. One row per (task_type, data_shape) cell.

| column | type | example |
| --- | --- | --- |
| `task_type` | TaskType enum string | `comparison` |
| `data_shape` | controlled string | `categorical_single` / `two_numeric` / `time_series` / `part_whole` |
| `effective_charts` | pipe-separated ChartType set | `bar\|table` |
| `source` | string | `Saket2019` / `KimHeer2018` |
| `confidence` | `high\|medium` | `high` |
| `notes` | string | coverage/interpretation note |

Set-valued gold: a prediction is correct if the primary chart ∈ `effective_charts`.
**L1 literature coverage is partial** — both sources cover only a subset of task types
(≈5: `comparison / correlation / ranking / deviation / distribution`) and a small set of
chart types; `trend / composition / part_to_whole / flow` and exotic charts are largely
uncovered, and `data_shape` is inferred. **Uncovered cells are always excluded from L1
accuracy (never counted as wrong), and `L1_coverage_rate` is reported alongside every L1
number.** L1 accuracy is only interpretable on covered cells with small-n caveats.

### 3.4 `data/eval/real_briefs_v1.jsonl` — one `DashboardBrief` per line, **no `recommendation`, no chart labels**

**Create 30–40 real briefs.** Reuse existing real briefs **only if** such a file is found
in the repo and verified (provenance + schema confirmed); otherwise author the full set.
Do not assume any pre-existing count of real briefs.

```jsonc
{
  "item_id": "rb_v1_###",
  "users": "string", "goals": ["string"], "kpis": ["string"],
  "columns": [{"name": "string", "dtype": "..."}],
  "constraints": "string | null",
  "extra": {"provenance_id": "rb_v1_###", "source": "Power BI / Tableau / Grafana ...",
            "domain": "string", "confidence": "high|medium"}
}
```

### 3.5 Non-data frozen artifacts

- `generation_spec.yaml` — generator version + seed; per-split target counts; domain list
  and target distribution; task_type/chart_type target distribution; list of source
  templates with citations; chart–task mapping references (Cleveland & McGill, Munzner,
  Saket, Kim & Heer); perturbation config. **This file documents the generation *process*;
  it does not guarantee bit-exact regeneration if any external/API step is involved.** The
  frozen artifacts are the ground truth, and their integrity is guaranteed by the SHA256
  values in `hashes.json` — reproducibility means "same committed files, verified by hash",
  not "re-derivable byte-for-byte from the spec".
- `dataset_card.md` — summary, composition tables, provenance, **intended use vs
  prohibited use** (§4), known limitations, links to `evaluation_protocol.md`.
- `validation_report.md` — human-readable pass/fail for every check in §5.
- `hashes.json` — `{file: {sha256, n_records}}` for all frozen files + `generator_version`
  + git commit at freeze time.

## 4. Training-eligible vs never-train

| File | May be used for training? |
| --- | --- |
| `data/frozen/dashboard_v2/train.jsonl` | **YES** — gradient updates (SFT) |
| `data/frozen/dashboard_v2/val.jsonl` | **VALIDATION-ONLY** — early stopping / loss curve, **no gradient updates** |
| `data/frozen/dashboard_v2/internal_test.jsonl` | **NO** — diagnostic only (circular for chart choice) |
| `data/frozen/dashboard_v2/test_paraphrased.jsonl` | **NO** |
| `data/frozen/dashboard_v2/test_missing_info.jsonl` | **NO** |
| `data/eval/l1_chart_effectiveness_v1.csv` | **NEVER** — independent eval gold |
| `data/eval/real_briefs_v1.jsonl` | **NEVER** — independent external eval |
| legacy `data/gold.jsonl` / `data/processed/*` | **NO** for thesis training — legacy/smoke only |

Enforced by: (a) `dashboard_v2.yaml` config exposes only `train` (gradient updates) + `val`
(validation-only) as training inputs; (b) a leakage check
(`src/data_pipeline/builders/leakage.py::fingerprint`) asserts zero `item_id` and zero
brief-fingerprint overlap between {train ∪ val} and {internal_test ∪ real_briefs}; failure
aborts the freeze.

## 5. Validation checks

Implemented by a new `validate_frozen_dataset.py` (writes `validation_report.md`, then
`hashes.json` only if all checks pass):

1. **Valid JSON** — every line parses.
2. **Pydantic schema validity** — each record loads as `GoldItem` (full contract), reusing
   `src/evaluation/metrics/schema_compliance.py::full_schema_valid` on the recommendation.
3. **Valid enums** — every `task_type ∈ TaskType`, `chart_type ∈ ChartType` (strict, no
   lenient repair); alternatives are valid ChartTypes.
4. **Non-empty required fields** — `users`, `goals`, `kpis`, `columns` non-empty; each
   `DesignOutput` required key present **and non-empty** (`completeness_fraction`).
5. **Duplicate detection** — no duplicate `item_id`; no duplicate brief fingerprint within
   or across splits (`leakage.fingerprint`).
6. **Balanced domain distribution** — each of the ~10 domains within a tolerance band of
   the `generation_spec.yaml` target; report the histogram.
7. **Balanced task_type / chart_type distribution** — all 9 task types present; no chart
   type exceeds a max-share cap; report both histograms.
8. **Deterministic hash split** — re-running `assign_split(item_id)` reproduces each
   record's stored `split`; no train/val/test leakage of the same `item_id`.
9. **SHA256 hashes** — compute and record SHA256 + record count for every frozen file into
   `hashes.json`; re-running the validator over the same frozen files reproduces identical hashes.

## 6. Scripts to add / modify (implementation phase — not now)

**New:**
- `src/data_pipeline/synth_generator_v2.py` — source-conditioned generator (templates +
  literature chart–task mappings + realistic seeds). Distinct from the legacy
  `synth_generator.py`.
- `experiments/scripts/generate_dataset_v2.py` — emits `train/val/internal_test.jsonl` +
  `generation_spec.yaml` into `data/frozen/dashboard_v2/`.
- `experiments/scripts/validate_frozen_dataset.py` — runs §5, writes `validation_report.md`
  + `hashes.json`.
- `data/eval/l1_chart_effectiveness_v1.csv` — hand-authored from Saket 2019 + Kim & Heer
  2018 (+ a small checker asserting enum validity and set membership coverage).
- `data/eval/real_briefs_v1.jsonl` — create 30–40 external briefs; reuse existing real
  briefs only if a file is found in the repo and verified.
- `src/config/data/dashboard_v2.yaml` — points training/eval at the frozen paths; exposes
  only `train` (gradient updates) + `val` (validation-only) for training. **Path note:** the
  current repo keeps Hydra data configs under `src/config/data/` (e.g.
  `src/config/data/dashboard_v1.yaml`), **not** `configs/data/`; the v2 config follows the
  existing location.

**Modify / reuse (no rewrite):**
- `experiments/scripts/build_perturbations.py` — point at `internal_test.jsonl`; reuse
  `perturbations.py` (paraphrase / drop_info) to emit the two variant files with
  `variant` + `base_item_id`.
- Reuse as-is: `dataset.py::compute_item_id`, `splits.py::assign_split`,
  `dataset.py::load_gold_items`, `builders/leakage.py::{fingerprint,filter_against}`,
  `schema_compliance.py::{full_schema_valid,completeness_fraction}`.
- `docs/datasets/real_briefs_provenance.md` — extend for the new external briefs.
- `data/frozen/dashboard_v2/dataset_card.md` + `docs/evaluation/evaluation_protocol.md`
  cross-links.

## 7. Scientific risk control

- **`dashboard_v2` is synthetic, not human-labeled.** Every brief and reference
  recommendation is machine-generated (source-conditioned). It is legitimate training and
  format/robustness-diagnostic material, but it is **not** independent evidence of design
  quality; only L1 (literature), the external real briefs, and L4 (human) provide that.
- **Circularity of synthetic data.** In v1 and v2 the synthetic `task_type → chart_type`
  labels are produced by a fixed generator rule shared by train and test. Measuring chart
  choice on any synthetic split therefore tests whether the model re-learned the
  generator, not whether it makes *good* charts. Consequently `internal_test`,
  `test_paraphrased`, `test_missing_info` are **internal diagnostic evidence only** —
  admissible for **L2** (JSON parse, schema validity, completeness, robustness/consistency)
  but **never** as a primary chart-quality claim.
- **Why L1 is required.** `l1_chart_effectiveness_v1.csv` encodes set-valued
  human-effectiveness results from **independent literature** (Saket 2019; Kim & Heer 2018),
  with a label lineage disjoint from the generator. Chart-selection claims (RQ1a) are made
  **only** against this table, on covered cells, with coverage reported.
- **Why real briefs are required.** `real_briefs_v1.jsonl` are **external** inputs with
  **no chart labels**, exercising format/schema/grounding on realistic, out-of-generator
  briefs and feeding the human-rated sample. They cannot leak into training.
- **Why human eval remains the validity anchor.** Usefulness/actionability (RQ2) is not
  decidable from synthetic labels or lexical proxies; it requires the L4 human rubric with
  inter-rater reliability. Automatic metrics are supportive only.
- **Enforcement:** the strict separation rule ("no artifact, label set, or
  label-generation lineage is used both for training/augmentation and independent eval
  gold", `evaluation_protocol.md`) is enforced by the leakage check in §5.5 and by
  `dashboard_v2.yaml` exposing only train/val as trainable.

## Verification (after implementation)

1. `python experiments/scripts/generate_dataset_v2.py` → frozen JSONL + `generation_spec.yaml`;
   confirm counts land in the §2 ranges.
2. `python experiments/scripts/build_perturbations.py` → both variant files, 1:1 paired.
3. `python experiments/scripts/validate_frozen_dataset.py` → `validation_report.md` all-pass;
   `hashes.json` written; re-run reproduces identical SHA256 (determinism).
4. Assert zero overlap between {train ∪ val} and {internal_test ∪ real_briefs} (leakage check).
5. Author + check `l1_chart_effectiveness_v1.csv` (enum-valid, ≥30 rows) and
   `real_briefs_v1.jsonl` (30–40, no `recommendation`/chart labels).
6. `pytest` (extend `test_splits`/`test_generator` for v2); run the import guard so
   data/eval code pulls no training deps.
7. Dry-run `load_gold_items` via `dashboard_v2.yaml` to confirm the trainer sees only
   train/val.
