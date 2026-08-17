# Human Evaluation Plan

Human evaluation is the validity anchor for RQ2. It compares exactly four
methods from one fixed dataset, model and seed:

| Method | Meaning |
|---|---|
| A | Prompt-only |
| B | RAG |
| C | QLoRA |
| D | QLoRA + RAG |

The final Professor run layout is:

```text
experiments/outputs/final/<dataset>/<model>/<A-D>/seed_<seed>/predictions.jsonl
```

The builder resolves all four paths automatically. It never combines methods
from different models, seeds, dataset versions or incompatible test sets.

## Fixed final study

For `dashboard_v4`, the canonical human-evaluation IDs are the existing
40-row `data/frozen/dashboard_v4/human_eval_test_items_40.csv`. Briefs come
only from `data/frozen/dashboard_v4/test.jsonl`; Train and Validation are not
used. The builder verifies all 40 IDs in the test file and in A/B/C/D
predictions before writing a study.

The recommended final design is 40 items × 4 methods × 3 independent ratings
per output = 160 rating units and 480 ratings. Six raters receive a balanced
80 ratings each. Smaller or otherwise different designs are marked
`study_type: pilot` and are not automatically pooled with a final study.

Each study is isolated at:

```text
experiments/results/human_eval/<dataset>/<model>/seed_<seed>/
```

It contains `study_manifest.json`, `items.jsonl`, `assignment.json`,
`ratings/`, and `analysis/`. The manifest records source prediction paths and
hashes, run IDs/config hashes, dataset and KB hashes, item IDs, rubric hash,
assignment settings, and expected counts. Existing studies are never
overwritten. Analysis refuses to use ratings if source predictions or run
metadata changed after study creation.

## Rubric and blindness

The rubric remains fixed at six 1–5 Likert dimensions:

- `chart_appropriateness`
- `layout_quality`
- `styling_accessibility`
- `interaction_design`
- `rationale_quality`
- `overall_usefulness`

The Streamlit app shows only the dashboard brief, anonymous recommendation,
the six dimensions, 1/3/5 anchors, optional comment, and progress. Method,
model, seed, automatic metrics and gold/reference recommendations are not
shown. Ratings append immediately to one file per rater, so stopping and
restarting resumes from the next unfinished unit.

## Analysis

Before final statistics, `rating_completion.json` checks the expected counts,
three distinct raters per output, six integer scores in 1–5 per rating,
duplicates, unknown items/methods, and missing units. Final analysis fails on
any incomplete or invalid study. `--allow-incomplete` exists only for pilot or
debug analysis.

The analysis writes:

- Krippendorff's ordinal α separately for all six dimensions;
- A/B/C/D mean and standard deviation for each dimension and an explicit
  `composite_mean` (the mean across dimensions; distinct from
  `overall_usefulness`);
- per-dimension and composite paired Friedman tests;
- pairwise Wilcoxon signed-rank tests with Holm correction;
- paired rank-biserial and Cohen `d_z` effects and bootstrap CIs for paired
  differences;
- `per_item_scores.csv` with one row per item × method;
- `human_chart_acceptability.json`, derived from
  `chart_appropriateness >= 4` per rater and majority aggregation, using
  Cochran-Q and exact McNemar-Holm tests.

Human chart acceptability is a supportive human measure, not objective chart
correctness.

## Operational workflow

1. Professor experiments finish
2. Build one fixed dataset/model/seed human study
3. Launch Streamlit
4. Collect ratings
5. Compute statistics
6. Find final result files

Build:

```bash
python experiments/scripts/build_human_eval.py \
  --dataset dashboard_v4 \
  --model qwen3_8b \
  --seed 42 \
  --n-items 40 \
  --n-raters 6 \
  --ratings-per-output 3
```

Launch rating UI:

```bash
python experiments/scripts/run_human_eval.py \
  --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8b/seed_42
```

Compute statistics after all ratings:

```bash
python experiments/scripts/compute_irr.py \
  --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8b/seed_42
```

Final files are under:

```text
experiments/results/human_eval/dashboard_v4/qwen3_8b/seed_42/analysis/
```

For a pilot, change `--n-items`, `--n-raters` or `--ratings-per-output`; the
result is explicitly marked `study_type: pilot`.
