# Memorization vs Generalization Protocol

How the thesis distinguishes **memorization** (repeating seen training examples/patterns)
from **generalization** (correct behaviour on unseen briefs, domains, and task
combinations). Companion to `scientific_dataset_validity_audit.md`.

## Why this matters
Fine-tuning on synthetic data risks the model learning the generator's fixed
`task_type → chart_type` mapping (see `rule_leakage_report.md`). A model can score high on
the synthetic test split by **reproducing** that mapping without any transferable design
skill. This protocol defines tests that separate the two.

## Test battery
1. **Exact & near-duplicate checks** — `experiments/scripts/check_dataset_leakage.py`
   (exact `item_id`/brief fingerprint + char-3gram Jaccard). Ensures no eval item is a copy
   of a training item. Status: implemented; active train vs eval is clean.
2. **Paraphrase consistency & accuracy** — `metrics/robustness.py` on `test_paraphrased`.
   Stable *and correct* output under meaning-preserving rewording indicates the model is not
   keyed to exact training strings. (Consistency alone can be "consistently wrong".)
3. **Missing-info behaviour** — `test_missing_info`: does the model degrade gracefully /
   ask for clarification rather than confidently emit a full schema on under-specified input?
4. **Held-out combination evaluation** — `experiments/scripts/analyze_generalization.py`:
   compute `(domain × task_type)` combinations present in **training** vs **evaluation**,
   identify **novel** combinations (in eval, not in train), and compare cached-prediction
   accuracy on **seen** vs **novel** combinations. A large seen≫novel gap is evidence of
   memorization; comparable accuracy is evidence of generalization. (This uses the synthetic
   gold, so it is an **internal diagnostic** — it measures generalization of the *generator
   mapping*, not real design quality.)
5. **Independent benchmark performance** — `eval_l1_independent.py` (covered items) and, once
   an approved inference pass exists, `benchmark_v1` scoring. Non-circular signal of whether
   chart choices transfer to independent, literature-backed acceptability.
6. **Human evaluation** — `human_eval_scientific_protocol.md`. The only evidence for
   usefulness/actionability/quality; required before any such claim.

## Reading the results
- Report **seen vs novel** accuracy with counts; flag small n.
- Never present synthetic seen/novel accuracy as evidence of real dashboard quality — it is a
  memorization probe on the generator mapping only.
- Generalization is **supported** only when: near-duplicate-clean **and** paraphrase accuracy
  holds **and** independent L1 / benchmark / human signals agree — not from the synthetic
  split alone.

## Artifacts
- `experiments/results/leakage_report.md` (dup/near-dup)
- `experiments/results/generalization_report.md` (seen vs novel combinations)
- `experiments/results/l1_independent_results.md` (independent chart selection)
- `experiments/results/rule_leakage_report.md` (why the synthetic split is circular)
