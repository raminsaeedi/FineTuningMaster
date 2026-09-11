# What is still missing — a plain-language checklist

Updated 2026-09-11 after the new Qwen3-8B prompt-only and RAG runs arrived.
This file is deliberately simple. It lists what is finished, what is missing,
and what you (or another assistant) have to do to close each gap. No jargon
where a plain word works. Nothing here is hidden in the chapters.

---

## 1. The short version

The nine chapters are written and complete. Every number in them comes from a
real file in the repository. Nothing is invented.

**The experiment matrix is now complete at the primary seed.** All four models
have all four methods (A, B, C, D) with seed 42, and two of them also have
seeds 43 and 44. Thirty-two runs, all clean.

Three things are **missing from the research itself**, and the chapters say so
openly:

1. **Human evaluation** — nobody has rated any output yet.
2. **The independent chart-effectiveness scorer** — designed, not programmed.
3. **Seeds 43 and 44 for the two big models** — only seed 42 exists there.

Two more things are **decisions you still have to make** (Section 5 below).

---

## 2. Missing experiments — what and why

### 2.1 Human evaluation (the biggest gap)

**What is missing.** Nobody has looked at the generated dashboards and said
whether they are good. The folder `experiments/results/human_eval/` contains the
list of items and the rater assignment, but the `ratings/` folder is empty.

**Why it matters.** The whole thesis can currently only say "the model matched
our reference answer X % of the time". It cannot say "the recommendation was
useful". Chapter 6, Section 6.8 explains this and Chapter 9 repeats it.

**What you need.** 40 items × 4 methods × 3 ratings = 480 ratings, from 6
people, 80 ratings each. Six questions per output, each scored 1 to 5: chart
appropriateness, layout, styling/accessibility, interactions, rationale quality,
overall usefulness. Raters must not see which method produced the output.

**How to run it.** Everything is already programmed:

```bash
python experiments/scripts/build_human_eval.py --dataset dashboard_v4 --model qwen3_8_27b --seed 42 --n-items 40 --n-raters 6 --ratings-per-output 3
python experiments/scripts/run_human_eval.py --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42
python experiments/scripts/compute_irr.py --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42
```

**If you cannot find 6 raters**, run a smaller pilot. The script marks it
automatically as `study_type: pilot`. A pilot is still much better than nothing,
but you must call it a pilot in the text, not a full study.

**Three questions the human evaluation would answer immediately:**

1. Both the 8B and the 27B model replaced "pie" with "donut" on 28 of 36 items
   when RAG was switched on, because the knowledge base says "prefer donut over
   pie". The automatic metric counted all 28 as errors. A human can say whether
   donut was actually worse. Right now nobody knows, and this single substitution
   is the whole reason RAG looks bad at both scales.
2. Fine-tuning makes every model stop offering alternative charts completely
   (from 274 items with alternatives down to 0). Is a single recommendation
   better or worse for a user than a ranked list?
3. At 8B, fine-tuning pushes the model toward bar charts. Agreement goes up,
   class-balanced F1 goes down. Which behaviour does a reader prefer?

### 2.2 The independent chart scorer (called "L1" in older documents)

**What is missing.** A small piece of code. There is a table of "which charts
work well for which task" taken from published studies. The idea was: instead of
checking whether the model matched *our* answer, check whether the model picked
*any* chart that research says works for that task.

**Why it matters.** Right now the only reference is the one the project built
itself. That is circular. This scorer would break the circle — and it might
settle the donut question without needing human raters.

**Where things stand.** The gold table exists in `data/eval/`. Every result line
currently says `l1_covered = not_applicable`, which means "never checked".

**What to do.** Write the scorer, run it over the existing predictions (no GPU
needed, the predictions are already saved), and add one table to Chapter 7. Also
report how many items the table actually covers — if it only covers 60 of 274
items, say 60.

### 2.3 Runs that are still missing

| Missing thing | Why | What to do | Needs training? |
| --- | --- | --- | --- |
| Bigger output limit | 512 tokens was too small. OLMo was cut off on 272 of 274 answers; Qwen3-8B on 0 answers before fine-tuning and 69 after | Re-run inference with a larger limit, e.g. 1024 or 1536 | No |
| Same code version for all runs | The runs were made over several weeks on four machines. Of 40 comparisons, only 7 come from one recorded code version | Re-run inference everywhere from one Git commit | No — adapters can be reused |
| Qwen3-8B seeds 43, 44 | Never run | Train two more adapters and run A–D | Yes |
| Qwen3.8-27B seeds 43, 44 | Never run | Train two more adapters and run A–D | Yes |

**Priority order if you only have limited GPU time:**
1. Re-run everything with a bigger output limit and one fixed code version
   (no training needed — this fixes the two biggest problems at once).
2. Only then think about extra seeds.

Note which comparisons are already clean: the RAG-versus-prompt comparison at
8B and at 27B, and the "RAG on top of fine-tuning" comparison at 8B, 27B and
OLMo seeds 42/43, all come from one recorded code version. Those carry the main
findings. The fine-tuning-versus-prompt comparison is **not** clean anywhere,
which is the strongest reason to do the re-run.

---

## 3. Things that are done and verified — do not redo them

- All nine chapters are written and updated for the new Qwen3-8B runs.
- The experiment matrix is complete at seed 42 for all four models (32 runs).
- All 77 bibliography entries are real and were checked against the original
  source. Twenty-one fake or placeholder entries (things like
  `author = {Author, A. and others}`) were deleted from `references.bib`.
- Every `\cite{}` in the text has an entry in `references.bib`, and every entry
  is cited somewhere. No duplicates.
- The paired statistics (McNemar, Cochran's Q, Holm correction, bootstrap
  confidence intervals) are computed and saved under
  `experiments/results/final/dashboard_v4/statistics/`, now including a
  code-version label for **each individual comparison**, not just per block.
- Every numeric table in Chapter 7 was checked automatically against the
  artifacts: 32 + 32 + 16 + 32 + 32 + 32 + 16 values, zero mismatches.
- The old Llama 3.1 and Qwen3-14B mentions are removed from the chapters.
- `.gitignore` now allows `experiments/results/final/` to be pushed (small
  tables and figures only), while the huge raw outputs stay out.

---

## 4. Two problems that were found while writing — read these

These are not bugs in the text. They are real findings, and they are already
written into Chapters 6, 7 and 8. Mentioning them here so you are not surprised.

### 4.1 The strict schema check can never pass

The "strict schema validity" number is 0 % for every fine-tuned run. That is not
the model's fault. The strict check forbids extra keys, but the reference
answers themselves have 13 extra keys in the encoding object. I ran the check
against the 274 reference answers: **0 of 274 pass**. So any model that copies
the reference correctly is guaranteed to score 0 %.

The proof is saved at
`experiments/results/final/dashboard_v4/statistics/strict_schema_ceiling.json`.

**Decision needed:** either widen the strict schema so it accepts the reference
format, or delete this metric. Right now the chapters report it with a warning.

### 4.2 The main accuracy number was measuring two things at once

The top-1 accuracy metric first parses the answer into an object, then reads the
chart from it. If parsing fails, the item counts as a wrong chart. That mixes
"wrong chart" with "wrong format" and with "answer was cut off".

Two examples:

- Qwen3-1.7B prompt-only scores **0.36 %**. Reading the chart straight out of
  the raw text gives **86.9 %**. The model picked the right chart almost every
  time and only wrote `"encoding": "column_name"` instead of an object.
- Qwen3-8B fine-tuning looks like it **hurt** (−7.66 points). Reading the raw
  text shows it **helped** by +4.38 points, and separately cost 36 answers that
  were cut off by the 512-token limit. The sign of the result flips.

Chapter 7 therefore reports three levels for every run, and Chapter 8 explains
what changes. This was only possible because the raw model text was saved.

---

## 5. Decisions only you can make

### 5.1 The AI-use declaration

I did **not** write an AI-use statement, because I do not know your university's
rules. Before writing it, answer these four questions:

1. Which AI tools did you use, and what are their exact names and versions?
2. What did you use each one for? (Your note so far: correcting and improving
   text you wrote or reviewed, help with debugging, suggestions for technical
   solutions.)
3. Does your university require a formal AI declaration? If yes, which form or
   wording?
4. Where must it appear — title page, declaration page, appendix, methods note?

Tell me the answers and I will write it. Do not let me guess.

**Note:** the dataset itself used an LLM to generate six fields (users, context
summary, layout, styling, interactions, rationales), and `gpt-5.6-luna` is
recorded in the frozen manifest as the repair model. That is a *methodology*
fact and is already documented in Chapter 4. It is separate from the personal
AI-use declaration.

### 5.2 The experiment plan in `src/config/matrix/final.yaml`

That config file still lists `qwen3_14b` and `llama3_1_8b` as final models.
Neither was ever run. The chapters describe the four models that were actually
used and do not mention these two.

**I did not change the config file**, because if those two models were part of
the plan your professor approved, deleting them silently would be wrong.

Choose one:
- (a) The plan changed officially → update `final.yaml` to the four real models
      and note the scope change in your defence.
- (b) The plan still applies → those two models are outstanding work and the
      thesis is incomplete until they are run or a scope change is approved.

---

## 6. Decisions already taken (so you know where things went)

**Top-3 accuracy was removed from the thesis.** It was in an earlier draft. It
cannot be reported as an A/B/C/D comparison, because no fine-tuned condition of
any model produces more than one recommendation, so the metric only exists for
two of thirty-two runs. Rather than report a fragment that would immediately
raise the question "why only there?", the metric is gone from every chapter. The
underlying *behaviour* — fine-tuning makes models stop offering alternatives —
is still reported in Chapter 7, Section 7.2.5 and Chapter 8, Section 8.3.4,
because that is a real finding about the systems and not a metric artifact.

---

## 7. Smaller items worth cleaning up

| Item | Detail | Effort |
| --- | --- | --- |
| Figures for the thesis | `docs/thesis/figures/` is empty. Chapter 7 refers to tables only. Figures F1–F5 exist under `experiments/results/final/dashboard_v4/figures/` and can be copied in. They were regenerated after the new runs | Low |
| A confusion-matrix figure | The data is in `statistics/chart_confusions.csv`; a heatmap for the 8B and 27B models would make the donut finding in Section 8.3.3 immediately visible | Low |
| Knowledge-base hash | The hash in the repository and the hash in the runs differ only because of Windows line endings (CRLF vs LF). Chapter 6 explains it. Fix properly by normalising line endings before hashing | Low |
| Pinned model versions | Two of four models have a pinned revision; Qwen3-1.7B and Qwen3-8B do not. Pin them before any re-run | Low |
| `experiments/results001.zip` | 1.2 GB sitting in the repository folder. Now ignored by Git, but move it out of the project folder anyway | Low |
| Old outline | `docs/thesis/proposal/thesis_outline.md` describes an older plan (Qwen2.5-0.5B, a synthetic generator). Keep it for history, but do not use it as a guide — it is out of date | — |

---

## 8. If you hand this to another AI assistant

Give it this instruction:

> Read `docs/thesis/README.md` first, then this file. The nine chapters are in
> `docs/thesis/chapters/`. The single bibliography is
> `docs/thesis/references.bib`. Every number in Chapters 6 to 8 comes from
> `experiments/results/final/dashboard_v4/` and from the run folders under
> `experiments/outputs/final/dashboard_v4/`. Do not add a number that is not in
> one of those files. Do not add a reference that you have not verified against
> the real paper. If you add a citation, put the entry in `references.bib` and
> then run `python experiments/scripts/build_chapter_reference_lists.py`. If new
> runs arrive, run `python experiments/scripts/analyze_final_results.py
> --dataset dashboard_v4` first and update the tables from its output.

That is enough for another assistant to continue without breaking anything.
