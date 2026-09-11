# Thesis folder — layout, references, and how to rebuild everything

This folder holds the written thesis. It is the single place where chapter text
and the bibliography live. Everything here is plain Markdown plus one BibTeX
file, so it can be moved into LaTeX or Overleaf without conversion work.

## 1. Where everything is

```text
docs/thesis/
├── README.md                     ← this file
├── OPEN_ITEMS_AND_TODO.md        ← what is still missing, in plain language
├── references.bib                ← THE central bibliography (only this one)
├── thesis.tex                    ← LaTeX entry point
├── chapters/                     ← the nine chapters, one Markdown file each
│   ├── Chapter_1_Introduction.md
│   ├── Chapter_2_Background.md
│   ├── Chapter_3_Related_Work.md
│   ├── Chapter_4_Dataset_Construction_and_Provenance.md
│   ├── Chapter_5_System_Design_and_Methods.md
│   ├── Chapter_6_Experimental_Setup_and_Evaluation_Protocol.md
│   ├── Chapter_7_Results.md
│   ├── Chapter_8_Discussion_Error_Analysis_and_Threats_to_Validity.md
│   └── Chapter_9_Conclusion_and_Future_Work.md
├── figures/                      ← figures for the written thesis (currently empty)
├── tables/                       ← exported tables, if any (currently empty)
├── chapters_edit/                ← scratch space for manual editing (currently empty)
└── proposal/                     ← the original proposal PDF and the old outline
```

## 2. References: one file, one key per source

**All references live in `docs/thesis/references.bib`. Do not create a second
`.bib` file.** When you move the thesis to Overleaf, upload exactly this file
and point `\bibliography{references}` at it.

The rules that the current file follows:

- every source has one entry and one stable key, for example `dettmers2023qlora`,
  `lewis2020rag`, `luo2021nvbenchsigmod`;
- the text cites with LaTeX commands, `\cite{key}` or `\cite{key1,key2}`;
- there are no duplicate keys and no duplicate DOIs;
- every key used in a chapter exists in the `.bib`, and every entry in the
  `.bib` is cited by at least one chapter.

Check all of that at any time:

```bash
python experiments/scripts/build_chapter_reference_lists.py --check
```

The end of every chapter has a section **"References Used in Chapter X"**. That
section is **generated**, not hand-written. It is rebuilt from `references.bib`
and from the `\cite{...}` commands in that chapter, so it can never drift out of
sync with the text. Regenerate all nine at once with:

```bash
python experiments/scripts/build_chapter_reference_lists.py
```

If you add a citation to a chapter, add the entry to `references.bib` first,
then run the command above.

## 3. Moving to LaTeX

The Markdown was written so that the conversion is mechanical:

- headings are `#`, `##`, `###` and map directly to `\chapter`, `\section`,
  `\subsection`;
- citations are already LaTeX (`\cite{...}`), so nothing has to be rewritten;
- display equations are already in `\[ ... \]` form;
- inline code is in backticks and becomes `\texttt{...}`;
- tables are GitHub-flavoured Markdown and convert cleanly to `tabular`.

A direct route with Pandoc, one chapter at a time:

```bash
pandoc docs/thesis/chapters/Chapter_7_Results.md --from gfm --to latex --output chapter7.tex
```

Then include the generated `.tex` files from `thesis.tex` and drop the generated
"References Used in Chapter X" sections, because LaTeX builds one bibliography
at the end from `references.bib`.

## 4. Where the numbers in Chapters 6–8 come from

No number in the thesis is typed by hand. Everything is read from generated
artifacts:

| What | Where |
| --- | --- |
| Per-run metrics | `experiments/outputs/final/dashboard_v4/<model>/<method>/seed_<seed>/metrics_auto.json` |
| Per-item outcomes | the same folders, `eval_per_item.jsonl` |
| Raw model responses | the same folders, `predictions*.jsonl` |
| Run provenance (git, config, training hashes, hardware) | the same folders, `manifest.json` |
| Aggregated comparison tables | `experiments/results/final/dashboard_v4/` |
| Paired statistics, three-level chart re-scoring, failure modes | `experiments/results/final/dashboard_v4/statistics/` |
| Figures F1–F5 | `experiments/results/final/dashboard_v4/figures/` |

Rebuild the statistics and the diagnostics that Chapters 6–8 cite (no GPU, no
network, about a minute):

```bash
python experiments/scripts/analyze_final_results.py --dataset dashboard_v4
```

Rebuild the aggregated tables and the figures:

```bash
python experiments/scripts/aggregate_results.py --dataset dashboard_v4
python experiments/scripts/make_figures.py --dataset dashboard_v4
```

## 5. What is versioned in Git

`experiments/outputs/` stays out of Git: it holds the raw predictions and the
adapters and is far too large. `experiments/results/final/` **is** versioned —
43 files, about 2.4 MB — because the thesis cites those tables and figures as
evidence and a reader must be able to open them. `experiments/results/packages/`
and any `experiments/*.zip` stay out.

If `git status` does not show the files under `experiments/results/final/`,
check the negation block in `.gitignore`; Git cannot re-include a file inside a
directory that was excluded with a trailing slash, which is why the pattern is
written as `experiments/results/**` plus explicit re-inclusions.

## 6. Conventions used in the text

- **Evidence class on every metric.** Agreement with the frozen reference
  recommendation is called an internal diagnostic, never evidence about
  visualization quality.
- **Configuration versus manifest.** A configuration file states intent; a run
  manifest states fact. Where they disagree, the chapters report the manifest
  and say so.
- **Seeds are not extra data.** Three seeds are three repetitions of the same
  274 items. They are shown individually; no confidence interval is computed
  across three values, and nothing is computed for a single-seed condition.
- **Unmeasured means unmeasured.** Human ratings, the independent
  chart-effectiveness scorer, retrieval relevance and top-3 agreement are named
  as missing wherever they would otherwise be expected.
