# Fine-Tuning LLMs for Structured Dashboard Design Recommendations

Master's thesis project (Hochschule Ruhr West). The system takes a **dashboard
brief** (audience, goals, KPIs, data context) and produces a **structured JSON
design recommendation** (chart choices, layout, styling, interactions,
rationales). It compares four methods under one evaluation protocol:

| ID | Method | Status |
|----|--------|-------------|
| A | Prompt-only | ✅ implemented |
| B | RAG (TF-IDF over guideline KB) | ✅ implemented |
| C | Fine-tuning (QLoRA) | ✅ implemented |
| D | Fine-tuning + RAG | ✅ implemented |

Primary model: **Qwen2.5-0.5B-Instruct**.

## Architecture

Everything is config-driven (Hydra), with pluggable components behind a registry
and a shared Pydantic data contract. All four methods implement one interface —
`generate(brief) -> GenerationResult` — so evaluation is written once and works
for every method.

```
configs/              Hydra config groups (model, method, training, data, eval, experiment)
src/
  core/               schemas (Pydantic), registry, interfaces, prompts, constants
  data/               data-loading CODE (not the datasets): hash splits, gold loading, formatter, perturbations
  models/             HuggingFace causal-LM wrapper (import-safe)
  methods/            A/B/C/D — all behind the METHODS registry, one generate() contract
  retrievers/         TF-IDF retriever over the guideline KB (RETRIEVERS registry)
  training/           QLoRA SFT trainer (the ONLY place importing peft/trl/bitsandbytes)
  inference/          JSON post-processing + cached/resumable batch runner
  evaluation/         metrics/ (schema, top-k, macro-F1, latency, robustness, grounding),
                      stats/ (Friedman, Wilcoxon+Holm, Cliff's δ, Cochran/McNemar, bootstrap),
                      human/ (Streamlit app, balanced assignment, Krippendorff's α)
  pipeline/           ExperimentRunner (local infer -> eval)
  utils/              seed, io, logging, config hashing, git hash, run artifacts
scripts/              CLI entry points (build_data, train, infer, eval_auto, eval_stats, run_experiment)
tests/                unit tests
data/                 DATASETS (not code): gold.jsonl pool, processed/ splits, knowledge_base/, raw_legacy/ fallback
docs/                 thesis PDF, masterplan, figures
outputs/              raw experiment runs (gitignored): experiments/ (current E01–E04), legacy_runs/ (old ablations)
results/              aggregated analysis: stats/, human_eval/, human_ratings/
archive/old_pipeline/ the previous codebase, preserved for reference
```

Note: `data/` (top level) holds the **datasets**; `src/data/` holds the **code**
that loads and processes them — same word, different role.

**Training and inference are decoupled.** Importing the inference/evaluation code
never pulls in `peft`/`trl`/`bitsandbytes`; those are imported lazily and only by
the training stack. This is what lets training run on a GPU machine while
inference + evaluation run locally on CPU.

### Reproducibility

Every run writes a self-describing folder under `outputs/experiments/<id>/`:
`config_snapshot.yaml`, `config_hash.txt`, `git_hash.txt`, `env.txt`, plus the
run's outputs (`adapter/`, `predictions*.jsonl`, `metrics_auto.json`, `logs/`).
Train/val/test splits are a deterministic hash of each item's content, so the
test set never leaks into training as the dataset grows. The split source is the
principled synthetic pool `data/gold.jsonl`; the superseded original split files
are kept under `data/raw_legacy/` only as a fallback.

---

## For the professor: how to train

You only need to run **one command**. Hydra is used internally — you do not need
to learn it.

**1. Prerequisites:** a CUDA GPU and Python ≥ 3.10. Install a CUDA-matched
PyTorch first, then the training requirements:

```bash
# Example for CUDA 12.4 — adjust to your CUDA version:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt
```

**2. Build the dataset** (deterministic splits from the gold data):

```bash
python scripts/build_data.py
```

**3. Train** (this is the one command):

```bash
python scripts/train.py --experiment E03_qwen0_5b_ft
```

Add `--debug` first for a 1-minute sanity run (10 samples, 1 epoch).

**4. Send back the whole run folder** that the script prints, e.g.:

```
outputs/experiments/E03_qwen0_5b_ft_42/
├── adapter/                 # LoRA weights + tokenizer + training_metadata.json
├── config_snapshot.yaml
├── config_hash.txt
├── git_hash.txt
├── env.txt                  # pip freeze of your environment
└── logs/train.log
```

That folder is everything needed to run and reproduce the fine-tuned model.

---

## Local inference & evaluation (no GPU)

Install the base stack only (no training dependencies needed):

```bash
pip install -e .
python scripts/build_data.py        # once: hash-split dataset
python scripts/build_kb.py          # once: build the RAG knowledge base
```

**Method A — prompt-only**, end to end (inference + metrics):

```bash
python scripts/run_experiment.py --experiment E01_qwen0_5b_prompt
```

**Method B — RAG** (retrieves guideline chunks, no adapter needed):

```bash
python scripts/run_experiment.py --experiment E02_qwen0_5b_rag
```

**Method C — fine-tuned:** drop the adapter folder sent by the professor at
`outputs/experiments/E03_qwen0_5b_ft_42/adapter`, then:

```bash
python scripts/run_experiment.py --experiment E03_qwen0_5b_ft
```

Inference is cached per item — re-running prints `CACHE HIT` and does no work.
For a fast check, cap the items and shorten generation:

```bash
python scripts/run_experiment.py --experiment E01_qwen0_5b_prompt \
    --override data.max_samples=2 eval=quick method.generate.max_new_tokens=128
```

**Statistical comparison** across methods (matched per-item tests):

```bash
python scripts/eval_stats.py --experiments E01_qwen0_5b_prompt E03_qwen0_5b_ft
```

This writes Friedman / Wilcoxon+Holm / Cliff's δ / bootstrap-CI (completeness)
and Cochran's Q / McNemar+Holm (top-1) to `results/stats/`.

---

## Human evaluation (blind, multi-rater)

A complete Streamlit rating workflow with a balanced, blind assignment and
Krippendorff's α.

```bash
pip install -e ".[human]"

# 1. Build the eval set + balanced rater assignment from the four method runs:
python scripts/build_human_eval.py \
    --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag \
    --n-items 60 --n-raters 6 --ratings-per-output 3

# 2. Each rater opens the app, picks their ID, and rates (auto-saves + resumes):
python scripts/run_human_eval.py

# 3. Aggregate inter-rater reliability + per-system scores + statistics:
python scripts/compute_irr.py
```

- Rating is **blind**: raters never see which method produced an output.
- Each (item, method) output is rated by ≥3 distinct raters; per-rater load is balanced.
- Rubric: 6 Likert dimensions (`src/evaluation/human/rubric.py`).
- `compute_irr.py` writes Krippendorff's α per dimension, per-system means, and a
  Friedman + Wilcoxon+Holm (with Cliff's δ and bootstrap CI) comparison of the
  per-item overall human scores, to `results/human_ratings/`.

---

## Commands at a glance

| Command | What it does |
|---|---|
| `python scripts/build_data.py` | Build `data/processed/{train,val,test}.jsonl` with hash splits |
| `python scripts/build_kb.py` | Build the RAG knowledge base (`data/knowledge_base/chunks.jsonl`) |
| `python scripts/build_perturbations.py` | Build paraphrase / missing-info test variants (enables the robustness metrics) |
| `python scripts/train.py --experiment E03_qwen0_5b_ft` | Fine-tune (GPU); writes the run folder |
| `python scripts/run_all.py --experiments E01_qwen0_5b_prompt E03_qwen0_5b_ft --seeds 42 43 44` | Run methods × seeds |
| `python scripts/infer.py --experiment E01_qwen0_5b_prompt` | Cached inference only |
| `python scripts/eval_auto.py --experiment E01_qwen0_5b_prompt` | Automatic metrics |
| `python scripts/run_experiment.py --experiment E01_qwen0_5b_prompt` | Infer + eval |
| `python scripts/eval_stats.py --experiments A B ...` | Cross-method statistics |
| `python scripts/aggregate_results.py` | Aggregate all runs → `results/comparison_table.csv`, `comparison_seeds.csv` (mean/std across seeds), `final_report.md` |
| `python scripts/generate_dataset.py --n 600` | Generate principled gold data (`data/gold.jsonl`) |
| `python scripts/build_human_eval.py --experiments E01_qwen0_5b_prompt E02_qwen0_5b_rag E03_qwen0_5b_ft E04_qwen0_5b_ft_rag` | Build blind human-eval set + assignment |
| `python scripts/run_human_eval.py` | Launch the Streamlit rating app |
| `python scripts/compute_irr.py` | Krippendorff's α + per-system human scores |
| `pytest` | Run the unit tests |

## Optional features (extras)

These are implemented and registered; each needs its optional dependency. None
are required for the core A/B/C/D study.

| Feature | Install | Use |
|---|---|---|
| **Constrained JSON decoding** (Outlines) — forces schema-valid output | `pip install -e ".[constrained]"` | add `method.generate.constrained=true` |
| **Dense retriever** (BGE embeddings) — semantic RAG, for a retriever ablation | `pip install -e ".[rag-dense]"` | `--override method.retriever.name=dense` |
| **DoRA / RSLoRA** — fine-tuning algorithm ablation | (uses the base train stack) | `--override training=dora` (or `training=rslora`) |
| **GaLore** — full-parameter fine-tuning ablation | `pip install -e ".[galore]"` | `--override training=galore` |
| **G-Eval (LLM-as-judge)** — auto rubric scoring vs. humans | set `OPENAI_API_KEY` | `--override eval=with_judge` |

## Scope

Implemented: all four methods (A/B/C/D) end-to-end with the 0.5B model, TF-IDF
**and** dense RAG retrievers, the automatic metric suite (schema, top-k, macro-F1,
latency, robustness, grounding, **G-Eval**), the full statistics module, a
principled synthetic gold-data generator, the complete blind human-evaluation
workflow (Streamlit + balanced assignment + Krippendorff's α), **constrained
decoding**, and the **QLoRA / DoRA / RSLoRA / GaLore** training algorithms. Not
implemented (out of scope): vLLM, RAFT, and larger models (add a model YAML — no
code). The main remaining levers for stronger results are a larger /
supervisor-approved gold dataset and running the experiments at scale.
