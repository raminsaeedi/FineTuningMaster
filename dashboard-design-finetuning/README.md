# dashboard-design-finetuning

Fine-tuning `Qwen/Qwen2.5-0.5B-Instruct` with **LoRA/QLoRA** to generate structured JSON dashboard design recommendations from short dashboard briefs.

**Master Thesis** | PEFT + RAG for Dashboard Design Quality

---

## What This Project Does

**Input** (Dashboard Brief):

```json
{
  "title": "E-Commerce Sales Dashboard",
  "target_audience": "Sales Managers",
  "business_goals": "Monitor daily revenue and conversion rates",
  "kpis": ["Revenue", "Conversion Rate", "Average Order Value"],
  "update_frequency": "Daily",
  "user_expertise": "Intermediate"
}
```

**Output** (Structured Recommendation):

```json
{
  "context_summary": "...",
  "kpi_task_chart_mapping": [...],
  "layout_hierarchy": {...},
  "labels_scales_colors": {...},
  "interactions": {...},
  "design_rationales": {...}
}
```

---

## Project Structure

```
dashboard-design-finetuning/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── .gitignore                         ← Files excluded from git
│
├── config/
│   └── train_config.yaml              ← All settings (model, LoRA, training)
│
├── data/
│   ├── raw/                           ← Original source data (not in git)
│   ├── processed/                     ← Formatted JSONL files for training
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── examples/                      ← Hand-crafted example briefs for testing
│       └── example_brief.json
│
├── scripts/                           ← Run these IN ORDER
│   ├── 01_check_environment.py        ← Verify GPU, libraries, versions
│   ├── 02_generate_synthetic_dataset.py ← Create training data
│   ├── 03_prepare_dataset.py          ← Format data for training
│   ├── 04_train_lora.py               ← Fine-tune with LoRA/QLoRA
│   ├── 05_inference_base_model.py     ← Test base model (no fine-tuning)
│   ├── 06_inference_finetuned_model.py ← Test fine-tuned model
│   └── 07_evaluate_schema_compliance.py ← Measure output quality
│
├── notebooks/
│   └── colab_finetuning_qwen_0_5b.ipynb ← Google Colab notebook
│
└── outputs/
    ├── models/                        ← Saved LoRA adapter weights
    ├── logs/                          ← Training logs
    └── predictions/                   ← Model outputs for evaluation
```

---

## Quick Start (Step by Step)

### Prerequisites

- Python 3.10 or 3.11
- Windows 10/11, Linux, or macOS
- NVIDIA GPU recommended (training on CPU is very slow)

### Step 1 – Set Up Environment

```bash
# Navigate to project folder
cd dashboard-design-finetuning

# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Linux/Mac)
# source .venv/bin/activate
```

### Step 2 – Install Dependencies

```bash
# Install PyTorch with CUDA 12.1 (NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install all other libraries
pip install -r requirements.txt
```

### Step 3 – Verify Environment

```bash
python scripts/01_check_environment.py
```

Expected output:

```
[OK] Python 3.11.x
[OK] PyTorch 2.x.x
[OK] CUDA available: NVIDIA GeForce RTX ...
[OK] transformers 4.40.x
[OK] peft 0.10.x
[OK] trl 0.8.x
[OK] bitsandbytes 0.43.x
```

### Step 4 – Generate Synthetic Dataset

```bash
python scripts/02_generate_synthetic_dataset.py
```

Creates 100 examples (80 train / 10 val / 10 test) in `data/raw/`.

### Step 5 – Prepare Dataset for Training

```bash
python scripts/03_prepare_dataset.py
```

Formats examples into instruction-response pairs in `data/processed/`.

### Step 6 – Fine-Tune the Model

```bash
# Local (requires GPU)
python scripts/04_train_lora.py

# Or use Google Colab (recommended for free GPU):
# Upload notebooks/colab_finetuning_qwen_0_5b.ipynb to colab.research.google.com
```

Training takes ~15–30 minutes on a T4 GPU.

### Step 7 – Compare Base vs. Fine-Tuned Model

```bash
# Test base model (no fine-tuning)
python scripts/05_inference_base_model.py

# Test fine-tuned model
python scripts/06_inference_finetuned_model.py
```

### Step 8 – Evaluate Schema Compliance

```bash
python scripts/07_evaluate_schema_compliance.py
```

Reports: JSON parse rate, schema validity rate, missing keys.

---

## Configuration

All settings are in [`config/train_config.yaml`](config/train_config.yaml).

Key parameters:

| Parameter                | Default                      | Description           |
| ------------------------ | ---------------------------- | --------------------- |
| `model.name`             | `Qwen/Qwen2.5-0.5B-Instruct` | Base model            |
| `model.load_in_4bit`     | `true`                       | QLoRA (saves VRAM)    |
| `lora.r`                 | `16`                         | LoRA rank             |
| `lora.lora_alpha`        | `32`                         | LoRA scaling          |
| `training.num_epochs`    | `3`                          | Training epochs       |
| `training.learning_rate` | `2e-4`                       | Learning rate         |
| `training.batch_size`    | `2`                          | Per-device batch size |

---

## Common Errors

| Error                                  | Fix                                        |
| -------------------------------------- | ------------------------------------------ |
| `CUDA out of memory`                   | Set `batch_size: 1` in `train_config.yaml` |
| `bitsandbytes` not found               | See install note in `requirements.txt`     |
| `data/processed/train.jsonl not found` | Run scripts 02 and 03 first                |
| `outputs/models/final not found`       | Run script 04 first                        |
| Model outputs invalid JSON             | Increase `num_epochs` to 5                 |

---

## Local vs. Colab

| Task                | Local CPU    | Local GPU | Google Colab T4 |
| ------------------- | ------------ | --------- | --------------- |
| Environment check   | Fast         | Fast      | Fast            |
| Dataset generation  | Fast         | Fast      | Fast            |
| Training (3 epochs) | Hours        | ~5–15 min | ~15–30 min      |
| Inference           | ~60s/example | ~2–5s     | ~3–8s           |
| Evaluation          | Very slow    | ~1–2 min  | ~2–5 min        |

**Recommendation**: Use Colab for training, local for everything else.
