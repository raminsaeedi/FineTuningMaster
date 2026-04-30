# Master Thesis – Dashboard Design Recommendation Fine-Tuning

**Research Question**: Does PEFT/Fine-Tuning (+ RAG) improve the quality of structured dashboard design recommendations?

**Model**: `Qwen/Qwen2.5-0.5B-Instruct` fine-tuned with **QLoRA** (LoRA on 4-bit quantized weights)

---

## Project Structure

```
master-thesis-finetuning/
│
├── config.yaml              # All settings (model, LoRA, training, paths)
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── generate_dataset.py      # STEP 1 – Generate synthetic training data
├── train.py                 # STEP 2 – Fine-tune the model with QLoRA
├── inference.py             # STEP 3 – Run inference & evaluate
│
├── colab_finetune.ipynb     # Google Colab notebook (all steps in one place)
│
├── data/                    # Auto-created by generate_dataset.py
│   ├── train.jsonl          # 80 training examples
│   ├── val.jsonl            # 10 validation examples
│   └── test.jsonl           # 10 test examples
│
├── outputs/                 # Auto-created during training
│   ├── checkpoints/         # Intermediate model checkpoints
│   ├── final_model/         # Final LoRA adapter weights (~10-50 MB)
│   ├── logs/                # Training and inference logs
│   └── results/             # Evaluation results (JSON)
│
└── utils/
    ├── __init__.py
    └── helpers.py           # Shared utilities (config, prompts, JSON parsing)
```

---

## Execution Order

### Local (VS Code)

Run these commands **in order** in your terminal:

```bash
# 1. Create virtual environment (do this once)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Install dependencies (do this once)
pip install torch --index-url https://download.pytorch.org/whl/cu121  # NVIDIA GPU
pip install -r requirements.txt

# 3. Generate synthetic dataset
python generate_dataset.py

# 4. Fine-tune the model
python train.py

# 5. Run inference demo
python inference.py

# 6. Evaluate on test set
python inference.py --evaluate

# 7. Compare base vs. fine-tuned model
python inference.py --base-only   # base model
python inference.py               # fine-tuned model
```

### Google Colab (Recommended for Training)

1. Upload `colab_finetune.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU**: Runtime → Change runtime type → T4 GPU
3. Run all cells top to bottom
4. Training takes ~15–30 minutes on a T4 GPU

---

## What Each Script Does

### `generate_dataset.py`

- Creates synthetic dashboard briefs across 10 industries
- Each example has a `brief` (input) and `recommendation` (expected output)
- Output format: JSONL (one JSON object per line)
- Produces 80 train / 10 val / 10 test examples

**Example brief:**

```json
{
  "title": "E-Commerce Sales Dashboard",
  "target_audience": "Sales Managers and Regional Directors",
  "business_goals": "Monitor daily revenue and conversion rates",
  "kpis": ["Revenue", "Conversion Rate", "Average Order Value"],
  "data_context": "Data from Shopify. Updated daily.",
  "update_frequency": "Daily",
  "user_expertise": "Intermediate"
}
```

**Expected output (6 structured keys):**

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

### `train.py`

- Loads `Qwen/Qwen2.5-0.5B-Instruct` with 4-bit quantization (QLoRA)
- Applies LoRA adapters to attention and MLP layers
- Trains with `SFTTrainer` from the TRL library
- Saves only the LoRA adapter weights (~10–50 MB, not the full 1 GB model)

### `inference.py`

- Loads the base model + LoRA adapter
- Formats a brief into a chat-template prompt
- Generates a structured JSON recommendation
- Parses and validates the output (checks all 6 required keys)
- Computes metrics: JSON parse rate, schema validity rate, latency

---

## Configuration (`config.yaml`)

Key settings you may want to adjust:

| Setting                                | Default                      | Description                                      |
| -------------------------------------- | ---------------------------- | ------------------------------------------------ |
| `model.name`                           | `Qwen/Qwen2.5-0.5B-Instruct` | Base model to fine-tune                          |
| `model.load_in_4bit`                   | `true`                       | Enable QLoRA (saves VRAM)                        |
| `lora.r`                               | `16`                         | LoRA rank (higher = more params, better quality) |
| `lora.lora_alpha`                      | `32`                         | LoRA scaling (usually 2× rank)                   |
| `training.num_train_epochs`            | `3`                          | Training epochs                                  |
| `training.learning_rate`               | `2e-4`                       | Learning rate                                    |
| `training.per_device_train_batch_size` | `2`                          | Reduce to `1` if OOM                             |
| `dataset_generation.num_train_samples` | `80`                         | Training examples to generate                    |

---

## Python Version & Environment

- **Python**: 3.10 or 3.11 recommended (3.12 works but some libs lag behind)
- **CUDA**: 11.8 or 12.1 for NVIDIA GPU training
- **bitsandbytes**: Required for 4-bit quantization (QLoRA)
  - Windows: needs special wheel (see `requirements.txt` notes)
  - Apple Silicon: not supported → set `load_in_4bit: false` in `config.yaml`

---

## What Works Where

| Task                     | Local (CPU)          | Local (GPU)  | Google Colab T4 |
| ------------------------ | -------------------- | ------------ | --------------- |
| Dataset generation       | ✅ Fast              | ✅ Fast      | ✅ Fast         |
| Model download           | ✅ (~1 GB)           | ✅ (~1 GB)   | ✅ (~1 GB)      |
| Training (3 epochs)      | ⚠️ Very slow (hours) | ✅ ~5–15 min | ✅ ~15–30 min   |
| Inference (1 example)    | ⚠️ Slow (~60s)       | ✅ ~2–5s     | ✅ ~3–8s        |
| Evaluation (10 examples) | ⚠️ Very slow         | ✅ ~1–2 min  | ✅ ~2–5 min     |

**Recommendation**: Generate data and inspect results locally. Run training in Colab.

---

## Common Errors & Fixes

### `CUDA out of memory`

```
RuntimeError: CUDA out of memory.
```

**Fix**: In `config.yaml`, reduce:

```yaml
training:
  per_device_train_batch_size: 1 # was 2
  gradient_accumulation_steps: 8 # increase to compensate
```

### `ModuleNotFoundError: No module named 'bitsandbytes'`

**Fix on Windows**:

```bash
pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-whl
```

**Fix on CPU-only machine**: Set `load_in_4bit: false` in `config.yaml`

### `FileNotFoundError: data/train.jsonl`

**Fix**: Run dataset generation first:

```bash
python generate_dataset.py
```

### `Adapter not found: ./outputs/final_model`

**Fix**: Run training first:

```bash
python train.py
```

### Model outputs invalid JSON

This is normal in early training. The model needs enough epochs to learn the JSON format.

- Increase `num_train_epochs` to 5
- Increase `num_train_samples` to 200
- Check that `temperature` is low (0.1) in `config.yaml`

### Loss not decreasing after epoch 1

- Try increasing `learning_rate` to `5e-4`
- Try increasing LoRA rank `r` to `32`
- Ensure `gradient_accumulation_steps` × `per_device_train_batch_size` ≥ 8

---

## Understanding the Training Output

```
{'loss': 2.4521, 'learning_rate': 0.0002, 'epoch': 0.1}   ← Start: high loss
{'loss': 1.2341, 'learning_rate': 0.00018, 'epoch': 1.0}  ← After epoch 1
{'loss': 0.6123, 'learning_rate': 0.0001, 'epoch': 2.0}   ← After epoch 2
{'loss': 0.3891, 'learning_rate': 0.00002, 'epoch': 3.0}  ← After epoch 3: good!
```

- **Loss > 1.5**: Model hasn't learned the format yet
- **Loss 0.5–1.0**: Model is learning, JSON output may be partially correct
- **Loss < 0.5**: Model reliably generates structured JSON
- **Loss < 0.2**: Possible overfitting on small dataset

---

## Thesis Context

This project is **Version 1 (Prototype)**. The research pipeline will expand to:

1. **V1 (this)**: Fine-tuning only, synthetic data, small model
2. **V2**: Add RAG (Retrieval-Augmented Generation) with a vector database of design guidelines
3. **V3**: Human evaluation of recommendation quality
4. **V4**: Comparison study: base model vs. fine-tuned vs. fine-tuned+RAG

---

## File Sizes (Approximate)

| File                           | Size      |
| ------------------------------ | --------- |
| Base model download            | ~1 GB     |
| LoRA adapter (saved)           | ~10–50 MB |
| Training dataset (80 examples) | ~500 KB   |
| Training logs                  | ~50 KB    |
| Evaluation results             | ~200 KB   |
