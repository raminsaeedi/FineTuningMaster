# Masterplan — Vollständige Architektur und Umsetzungsplan der Masterarbeit

> **Thema:** Fine-Tuning von Large Language Models für strukturierte Dashboard-Designempfehlungen basierend auf Benutzerzielen und Datenkontext
>
> **Verfasser:** Ramin Saeedi Valashani, Hochschule Ruhr West
>
> Dieses Dokument ist das zentrale Referenzdokument für die gesamte Masterarbeit. Wenn Sie den Überblick verlieren, lesen Sie zuerst dieses Dokument. Ziel: Vier Methoden × mehrere Modelle × mehrere Ablations **ohne Chaos**, mit **vollständiger Reproduzierbarkeit** und **korrekter statistischer Auswertung** durchführen.

---

## Inhaltsverzeichnis

- [Teil 0: Architekturphilosophie](#teil-0-architekturphilosophie)
- [Teil 1: Vollständige Projektstruktur](#teil-1-vollständige-projektstruktur)
- [Teil 2: Konfigurationssystem (Hydra)](#teil-2-konfigurationssystem-hydra)
- [Teil 3: Abstraktionen und Schnittstellen](#teil-3-abstraktionen-und-schnittstellen)
- [Teil 4: Module im Detail](#teil-4-module-im-detail)
- [Teil 5: Moderne Algorithmen und Auswahl](#teil-5-moderne-algorithmen-und-auswahl)
- [Teil 6: Experiment-Tracking und Reproduzierbarkeit](#teil-6-experiment-tracking-und-reproduzierbarkeit)
- [Teil 7: 6-Monats-Umsetzungsplan](#teil-7-6-monats-umsetzungsplan)
- [Teil 8: Vollständige Checkliste](#teil-8-vollständige-checkliste)
- [Teil 9: Algorithmen- und Formelverzeichnis](#teil-9-algorithmen--und-formelverzeichnis)
- [Teil 10: Ehrliche Einschätzung und Risiken](#teil-10-ehrliche-einschätzung-und-risiken)

---

## Teil 0: Architekturphilosophie

Fünf Grundprinzipien, die durchgehend einzuhalten sind:

### Prinzip 1 — Config-Driven Everything

**Kein Hyperparameter, kein Dateipfad, kein Modellname** darf hartkodiert werden. Alles kommt aus YAML-Dateien. Das bedeutet:
- Basismodell wechseln = eine Zeile YAML ändern
- Neues Experiment hinzufügen = neue Datei anlegen
- Experiment von vor drei Monaten reproduzieren = dieselbe Config ausführen

### Prinzip 2 — Registry Pattern

Für alles mit mehreren Varianten (Modelle, Methoden, Retriever, Metriken) gibt es eine **Registry**. Neue Varianten hinzufügen bedeutet nur Eintragen in die Registry, ohne den Hauptcode zu ändern.

```python
@register_method("rag_dense")
class DenseRAGMethod(BaseMethod): ...
```

### Prinzip 3 — Standardisierte Schnittstellen

Alle Methoden (A, B, C, D) müssen **dieselbe** Methode haben:
```python
def generate(self, brief: DashboardBrief) -> GenerationResult
```
Das heißt: Der Evaluierungscode wird **einmal** geschrieben und funktioniert mit allen Methoden.

### Prinzip 4 — Separation of Concerns

- `data/` weiß nicht, was das Modell ist
- `models/` weiß nicht, wie die Evaluierung funktioniert
- `eval/` weiß nicht, wie das Modell gebaut ist
- Jedes Modul macht nur seine Aufgabe und nutzt die Schnittstellen der anderen

### Prinzip 5 — Cache und Resume

- Inference auf dem Test-Set dauert manchmal 1–2 Stunden. Das darf **niemals** erneut ausgeführt werden müssen.
- Die Ausgabe jedes Experiments wird als JSONL gespeichert, mit einem Hash der Config.
- Wenn die Config unverändert ist → vorheriges Ergebnis laden.
- Wenn ein Lauf zur Hälfte fertig ist und abstürzt → von dort fortsetzen.

---

## Teil 1: Vollständige Projektstruktur

```
thesis-dashboard-llm/
│
├── README.md
├── pyproject.toml                  # oder requirements.txt
├── .env.example                    # API-Keys (HuggingFace, W&B)
├── .gitignore
├── Makefile                        # Schnellbefehle
│
├── configs/                        # ⭐ Herz des Projekts
│   ├── config.yaml                 # Standard-Config, die überschrieben wird
│   ├── experiment/                 # Vollständige Experimente (Modell + Methode + Daten)
│   │   ├── E01_qwen7b_prompt.yaml
│   │   ├── E02_qwen7b_rag.yaml
│   │   ├── E03_qwen7b_ft.yaml
│   │   ├── E04_qwen7b_ft_rag.yaml
│   │   ├── E05_llama8b_prompt.yaml
│   │   ├── E06_llama8b_rag.yaml
│   │   ├── ...
│   │   └── ablation/
│   │       ├── A01_lora_r8.yaml
│   │       ├── A02_lora_r32.yaml
│   │       ├── A03_retriever_bm25.yaml
│   │       └── A04_retriever_hybrid.yaml
│   ├── model/                      # ⭐ Definition der Basismodelle
│   │   ├── qwen2_5_7b.yaml
│   │   ├── qwen2_5_14b.yaml
│   │   ├── llama3_1_8b.yaml
│   │   ├── llama3_1_70b.yaml
│   │   ├── mistral_7b.yaml
│   │   ├── phi3_5_mini.yaml
│   │   └── gemma2_9b.yaml
│   ├── method/                     # ⭐ Die vier Methoden
│   │   ├── prompt_only.yaml
│   │   ├── rag.yaml
│   │   ├── ft.yaml
│   │   └── ft_rag.yaml
│   ├── retriever/
│   │   ├── bge_small.yaml
│   │   ├── bge_m3.yaml
│   │   ├── e5_large.yaml
│   │   ├── bm25.yaml
│   │   ├── hybrid_rrf.yaml
│   │   └── bge_with_reranker.yaml
│   ├── training/
│   │   ├── lora_default.yaml
│   │   ├── qlora_default.yaml
│   │   ├── dora.yaml
│   │   └── full_ft.yaml
│   ├── data/
│   │   ├── dashboard_v1.yaml
│   │   └── dashboard_v1_augmented.yaml
│   └── eval/
│       ├── full.yaml
│       └── quick.yaml              # für Smoke-Tests
│
├── src/
│   ├── __init__.py
│   ├── core/                       # ⭐ Kern – Schnittstellen und Basisklassen
│   │   ├── __init__.py
│   │   ├── interfaces.py           # BaseMethod, BaseRetriever, ...
│   │   ├── registry.py             # Decorator @register
│   │   ├── schemas.py              # Pydantic-Modelle (DashboardBrief, ...)
│   │   ├── types.py                # Type-Aliases
│   │   └── constants.py            # CHART_TYPES, TASK_TYPES, ...
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── builders/               # Dataset-Erstellung aus Rohquellen
│   │   │   ├── vizml_builder.py
│   │   │   ├── nvbench_builder.py
│   │   │   └── plotly_builder.py
│   │   ├── splits.py               # Train/Val/Test, deterministisch
│   │   ├── perturbations.py        # für Robustheitstests
│   │   └── kb_builder.py           # Knowledge Base für RAG
│   │
│   ├── models/                     # Wrapper für Modelle
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── hf_causal.py            # HuggingFace Causal LM Wrapper
│   │   ├── vllm_engine.py          # für schnelle Inference
│   │   └── loaders.py
│   │
│   ├── methods/                    # ⭐ Die vier Methoden
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseMethod
│   │   ├── prompt_only.py
│   │   ├── rag.py
│   │   ├── ft.py
│   │   └── ft_rag.py
│   │
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dense.py                # BGE, E5
│   │   ├── sparse.py               # BM25
│   │   ├── hybrid.py               # RRF
│   │   ├── rerank.py               # Cross-Encoder
│   │   └── hyde.py                 # Hypothetical Document Embeddings
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py          # SFT mit LoRA/QLoRA/DoRA
│   │   ├── raft_trainer.py         # Retrieval-Augmented FT
│   │   ├── data_formatter.py
│   │   └── callbacks.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── runner.py               # Batch-Inference mit Caching
│   │   ├── postprocess.py          # JSON parsen, defektes JSON reparieren
│   │   └── decoders.py             # Constrained Decoding (Outlines)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics/                # eine Datei pro Metrik
│   │   │   ├── base.py
│   │   │   ├── topk_accuracy.py
│   │   │   ├── macro_f1.py
│   │   │   ├── schema_compliance.py
│   │   │   ├── robustness.py
│   │   │   ├── grounding.py
│   │   │   └── llm_judge.py        # G-Eval-Stil
│   │   ├── human/
│   │   │   ├── streamlit_app.py
│   │   │   ├── assignment.py       # Balanced Incomplete Block Design
│   │   │   └── irr.py              # Krippendorffs Alpha
│   │   ├── stats/
│   │   │   ├── friedman.py
│   │   │   ├── wilcoxon_holm.py
│   │   │   ├── cliff_delta.py
│   │   │   ├── cochran_mcnemar.py
│   │   │   └── bootstrap_ci.py
│   │   └── aggregator.py           # alle Ergebnisse → DataFrame
│   │
│   ├── pipeline/                   # ⭐ Orchestrierung
│   │   ├── __init__.py
│   │   ├── runner.py               # ExperimentRunner
│   │   ├── stages.py               # data → train → infer → eval
│   │   └── cache.py                # Caching-Logik
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                 # Seeds überall setzen
│       ├── logging.py              # strukturiertes Logging
│       ├── io.py                   # JSONL, YAML, Parquet
│       ├── timer.py
│       └── tracking.py             # W&B-/MLflow-Integration
│
├── scripts/                        # ⭐ CLI-Einstiegspunkte
│   ├── build_data.py               # Dataset aus Quellen aufbauen
│   ├── build_kb.py                 # Knowledge Base aufbauen
│   ├── train.py                    # Fine-Tuning durchführen
│   ├── infer.py                    # Inference durchführen
│   ├── eval_auto.py                # automatische Evaluierung
│   ├── eval_stats.py               # statistische Tests
│   ├── run_experiment.py           # End-to-End ein Experiment durchführen
│   └── run_all.py                  # Batch aller Experimente
│
├── notebooks/                      # nur für Exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_kb_inspection.ipynb
│   ├── 03_error_analysis.ipynb
│   └── 04_final_figures.ipynb
│
├── tests/                          # ⭐ Unit-Tests – optional, aber empfohlen
│   ├── test_schema.py
│   ├── test_methods.py
│   ├── test_metrics.py
│   └── test_stats.py
│
├── data/                           # gitignored
│   ├── raw/
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── knowledge_base/
│   │   ├── chunks.jsonl
│   │   └── index/                  # FAISS-/BM25-Indizes
│   └── eval_set/
│       ├── items.jsonl
│       └── assignments.json        # Rater × Item × Variante
│
├── models/                         # gitignored
│   └── adapters/
│       ├── qwen7b_lora_r16/
│       └── llama8b_qlora_r16/
│
├── results/                        # ⭐ Kritische Struktur
│   ├── experiments/
│   │   ├── E01_qwen7b_prompt_only/
│   │   │   ├── config_snapshot.yaml    # Kopie der Config zum Ausführungszeitpunkt
│   │   │   ├── config_hash.txt
│   │   │   ├── predictions.jsonl       # Output je Item
│   │   │   ├── metrics_auto.json
│   │   │   ├── retrieved_docs.jsonl    # für RAG-Methoden
│   │   │   └── logs/
│   │   ├── E02_qwen7b_rag/
│   │   └── ...
│   ├── human_ratings/
│   │   ├── rater_01.jsonl
│   │   └── ...
│   ├── stats/
│   │   ├── friedman_results.csv
│   │   ├── posthoc_wilcoxon.csv
│   │   ├── irr_alphas.csv
│   │   └── final_table.csv
│   └── figures/
│       ├── F01_topk_per_system.pdf
│       └── ...
│
└── thesis_doc/                     # Verfassen der Masterarbeit (LaTeX/Word)
    ├── chapters/
    ├── figures/
    └── refs.bib
```

---

## Teil 2: Konfigurationssystem (Hydra)

### Warum Hydra?

- Komposition von Configs
- Override über Kommandozeile (`python train.py model=llama8b training.lr=1e-4`)
- Multirun: ein Befehl = 20 Experimente (`-m model=qwen7b,llama8b`)
- Kompatibel mit Dataclasses (typsicher)

### Installation

```bash
pip install hydra-core==1.3.* omegaconf
```

### Beispiel-Configs

**`configs/config.yaml`** (Haupt):
```yaml
defaults:
  - model: qwen2_5_7b
  - method: prompt_only
  - data: dashboard_v1
  - eval: full
  - _self_

seed: 42
output_root: results/experiments
experiment_id: ${experiment_name}_${seed}
experiment_name: default

tracking:
  use_wandb: true
  project: thesis-dashboard-llm
  entity: ramin-thesis
```

**`configs/model/qwen2_5_7b.yaml`**:
```yaml
name: qwen2_5_7b
hf_id: Qwen/Qwen2.5-7B-Instruct
size_billions: 7
context_length: 32768
chat_template: qwen
dtype: bfloat16
load_in_4bit: false        # wird für Training auf True gesetzt
trust_remote_code: false
```

**`configs/method/ft_rag.yaml`**:
```yaml
name: ft_rag
type: fine_tuned_rag
defaults:
  - /retriever: bge_small
  - /training: qlora_default
adapter_path: ${output_root}/${experiment_id}/adapter
generate:
  max_new_tokens: 1500
  temperature: 0.2
  top_p: 0.9
  do_sample: true
  repetition_penalty: 1.05
```

**`configs/training/qlora_default.yaml`**:
```yaml
type: qlora
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: none
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true
sft:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_seq_length: 2048
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  eval_steps: 200
  save_steps: 200
  save_total_limit: 3
```

**`configs/experiment/E04_qwen7b_ft_rag.yaml`**:
```yaml
# @package _global_
defaults:
  - override /model: qwen2_5_7b
  - override /method: ft_rag
  - override /data: dashboard_v1
  - _self_

experiment_name: E04_qwen7b_ft_rag
seed: 42
```

### Experimente ausführen

```bash
# Ein Experiment
python scripts/run_experiment.py +experiment=E04_qwen7b_ft_rag

# Schnelle Parameteränderung
python scripts/run_experiment.py +experiment=E04_qwen7b_ft_rag \
    training.sft.learning_rate=1e-4 seed=43

# Multirun: Modell × Methode × Seed
python scripts/run_experiment.py --multirun \
    +experiment=E01,E02,E03,E04 seed=42,43,44
```

### Config-Hash für Reproduzierbarkeit

```python
# src/utils/config_hash.py
import hashlib, json
from omegaconf import OmegaConf

def hash_config(cfg) -> str:
    """Stabiler Hash der Config zum Caching."""
    serial = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(serial.encode()).hexdigest()[:12]
```

Speichern Sie diesen Hash in `config_hash.txt` neben jedem Experiment. Wird dieselbe Config erneut ausgeführt → aus dem Cache laden.

---

## Teil 3: Abstraktionen und Schnittstellen

### Schemas mit Pydantic

```python
# src/core/schemas.py
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict, Any, Optional

class TaskType(str, Enum):
    TREND = "trend"
    COMPARISON = "comparison"
    COMPOSITION = "composition"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    RANKING = "ranking"
    DEVIATION = "deviation"
    PART_TO_WHOLE = "part_to_whole"
    FLOW = "flow"

class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    STACKED_BAR = "stacked_bar"
    GROUPED_BAR = "grouped_bar"
    AREA = "area"
    PIE = "pie"
    DONUT = "donut"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    KPI_CARD = "kpi_card"
    TABLE = "table"
    GAUGE = "gauge"
    SANKEY = "sankey"
    TREEMAP = "treemap"
    MAP = "map"

class DashboardBrief(BaseModel):
    """Standardisierte Eingabe für alle Methoden."""
    item_id: str
    users: str
    goals: List[str]
    kpis: List[str]
    columns: List[Dict[str, str]]      # [{"name":"...", "dtype":"..."}]
    constraints: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class KPIChartMapping(BaseModel):
    kpi: str
    task_type: TaskType
    chart_type: ChartType
    alternatives: List[ChartType] = []
    encoding: Dict[str, Any] = {}

class Rationale(BaseModel):
    claim: str
    principle: str

class DesignOutput(BaseModel):
    """Standardisierte Ausgabe aller Methoden."""
    context_summary: Dict[str, Any]
    kpi_chart_mapping: List[KPIChartMapping]
    layout: Dict[str, Any]
    styling: Dict[str, Any]
    interactions: List[str]
    rationales: List[Rationale]

class GenerationResult(BaseModel):
    """Was jede method.generate() zurückgibt."""
    item_id: str
    method_name: str
    model_name: str
    config_hash: str
    raw_text: str                                # rohe Modellausgabe
    parsed: Optional[DesignOutput] = None        # geparstes JSON
    parse_error: Optional[str] = None
    retrieved_docs: Optional[List[Dict]] = None  # nur RAG
    latency_ms: float
    seed: int
```

### Registry

```python
# src/core/registry.py
class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict = {}

    def register(self, key: str):
        def decorator(cls):
            if key in self._items:
                raise ValueError(f"{key} already in {self.name}")
            self._items[key] = cls
            return cls
        return decorator

    def get(self, key: str):
        if key not in self._items:
            raise KeyError(f"{key} not in {self.name}. "
                           f"Available: {list(self._items)}")
        return self._items[key]

METHODS = Registry("methods")
RETRIEVERS = Registry("retrievers")
METRICS = Registry("metrics")
TRAINERS = Registry("trainers")
```

### Haupt-Schnittstelle der Methoden

```python
# src/core/interfaces.py
from abc import ABC, abstractmethod
from src.core.schemas import DashboardBrief, GenerationResult

class BaseMethod(ABC):
    """Einheitlicher Vertrag für A, B, C, D."""
    name: str

    def __init__(self, cfg): self.cfg = cfg

    @abstractmethod
    def setup(self) -> None:
        """Modell, Index, Adapter laden – was auch immer nötig ist."""
        ...

    @abstractmethod
    def generate(self, brief: DashboardBrief) -> GenerationResult:
        ...

    def teardown(self) -> None:
        """Ressourcen freigeben (GPU-Speicher)."""
        ...

class BaseRetriever(ABC):
    @abstractmethod
    def setup(self) -> None: ...
    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[dict]: ...

class BaseTrainer(ABC):
    @abstractmethod
    def train(self, train_data, val_data, output_dir: str) -> str:
        """Gibt den Pfad des gespeicherten Adapters zurück."""
        ...

class BaseMetric(ABC):
    name: str
    @abstractmethod
    def compute(self, results: list[GenerationResult],
                references: list[dict]) -> dict: ...
```

### Beispiel einer Methode

```python
# src/methods/rag.py
import time, json
from src.core.interfaces import BaseMethod
from src.core.registry import METHODS, RETRIEVERS
from src.core.schemas import DashboardBrief, GenerationResult
from src.models.hf_causal import HFCausalModel
from src.inference.postprocess import parse_json_safe

@METHODS.register("rag")
class RAGMethod(BaseMethod):
    name = "rag"

    def setup(self):
        self.model = HFCausalModel(self.cfg.model)
        self.model.load()
        retriever_cls = RETRIEVERS.get(self.cfg.method.retriever.type)
        self.retriever = retriever_cls(self.cfg.method.retriever)
        self.retriever.setup()
        self.system_prompt = self._load_system_prompt()

    def generate(self, brief: DashboardBrief) -> GenerationResult:
        t0 = time.perf_counter()
        query = self._brief_to_query(brief)
        passages = self.retriever.retrieve(query, k=self.cfg.method.retriever.top_k)
        user_msg = self._build_user_msg(brief, passages)
        raw = self.model.chat(self.system_prompt, user_msg,
                              **self.cfg.method.generate)
        parsed, err = parse_json_safe(raw)
        return GenerationResult(
            item_id=brief.item_id,
            method_name=self.name,
            model_name=self.cfg.model.name,
            config_hash=self.cfg._hash,
            raw_text=raw, parsed=parsed, parse_error=err,
            retrieved_docs=passages,
            latency_ms=(time.perf_counter()-t0)*1000,
            seed=self.cfg.seed,
        )
```

**Hinweis:** Alle vier Methoden folgen diesem Muster. Der Evaluierungscode funktioniert **ohne Änderung** mit allen.

---

## Teil 4: Module im Detail

### 4.1 Data-Modul

**Ziel:** Deterministische Dataset-Erstellung aus öffentlichen Quellen.

```python
# src/data/builders/vizml_builder.py
class VizMLBuilder:
    def __init__(self, source_path: str, schema_path: str):
        self.source = source_path
        self.schema = json.load(open(schema_path))

    def iter_pairs(self):
        for row in self._iter_raw():
            brief = self._row_to_brief(row)
            gold = self._row_to_design(row)
            yield {"brief": brief.dict(), "gold": gold.dict()}

    def _row_to_brief(self, row) -> DashboardBrief:
        return DashboardBrief(
            item_id=f"vizml_{row['id']}",
            users=self._infer_users(row),
            goals=[row["analytical_task"]],
            kpis=[row["target_field"]],
            columns=row["columns_meta"],
        )
```

**Splits müssen deterministisch sein:**

```python
# src/data/splits.py
import hashlib
def assign_split(item_id: str) -> str:
    """Basierend auf Hash, nicht Random. Wird später eine Zeile
    hinzugefügt, ändern sich die alten Splits nicht."""
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16)
    p = (h % 100) / 100.0
    if p < 0.8: return "train"
    elif p < 0.9: return "val"
    else: return "test"
```

Diese Methode ist kritisch: Bei Verwendung von `random.seed` und Hinzufügen einer neuen Zeile später ändern sich **alle** Splits, und es kann passieren, dass das Test-Set ins Training leakt.

### 4.2 Models-Modul

```python
# src/models/hf_causal.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class HFCausalModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self, adapter_path: str | None = None):
        kwargs = {"torch_dtype": getattr(torch, self.cfg.dtype),
                  "device_map": "auto"}
        if self.cfg.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.hf_id, **kwargs)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    @torch.inference_mode()
    def chat(self, system: str, user: str, **gen_kwargs) -> str:
        msgs = [{"role":"system","content":system},
                {"role":"user","content":user}]
        inputs = self.tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)
        out = self.model.generate(inputs, **gen_kwargs)
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
```

**Für Massen-Inference: vLLM**

Bei 100 Items × 4 Methoden × 5 Modellen = 2000 Generationen ist vLLM deutlich schneller:

```python
# src/models/vllm_engine.py
from vllm import LLM, SamplingParams

class VLLMEngine:
    def __init__(self, cfg):
        self.llm = LLM(model=cfg.hf_id,
                       dtype=cfg.dtype,
                       gpu_memory_utilization=0.9,
                       max_model_len=cfg.max_model_len,
                       enable_lora=cfg.get("enable_lora", False))
        self.sampling = SamplingParams(
            temperature=cfg.temperature, top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens, seed=cfg.seed)

    def batch_chat(self, prompts: list[str]) -> list[str]:
        outs = self.llm.generate(prompts, self.sampling)
        return [o.outputs[0].text for o in outs]
```

vLLM kann auch LoRA-Adapter ausliefern – ideal für Methode D.

### 4.3 Methoden (die vier Methoden)

Alle folgen einem einheitlichen Vertrag. Die Unterschiede liegen nur in Setup und Prompt-Aufbau:

| Methode | Setup | Prompt-Aufbau |
|---|---|---|
| **PromptOnly** | nur Basismodell | `[system, user(brief)]` |
| **RAG** | Basismodell + Retriever + Index | `[system, user(passages + brief)]` |
| **FT** | Basismodell + LoRA-Adapter | `[system, user(brief)]` mit FT-Modell |
| **FT+RAG** | Basismodell + Adapter + Retriever | `[system, user(passages + brief)]` mit FT-Modell |

### 4.4 Retrievers-Modul

Mehrere Varianten mit derselben Schnittstelle:

```python
# src/retrievers/dense.py
@RETRIEVERS.register("dense")
class DenseRetriever(BaseRetriever):
    def setup(self):
        from sentence_transformers import SentenceTransformer
        import faiss, json
        self.embedder = SentenceTransformer(self.cfg.embedder_id)
        self.chunks = [json.loads(l) for l in open(self.cfg.chunks_path)]
        self.index = faiss.read_index(self.cfg.index_path)

    def retrieve(self, query: str, k: int) -> list[dict]:
        qv = self.embedder.encode([query], normalize_embeddings=True)
        D, I = self.index.search(qv.astype("float32"), k)
        return [{**self.chunks[i], "score": float(d)}
                for i, d in zip(I[0], D[0])]

# src/retrievers/sparse.py
@RETRIEVERS.register("bm25")
class BM25Retriever(BaseRetriever):
    def setup(self):
        from rank_bm25 import BM25Okapi
        self.chunks = [json.loads(l) for l in open(self.cfg.chunks_path)]
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, k):
        scores = self.bm25.get_scores(query.lower().split())
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

# src/retrievers/hybrid.py
@RETRIEVERS.register("hybrid_rrf")
class HybridRRFRetriever(BaseRetriever):
    """Reciprocal Rank Fusion – Kombination von Dense und Sparse."""
    def setup(self):
        self.dense = RETRIEVERS.get("dense")(self.cfg.dense).setup()
        self.sparse = RETRIEVERS.get("bm25")(self.cfg.sparse).setup()

    def retrieve(self, query, k):
        d_results = self.dense.retrieve(query, k=k*2)
        s_results = self.sparse.retrieve(query, k=k*2)
        # RRF-Score: sum(1/(rank+60))
        scores = {}
        for rank, r in enumerate(d_results):
            scores[r["id"]] = scores.get(r["id"], 0) + 1/(rank+60)
        for rank, r in enumerate(s_results):
            scores[r["id"]] = scores.get(r["id"], 0) + 1/(rank+60)
        # ... sortieren und Top k zurückgeben
```

### 4.5 Training-Modul

```python
# src/training/sft_trainer.py
@TRAINERS.register("qlora_sft")
class QLoRASFTTrainer(BaseTrainer):
    def train(self, train_data, val_data, output_dir):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig

        model = self._load_base_4bit()
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(**self.cfg.training.lora,
                              task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)

        sft_cfg = SFTConfig(output_dir=output_dir,
                            **self.cfg.training.sft,
                            report_to="wandb")

        trainer = SFTTrainer(
            model=model, args=sft_cfg,
            train_dataset=train_data, eval_dataset=val_data,
            tokenizer=self.tokenizer,
            formatting_func=self._format,
        )
        trainer.train()
        trainer.save_model()
        return output_dir
```

### 4.6 Inference-Modul – grundlegendes Caching

```python
# src/inference/runner.py
class InferenceRunner:
    def __init__(self, method: BaseMethod, cfg):
        self.method = method
        self.cfg = cfg
        self.out_path = Path(cfg.output_root) / cfg.experiment_id / "predictions.jsonl"

    def run(self, items: list[DashboardBrief]):
        done_ids = self._load_done()
        if len(done_ids) == len(items):
            print(f"[CACHE HIT] {self.out_path} already complete.")
            return

        self.method.setup()
        with self.out_path.open("a") as f:
            for brief in items:
                if brief.item_id in done_ids:
                    continue
                try:
                    res = self.method.generate(brief)
                    f.write(res.json() + "\n")
                    f.flush()
                except Exception as e:
                    self._log_error(brief.item_id, e)
        self.method.teardown()

    def _load_done(self) -> set:
        if not self.out_path.exists(): return set()
        return {json.loads(l)["item_id"] for l in self.out_path.open()}
```

**Bei einem Absturz → erneut ausführen, der Lauf wird fortgesetzt.**

### 4.7 Evaluation-Modul

Jede Metrik ist eine eigenständige Klasse:

```python
# src/evaluation/metrics/topk_accuracy.py
@METRICS.register("top_k_accuracy")
class TopKAccuracy(BaseMetric):
    name = "top_k_accuracy"
    def compute(self, results, refs):
        out = {"top_1": 0.0, "top_3": 0.0, "n": 0}
        for r, ref in zip(results, refs):
            if r.parsed is None: continue
            out["n"] += 1
            pred_main = r.parsed.kpi_chart_mapping[0].chart_type
            pred_alts = r.parsed.kpi_chart_mapping[0].alternatives
            gold = ref["kpi_chart_mapping"][0]["chart_type"]
            if pred_main == gold: out["top_1"] += 1
            if gold in ([pred_main] + pred_alts)[:3]: out["top_3"] += 1
        out["top_1"] /= max(out["n"], 1)
        out["top_3"] /= max(out["n"], 1)
        return out
```

```python
# src/evaluation/metrics/llm_judge.py
@METRICS.register("g_eval")
class GEvalMetric(BaseMetric):
    """G-Eval: Verwendung eines stärkeren LLM als Bewerter – für
    Dimensionen ohne klare Regel (z. B. Rationale-Qualität)."""
    def __init__(self, cfg):
        self.judge_model = cfg.judge_model  # z. B. GPT-4o-mini oder Claude
        self.dimensions = cfg.dimensions

    def compute(self, results, refs):
        # Pro Item via CoT-Prompt eine Bewertung 1–5 vom Bewerter holen.
        # Ersetzt nicht die menschliche Bewertung, sondern ergänzt sie.
        ...
```

### 4.8 Pipeline-Orchestrator

```python
# src/pipeline/runner.py
class ExperimentRunner:
    """End-to-End: data → (train if needed) → infer → eval"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.exp_dir = Path(cfg.output_root) / cfg.experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_config()

    def run(self):
        stages = ["data", "train", "infer", "eval"]
        for stage in stages:
            if self._is_done(stage):
                logger.info(f"[SKIP] {stage} already done")
                continue
            getattr(self, f"_run_{stage}")()
            self._mark_done(stage)

    def _run_data(self):
        # sicherstellen, dass Daten-Dateien existieren
        ...

    def _run_train(self):
        if self.cfg.method.type not in ("fine_tuned", "fine_tuned_rag"):
            return
        trainer_cls = TRAINERS.get(self.cfg.training.type)
        trainer = trainer_cls(self.cfg)
        adapter_path = trainer.train(...)
        self.cfg.method.adapter_path = adapter_path

    def _run_infer(self):
        method_cls = METHODS.get(self.cfg.method.name)
        method = method_cls(self.cfg)
        runner = InferenceRunner(method, self.cfg)
        runner.run(self._load_test_items())

    def _run_eval(self):
        results = self._load_predictions()
        refs = self._load_references()
        all_metrics = {}
        for m_name in self.cfg.eval.metrics:
            m_cls = METRICS.get(m_name)
            all_metrics[m_name] = m_cls(self.cfg).compute(results, refs)
        json.dump(all_metrics, (self.exp_dir/"metrics_auto.json").open("w"))
```

---

## Teil 5: Moderne Algorithmen und Auswahl

### 5.1 Fine-Tuning: welche Methode?

| Methode | Wann verwenden? | Hinweise |
|---|---|---|
| **Full FT** | habe ich nicht / will ich nicht (sehr teuer) | nur als Referenz in Ablations bei genügend Ressourcen |
| **LoRA** | wenn GPU mit ≥24 GB | rank=16, target = alle Linear-Module |
| **QLoRA** | ⭐ Standard für diese Arbeit | 4-Bit NF4, rank=16, Double Quant |
| **DoRA** | für Ablation im Ergebnisteil | DoRA = LoRA + Magnitude-Decomposition; meist 1–2 % besser |
| **GaLore** | wenn Full FT geplant, aber RAM-Mangel | Gradient Low-Rank Projection |
| **Llama-Pro / Block Expansion** | Wissen ohne Vergessen ergänzen | komplexer, seltener |

**Empfehlung für die Masterarbeit:** QLoRA (Standard) + Ablation mit DoRA und unterschiedlichen Ranks.

### 5.2 RAG: verschiedene Generationen

| Generation | Technik | Wann verwenden? |
|---|---|---|
| Naive | Dense Retrieval (BGE) → in Kontext einfügen | RAG-Baseline |
| Hybrid | Dense + BM25 + RRF | bei schwachem Retrieval |
| **Rerank** | k=50 retrieven, mit Cross-Encoder reranken, Top 5 behalten | ⭐ empfohlen |
| HyDE | LLM erzeugt eine hypothetische Antwort → diese wird abgefragt | bei kurzen/ambigen Queries |
| Self-RAG | Modell entscheidet selbst, ob Retrieval nötig | komplex, evtl. übertrieben für die Arbeit |
| CRAG | Retrieve → Evaluator-Score → bei Schwäche Websuche | externe Werkzeuge nötig |
| GraphRAG | Knowledge Graph statt Flat Chunks | bei relationalen Domänen |

**Empfehlung für die Masterarbeit:** Dense (BGE-small oder BGE-m3) + Cross-Encoder-Reranker + Ablation mit BM25 und Hybrid.

```python
# src/retrievers/rerank.py
@RETRIEVERS.register("dense_with_reranker")
class RerankedRetriever(BaseRetriever):
    def setup(self):
        from sentence_transformers import CrossEncoder
        self.base = RETRIEVERS.get("dense")(self.cfg.base)
        self.base.setup()
        self.reranker = CrossEncoder(self.cfg.reranker_id)

    def retrieve(self, query, k):
        candidates = self.base.retrieve(query, k=self.cfg.rerank_pool)  # k=50
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        idx = np.argsort(scores)[::-1][:k]
        return [{**candidates[i], "rerank_score": float(scores[i])} for i in idx]
```

### 5.3 RAFT – Retrieval-Augmented Fine-Tuning

Dieser Algorithmus ist besonders stark für **Methode D**. Idee:

- In den Trainingsdaten werden neben Brief und Gold auch **retrievete Passagen** mitgegeben
- Manche Passagen sind **relevant** (Oracle), manche sind **Distraktoren**
- Das Modell lernt, auf Passagen zu achten und Distraktoren zu ignorieren

```python
# src/training/raft_trainer.py
def build_raft_example(brief, gold, oracle_passages, distractor_passages, p_oracle=0.8):
    """Mit Wahrscheinlichkeit p_oracle wird Oracle behalten,
    sonst nur Distraktoren."""
    if random.random() < p_oracle:
        passages = oracle_passages + random.sample(distractor_passages, k=2)
    else:
        passages = random.sample(distractor_passages, k=5)
    random.shuffle(passages)

    user_msg = f"Passages:\n{format_passages(passages)}\n\nBrief: {brief}"
    return {"messages": [
        {"role":"system","content":SYSTEM_RAFT},
        {"role":"user","content":user_msg},
        {"role":"assistant","content":json.dumps(gold)}
    ]}
```

Dies ist ein guter Beitrag für die Masterarbeit: Vergleich von einfachem D (FT + Simple RAG) mit D (RAFT).

### 5.4 Constrained Decoding

Für hohe **Schema-Compliance** kann Constrained Decoding eingesetzt werden:

```python
# Mit Outlines:
import outlines
schema = open("schemas/dashboard_schema.json").read()
generator = outlines.generate.json(model, schema)
result = generator(prompt)  # garantiert gültiges JSON
```

Das bedeutet: **Schema-Compliance stets 100 %**. Kosten: etwas langsamer und teils geringere Inhaltsqualität. In der Arbeit kann sowohl *ohne* Constrained (Aussage über inhärente Qualität) als auch *mit* (Aussage über die Wirkung des Decodings) berichtet werden.

### 5.5 Decoding-Hyperparameter

| Parameter | Empfehlung |
|---|---|
| temperature | 0.2 für strukturierte Aufgaben |
| top_p | 0.9 |
| repetition_penalty | 1.05 |
| max_new_tokens | 1500 (ausreichend für vollständiges JSON) |
| seed | stets dokumentieren |

Für Robustheitstests kann temperature=0 (greedy) gewählt werden, um Varianz zu eliminieren.

---

## Teil 6: Experiment-Tracking und Reproduzierbarkeit

### 6.1 Weights & Biases

```python
# src/utils/tracking.py
import wandb
def init_run(cfg):
    return wandb.init(
        project=cfg.tracking.project,
        name=cfg.experiment_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.model.name, cfg.method.name],
    )
```

W&B speichert automatisch: Trainingskurven, GPU-Nutzung, Gradienten, Metriken. Bei der Verteidigung kann das W&B-Dashboard geöffnet werden.

### 6.2 Seeds – überall

```python
# src/utils/seed.py
import random, numpy as np, torch
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Jedes Experiment mit drei Seeds wiederholen** (42, 43, 44), um die Varianz zu berichten.

### 6.3 Version-Pinning

```bash
# pyproject.toml oder requirements.txt
torch==2.4.0
transformers==4.44.2
peft==0.12.0
bitsandbytes==0.43.3
trl==0.10.1
accelerate==0.33.0
vllm==0.6.0  # optional
```

Den Output von `pip freeze` in `results/experiments/EXX/env.txt` ablegen.

### 6.4 Git-Hash

In Config-Snapshot zusätzlich den Git-Commit-Hash mitspeichern:
```python
import subprocess
git_hash = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
```

---

## Teil 7: 6-Monats-Umsetzungsplan

### Monat 1 — Fundament

**Woche 1:**
- [ ] Repo anlegen, Hydra einrichten, einfache CI (Unit-Tests)
- [ ] Umgebung, GPU, W&B, HF-Tokens
- [ ] Literatur lesen: Few, Munzner, Cleveland & McGill, LoRA, QLoRA, RAG-Survey, RAFT

**Woche 2:**
- [ ] Finales Rubric (10 Dimensionen, mit Referenzbeispielen)
- [ ] Finales JSON-Schema
- [ ] Einige manuelle Musterbeispiele (~10) als Few-Shot-Prompts

**Woche 3:**
- [ ] `data/builders/` implementieren
- [ ] Erstes Dataset (≥1000 Paare) mit deterministischem Split
- [ ] Knowledge Base sammeln (~30–50 Dokumente → ~500–1000 Chunks)

**Woche 4:**
- [ ] Embedder + FAISS-Index
- [ ] BM25-Index
- [ ] Hybrid-Retriever testen (manuelle Queries, prüfen, ob Passagen sinnvoll)

**Ergebnis Monat 1:** Funktionierendes Repo, fertiges Dataset, indizierte KB.

### Monat 2 — Baselines und RAG

**Woche 5:**
- [ ] HFCausalModel-Wrapper
- [ ] PromptOnlyMethod
- [ ] Finale Few-Shot-Prompts
- [ ] Ausführung auf Val-Set als Smoke-Test

**Woche 6:**
- [ ] RAGMethod (Dense)
- [ ] RAGMethod mit Reranker
- [ ] Erste Ablation: BM25 vs. Dense vs. Hybrid (auf Val)
- [ ] Auswahl der besten Retriever-Variante

**Woche 7:**
- [ ] Einfache Metriken (top_k, schema_compliance, macro_f1)
- [ ] `eval_auto.py`
- [ ] Erste Tabelle: A vs. B auf Val (als Sanity Check)

**Woche 8:**
- [ ] vLLM-Integration für Massen-Inference
- [ ] Finale Auswahl der zu fine-tuneenden Basismodelle (2–3 Modelle reichen)

**Ergebnis Monat 2:** Systeme A und B funktionieren, erste Ergebnisse auf Val vorhanden.

### Monat 3 — Fine-Tuning und FT+RAG

**Woche 9:**
- [ ] QLoRA-Training-Pipeline
- [ ] Erstes Training auf Qwen2.5-7B (Smoke-Run, 1 Epoche, kleines Sample)
- [ ] Bugfixing

**Woche 10:**
- [ ] Vollständiges Training für 2–3 Modelle (Qwen7B, Llama8B, optional Qwen14B mit QLoRA)
- [ ] Kleiner Hyperparameter-Sweep (r ∈ {8,16,32}, lr ∈ {1e-4, 2e-4})
- [ ] Auswahl der besten Hyperparameter basierend auf Val-Loss + Val-Top-1

**Woche 11:**
- [ ] FT-Inference auf Test-Set
- [ ] FT+RAG-Inference auf Test-Set
- [ ] Optional: RAFT-Training und Vergleich mit Simple FT+RAG

**Woche 12:**
- [ ] Vollständige Tabelle automatischer Metriken (alle 4 Methoden × alle Modelle × 3 Seeds)
- [ ] Erste Fehleranalyse (manuelle Betrachtung von 20–30 Beispielen)
- [ ] Reparaturen: bei niedriger Schema-Compliance Constrained Decoding einbauen

**Ergebnis Monat 3:** Alle vier Systeme für die ausgewählten Modelle funktionsfähig, automatische Ergebnisse bereit.

### Monat 4 — Menschliche Evaluation

**Woche 13:**
- [ ] Streamlit-App
- [ ] Pilotstudie (10–15 Items, 2–3 Rater)
- [ ] Rubric basierend auf Pilot-Feedback anpassen
- [ ] Bewertungsvarianz schätzen → finale Stichprobengröße bestimmen

**Woche 14:**
- [ ] Finale Auswahl von 100 Items für die menschliche Evaluation (Diversität gewährleisten)
- [ ] Assignment-Design (Balanced Incomplete Block Design):
  - 100 Items × 4 Systeme = 400 Outputs
  - jedes Output 3-mal bewerten lassen → 1200 Ratings nötig
  - mit 8 Ratern à 150 Ratings → vertretbarer Aufwand
- [ ] Rater-Recruitment (HCI/UX-Studierende oder Praktiker)

**Woche 15:**
- [ ] Kalibrierungssitzung mit Ratern (1–2 Stunden): Referenzbeispiele zeigen, Meinungsverschiedenheiten diskutieren
- [ ] Start der Ratings
- [ ] Fortschritt täglich überwachen

**Woche 16:**
- [ ] Alle Ratings einsammeln
- [ ] Krippendorffs α pro Dimension berechnen
- [ ] Bei α < 0.67 für einzelne Dimensionen → Diskussionsabschnitt in der Arbeit

**Ergebnis Monat 4:** Vollständige Ratings, IRR berechnet.

### Monat 5 — Statistische Analyse und Robustheit

**Woche 17:**
- [ ] Friedman-Test pro Rubric-Dimension
- [ ] Post-hoc Wilcoxon + Holm
- [ ] Cliffs δ für signifikante Paare
- [ ] Bootstrap-Konfidenzintervalle

**Woche 18:**
- [ ] Cochrans Q + McNemar für Top-1
- [ ] Robustheitstests: Paraphrase, Drop-Info, Noise
- [ ] Grounding-Analyse für B und D

**Woche 19:**
- [ ] Systematische Fehleranalyse (Kategorisierung von 100 Fehlern)
- [ ] Abbildungen (Matplotlib/Seaborn), hochauflösendes PDF
- [ ] Finale Ablations (LoRA-Rank, Retriever-Typ, RAFT vs. FT+RAG)

**Woche 20:**
- [ ] Zusammenführung aller Ergebnisse in einem zentralen DataFrame
- [ ] Sanity Checks: Stimmen Zahlen und Diagramme überein? Passen p-Werte zu Konfidenzintervallen?

**Ergebnis Monat 5:** Alle statistischen Ergebnisse, Abbildungen und Tabellen fertig.

### Monat 6 — Schreiben und Verteidigung

**Woche 21–23:**
- [ ] Verfassen der Kapitel (Einleitung → Schluss)
- [ ] Feedback vom Betreuer → Überarbeitung
- [ ] Reproduzierbarkeits-Check: jemand anderes (oder Sie selbst auf einem sauberen System) muss das Repo klonen und ausführen können

**Woche 24:**
- [ ] Vortragsfolien
- [ ] Generalprobe
- [ ] Verteidigung 🎉

---

## Teil 8: Vollständige Checkliste

### Daten
- [ ] Test-Set mit deterministischem Split (hash-basiert)
- [ ] Keine Test-Beispiele in Train oder Val (über Set-Mitgliedschaft prüfen)
- [ ] Augmented Daten (Paraphrasen) nur im Train, nicht in Val/Test
- [ ] KB-Chunks mit stabiler ID (bei Aktualisierung der KB ändern sich alte IDs nicht)

### Modelle & Training
- [ ] Jeder Training-Lauf wird mit W&B geloggt
- [ ] env.txt und git_hash liegen im Ordner jedes Laufs
- [ ] LoRA-Adapter, Basismodell und Tokenizer alle drei reproduzierbar abrufbar
- [ ] Eval-Loss kontinuierlich gefallen (kein Overfitting)
- [ ] Drei Seeds pro Konfiguration

### Inference
- [ ] Cache-Mechanismus funktioniert (Test: ein Lauf, erneuter Lauf → "CACHE HIT")
- [ ] `retrieved_docs` werden für RAG-Methoden gespeichert
- [ ] `latency_ms` wird für alle erfasst (für Effizienz-Tabelle)
- [ ] JSON-Parse-Fehler werden separat erfasst (nicht als Unter-Qualität, sondern als eigene Metrik)

### Evaluation – Automatisch
- [ ] Top-1, Top-3, Macro-F1 (über Chart-Typ-Klassen)
- [ ] Schema-Compliance: Valid-JSON-Rate + durchschnittliche Vollständigkeit
- [ ] Robustheit: Konsistenz unter Paraphrase / Drop / Noise
- [ ] Grounding-Proxy (nur B und D): unsupported_claim_rate
- [ ] Für RAG-Methoden: Retrieval-Precision@k (bei vorhandenen Gold-Passagen) oder Recall@k

### Evaluation – Menschlich
- [ ] Pilot durchgeführt
- [ ] Kalibrierungssitzung mit Ratern abgehalten
- [ ] Blind: Rater wissen nicht, welches System welches ist
- [ ] Varianten-Labels zufällig gemischt
- [ ] Jedes Item ≥3 Rater
- [ ] Krippendorffs α pro Dimension berichtet
- [ ] Inter-Rater-Varianz untersucht

### Statistische Analyse
- [ ] Friedman-Test pro Rubric-Dimension (Likert-ordinal)
- [ ] Wilcoxon Signed-Rank Post-hoc mit Holm-Korrektur
- [ ] Cliffs δ für Effektgröße
- [ ] Bootstrap-95-%-CIs (≥10 000 Iterationen)
- [ ] Cochrans Q für Top-1 (binäres Ergebnis)
- [ ] McNemar Post-hoc mit Holm
- [ ] Keine Signifikanzbehauptung ohne CI und Effektgröße

### Reproduzierbarkeit
- [ ] Jeder Lauf hat `config_snapshot.yaml`
- [ ] `config_hash.txt` für schnellen Abgleich
- [ ] `env.txt`
- [ ] `git_hash`
- [ ] Klare README: wie reproduziert man Tabelle X
- [ ] Einzeiler zur Reproduktion: `python scripts/run_experiment.py +experiment=EXX seed=42`

### Masterarbeitsdokument
- [ ] Formeln (LaTeX): LoRA, RRF, RAFT, Cliffs δ, Krippendorffs α
- [ ] Referenztabelle mit Chart-Typ-Labels
- [ ] Threats to Validity: subjektive Ratings, Dataset-Bias, Modellgröße, Compute
- [ ] Limitations-Abschnitt explizit
- [ ] Future Work: weitere Chart-Typen, mehrseitige Dashboards, End-to-End-UI-Generierung
- [ ] Link zum öffentlichen Repo (falls erlaubt)

---

## Teil 9: Algorithmen- und Formelverzeichnis

Verzeichnis aller Algorithmen und Formeln, die in der Masterarbeit verwendet oder zumindest erwähnt werden:

### Adaptationsalgorithmen
1. **LoRA** — Hu et al. 2021. \( W' = W + BA \), mit \( B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k} \), \( r \ll \min(d,k) \).
2. **QLoRA** — Dettmers et al. 2023. NF4-Quantisierung + Double Quant + Paged Optimizers.
3. **DoRA** — Liu et al. 2024. \( W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c} \) — Trennung von Magnitude und Direction.

### Retrieval-Algorithmen
4. **BM25** — Robertson & Spärck Jones. Klassisch, transparent.
5. **Dense Retrieval (BGE/E5)** — Bi-Encoder mit kontrastivem Training.
6. **Cross-Encoder Reranking** — Query und Dokument werden gemeinsam enkodiert für präziseren Score.
7. **Reciprocal Rank Fusion (RRF)** — \( \text{score}(d) = \sum_i \frac{1}{k + \text{rank}_i(d)} \), üblicherweise \( k=60 \).
8. **HyDE** — Hypothetical Document Embeddings: LLM erzeugt eine hypothetische Antwort, die abgefragt wird.

### Generation-/Decoding-Algorithmen
9. **Nucleus (Top-p) Sampling**
10. **Constrained Decoding (Outlines / lm-format-enforcer)** — Garantierte Grammar-/JSON-Schema-Konformität.

### Fortgeschrittene RAG-Algorithmen
11. **RAFT** — Zhang et al. 2024. Training mit Oracle + Distraktor-Passagen.
12. **Self-RAG** — Asai et al. 2024. Modell setzt Reflection-Tokens selbst.
13. **CRAG** — Yan et al. 2024. Fallback bei schwachem Retrieval.

### Evaluierungsalgorithmen
14. **Top-k Accuracy**
15. **Macro-F1**
16. **JSON Schema Validation (Draft-07)**
17. **CheckList Behavioral Testing**
18. **G-Eval / LLM-as-Judge** (Liu et al. 2023) — als Ergänzung, nicht als Ersatz für die menschliche Bewertung.
19. **RAGAS-Metriken**: Faithfulness, Answer Relevance, Context Precision/Recall.

### Statistik
20. **Krippendorffs α (ordinal)** — \( \alpha = 1 - \frac{D_o}{D_e} \).
21. **Friedman-Test** — \( \chi^2_F = \frac{12}{nk(k+1)} \sum_j R_j^2 - 3n(k+1) \).
22. **Wilcoxon Signed-Rank Test**.
23. **Holm-Bonferroni-Korrektur** — schrittweise verwerfend.
24. **Cliffs δ** — \( \delta = \frac{\#(x_i > y_j) - \#(x_i < y_j)}{n_x n_y} \).
25. **Bootstrap-Perzentil-CI** — \( B \geq 10\,000 \).
26. **Cochrans Q** — für k matched dichotome Stichproben.
27. **McNemar-Test (exakt)** — für 2 matched dichotome Stichproben.

### Behavioral Testing
28. **Paraphrase-Robustheit** — Invarianz-Test.
29. **Drop-Info-Robustheit** — Minimum-Functionality-Test.

---

## Teil 10: Ehrliche Einschätzung und Risiken

Dieser Abschnitt fasst die kritische Reflexion zum Plan zusammen. Bei einer Masterarbeit zählt die wissenschaftliche Tiefe – nicht die Eleganz der Architektur.

### Größte Risiken

1. **Over-Engineering**: Hydra + Registry + vLLM + W&B + RAFT + ... ist Production-Grade-ML-Engineering. Für **eine Person in 6 Monaten** ist das möglicherweise überdimensioniert. Wird der erste Monat ausschließlich für die Infrastruktur aufgewendet, fehlt am Ende Zeit für die wissenschaftliche Tiefe.

2. **Dataset-Mismatch**: Die im Proposal genannten öffentlichen Quellen (VizML, nvBench) liefern in der Regel **Einzelchart-Labels**, nicht **vollständige Dashboard-JSONs mit sechs Abschnitten**. Vor Beginn der Implementierung ist zu klären, ob das Gold-Dataset überhaupt aufgebaut werden kann (ggf. via synthetischer Daten durch ein leistungsstarkes LLM wie GPT-4o oder Claude). **Dies ist die wichtigste Entscheidung im Projekt.**

3. **Skalierung der Experimente**: 30–50 Experimente (mehrere Modelle × mehrere Methoden × Seeds × Ablations) sind realistisch nicht abbildbar. Empfehlung: **Tiefe statt Breite** – ein zentrales Modell (Qwen2.5-7B), eine Retriever-Variante (Dense + Reranker), und alle vier Methoden gründlich untersuchen. Zusätzliche Modelle nur in kleinen Ablations.

4. **Menschliche Evaluation als Flaschenhals**: 8 Rater zu rekrutieren, zu schulen und zur Mitarbeit zu motivieren, ist anspruchsvoll. Diese Phase sollte **früher** beginnen (Pilot bereits in Monat 2/3), nicht erst in Monat 4.

5. **Inter-Rater-Reliability (IRR)**: Erste Pilots erreichen typischerweise α<0.5. Ohne eine ordentliche Kalibrierungsphase besteht die Gefahr, dass die endgültigen Ergebnisse statistisch nicht aussagekräftig sind.

### Empfehlung für ein minimales realistisches Setup

Wenn die Zeit knapp wird, ist folgendes Minimal-Setup vertretbar:

- **Ein Basismodell** (Qwen2.5-7B-Instruct)
- **Vier Hauptmethoden** (A, B, C, D) wie im Proposal beschrieben
- **Ein Retriever** (Dense BGE-small + Cross-Encoder-Reranker)
- **QLoRA r=16** für Fine-Tuning
- **60 Items** in der menschlichen Evaluation (Untergrenze des Proposals)
- **Drei Seeds** pro System
- **Eine** Ablation (z. B. Retriever-Variante oder LoRA-Rank)

Dies ist solide, defensibel und in 6 Monaten realistisch umsetzbar.

### Vor Beginn unbedingt klären

Bevor mit der Implementierung begonnen wird, sollten folgende Fragen mit dem Betreuer geklärt werden:

1. Ist die Verwendung synthetischer Daten (von GPT-4o oder Claude) als Gold-Daten für das Fine-Tuning akzeptabel?
2. Welcher Aspekt ist für die Verteidigung wichtiger: methodische Breite (viele Modelle/Varianten) oder methodische Tiefe (eine sorgfältige Analyse)?
3. Ist das Open-Sourcen des Codes erlaubt/gewünscht?
4. Welche GPU-Ressourcen stehen tatsächlich zur Verfügung?
5. Gibt es spezielle Anforderungen der HRW an den formalen Aufbau der Arbeit?

### Häufige Fehler, die vermieden werden müssen

1. **Pfade hartkodieren** → stets `cfg.paths.X` verwenden.
2. **Experimente ohne W&B-Tracking ausführen** → später nicht mehr rekonstruierbar.
3. **`random.seed` ohne `set_all_seeds`** → Ergebnisse nicht reproduzierbar.
4. **Test-Set in Ablations verwenden** → Kontamination. Ablations nur auf Val.
5. **p-Wert ohne Effektgröße berichten** → der Gutachter fragt: "Signifikant, aber wie stark?"
6. **Automatische Metriken ohne Korrelation zu menschlicher Bewertung** → die Korrelation (Spearman/Pearson) muss berichtet werden.
7. **Nur ein Seed** → starke Varianz. Mindestens drei Seeds.
8. **JSON-Parse-Fehler ignorieren** → bei 30 % unparsbaren Outputs ist Top-1 auf den restlichen 70 % irreführend. Parse-Failure-Rate separat berichten.
9. **Schwache Knowledge Base** → bei nur 5 Dokumenten kann RAG keinen Mehrwert leisten. ≥30 vielfältige Dokumente.
10. **Keine Rater-Kalibrierung** → niedriges α, Gutachter hinterfragt zu Recht.

---

## Schlussübersicht

```
                    DATEN (deterministischer Split)
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
     TRAIN          VAL             TEST
       │               │               │
       │       (Hyperparameter-Tuning) │
       │               │               │
       ▼                               ▼
   ┌──────┐                         ┌──────────────────────────────┐
   │  FT  │ ─── Adapter ───┐        │ Inference aller 4 Methoden    │
   │QLoRA │                │        │ A: PromptOnly                 │
   └──────┘                │        │ B: RAG (Retriever-Varianten) │
                           ▼        │ C: FT (mit Adapter)           │
                         ┌──────┐   │ D: FT + RAG                   │
                         │ KB   │ ──┴→ predictions.jsonl × 4×N×M   │
                         │FAISS │                                    │
                         │ BM25 │                                    │
                         └──────┘                                    │
                                                                     ▼
                                                    ┌─────────────────────────┐
                                                    │ AUTOMATISCHE METRIKEN   │
                                                    │ • Top-1/3, F1           │
                                                    │ • Schema-Compliance     │
                                                    │ • Robustheit            │
                                                    │ • Grounding (B,D)       │
                                                    └────────────┬────────────┘
                                                                 │
                                       ┌─────────────────────────┤
                                       ▼                         ▼
                              ┌────────────────┐     ┌──────────────────────┐
                              │ HUMAN EVAL     │     │ STATISTISCHE TESTS   │
                              │ 100 Items × 4  │     │ Friedman/Wilcoxon    │
                              │ ≥3 Rater je    │ ─→  │ Cochran/McNemar      │
                              │ Krippendorffs  │     │ Cliffs δ, Bootstrap  │
                              │     α          │     └──────────┬───────────┘
                              └────────────────┘                │
                                                                ▼
                                                    ┌─────────────────────┐
                                                    │ ERGEBNISSE DER      │
                                                    │ MASTERARBEIT        │
                                                    │ • Tabellen          │
                                                    │ • Abbildungen       │
                                                    │ • Schlussfolgerung  │
                                                    └─────────────────────┘
```

Viel Erfolg. Bei Problemen mit einer konkreten Phase (z. B. "Woche 10 – QLoRA-Training stürzt mit OOM ab") gezielt nachfragen, um zielgerichtet zu helfen.
