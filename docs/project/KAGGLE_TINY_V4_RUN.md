# Kaggle Tiny dashboard_v4 Run

Zweck: vollständigen Pipeline-Test auf einer kleinen GPU. Kein Thesis-Endergebnis.

## Daten

- `dashboard_v4_tiny`: 100 Train, 50 Validation, 50 kanonischer In-Domain-Test.
- `sports_v4_tiny`: 50 Sport-Holdout-Beispiele für Cross-Domain-Evaluation.
- Train und Validation enthalten keine explizit markierten Sport-Daten.
- Alle Auswahlschritte sind deterministisch mit Seed `20260820`.
- `manifest.json`, `hashes.json`, Reports und zwei Human-Eval-CSV-Vorlagen liegen unter `data/frozen/dashboard_v4_tiny/`.

## Kaggle-Kommandos

```bash
python experiments/scripts/build_dashboard_v4_tiny.py --verify
python experiments/scripts/run_tiny_v4_kaggle.py
```

Standard: Qwen2.5-0.5B, QLoRA, 1 Epoche, 100 Train, A/B/C/D auf 50 In-Domain-Beispielen, danach A/C auf 50 Sport-Beispielen. Falls RAG-Dateien fehlen, baut Runner `chunks.jsonl` selbst.

Schneller Kernlauf nur für Baseline und Fine-Tuning:

```bash
python experiments/scripts/run_tiny_v4_kaggle.py --methods A C
```

Nur Kommandos und Konfiguration prüfen:

```bash
python experiments/scripts/run_tiny_v4_kaggle.py --dry-run --methods A C
```

## Kleine GPU

Default nutzt `max_seq_length=1024`, Batch 1, Gradient Accumulation 8, 4-bit NF4 QLoRA und 512 maximale Ausgabetokens. Für Kaggle T4/L4 mit mehr VRAM kann längerer Kontext sinnvoll sein:

```bash
python experiments/scripts/run_tiny_v4_kaggle.py --max-seq-length 1536
```

Bei Unterbrechung:

```bash
python experiments/scripts/run_tiny_v4_kaggle.py --resume
```

## Erfolgsbedingung

Runner endet nur erfolgreich, wenn jeder angeforderte Ergebnisordner `predictions.jsonl`, `metrics_auto.json`, Manifest, Config-Snapshot, Cache-Identität und vollständige 50/50 Prediction-Coverage enthält. Parse-Fehler des Modells zählen als Qualitätsproblem, nicht als fehlende Ausführung.

## Interpretation

Tiny-Metriken testen Pipeline, Adapter, Caching, Reports und Cross-Domain-Wiring. Sie sind wegen 100 Trainingsbeispielen, einer Epoche und einem Seed keine belastbaren Qualitäts- oder Thesis-Ergebnisse.
