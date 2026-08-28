# Final-Run-Readiness v4

Stand: 2026-08-26. Original Go/No-Go plan created 2026-08-20.

## Current update: 2026-08-26

New full-data artifacts change run inventory:

- Qwen3-1.7B A: seeds 42, 43, 44 complete.
- Qwen3-1.7B B: seeds 42, 43, 44 complete.
- Qwen3-1.7B C: seed 42 evaluated; seeds 43 and 44 have training artifacts only.
- Qwen3-1.7B D: seed 42 complete with C seed-42 adapter and RAG.
- OLMo A: seeds 42, 43, 44 complete.
- OLMo B: all seeds failed after 35/274 predictions because RAG passage headings exceeded input-token budget.
- OLMo C/D: no results.

Qwen seed-42 A-D execution is now present, but it is not thesis-ready. Strict schema validity is A 0.36%, B 1.09%, C 1.46%, and D 0%. D raw chart diagnostic is 92.78%, below C's approximately 97.0%; no C-to-D quality improvement is shown. C/D multi-seed evaluation, independent quality evaluation, confidence-interval repair, real git revision capture, and D adapter provenance cleanup remain open.

OLMo A is faster than Qwen A but has much lower JSON parse and completeness. OLMo B is not a valid quality comparison until its model-specific RAG prompt-budget failure is fixed. Full comparison: [dashboard v4 Qwen/OLMo analysis](dashboard_v4_qwen3_1_7b_full_analysis.md).

## Historische Entscheidung (2026-08-20)

Noch kein großer Final-Run.

Der konkrete B/D-Formatfehler ist repariert. Der neue 20-Item-B-Pilot erreicht 100% JSON-Parse und 90% strict schema validity statt zuvor 25%/0% auf denselben 20 IDs. Der Gesamt-Output ist trotzdem noch nicht thesis-ready: Completeness bleibt 0.333, D wurde nach der Promptänderung noch nicht mit einem neu trainierten C-Adapter geprüft, Robustheit und der finale unabhängige Benchmark fehlen, Human Ratings sind null.

## Bewiesene Ursache

Tiny-Konfiguration:

- `max_seq_length = 1024`
- `max_new_tokens = 512`
- verfügbares Input-Budget = 512 Tokens
- Tokenizer-Trunkierung: rechts

Vor dem Fix:

- A/C-Prompt: 499–537 Tokens; 25/50 über Budget.
- B/D-Prompt: 671–839 Tokens; 50/50 über Budget.
- Der 512-Token-Schnitt lag beim ersten B-Fall mitten im JSON-Beispiel nach `"task_type": "trend`.
- Das Modell vervollständigte deshalb ein bereits begonnenes Fragment. Das erklärt Antworten, die mit `"}, {` oder anderen JSON-Mittelteilen begannen.

Hugging Face dokumentiert, dass `truncation=True` auf `max_length` kürzt; `max_new_tokens` reserviert separat die maximale neue Ausgabe. Das alte Zusammenspiel war daher deterministisch falsch, kein RAG-Retrievalfehler: [Padding and truncation](https://huggingface.co/docs/transformers/pad_truncation), [GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation).

## Implementierter Minimalfix

- Prompt von 499–537 auf 322–360 Tokens komprimiert. Sechs Pflichtfelder, Typen und Enums bleiben erhalten.
- RAG-Kontext wird token-genau gekürzt. Das Restbudget wird gleichmäßig über alle drei Top-k-Passagen verteilt.
- Keine stille Modell-Trunkierung mehr. Budgetüberlauf erzeugt einen harten, erklärenden Fehler.
- Jede neue Prediction speichert `prompt_input_tokens`, `prompt_input_budget`, `rag_context_truncated`.
- Tiny-Runner prüft Coverage, JSON parse, strict schema, required keys und Completeness.
- `eval_max_samples` trennt 10–20-Item-Evaluation von den 100 Tiny-Train-Items.
- Validation wird geladen; Evaluation und Checkpointing erfolgen pro Epoche; bestes `eval_loss`-Checkpoint wird geladen und mit Verlauf gespeichert. Das entspricht dem dokumentierten Trainer-Vertrag: [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

Nach dem Fix passen alle geprüften Prompts:

- A/C: 322–360/512 Tokens; kein Überlauf.
- B/D: 494–512/512 Tokens; kein Überlauf.
- 48/50 RAG-Prompts werden kontrolliert gekürzt; zwei passen vollständig.
- Alle drei Retrieval-Passagen bleiben in jedem RAG-Prompt vertreten.

RAG bleibt methodisch sinnvoll: Retrieval und Generator werden getrennt betrachtet; Provenienz und Konditionierung auf externe Passagen sind zentrale Eigenschaften des ursprünglichen RAG-Ansatzes: [Lewis et al. 2020](https://arxiv.org/abs/2005.11401).

## Alt/Neu auf denselben 20 Dashboard-IDs

- A alt: parse 45%, schema 45%, Top-1 35%, Macro-F1 0.2111.
- A neu: parse 90%, schema 90%, Top-1 60%, Macro-F1 0.2857.
- B alt: parse 25%, schema 0%, Top-1 0%, Macro-F1 0.
- B neu: parse 100%, schema 90%, Top-1 75%, Macro-F1 0.6236.
- B neu: Coverage 20/20; drei Retrieval-Dokumente pro Item; kein OOM; keine Laufzeitfehler.
- B neu: lexical grounding support 61.03% über 32 Claims. Nur Proxy, keine Faithfulness-Garantie.

Zwei strict-schema-Fehler bleiben:

- `task_type = "comparing"` statt `comparison`.
- `chart_type = "count"` statt erlaubtem Chart-Typ.

Das sind jetzt echte Modell-/Enumfehler, keine abgeschnittenen Fragmente. Das ist die richtige Fehlerklasse für eine Evaluation.

## Was die aktuellen KPIs wirklich bedeuten

- `json_parse_rate`: vollständiges JSON-Objekt extrahierbar. Markdown-Fences können trotzdem vorhanden sein.
- `schema_validity_rate`: Pydantic-Typen und Enums gültig. Leere Dicts/Strings sind noch möglich.
- `completeness_score`: sechs Top-Level-Felder nicht leer. Aktuell 0.30–0.333; deshalb kein Release.
- `exact_encoding = 0%`: gesamte Gold-Encoding-Dictionaries müssen exakt identisch sein. Sehr streng; kein Beweis allein, dass die Pipeline defekt ist.
- `exact_aggregate = 0%`: Aggregate werden im `encoding` nicht zuverlässig reproduziert. Vor dem Final-Run Prompt/Metric sichtbar ausrichten.
- `top_3_valid = false`: korrekt. Dashboard-v4-Test-Gold enthält in 274/274 Fällen keine Alternativen; Training enthält nur bei 27% der Mappings mindestens zwei. Top-3 ist daher Diagnostik, kein primärer Thesis-KPI.
- Interne Top-1/Macro-F1-Werte gegen synthetisches Gold sind zirkuläre Diagnostik. Keine Hauptaussage über reale Designqualität.

Grammar-constrained decoding kann syntaktische Gültigkeit garantieren, nicht semantisch gute KPI-, Encoding- oder Dashboard-Entscheidungen. Falls genutzt, dann für A–D identisch und als separates „constrained“-Setting; unconstrained Resultate getrennt berichten: [Park et al. 2025](https://proceedings.mlr.press/v267/park25l.html).

## Korrektur der sechs ursprünglichen Selections

1. Final thesis run: weiterhin nein. B-Formatblocker repariert; D-Neutraining und Evaluationsebenen offen.
2. B/D RAG wiring: bestätigt. Retrieval selbst war korrekt; Promptbudget war falsch.
3. B/D 0% schema-valid: historischer Tiny-Befund. B erreicht nach Fix 90% im 20-Item-Pilot. D noch nicht neu gemessen.
4. Fragmentfehler: behoben. Exact encoding/aggregate, Robustheit, L3, Human und Multi-Seed bleiben getrennte offene Punkte.
5. Structured JSON goal: B nicht mehr 0%; trotzdem Completeness-Gate nicht bestanden.
6. Required gates: Prompt-/Format-/Validation-Gates implementiert. Multi-Seed, Robustheit, finaler unabhängiger Benchmark, L3-Entscheidung und Human Ratings noch auszuführen.

## Verbindliche Gates

Diese Grenzwerte sind vorab definierte Engineering-Release-Kriterien, keine aus den Testdaten optimierten wissenschaftlichen Schwellen:

- Coverage = 100%; `n_missing = 0`; keine `errors.jsonl`-Einträge.
- Jeder Prompt: `prompt_input_tokens <= prompt_input_budget`.
- RAG: genau drei Retrieval-Ergebnisse; alle drei im budgetierten Kontext vertreten.
- JSON parse ≥95%.
- Strict schema validity ≥90%.
- Required keys ≥95%.
- Completeness ≥0.80.
- C: nicht-leerer Validation-Verlauf; endliches `eval_loss`; bestes Checkpoint dokumentiert.
- D: Adapter-Hash/Seed/Dataset kompatibel mit C.
- Keine Methoden-spezifische JSON-Reparatur. Gleiche Post-Processing-Regeln für A–D.
- Frozen dataset/KB hashes unverändert; sauberer Commit-Hash im Run.

Gate-Werte dürfen nach Sichtung der Final-Testresultate nicht nachträglich angepasst werden.

## Ausführungsplan

### Phase 1 — Kaggle-Pilot, 20 Items

Ein Modell, Seed 42, A–D. C trainiert auf allen 100 Tiny-Train-Items und validiert auf 50; Evaluation nur 20 Items:

```powershell
python experiments/scripts/run_tiny_v4_kaggle.py --eval-items 20 --force
```

Go nur, wenn alle Gates bestehen. Falls nur Completeness scheitert: Prompt und Trainingsziel einmal prüfen; nicht auf dem Final-Test iterativ tunen.

### Phase 2 — Tiny-Full, 50 Items

Gleicher Code, gleiche Einstellungen, A–D plus Sports A/C. Zweck: Artefaktvollständigkeit, Adapter-Reuse, Streamlit-Eingaben, Laufzeit/P95. Keine Thesis-Hypothesentests.

### Phase 3 — Dashboard-v4 Seed 42

Zuerst ein ausgewähltes Finalmodell, nicht vier Modelle gleichzeitig. A–D auf 274 Testfällen plus vorhandene `robustness_v4`-Varianten. Danach:

- L2: parse/schema/completeness, robustness, grounding mit Modus.
- Interne Diagnostik: Top-1, Macro-F1, exact fields; ausdrücklich zirkulär markieren.
- Unabhängig: `benchmark_v1` für alle A–D; 22/30 automatisch L1-scorable, Coverage 0.7333.

Der unabhängige L1-Scorer ist bereits implementiert. Fehlend ist der finale A–D-Benchmarklauf; vorhanden ist nur ein alter A-Smoke-Lauf. Die Literaturbasis ist task- und datenabhängig: [Saket et al.](https://arxiv.org/abs/1709.08546), [Kim & Heer](https://idl.uw.edu/papers/task-data-effectiveness).

### Phase 4 — Seeds 43/44

Erst nach bestandenem Seed-42-Lauf. Gleiche Konfiguration, keine Promptänderung. Mittelwert/Streuung pro Seed; paarweise Item-Tests und Bootstrap-CIs. Ein einzelner Seed genügt nicht für Stabilitätsaussagen: [Reimers & Gurevych 2017](https://aclanthology.org/D17-1035/). Testwahl und Mehrfachvergleiche vorab festlegen: [Dror et al. 2018](https://aclanthology.org/P18-1128/).

### Phase 5 — Human Evaluation

- 5-Item-Pilot, 2–3 Rater; Pilotdaten aus finaler Analyse ausschließen.
- Danach 40 Items × 4 Methoden × 3 Rater = 480 Ratings.
- Blind, randomisierte Reihenfolge, gleiche Items je Methode.
- Sechs getrennte 1–5-Dimensionen; Krippendorff α ordinal; Friedman; Wilcoxon + Holm; Bootstrap-CIs; Effektgrößen.
- Erst nach erfolgreichem automatischem Gate das Streamlit-Paket bauen. Sonst bewerten Menschen Parsermüll.

Mehrere Annotatoren, getrennte Kriterien, begründete Stichprobe, Agreement und Randomisierung entsprechen etablierten NLG-Empfehlungen: [van der Lee et al. 2019](https://aclanthology.org/W19-8643/).

### Phase 6 — L3-Entscheidung

L3 ist nicht implementiert; Tableau-Corpus/Lizenz-Mapping fehlt. Minimal vertretbare Wahl:

- entweder L3 vorab aus dem Claim-Scope entfernen und als Future Work deklarieren;
- oder Census-Daten erwerben/mappen und nur deskriptive Strukturähnlichkeit berichten: Chart-/Interaktionsverteilungen, View-/KPI-Anzahl, Jensen-Shannon/Total-Variation plus Bootstrap-CI.

Der Census enthält 25.620 Tableau-Public-Dashboards und beschreibt reale Nutzung, nicht optimale Wirksamkeit: [Purich et al. 2023](https://arxiv.org/abs/2306.16513).

## Finales Go

Großer Lauf erst, wenn:

- frischer C/D-20-Pilot mit neuem Adapter besteht;
- Validation-Metadaten vorhanden sind;
- Format- und Completeness-Gates bestehen;
- unabhängiger Benchmark-Befehl auf Pilot-Artefakten funktioniert;
- Robustheitsdateien tatsächlich Prediction-Zeilen erzeugen;
- Human-Eval-Paket aus vier kompatiblen Runs gebaut und in Streamlit geöffnet werden kann;
- Git-Status/Commit und Frozen-Hashes festgehalten sind.

Dann ist der Code experimentbereit. Thesis-ready werden die Resultate erst nach drei Seeds, unabhängiger L1-Auswertung und Human Ratings. L3 ist nur nötig, wenn RQ1d im Claim-Scope bleibt.

## Implementierungsstand 2026-08-28

Code-seitig umgesetzt:

- Separates striktes Generation-Schema. `encoding` muss Objekt mit nicht-leerem `x`, `y` sowie `aggregate` aus `SUM|AVG|COUNT|MIN|MAX|null` sein. Lenienter Parser für historische Artefakte bleibt unverändert.
- Outlines-Pfad auf API 1.3 aktualisiert; Dependency exakt auf `outlines==1.3.3` fixiert. Constrained Decoding bleibt optional und verändert alte Runs nicht.
- `schema_validity_rate` validiert jetzt dasselbe strikte Schema wie Constrained Decoding.
- Explizite `encoding_object_rate` sowie Mapping-Metriken für `x`, `y`, `aggregate` und gemeinsamen Mapping-Treffer ergänzt. Fehlende/ungültige Vorhersagen bleiben im Nenner.
- `json_parse_rate`-Konfidenzintervall nutzt rohe JSON-Extraktion; Pydantic-/Schemafehler werden nicht länger als JSON-Parsefehler gezählt.
- Optionales TRL Prompt-Completion-Training über `training.sft.completion_only_loss=true`. Standard bleibt `false`, damit alte C/D-Runs reproduzierbar bleiben. Neue Loss-Variante braucht neuen Adapter und neue Run-ID.
- RAG-Relevanzmetrik mit manuellen Qrels: Recall@3, MRR@3, nDCG@3, Query-Coverage und Anteil von drei eindeutigen Retrieval-IDs. Ohne Qrels keine erfundene Zahl; A/C werden als nicht anwendbar markiert.
- Verbindlicher Pilot-/HPC-Plan: [Constrained Encoding and HPC Validation Protocol](constrained_encoding_hpc_protocol.md).

Lokale Verifikation:

- `907 passed` im vollständigen CPU-Testlauf.
- Poetry-Lockdatei konsistent; `outlines==1.3.3` reproduzierbar fixiert.
- A-D/Seed-42/20-Item-Matrix mit constrained Decoder, 1024 Output-Tokens, 4096 Kontext und response-only Loss erfolgreich als Dry-Run komponiert. Kein GPU-Lauf ausgeführt.

Noch nicht empirisch bewiesen:

- Kein GPU-Inferenzlauf mit neuem striktem Decoder wurde in dieser Änderung ausgeführt.
- Kein response-only C-Adapter wurde trainiert.
- Keine manuellen Retrieval-Qrels wurden erstellt.
- Vorhandene A-D-Ergebniszahlen bleiben historische unconstrained/full-sequence Resultate.

Urteil: Implementierung ist bereit für 20-Item-Test. 27B bleibt No-Go, bis 20er und 50er Gates bestanden sind. Ein erfolgreicher Format-Gate beweist nur Output-Zuverlässigkeit; fachliche Encoding-, RAG- und Dashboard-Qualität bleiben getrennte Messungen.
