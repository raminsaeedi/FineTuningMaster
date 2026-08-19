# CUDA-Inference- und Coverage-Korrektur

Stand: 19. August 2026

## Zweck

Diese Änderung behebt zwei voneinander getrennte Probleme:

1. Method C erzeugte im offiziellen Lauf keine Prediction. Der erste Fehler
   entstand beim probabilistischen CUDA-Sampling. Danach war der CUDA-Kontext
   nicht mehr zuverlässig, trotzdem verarbeitete die alte Schleife weitere
   Datensätze und protokollierte insgesamt 274 Fehler.
2. Automatische Metriken verwendeten teilweise nur vorhandene Predictions als
   Nenner. Vollständig fehlende Zeilen konnten dadurch aus dem Ergebnis
   verschwinden. Ein unvollständiger Run konnte besser aussehen, als er war.

Änderung bleibt absichtlich klein. Gemeinsame Inference- und
Evaluationsschnittstellen wurden korrigiert; keine Methode erhielt einen
GPU- oder Dataset-spezifischen Sonderweg.

## Beobachtete Ausgangsdaten

### Method A

- Erwartete Testfälle: 274
- Vorhandene Predictions: 262
- Vollständig fehlende Predictions: 12
- Alle 262 vorhandenen Zeilen waren schema-valide.
- Ursache der Unterbrechung ist in den übertragenen Artefakten nicht sicher
  dokumentiert. Fehlende Zeilen müssen deshalb als fehlend behandelt werden;
  eine Ursache darf nicht erfunden werden.

### Method C

- Offizielle Predictions: 0/274
- Erster Fehler: `CUDA error: device-side assert triggered`
- Entscheidender Stackframe: `torch.multinomial(probs, num_samples=1)`
- Weitere Fehler entstanden nach dem ersten CUDA-Assert teilweise bereits beim
  Übertragen neuer Inputs auf die GPU. Das ist typisches Verhalten eines nach
  einem fatalen CUDA-Fehler unbrauchbaren Prozesskontexts.
- Gespeicherter Adapter wurde direkt geprüft: 392 Tensoren, 17.432.576
  Parameter, kein NaN und kein Inf.
- Training wurde erfolgreich beendet und meldete `train_loss = 0.106767...`.

Damit ist kein beschädigtes Adapterfile belegt. Belegt ist ein numerischer
Fehler an der Sampling-Grenze unter FP16 sowie eine falsche Weiterverarbeitung
nach dem ersten fatalen CUDA-Fehler.

## Codeänderungen und Begründung

### 1. Sampling stabilisieren

Datei: `src/models/hf_causal.py`

`model.generate()` erhält standardmäßig:

```python
remove_invalid_values=True
renormalize_logits=True
```

`remove_invalid_values` bereinigt nicht-endliche Sampling-Scores vor
`torch.multinomial`. `renormalize_logits` normalisiert die danach veränderten
Scores erneut. Beide Werte können weiterhin über Generation-Konfigurationen
überschrieben werden.

Lösung ist modell- und GPU-unabhängig. Vorhandene dynamische Precision-Auswahl
bleibt zuständig:

- P100, V100 und T4 verwenden FP16.
- A100, L40S und andere BF16-fähige GPUs verwenden BF16.
- CPU verwendet FP32.
- Bei mehreren unterschiedlichen GPUs entscheidet das schwächste sichtbare
  Gerät.

### 2. Nach fatalem CUDA-Fehler sofort stoppen

Datei: `src/inference/runner.py`

Recoverable Fehler einzelner Items werden weiter protokolliert. Fehler wie
`device-side assert triggered`, `illegal memory access` oder
`unspecified launch failure` werden dagegen nach dem ersten Fehler erneut
ausgelöst. Der Prozess endet und gibt GPU-Ressourcen frei.

Grund: Derselbe CUDA-Prozess darf danach nicht für weitere Items verwendet
werden. Ein neuer Prozess kann den resumierbaren Run fortsetzen. Bereits
erfolgreiche Predictions bleiben im JSONL-Cache erhalten.

### 3. Vollständige Referenzkohorte verwenden

Dateien:

- `src/evaluation/metrics/base.py`
- `src/evaluation/metrics/schema_compliance.py`
- `src/evaluation/metrics/topk_accuracy.py`
- `src/evaluation/metrics/macro_f1.py`

Predictions werden über `item_id` an alle erwarteten Referenzen angefügt.
Fehlende Predictions werden nicht mehr entfernt:

- Schema- und Chart-Metriken zählen sie als Fehler.
- Macro-F1 verwendet dafür die explizite Klasse `(none)`.
- Per-KPI-Chartgenauigkeit verwendet alle erwarteten KPI-Zuordnungen als
  Nenner.

Macro-F1 wird vollständig in Python berechnet. Dadurch bleibt diese Kernmetrik
unabhängig vom schweren optionalen Scikit-learn/Pandas-Importpfad.

### 4. Latenz wissenschaftlich korrekt berichten

Datei: `src/evaluation/metrics/latency.py`

Mean, Median und p95 werden nur aus tatsächlich gemessenen Predictions
berechnet. Für fehlende Predictions wird keine künstliche Laufzeit eingesetzt.
Zusätzlich werden `n_measured`, `n_requested` und `coverage_rate` ausgegeben.

Beispiel: 262 gemessene Laufzeiten bei 274 erwarteten Predictions ergeben
95,62 Prozent Latenz-Coverage.

### 5. Strukturierte Exact-Match-Metriken ergänzen

Datei: `src/evaluation/metrics/structured_exact_match.py`

Neue vollständige, zeilenbasierte Metriken:

- `exact_task_classification`: geordnete Task-Liste exakt gleich
- `exact_kpi_selection`: geordnete KPI-Liste exakt gleich
- `exact_mapping_count`: Anzahl Zuordnungen exakt gleich
- `exact_encoding`: komplette geordnete Encoding-Liste exakt gleich
- `exact_aggregate`: Aggregate-Felder exakt gleich, wo Gold anwendbar ist

Zusätzlich werden `db_id`, `source` und `n_kpis` diagnostisch geprüft. Diese
Felder sind im normalen Benutzerprompt nicht sichtbar. Ihre Werte dürfen daher
nicht als Beleg für factual grounding interpretiert werden.

Die Metrik ist in `full.yaml` und `with_judge.yaml` aktiviert und erscheint
auch in der Multi-Seed-Zusammenfassung.

### 6. Coverage speichern und unvollständige Runs ablehnen

Datei: `src/pipeline/runner.py`

`metrics_auto.json` enthält jetzt für Original und konfigurierte Robustness-
Varianten:

- `n_requested`
- `n_predictions`
- `n_missing`
- `prediction_coverage_rate`
- `missing_item_ids`

Metriken und Reports werden auch bei unvollständiger Coverage geschrieben,
damit Diagnose möglich bleibt. Danach schlägt der Run bewusst fehl. So bleibt
er resumierbar und wird nicht als abgeschlossen veröffentlicht.

Datei: `experiments/scripts/run_final_matrix.py`

Ein Run gilt nur noch als vollständig, wenn:

- Coverage-Daten vorhanden und valide sind,
- `n_requested > 0` gilt,
- `n_predictions == n_requested` gilt,
- `n_missing == 0` gilt,
- dieselben Bedingungen für jede vorhandene Robustness-Variante gelten.

Alte `metrics_auto.json` ohne Coverage werden nicht mehr als vollständig
akzeptiert. Sie werden beim nächsten Matrixlauf neu evaluiert; vorhandene
kompatible Predictions bleiben nutzbar.

## Verifizierte Method-A-Ergebnisse

Neuberechnung über vollständige 274er-Testkohorte:

| Metrik | Ergebnis |
|---|---:|
| Prediction-Coverage | 262/274 = 95,62 % |
| Schema validity | 95,62 % |
| Chart Top-1 | 83,94 % |
| Macro-F1 | 0,4656 |
| Exact task classification | 12,77 % |
| Exact KPI selection | 93,80 % |
| Exact mapping count | 94,53 % |
| Exact encoding | 0,00 % |
| Exact aggregate | 0,00 % |
| Mean latency | 12,385 s |
| Median latency | 12,278 s |
| p95 latency | 13,475 s |
| Latenz-Coverage | 262/274 = 95,62 % |

`exact_encoding = 0` und `exact_aggregate = 0` sind echte strikte Ergebnisse.
Beispiel: Gold enthält vollständige Encoding-Angaben wie `source_x`,
`source_y`, `aggregate` und Gruppierung, während eine Prediction teilweise nur
`{"value": "count"}` enthält. Berechnung ist vorhanden; Strukturen stimmen
nicht exakt überein.

## Was jetzt funktioniert

- Gemeinsamer Evaluationsweg gilt für A, B, C und D.
- Fehlende Original- und Robustness-Predictions bleiben sichtbar.
- Unvollständige Runs können nicht als vollständig übersprungen werden.
- Alle verlangten automatischen Strukturmetriken werden berechnet und in
  Einzel- sowie Multi-Seed-Ergebnissen gespeichert.
- Inference bleibt resumierbar.
- GPU-Precision und Pfade bleiben dynamisch für HPC, Kaggle, Colab und lokale
  Systeme.

## Was noch nicht als Ergebnis existiert

- Historischer C-Run bleibt 0/274 und ist wissenschaftlich unbrauchbar. Er muss
  mit geändertem Code erneut ausgeführt werden.
- Ein echter GPU-Lauf ist für die empirische Bestätigung der C-Korrektur nötig;
  Unit-Tests können keinen P100/V100-CUDA-Kernel ersetzen.
- Human-perceived quality benötigt weiterhin echte menschliche Ratings. Der
  Full Run erzeugt Predictions; Blind-Study-Paket und Ratings sind getrennte
  Schritte und verlangen vollständige A/B/C/D-Ausgaben.
- Factual-grounding-Metrik gilt aktuell für RAG-Evidenz von B und D. Für A und C
  darf ohne sichtbare Quellen oder unabhängiges Grounding-Gold keine künstliche
  Zahl erzeugt werden.

## Empfohlener nächster C-Lauf

Vorhandenen endlichen Adapter zuerst wiederverwenden; kein erneutes Training
erzwingen:

```bash
python experiments/scripts/run_final_matrix.py \
  --profile final \
  --model qwen3_1_7b \
  --method C \
  --seed 42 \
  --skip-training \
  --input-model-weights /absolute/path/to/C/seed_42/adapter
```

Für einen völlig sauberen Diagnoseverlauf sollte ein neuer Output-Root benutzt
oder die alte `errors.jsonl` vorher separat archiviert werden. Predictions und
Metriken des alten C-Laufs dürfen nicht als Resultat wiederverwendet werden.

## Double-Check

Ausgeführt nach letzter Korrektur:

- Gesamte Testsuite: 835 Tests bestanden
- A/B/C/D Hydra-Konfigurationen erfolgreich zusammengesetzt
- `structured_exact_match` in allen vier Full-Profilen aktiv
- Alle YAML-Dateien erfolgreich geparst
- Python-Syntaxprüfung erfolgreich
- `git diff --check` ohne Fehler
- C-Adapter vollständig auf NaN/Inf geprüft
- Method-A-Metriken direkt aus übertragenen HPC-Predictions und gefrorenem
  `dashboard_v4`-Testset neu berechnet

## Grenzen der Bestätigung

Codepfade, Metriken, Konfigurationen und gespeicherte Artefakte sind lokal
vollständig geprüft. Frische Modellgeneration auf P100, V100, T4 oder A100 war
in dieser lokalen Umgebung nicht möglich. Aussage deshalb präzise:

- Code- und Evaluationskorrektur ist getestet.
- GPU-Portabilität folgt gemeinsamer dynamischer Precision-Logik.
- Endgültiger empirischer Nachweis für Method C entsteht erst durch frischen
  vollständigen GPU-Run mit 274/274 Predictions und vollständigen Varianten.
