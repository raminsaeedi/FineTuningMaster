# HPC Results: Konsolidierung und wissenschaftlicher Status

**Stand:** 2026-09-09  
**Datensatz:** `dashboard_v4`  
**Originalquelle:** `C:\Projects\MasterArbeit\HPC Results`  
**Neu eingearbeitet:** `new result 001` und `new result 002` (Spark-Exporte)  
**Wichtig:** Die Originaldateien wurden nicht gelöscht oder verändert. Relevante Dateien wurden in die Projektstruktur kopiert.

## 1. Kurzes Ergebnis

- Die verstreuten Resultate wurden nach vollständigen Läufen, unvollständigen Läufen, Adaptern und aggregierten Auswertungen getrennt.
- **32 Läufe sind technisch vollständig:** jeweils 274 Original-, 274 Paraphrase- und 274 Missing-Info-Vorhersagen, gültige JSONL-Dateien, eindeutige IDs und eine automatische Metrikdatei.
- **8 alte, partielle oder dublettenbelastete Laufordner** liegen getrennt unter `incomplete` und werden nicht als finale Ergebnisse behandelt.
- **8 trainierte C-Adapter sind vollständig vorhanden:** Adapterkonfiguration, Safetensors-Gewichte und Trainingsmetadaten existieren.
- Aggregation und Figurenerzeugung funktionieren. Es wurden Vergleichstabellen, ein automatischer Bericht und fünf Figuren in PNG und PDF erzeugt.
- Die Sammlung ist **noch nicht vollständig thesis-final**, weil wichtige Modell-/Methoden-Kombinationen fehlen und einige vorhandene Läufe aus unterschiedlichen Code- und Konfigurationsständen stammen.

## 2. Neue, geordnete Ablage

| Inhalt | Projektpfad | Bedeutung |
|---|---|---|
| Technisch vollständige Läufe | `experiments/outputs/final/dashboard_v4/` | Nur Läufe mit 274/274/274 Vorhersagen |
| Unvollständige Läufe | `experiments/outputs/incomplete/dashboard_v4/` | Teilresultate; nicht als finales Thesis-Ergebnis verwenden |
| Trainierte Adapter | `experiments/outputs/adapters/dashboard_v4/` | Wiederverwendbare LoRA/QLoRA-Adapter von Methode C; D verwendet den passenden C-Adapter |
| Aggregierte Tabellen und Berichte | `experiments/results/final/dashboard_v4/` | Automatisch erzeugte Vergleiche |
| Figuren | `experiments/results/final/dashboard_v4/figures/` | F1 bis F5 als PNG und PDF |
| Legacy-Sicherungen | `experiments/outputs/legacy/` und `experiments/outputs/legacy_adapters/` | Ersetzte ältere Protokollstände; nicht gelöscht |
| Weitergabepaket | `experiments/results/packages/dashboard_v4/professor_results_dashboard_v4_consolidated.zip` | Sichere Resultatdateien ohne Modellcache und virtuelle Umgebung |

Nicht kopiert wurden Hugging-Face-Modellcaches, `.venv`, Checkpoints, Pilotläufe und `_stale_cache`. Diese Dateien sind groß, für die Resultatanalyse unnötig oder nicht final.

## 3. Herkunft der übernommenen Daten

| Quellordner | Übernommener Inhalt |
|---|---|
| `ftm_runtime` | Qwen 1.7B A/B, ältere C/D-Seed-42-Läufe, OLMo A sowie Adapter |
| `lightingAi` | Qwen 1.7B C/D für Seeds 43 und 44 sowie Adapter |
| `ramin_khareh` | Qwen 27B A/B, unvollständige C/D-Ausgaben und vollständiger C-Adapter |
| `ramin_khareh2` | OLMo C/D und die zugehörigen Adapter |
| `ftm-spark` | Nichts übernommen; die gefundenen finalen Dateien waren Null-Byte-Platzhalter |
| `new result 001` | Neue vollständige Qwen-27B-C/D-, OLMo-B- und Qwen-8B-C/D-Läufe; aktueller Qwen-8B-Adapter |
| `new result 002` | Bereinigte, vollständige Qwen-8B-A/B-Läufe sowie nicht benötigte Zweitkopie der bereits vorhandenen C/D-Dateien |

## 4. Technisch vollständige Läufe

| Modell | A | B | C | D | Technischer Status |
|---|---:|---:|---:|---:|---|
| Qwen3 1.7B | Seeds 42, 43, 44 | Seeds 42, 43, 44 | Seeds 42, 43, 44 | Seeds 42, 43, 44 | Alle Dateien vollständig; Protokollkonflikt siehe Abschnitt 7 |
| OLMo 2 1.49B | Seeds 42, 43, 44 | Seeds 42, 43, 44 | Seeds 42, 43, 44 | Seeds 42, 43, 44 | Komplette A–D-Matrix; A-Protokollstände beachten |
| Qwen3.8 27B | Seed 42 | Seed 42 | Seed 42 | Seed 42 | Komplette A–D-Matrix für einen Seed; Protokollstände beachten |
| Qwen3 8B | Seed 42 | Seed 42 | Seed 42 | Seed 42 | Komplette A–D-Matrix; Inferenz-Protokollstände beachten |

Jeder hier als vollständig markierte Lauf wurde auf folgende Punkte geprüft:

1. `predictions.jsonl`: 274 Datensätze
2. `predictions_paraphrased.jsonl`: 274 Datensätze
3. `predictions_missing_info.jsonl`: 274 Datensätze
4. Jede Zeile ist gültiges JSON
5. Item-IDs sind innerhalb einer Datei eindeutig
6. `metrics_auto.json` ist vorhanden

Ergebnis der Prüfung: **32 Läufe, 0 Dateiprobleme**.

## 5. Unvollständige Läufe

| Modell | Methode | Seed(s) | Original | Paraphrase | Missing info | Bewertung |
|---|---|---|---:|---:|---:|---|
| OLMo 2 1.49B | B | 42, 43, 44 | je 35 | je 35 | je 43 | Abgebrochen; nicht final verwenden |
| Qwen3.8 27B | C | 42 | 274 | 0 | 0 | Nur Originaltest fertig |
| Qwen3.8 27B | D | 42 | 272 | 0 | 0 | Originaltest fast fertig, Robustheit fehlt vollständig |
| OLMo 2 1.49B | A | 43 | 36 | 0 | 0 | Neuer Spark-Versuch abgebrochen; alter vollständiger Lauf bleibt separat vorhanden |
| Qwen3 8B | A | 42 | 548 | 548 | 548 | Je 274 IDs doppelt; 263–269 IDs haben widersprüchliche Generationen |
| Qwen3 8B | B | 42 | 274 | 548 | 548 | Robustheitsdateien enthalten je 274 doppelte IDs mit widersprüchlichen Generationen |

Diese Dateien bleiben erhalten und können zur Fehlersuche oder für echtes Resume dienen. Sie liegen bewusst nicht im finalen Ordner.

## 6. Adapterbestand

| Modell | Methode | Seeds | Status |
|---|---|---|---|
| Qwen3 1.7B | C | 42, 43, 44 | Gewichte, Konfiguration und Metadaten vorhanden |
| OLMo 2 1.49B | C | 42, 43, 44 | Gewichte, Konfiguration und Metadaten vorhanden |
| Qwen3 8B | C | 42 | Aktueller Spark-Adapter, 4096 Kontext, Gewichte und Metadaten vorhanden; passt zu C/D |
| Qwen3.8 27B | C | 42 | Gewichte, Konfiguration und Metadaten vorhanden; C/D-Inferenz vollständig |

Der ältere Qwen-8B-Adapter mit 512 Kontext wurde vor Ersetzung unter `legacy_adapters` gesichert. Der neue 4096-Kontext-Adapter erzeugte die übernommenen Qwen-8B-C/D-Läufe.

## 7. Wissenschaftliche Vergleichbarkeit

### 7.1 Qwen3 1.7B

Die Dateien sind technisch komplett, aber nicht alle Läufe gehören zum gleichen Versuchsprotokoll:

| Gruppe | Training-Hash | Inferenz-Hash | Git-Stand | Bedeutung |
|---|---|---|---|---|
| A/B, Seeds 42–44 | `5884a6f07aae` | `8c76c135fc2e` | `unknown` | Älterer Laufstand |
| C/D, Seed 42 | `39db658e20fa` | `1514f2fd67d6` | `unknown` | Älterer Laufstand |
| C, Seeds 43–44 | `f154faeeb80a` | `6fbcef40f37d` | `e07022e...` | Neuer Laufstand |
| D, Seeds 43–44 | `f154faeeb80a` | `8bdeef23a499` | `449cd9e...` | Neuer Laufstand |

Die starken Ergebnisunterschiede bestätigen, dass diese Gruppen nicht blind gemittelt werden dürfen:

| Methode/Seed | JSON parse | Top-1 | Encoding object | Interpretation |
|---|---:|---:|---:|---|
| C/42, alt | 97.45% | 1.46% | nicht verfügbar | Technisch parsebar, aber alter Evaluations-/Trainingsstand |
| C/43, neu | 83.94% | 80.66% | 83.94% | Deutlich bessere Aufgabenleistung |
| C/44, neu | 87.96% | 85.40% | 87.96% | Deutlich bessere Aufgabenleistung |
| D/42, alt | 95.99% | 0.00% | nicht verfügbar | Nicht mit D/43–44 zusammenfassen |
| D/43, neu | 98.54% | 89.42% | 98.54% | Bestes vorhandenes Qwen-1.7B-Ergebnis |
| D/44, neu | 100.00% | 89.05% | 100.00% | Sehr stabil gegenüber Seed 43 |

**Folgerung:** Für einen sauberen A–D-Vergleich muss mindestens Qwen 1.7B Seed 42 für C/D mit dem aktuellen Protokoll neu inferiert werden. Falls A/B direkt gegen die neuen C/D-Läufe verglichen werden sollen, sollten auch A/B unter exakt demselben aktuellen Inferenzprotokoll erneut laufen. Das erfordert für A/B kein Training.

### 7.2 OLMo 2 1.49B

- C und D sind über Seeds 42–44 intern konsistent konfiguriert.
- B ist jetzt für Seeds 42–44 vollständig und verwendet denselben aktuellen Training-/Inferenz-Hash wie C/D.
- A Seed 42 wurde durch einen vollständigen aktuellen Spark-Lauf ersetzt. A Seeds 43/44 stammen weiterhin aus dem älteren Protokollstand.
- B erreicht 21.53% bis 24.09% Top-1 und 46.35% bis 49.64% Parse-Rate. Das ist klar besser als OLMo C/D.
- C und D zeigen sehr niedrige Parse-Raten von ungefähr 1.09% bis 3.28%, Schema-Gültigkeit 0% und Top-1 ungefähr 0.73% bis 2.92%.

Das ist ein echtes negatives Ergebnis oder ein Format-/Prompt-Kompatibilitätsproblem. Es ist nicht sinnvoll, daraus bereits zu behaupten, dass Fine-Tuning oder RAG bei OLMo generell nicht funktioniert. Zuerst müssen einige Rohantworten manuell geprüft werden. Für einen vollständig homogenen Drei-Seed-Vergleich fehlen nur aktuelle A-Läufe für Seeds 43/44; diese benötigen kein Training.

### 7.3 Qwen3.8 27B

- A und B sind für Seed 42 technisch vollständig.
- A erreicht 92.34% Top-1 und 46.72% Schema-Gültigkeit.
- B erreicht 83.58% Top-1 und 84.67% Schema-Gültigkeit.
- C erreicht 96.72% Top-1, 100% Parse-Rate und 100% Encoding-Objektrate.
- D erreicht 97.45% Top-1, 100% Parse-Rate und 100% Encoding-Objektrate. D ist damit der beste vorhandene 27B-Lauf.

Die A–D-Matrix ist technisch vollständig. A/B verwenden jedoch ältere Training-/Inferenz-Hashes als C/D. Ergebnisse dürfen beschreibend verglichen werden; eine starke kausale Aussage „Methode allein verursacht Verbesserung“ braucht identische Inferenzparameter. A/B könnten dafür ohne Training neu inferiert werden.

### 7.4 Qwen3 8B

- A ist jetzt vollständig: 91.61% Top-1, 97.45% Top-3, 100% Parse-Rate und 100% Encoding-Objektrate.
- B ist jetzt vollständig: 82.12% Top-1, 98.91% Top-3, 100% Parse-Rate und 100% Encoding-Objektrate.
- C ist vollständig: 83.94% Top-1, 86.13% Parse-Rate, 83.58% Paraphrase-Konsistenz.
- D ist vollständig: 58.39% Top-1, 74.09% Parse-Rate, 69.34% Paraphrase-Konsistenz.
- D ist in diesem Lauf klar schlechter als C. RAG verbessert dieses 8B-Ergebnis nicht.
- Bereinigte A/B-Läufe aus `new result 002` enthalten je Datei exakt 274 Zeilen, 274 eindeutige IDs, gültiges JSON, vollständige Metriken und `status=completed`. Sie wurden in den finalen Ordner übernommen.
- Frühere fehlerhafte A/B-Exporte bleiben nur unter `incomplete`: A hatte 548 Zeilen je Datei; B hatte 548 Zeilen in beiden Robustheitsdateien. Logs zeigten zwei gleichzeitig schreibende Prozesse.
- Der mitgelieferte Spark-Ordner `new result 001/results` enthält 16 Dateien mit jeweils 0 Byte. Diese Dateien sind Platzhalter, keine verwendbaren Tabellen oder Berichte.
- C-Training ist vollständig: finaler Adapter vorhanden, `global_step=1101` von `max_steps=1101`, Trainingsmetadaten und alle drei Checkpoints vorhanden. C und D wurden jeweils nur einmal gestartet und besitzen saubere 274/274/274-Dateien.
- A/B verwenden Inferenz-Hash `8bdeef23a499` und Git-Stand `449cd9e...`; C/D verwenden Inferenz-Hash `6fbcef40f37d` und Git-Stand `265109b...`. Dateitechnisch ist A–D komplett; strenge kausale Methodenvergleiche verlangen denselben Protokollstand.

Für vollständige Dateiergebnisse ist kein weiterer Qwen-8B-Lauf nötig. Für einen streng protokollgleichen A–D-Vergleich müssten nur C/D mit aktuellem Inferenzprotokoll wiederholt werden; kein Training nötig.

## 8. Top-3, Encoding und andere KPIs

### Top-3

Top-3 ist jetzt für Qwen-8B-A/B gültig: A erreicht 97.45%, B 98.91%. Bei Qwen-8B-C/D und vielen älteren Läufen bleibt `top_3_valid=False`, weil nicht genügend Beispiele drei verschiedene, geordnete Empfehlungen enthalten.

**Konsequenz:** Top-3 darf für A/B berichtet werden, aber nicht als vollständige A–D-Vergleichsmetrik. Für C/D ist sie weiterhin nicht auswertbar.

### Encoding

- Die neueren Qwen-1.7B-C/D-Läufe liefern eine messbare `encoding_object_rate` von 83.94% bis 100%.
- Ältere A/B- und Seed-42-Läufe enthalten diese neue Encoding-Metrik nicht.
- Deshalb darf Encoding nicht über alle Seeds und Methoden gemeinsam verglichen werden, bevor die alten Inferenzläufe mit dem neuen Evaluator reproduziert wurden.

### Schema-Gültigkeit

Bei neueren Qwen-1.7B-C/D-Läufen steht trotz hoher Parse- und Top-1-Werte eine Schema-Gültigkeit von 0%. Das ist kein Dateifehler. Es bedeutet, dass der strenge Schema-Validator mindestens eine geforderte Struktur verletzt sieht. Vor einer Thesis-Aussage müssen Stichproben der Rohantworten und die genaue Validator-Definition geprüft werden.

### Robustheit

Paraphrase- und Missing-Info-Dateien sind bei den 32 vollständigen Läufen vorhanden. Die Missing-Info-Clarification-Rate liegt in vielen Läufen jedoch bei 0% oder sehr niedrig. Das spricht gegen eine bereits erreichte sichere Rückfragefähigkeit und muss als Limitation berichtet werden.

## 9. Was für die Masterarbeit noch fehlt

### Mindestprogramm

1. **Qwen 8B optional protokollgleich machen:** Nur C/D unter demselben aktuellen Inferenzprotokoll wie A/B wiederholen, falls strenger kausaler A–D-Vergleich nötig ist; kein Training.
2. **Qwen 1.7B Protokoll vereinheitlichen:** mindestens C/D Seed 42 mit aktuellem Code; idealerweise A/B ebenfalls neu inferieren.
3. **OLMo A Seeds 43/44 aktualisieren**, falls der Drei-Seed-A–D-Vergleich vollständig protokollgleich sein soll; kein Training nötig.
4. **Qwen 27B A/B optional neu inferieren**, falls A–D als streng kontrollierter Methodenvergleich berichtet werden soll. C/D sind bereits fertig.
5. **Top-3 entscheiden:** entweder korrekt neu erzeugen oder transparent als nicht auswertbar dokumentieren.
6. **Encoding-Vergleich vereinheitlichen:** alte A/B-Inferenz mit aktuellem Evaluator/Outputformat wiederholen, falls Encoding eine Hauptmetrik ist.
7. **Human Evaluation durchführen**, mindestens auf einer klar begründeten Stichprobe. Automatische interne Referenztreffer beweisen nicht allein, dass Dashboard-Empfehlungen fachlich nützlich sind.
8. **Rohantwort-Stichprobe prüfen**, besonders OLMo und alle Läufe mit 0% Schema-Gültigkeit.

### Falls die definierte finale Modellmatrix verpflichtend bleibt

Die aktuelle `final.yaml` nennt außerdem Qwen3 8B, Qwen3 14B und Llama 3.1 8B als primäre Modelle. Gefunden wurde:

| Modell | Gefundener Stand | Fehlend |
|---|---|---|
| Qwen3 8B | A–D Seed 42 vollständig, aktueller C-Adapter vorhanden | Optional protokollgleiche C/D-Wiederholung |
| Qwen3 14B | nichts | Training/Adapter und A–D-Resultate |
| Llama 3.1 8B | nichts | Training/Adapter und A–D-Resultate |

Wenn diese Modelle im genehmigten Versuchsplan stehen, ist die Arbeit erst nach ihrer Ausführung oder nach einer begründeten Scope-Änderung vollständig.

## 10. Wissenschaftliches Urteil

**Technisch:** Ja, die Pipeline erzeugt vollständige Vorhersagen, Metriken, Robustheitsdateien, Aggregationen und Figuren. Die Adapter sind nutzbar.

**Für eine Masterarbeit verwendbar:** Teilweise bis gut. Qwen 8B und Qwen 27B besitzen jetzt vollständige A–D-Matrizen für Seed 42. OLMo besitzt A–D-Dateien für drei Seeds. Für eine saubere endgültige Hauptaussage bleiben gemischte Protokollstände, unvollständige Top-3-Unterstützung und fehlende Human Evaluation zu behandeln.

**Pragmatische Empfehlung:** Nicht alles neu trainieren. Vorhandene Adapter behalten. Nur die fehlenden oder inkompatiblen Inferenzläufe wiederholen, anschließend aggregieren und eine kleine Human Evaluation durchführen. Das ist der schnellste wissenschaftlich vertretbare Weg.

## 11. Erzeugte Auswertungsdateien

- `comparison_table.csv` und `comparison_table.md`: alle Einzelruns
- `comparison_seeds.csv`: Vergleich nach Seeds
- `multi_seed_summary.csv` und `.md`: deskriptive Seed-Zusammenfassung
- `final_report.md`: automatischer technischer Bericht
- `per_model/`: modellspezifische Tabellen
- `cross_model/`: modellübergreifende Tabellen
- `figures/`: Accuracy, Schema, Seed-Variabilität, Robustheit und Laufzeit
- `experiments/results/packages/dashboard_v4/professor_results_dashboard_v4_consolidated.zip`: kompaktes Weitergabepaket

Das ZIP enthält 538 Einträge und hat SHA-256:

`E1C97F9A659BCDD624456F648F8BE63284B31219A1DD2F9C4B5138E9F0A95C8E`

## 12. Regeln für neue Resultate

Neue vollständige Läufe nur unter folgendem Schema ablegen:

```text
experiments/outputs/final/dashboard_v4/<model>/<method>/seed_<seed>/
```

Unvollständige Läufe gehören nach:

```text
experiments/outputs/incomplete/dashboard_v4/<model>/<method>/seed_<seed>/
```

Adapter gehören getrennt nach:

```text
experiments/outputs/adapters/dashboard_v4/<model>/C/seed_<seed>/adapter/
```

Danach Aggregation und Figuren neu erzeugen. Vor der wissenschaftlichen Zusammenfassung müssen Modellrevision, Dataset-Hashes, Training-Hash, Inferenz-Hash und Git-Commit zwischen den verglichenen Läufen kontrolliert werden.
