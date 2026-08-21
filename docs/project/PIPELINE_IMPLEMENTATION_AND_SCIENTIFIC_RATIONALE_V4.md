# Dashboard-v4-Pipeline: Implementierung, wissenschaftliche Begründung und Status

**Stand:** 20.08.2026  
**Zweck:** Reproduzierbare Projektdokumentation für die Masterarbeit  
**Geltungsbereich:** Dashboard v4, `dashboard_v4_tiny`, Methoden A–D, RAG, QLoRA, automatische Evaluation und Vorbereitung der Human Evaluation

## 1. Zusammenfassung

Die Pipeline wurde als konfigurationsgesteuerte Experimentpipeline aufgebaut, damit vier Bedingungen mit identischen Daten-, Prompt-, Generierungs- und Bewertungsregeln vergleichbar bleiben. Methode A ist die Prompt-only-Baseline, Methode B ergänzt Retrieval-Augmented Generation (RAG), Methode C trainiert einen QLoRA-Adapter und Methode D kombiniert den trainierten Adapter mit RAG. Die Pipeline trennt Training, Validation, kanonischen Test, Cross-Domain-Test, automatische Metriken und Human Evaluation. Jeder Lauf speichert zusätzlich Konfigurations-, Daten-, Knowledge-Base- und Adapterinformationen.

Der wichtigste behobene technische Fehler war kein Fehler des TF-IDF-Retrievers. Die Eingabeprompts waren für das verfügbare Kontextfenster zu lang. Bei mehreren B- und D-Prompts wurde deshalb nicht nur der Retrieval-Text, sondern auch ein beginnendes JSON-Beispiel abgeschnitten. Das Modell erhielt damit teilweise ein bereits begonnenes JSON-Fragment und erzeugte Antworten, die mitten in einem Objekt starteten oder nicht geschlossen wurden. Die Pipeline begrenzt jetzt zuerst den Prompt budgetiert, kürzt RAG-Passagen kontrolliert und bricht bei verbleibender Überschreitung mit einer erklärenden Fehlermeldung ab.

Damit ist die konkrete Ursache der historischen B/D-Formatfehler behoben. Die Pipeline ist dadurch reproduzierbarer und diagnostisch aussagekräftiger. Das bedeutet jedoch nicht, dass bereits jedes Modell gute Dashboard-Entscheidungen erzeugt oder dass ein Final-Run für die Masterarbeit abgeschlossen ist. Formatzuverlässigkeit, semantische Qualität, Human Usefulness und Generalisierung sind getrennte Eigenschaften und werden deshalb getrennt geprüft.

## 2. Ausgangsproblem

Der ursprüngliche Tiny-Lauf sollte mit sehr wenig Compute ausschließlich prüfen, ob die gesamte Experimentkette funktioniert. Dabei zeigte sich eine wichtige Vermischung von drei Fehlerklassen:

1. Ein Prozess kann technisch erfolgreich laufen, während seine Modellantworten die Qualitätsanforderungen nicht erfüllen.
2. Ein Lauf kann wegen fehlendem oder inkompatiblem Adapter korrekt blockiert werden, ohne dass die RAG-Implementierung fehlerhaft ist.
3. Alte Prediction-Dateien können vorhanden und vollständig sein, aber zu einer anderen Konfiguration gehören. Ihre Wiederverwendung wäre wissenschaftlich falsch.

Die ersten Ergebnisse zeigten deshalb gleichzeitig erfolgreiche Ausführung, unzureichende strukturierte Ausgaben, einen fehlgeschlagenen Fine-Tuning-Lauf und Cache-Konflikte. Diese Fälle werden in der neuen Dokumentation nicht als ein gemeinsamer „Model Failure“ zusammengefasst.

### 2.1 Historische Befunde

Im ursprünglichen Tiny-Befund war die Retrieval-Verdrahtung für B und D funktionsfähig. Für jedes passende Item wurden drei Knowledge-Base-Chunks abgerufen, und gleiche Items erhielten gleiche Retrieval-Ergebnisse. Die schlechte Qualität entstand anschließend in der Generationsstufe.

Bei B wurden in der historischen 50-Item-Auswertung neun rohe JSON-Objekte gefunden, aber kein Output erfüllte das vollständige Schema. Bei D wurden vier rohe JSON-Objekte gefunden, ebenfalls ohne schema-validen Output. Viele Antworten begannen mitten in einem JSON-Objekt, enthielten kopierte Beispiel-Fragmente oder normalen Text außerhalb des erwarteten Formats.

Die historische Bewertung zeigte außerdem `exact_encoding = 0%` und `exact_aggregate = 0%` für alle Methoden. Diese Metriken sind sehr streng: Das gesamte Encoding-Dictionary beziehungsweise die Aggregatdarstellung muss exakt mit dem synthetischen Gold übereinstimmen. Ein Wert von null beweist daher nicht allein einen Pipeline-Defekt, zeigt aber, dass die strukturierte Reproduktion in diesem Lauf nicht zuverlässig war.

Der historische Top-3-KPI war für diese Daten ebenfalls nicht als primärer Erfolgskriterium geeignet. Weniger als 80% der Antworten lieferten drei unterschiedliche Chart-Empfehlungen. Das Test-Gold enthält in der Regel keine drei menschlich validierten Alternativen. Der Top-3-Wert bleibt deshalb eine Diagnostik, nicht die zentrale Thesis-Aussage.

### 2.2 Nachweis der unmittelbaren Ursache

Die Tiny-Konfiguration verwendete `max_seq_length = 1024` und `max_new_tokens = 512`. Dadurch standen ungefähr 512 Tokens für den Eingabeprompt zur Verfügung, wenn eine maximale Ausgabe von 512 Tokens reserviert wurde. Die alten A/C-Prompts lagen teilweise bei 499–537 Tokens. Die B/D-Prompts lagen mit Retrieval-Kontext bei etwa 671–839 Tokens. Die automatische Trunkierung schnitt dadurch bei B/D in den strukturellen Instruktionen oder im JSON-Beispiel.

Der kritische Fehler war damit ein deterministischer Budgetierungsfehler. Retrieval selbst lieferte die erwartete Anzahl von Passagen. Der Generator erhielt aber nicht immer einen vollständigen, syntaktisch sinnvollen Prompt.

## 3. Experimentdesign

### 3.1 Methoden

| Methode | Beschreibung | Wissenschaftliche Funktion |
|---|---|---|
| A | Prompt-only | Baseline ohne Retrieval und ohne Fine-Tuning |
| B | Prompt + RAG | Isoliert den zusätzlichen Einfluss externer Knowledge-Base-Passagen |
| C | QLoRA Fine-Tuning | Isoliert den Einfluss parametereffizienter Aufgabenanpassung |
| D | QLoRA + RAG | Kombiniert Aufgabenanpassung und externe Evidenz |

Die Reihenfolge ist absichtlich relevant: C muss den Adapter erzeugen, bevor D ihn verwenden kann. D darf nicht mit einem beliebigen vorhandenen Adapter laufen. Adapter müssen zu Modell, Seed, Dataset, Trainingskonfiguration und Adapterquelle passen.

### 3.2 Frozen-Datasets

`dashboard_v4` ist der vorgesehene vollständige Datensatz für die Thesis. Er enthält das vollständige v4-Training und die Validierung. Der kanonische Test bleibt für Evaluation reserviert. `dashboard_v4_tiny` ist eine deterministisch gezogene Entwicklungsfassung aus dem Frozen-Dataset und kein Ersatz für die Thesis-Evaluation.

Die Tiny-Auswahl verwendet Seed `20260820`, sortiert Item-IDs vor der Stichprobenziehung und stratifiziert nach verfügbaren Chart-Typen. Es werden mindestens drei Beispiele pro verfügbarer Chart-Kategorie berücksichtigt. Die Tiny-Datei enthält:

| Split | Umfang | Verwendung |
|---|---:|---|
| Train | 100 | QLoRA-Training |
| Validation | 50 | Epoch-Validation und Checkpoint-Auswahl |
| Test | 50 | kanonischer kleiner In-Domain-Test |
| Sports test | 50 | separater Cross-Domain-Diagnosetest |

Die Records werden aus dem Parent-Dataset kopiert; die Sports-Aufteilung wird nur durch das Split-Label gekennzeichnet. Es findet für diese Tiny-Auswahl keine neue LLM-Generierung statt. Die Frozen-Metadaten dokumentieren Duplikatprüfungen, Leakage-Prüfungen, Chart-Verteilung, Parent-Lineage und SHA-256-Hashes. Train und Validation enthalten keine explizit erlaubten Sports-Beispiele. Test und Sports-Test werden nicht für Training verwendet.

Diese Trennung ist wissenschaftlich notwendig. Ein Modell darf weder Testinformationen sehen noch indirekt über Duplikate oder identische Brief-Fingerprints auf Testantworten zugreifen. Der Sports-Test misst nicht dieselbe Eigenschaft wie der kanonische Test, sondern dient als Cross-Domain-Diagnostik.

## 4. Implementierte Änderungen

### 4.1 Prompt- und Kontextbudget

Die Promptstruktur wurde komprimiert, ohne die sechs erforderlichen Top-Level-Felder, Datentypen und erlaubten Enums zu entfernen. Damit sinkt der statische A/C-Prompt auf ungefähr 322–360 Tokens.

Für B und D wird der RAG-Kontext token-genau an das verfügbare Budget angepasst. Das Restbudget wird gleichmäßig über die drei Top-k-Passagen verteilt. Alle drei Retrieval-Ergebnisse bleiben im Prompt repräsentiert; längere Passagen werden gekürzt, statt den gesamten Prompt unkontrolliert abschneiden zu lassen.

Die Generierung speichert zusätzlich technische Diagnosefelder wie `prompt_input_tokens`, `prompt_input_budget` und `rag_context_truncated`. Überschreitet ein Prompt das Budget nach kontrollierter Kürzung weiterhin, wird der Lauf mit einer erklärenden Fehlermeldung beendet. Eine stille Trunkierung, die ein JSON-Beispiel unbemerkt beschädigt, ist damit ausgeschlossen.

Diese Änderung verbessert nicht automatisch die inhaltliche Qualität des Modells. Sie sorgt dafür, dass ein gemessener JSON-Fehler tatsächlich aus der Modellgenerierung stammt und nicht aus einem beschädigten Input.

### 4.2 Trennung von Training und Evaluation

Vor der Änderung konnte eine Sample-Begrenzung missverständlich sowohl als Trainings- als auch als Evaluationsbegrenzung interpretiert werden. Die Pipeline verwendet jetzt `data.max_samples` für Training und `data.eval_max_samples` für Test-Inferenz. Der Parameter `--eval-items` des Tiny-Runners begrenzt deshalb nur die Anzahl evaluierter Testitems. Das Training verwendet weiterhin alle 100 Tiny-Train-Items.

Diese Trennung verhindert, dass ein schneller 20-Item-Pilot versehentlich ein anderes Trainingsproblem ausführt. Sie macht außerdem Run-Größen und Ergebnisse zwischen Pilot, Tiny-Full-Run und finalem Dataset vergleichbar.

### 4.3 Validation und Checkpoint-Auswahl

QLoRA- und LoRA-Training laden nun bei aktivierter Evaluation den konfigurierten Validation-Split. Evaluation und Checkpointing finden pro Epoche statt. `eval_loss` ist als Auswahlmetrik vorgesehen; das beste Modell wird nach dem Training geladen. Training-Metadaten speichern zusätzlich Evaluation-Strategie, Validation-Verlauf, bestes Ergebnis und besten Checkpoint.

Nicht-finite Trainingsmetriken, nicht-finite Werte in der Historie und nicht-finite trainierbare Parameter führen dazu, dass kein scheinbar gültiger Adapter gespeichert wird. Diese Prüfung schützt vor einem formal vorhandenen, aber numerisch unbrauchbaren Adapter.

Validation dient der Entwicklungsentscheidung und Modell-/Checkpoint-Auswahl. Der Testsplit bleibt davon getrennt. Testmetriken dürfen nicht verwendet werden, um nachträglich Hyperparameter, Prompts oder Gates anzupassen.

### 4.4 Cache-, Hash- und Adapter-Provenienz

Jeder Run erhält eine eigene Struktur nach Dataset, Modell, Methode und Seed. Die gespeicherten Artefakte umfassen unter anderem:

- `predictions.jsonl`
- `metrics_auto.json`
- `manifest.json`
- `config_snapshot.yaml`
- `config_hash.txt`
- `cache_identity.json`
- Dataset- und Knowledge-Base-Hashes
- bei C/D Adapter- und Training-Metadaten

Die Cache-Identität umfasst Dataset-Version und Dataset-Hashes, Modell und Revision, Methode, Seed, Trainingskonfiguration, Inferenzkonfiguration und Knowledge-Base-Hashes. Existiert eine alte Prediction-Datei mit anderem Hash, wird sie nicht wiederverwendet. Das erklärt die früheren Sports-A/C-Meldungen mit unterschiedlichen `config_hash`-Werten. Diese Meldung ist ein Schutz gegen Ergebnisvermischung, kein Beweis für einen Modellfehler.

D wird nur ausgeführt, wenn ein kompatibler C-Adapter vorhanden ist. Wenn C fehlschlägt, wird D mit `FAILED(no-adapter)` blockiert. Dieses Fail-fast-Verhalten verhindert, dass D unbeabsichtigt einen alten oder unpassenden Adapter verwendet.

### 4.5 Format- und Coverage-Gates

Der Tiny-Runner prüft nicht nur, ob ein Prozess mit Exit Code 0 endet. Er prüft auch, ob der Ergebnisordner vollständig ist, ob die erwartete Item-Anzahl abgedeckt ist und ob die generierten Antworten die Mindestanforderungen erfüllen.

Die aktuellen projektdefinierten Smoke-Gates sind:

| Gate | Mindestwert | Bedeutung |
|---|---:|---|
| Coverage | 100% | Kein angefordertes Item fehlt |
| JSON Parse Rate | 95% | JSON-Objekt ist extrahierbar |
| Strict Schema Validity | 90% | Vollständiges Pydantic-Schema und erlaubte Enums |
| Required Keys Rate | 95% | Erforderliche Schlüssel vorhanden |
| Completeness | 0,80 | Durchschnittlicher Anteil nichtleerer Top-Level-Felder |

Diese Werte sind Engineering-Release-Kriterien für einen Smoke-Test. Sie sind keine aus einer Stichprobe abgeleiteten statistischen Signifikanzgrenzen und dürfen nicht als wissenschaftliche Effektgrenzen interpretiert werden. Sie sollen verhindern, dass ein offensichtlich unvollständiger Output als „funktionierender“ Run weitergereicht wird.

### 4.6 Bedeutung von `no_json_found`

`no_json_found` bezeichnet einen Fehler der strukturierten Ausgabe. Der Parser hat im Modelltext kein vollständiges, extrahierbares JSON-Objekt gefunden. Das bedeutet nicht automatisch, dass der Parser oder das Schema falsch implementiert ist. Häufige Ursachen sind eine normale Textantwort trotz JSON-Anweisung, ein abgeschnittenes JSON-Objekt, ein zu langer Prompt oder das Erreichen von `max_new_tokens`.

Der Fehler ist von späteren Validierungsstufen zu unterscheiden. Bei `no_json_found` wurde noch kein verwertbares JSON gefunden. Bei `json_parse_error` wurde ein JSON-Kandidat gefunden, der syntaktisch ungültig ist. Bei `schema_invalid` ist das JSON syntaktisch gültig, verletzt aber Typen, erlaubte Enums oder Pflichtfelder. Ein `completeness`-Fehler bedeutet, dass das JSON grundsätzlich verarbeitet werden konnte, aber relevante Felder leer oder unvollständig sind.

Ein einzelner `no_json_found`-Fall kann in einem kleinen Smoke-Test vorkommen und wird als ungültige Modellantwort gezählt. Er darf nicht gelöscht, still repariert oder als korrektes Ergebnis gezählt werden. Bei einem 50-Item-Lauf und genau einem solchen Fall wären 49 von 50 Antworten parsebar, also 98% JSON Parse Rate. Das würde das Parse-Gate von 95% bestehen; Schema- und Completeness-Gates müssen trotzdem separat erfüllt sein. Für eine wissenschaftliche Auswertung muss die Zahl aller `no_json_found`-Fälle, ihre Item-IDs und ihre Ursachen transparent berichtet werden.

Die Laufzeitangabe neben dem Status, beispielsweise `63487 ms`, ist eine separate Latenzmessung. Sie bedeutet ungefähr 63 Sekunden Verarbeitungszeit und erklärt den JSON-Fehler nicht automatisch. `no_json_found` zeigt zunächst nur, dass die Modellantwort die erwartete maschinenlesbare Schnittstelle nicht erfüllt hat. Die Pipeline erkennt diesen Zustand explizit, damit die Ausfallrate in den Metriken sichtbar bleibt.

## 5. Wissenschaftliche Begründung

### 5.1 Vergleichbarkeit durch kontrollierte Bedingungen

Die vier Methoden werden als faktorielles Vergleichsdesign interpretiert. A liefert den Baseline-Wert. B fügt externe Information hinzu, ohne die Modellparameter zu ändern. C fügt Aufgabenanpassung durch QLoRA hinzu, ohne RAG einzuschalten. D kombiniert beide Erweiterungen. Diese Struktur erlaubt eine getrennte Diskussion des Effekts von Retrieval und Fine-Tuning, statt beide Änderungen in einem einzigen Vergleich zu vermischen.

### 5.2 Parameter-effizientes Fine-Tuning

LoRA hält die vortrainierten Gewichte weitgehend eingefroren und lernt niedrig-rangige Gewichtsaktualisierungen. QLoRA kombiniert diesen Ansatz mit 4-bit-Quantisierung des Basismodells. Das reduziert Speicherbedarf und macht kontrollierte Experimente mit größeren Modellen realistischer. Es ersetzt jedoch keine Hardwareplanung: Ein größeres Modell kann trotz Quantisierung mehr VRAM, Hauptspeicher und Laufzeit benötigen.

### 5.3 RAG und externe Evidenz

RAG verbindet einen generativen Sprachmodellteil mit einer nichtparametrischen Knowledge Base. Für diese Arbeit ist die Trennung wichtig, weil Dashboard-Empfehlungen durch explizite Visualisierungsprinzipien, KPI-Regeln und Designwissen begründet werden sollen. Die Pipeline speichert daher Retrieval-Ergebnisse und Knowledge-Base-Hashes. Eine Grounding-Zahl wird nur zusammen mit ihrem Modus interpretiert: Die aktuelle lexikalische Unterstützung ist ein Wortüberlappungs-Proxy und keine vollständige Faithfulness-Bewertung.

### 5.4 Strukturvalidität als notwendige, aber nicht ausreichende Bedingung

Das Zielsystem erzeugt maschinenlesbare Dashboard-Empfehlungen. JSON-Parsing und Schema-Validität sind deshalb notwendige technische Voraussetzungen für nachgelagerte KPI-, Chart-, Layout- und Human-Evaluation-Schritte. Sie beweisen aber nicht, dass ein Chart semantisch passend, wahrheitsgetreu, verständlich oder nützlich ist.

Die Metriken werden deshalb hierarchisch gelesen. Erst werden Coverage und Parsing geprüft. Danach folgen Schema, Required Keys und Completeness. Erst bei ausreichender strukturierter Qualität sind exakte Encoding-Metriken, Grounding, unabhängige Chart-Effektivität und Human Usefulness sinnvoll interpretierbar.

### 5.5 Mehrere Seeds und unabhängige Bewertung

Ein einzelner Seed kann zufällige Eigenschaften der Datenreihenfolge, Optimierung und Generierung widerspiegeln. Für die finale Matrix sind deshalb Seeds 42, 43 und 44 vorgesehen. Berichtet werden sollen Mittelwerte und Streuung, nicht nur der beste Seed.

Die synthetischen Gold-Metriken sind interne Diagnostik, weil Gold-Annotation und Modellaufgabe aus derselben Benchmarkfamilie stammen können. Für Aussagen über menschliche Wirksamkeit sind eine unabhängige L1-Chart-Evaluation, Real-Briefs und eine blinde Human Evaluation erforderlich. Automatische Formatmetriken können Human Usefulness nicht ersetzen.

## 6. Empirische Wirkung des Minimalfixes

Ein 20-Item-Pilot auf denselben Dashboard-IDs zeigte folgende Veränderung. Diese Zahlen sind interne Smoke-Ergebnisse und keine finalen Thesis-Ergebnisse:

| Methode | Zustand | JSON Parse | Schema | Top-1 | Macro-F1 |
|---|---|---:|---:|---:|---:|
| A | vor Budgetfix | 45% | 45% | 35% | 0,2111 |
| A | nach Budgetfix | 90% | 90% | 60% | 0,2857 |
| B | vor Budgetfix | 25% | 0% | 0% | 0,0000 |
| B | nach Budgetfix | 100% | 90% | 75% | 0,6236 |

B erreichte nach dem Fix vollständige Coverage auf 20 Items, drei Retrieval-Dokumente pro Item und keine Laufzeit- oder Speicherfehler. Die lexikalische Grounding-Unterstützung lag bei 61,03% über 32 Claims. Diese Zahl bleibt ein Proxy und darf nicht als semantische Faithfulness-Garantie berichtet werden.

Zwei Schemafehler blieben im Pilot als echte Modellfehler bestehen: `task_type = "comparing"` statt des erlaubten Werts `comparison` sowie `chart_type = "count"` statt eines erlaubten Chart-Typs. Diese Fehlerklasse ist diagnostisch nützlich, weil sie nach Entfernung der Prompttrunkierung nicht mehr durch abgeschnittene JSON-Fragmente verdeckt wird.

Die mittlere Completeness blieb im Pilot bei etwa 0,333 und damit deutlich unter dem Smoke-Gate von 0,80. Der Fix hat somit die Formatursache verbessert, aber nicht automatisch die Vollständigkeit der Modellantworten gelöst.

## 7. Herausforderungen und aktueller Lösungsstatus

### 7.1 Gelöst oder technisch abgesichert

Die unkontrollierte Prompttrunkierung ist durch Budgetierung, kontrollierte RAG-Kürzung und harte Überlaufprüfung abgesichert. Die Verwechslung von Trainingsgröße und Evaluationsgröße ist durch `eval_max_samples` und den expliziten `--eval-items`-Parameter behoben. Alte Prediction-Dateien können nicht mehr still mit neuen Konfigurationen vermischt werden. D wird ohne kompatiblen C-Adapter nicht ausgeführt. Validation-Metadaten und Best-Checkpoint-Informationen werden vorgesehen und in den Training-Metadaten festgehalten. Format-Gates sind automatisiert und als Projektkriterien gekennzeichnet.

### 7.2 Noch offen

Ein früherer C-Tiny-Lauf endete mit `FAILED(1)`, und D wurde deshalb korrekt mit `FAILED(no-adapter)` blockiert. Der übermittelte Log-Ausschnitt enthielt keinen vollständigen Traceback. Daher darf die genaue Ursache nicht als bewiesen bezeichnet werden. Der wahrscheinlichste Prüfpunkt ist die neue Epoch-Validation unter knapper GPU-Ressource; ein GPU-Speicherfehler bleibt ohne vollständigen Traceback jedoch nur eine Hypothese. Ein frischer Lauf muss die vollständige Standardfehlerausgabe speichern.

`exact_encoding` und `exact_aggregate` lagen in historischen Läufen bei null. Vor einem finalen Lauf muss geprüft werden, ob Prompt, Schema und Metrik dieselbe Aggregat- und Encoding-Semantik verlangen. Die Metriken dürfen nicht stillschweigend abgeschwächt werden, sollten aber als sehr strenge exakte Übereinstimmung dokumentiert werden.

Robustheitsvarianten für Paraphrase und Missing Information, der unabhängige `benchmark_v1`-Lauf, Real-Briefs, L3-Realism und blinde Human Ratings sind für eine Thesis-Aussage noch auszuführen oder ausdrücklich als nicht im Claim-Scope zu deklarieren. Die Offline-L1-Scorer-Funktion ist im Code vorhanden, wurde aber nicht automatisch in den Tiny-Run integriert. Ein historischer v1-L1-Bericht und ein Benchmark-Smoke-Bericht sind daher nicht mit einem finalen v4-L1-Ergebnis gleichzusetzen. Streamlit-Human-Evaluation soll erst mit vollständigen, kompatiblen Outputs gestartet werden; sonst würden Rater überwiegend Parserfehler bewerten.

### 7.3 Challenge: Sind `layout` und `styling` im Dataset vorhanden?

Die erste Interpretation der leeren Modellfelder war, dass das Tiny-Dataset möglicherweise keine Layout- und Styling-Informationen enthält. Diese Vermutung wurde durch einen Split-Audit geprüft und nicht bestätigt. In `dashboard_v4_tiny` sind beide Felder befüllt: Train 100/100, Validation 50/50, Test 50/50 und Sports-Test 50/50. Auch die geprüften v4-Validation- und Testrecords enthalten diese Felder. Das Dataset liefert daher Trainings- und Referenzinformationen; fehlende Dataset-Felder erklären die leeren Modelloutputs nicht.

Die Prüfung zeigte zusätzlich, dass der Trainings-Formatter die vollständige `recommendation` als JSON-Referenz serialisiert. `layout` und `styling` werden beim Training somit nicht absichtlich entfernt. Der konkrete Modelloutput zeigte stattdessen bei einem Prompt mit 329 von 512 verfügbaren Input-Tokens:

```json
"layout": {},
"styling": {},
"encoding": {}
```

Der Output entsprach damit stark dem leeren Strukturbeispiel aus `src/core/prompts.py`. Das kleine Modell kopierte wahrscheinlich die vorgegebenen Platzhalter, ergänzte teilweise normalen Text und ließ die semantischen Inhalte leer. Die gemeinsame Ursache liegt damit eher im Output-Vertrag aus Prompt, Modellverhalten und Schema-Defaults als in fehlenden Dataset-Feldern. Die geringe Tiny-Trainingsmenge und eine Epoche können dieses Verhalten verstärken, sind aber nicht die alleinige Erklärung, weil auch die nicht fine-getunte Baseline A denselben gemeinsamen Prompt verwendet.

Diese Challenge wird durch vier getrennte Methoden kontrolliert. Erstens prüft ein Dataset-Preflight vor dem Training die Anzahl und Nicht-Leerheit von `layout` und `styling` pro Split. Zweitens vergleicht ein Raw-Output-Audit `raw_text`, geparstes JSON, `parse_error`, Prompt-Tokens und Budget, damit unterschieden wird, ob Information im Modelloutput fehlt oder erst beim Parsing verloren geht. Drittens muss der Prompt ein knappes, befülltes Minimalbeispiel für `layout`, `styling`, `encoding` und `rationales` zeigen und leere Platzhalter ausdrücklich verbieten, ohne das Kontextbudget erneut zu überlasten. Viertens soll die Validierung für diese Pflichtbereiche nicht nur syntaktisches JSON akzeptieren: leere Objekte müssen als Inhaltsfehler markiert werden und in `completeness` sowie in der Fehlerdiagnose sichtbar bleiben.

Die vierte Methode ist teilweise bereits durch das Completeness-Gate vorhanden. Nicht-leere Dataset-Felder beweisen außerdem noch nicht, dass jede Annotation semantisch optimal ist; ihre inhaltliche Qualität bleibt eine separate Dataset- und Human-Evaluation-Frage.

**Umgesetzter Minimalfix (Stand 21.08.2026):** In [`src/core/prompts.py`](../../src/core/prompts.py) (`build_user_message`) wurde das leere Struktur-Beispiel (`"layout":{}`, `"styling":{}`, `"encoding":{}`, `"interactions":[]`) durch ein kurzes, tatsächlich schema-valides Beispiel mit befüllten Feldern ersetzt und um eine explizite Nicht-Leer-Anforderung für `context_summary`, `layout`, `styling`, `encoding`, `rationales` und `interactions` ergänzt. Die Änderung betrifft ausschließlich diese eine Funktion, die einzige Quelle des Nutzer-Prompts für Training (`src/data_pipeline/formatter.py`) und Inferenz (`src/methods/base.py`) — sie wirkt damit identisch auf A, B, C und D, keine Methode wird bevorzugt.

Geprüfte Nebenwirkungen vor Übernahme:
- Das Beispiel wurde gegen `DesignOutput` validiert (`pytest`-Ad-hoc-Check) — schema-valide.
- Der volle Chat-Prompt für A/C stieg von ca. 322–360 auf 422 Tokens (Qwen2.5-0.5B-Tokenizer, `add_generation_prompt=True`). Bei `max_seq_length=1024` und `max_new_tokens=512` bleiben weiterhin 512 Tokens Budget übrig — das dynamische Budgetierungssystem aus Abschnitt 4.1 (`model.input_token_budget`, `prompt_token_count`) berechnet das RAG-Restbudget für B/D zur Laufzeit neu und kürzt ggf. die Retrieval-Passagen entsprechend stärker; es gibt keinen hartkodierten Token-Wert, der hätte veralten können.
- Bestehende Tests (`src/tests/test_metric_semantics.py`, `src/tests/test_scientific_validity.py`, `src/tests/test_postprocess.py`) — 22 Tests — laufen unverändert grün. Kein Test fixiert den exakten Prompt-Text, daher keine Snapshot-Brüche.
- `src/core/schemas.py`, `src/inference/postprocess.py`, `src/evaluation/metrics/schema_compliance.py` wurden **nicht** geändert: Der Non-Empty-Validator (`_nonempty`, `completeness_fraction`) existierte bereits und ist bereits regressionsgetestet; der Parser füllt fehlende Inhalte bereits korrekt nicht künstlich auf.

Damit ist die im vorigen Absatz offene vierte Methode umgesetzt, ohne Schema, Parser oder Metrik anzufassen. Der nächste Schritt ist ein 10–20-Item-Pilot mit unveränderter Trainings- und Evaluationslogik (siehe Abschnitt 9), um die empirische Wirkung zu messen, bevor der volle 50-Item-Tiny-Run und danach `dashboard_v4` folgen.

### 7.4 Challenge: Top-3 ist nicht „dreimal ausführen"

Eine zwischenzeitliche Bezeichnung als „Tab3“ war eine sprachliche Verwechslung. Gemeint ist die `Top-3`-Metrik der Chart-Auswahl. Top-3 bedeutet nicht, dass das Modell drei komplette Experimentierruns ausführt. Top-3 ist nur dann definiert, wenn ein einzelner Modelloutput drei verschiedene, geordnete Chart-Empfehlungen enthält.

Die Pipeline behandelt Top-3 bewusst konservativ. Der globale Wert wird nur berichtet, wenn mindestens 80% der bewerteten Items drei verschiedene Empfehlungen enthalten. Andernfalls werden `synthetic_top3.value = null` und `valid = false` ausgegeben. Das ist kein Laufzeitfehler, sondern verhindert eine irreführende Metrik. Im aktuellen Dashboard-v4-Testgold gibt es außerdem keine drei unabhängigen, menschlich validierten Alternativen pro Item. Top-3 ist deshalb nur interne Diagnostik und kein primärer Thesis-KPI.

Top-3 darf nicht durch drei Wiederholungen, künstliche Alternativen oder Wiederverwendung desselben Charts repariert werden. Für die Arbeit wird entweder `not valid` mit Support-Rate berichtet oder die Metrik aus den primären Vergleichstabellen ausgeschlossen. Der interne Code darf für Diagnosezwecke erhalten bleiben. Top-1 und Top-3 müssen getrennt bleiben: Ein Top-1-Treffer ist kein Top-3-Nachweis.

### 7.5 Challenge: Diagnostischer L1-Score für Qwen3-1.7B-Tiny

Der Qwen3-1.7B-Tiny-Ordner
`experiments/outputs/laptop_tiny_v4_qwen3_1_7b/dashboard_v4_tiny/qwen3_1_7b`
enthält vier Methoden A–D mit jeweils 50 Vorhersagen und Seed 42. Die L1-Auswertung wurde offline ausgeführt. Sie verwendet die gespeicherten `predictions.jsonl`-Dateien, das Tiny-Testgold `data/frozen/dashboard_v4_tiny/test.jsonl` und die literaturbasierte Tabelle `data/eval/l1_chart_effectiveness_v1.csv`. Es findet dabei keine neue Modellinferenz statt.

Der Score prüft pro KPI nur die primäre Chart-Auswahl. Liegt der Chart in der für den Task-Typ akzeptablen Literaturmenge, zählt der KPI als korrekt. Parse-Fehler oder fehlende primäre Charts zählen als falsch. Task-Typen ohne L1-Abdeckung werden ausgeschlossen, aber in der Coverage ausgewiesen.

Die erhaltenen Werte waren:

| Methode | Abdeckung | Abgedeckte KPIs | Korrekt | L1-Accuracy auf abgedeckten KPIs |
|---|---:|---:|---:|---:|
| A | 40/50 = 0,80 | 40 | 37 | 0,925 = 92,5% |
| B | 40/50 = 0,80 | 40 | 37 | 0,925 = 92,5% |
| C | 40/50 = 0,80 | 40 | 29 | 0,725 = 72,5% |
| D | 40/50 = 0,80 | 40 | 34 | 0,850 = 85,0% |

Die zehn nicht abgedeckten KPI-Einträge bestanden aus `composition` (3) und `part_to_whole` (7). Die Werte sind intern konsistent: `covered_accuracy` ist `covered_correct / n_covered`, also beispielsweise `37 / 40`, nicht `37 / 50`. Kleine Untergruppen sind instabil; im Tiny-Test kamen beispielsweise nur zwei `correlation`-Einträge vor.

Diese Zahlen sind **diagnostische L1-Werte für `dashboard_v4_tiny`**, keine endgültige unabhängige Thesis-Evidenz. Das Tiny-Testgold besitzt weiterhin synthetische Task-Label-Lineage. Die akzeptablen Chart-Mengen stammen zwar aus einer unabhängigen Literaturtabelle, aber die Task-Zuordnung ist nicht vollständig unabhängig vom Generator. Der Score darf daher als technischer Vergleich der Chart-Auswahl bezeichnet werden, nicht als Beweis für allgemeine Dashboard-Qualität.

Der Score bewertet weder Layout noch Styling, Interaktionen, Rationales, JSON-Vollständigkeit oder Human Usefulness. Das erklärt, warum ein Modell einen hohen diagnostischen L1-Wert und gleichzeitig ein schlechtes Completeness- oder Schema-Ergebnis haben kann. Im gleichen Qwen3-1.7B-Tiny-Lauf lagen die Schema-Raten bei A/B bei 100%, bei C bei 84% und bei D bei 88%; die Completeness-Werte lagen bei 0,5167, 0,5433, 0,4000 und 0,4733. Die L1-Zahl hebt diese Probleme nicht auf.

Der vorhandene historische CLI-Wrapper `experiments/scripts/eval_l1_independent.py` verwendet fest eingetragene v1-Pfade und ist deshalb nicht direkt für diesen Qwen3-Ordner zuständig. Die aktuelle Tiny-Zahl wurde durch direkten Offline-Aufruf von `src/evaluation/l1_independent.py::score_l1` mit den oben genannten drei Eingaben berechnet. Für einen wissenschaftlich unabhängigen finalen L1-Lauf muss stattdessen `benchmark_v1` verwendet werden:

```powershell
python experiments/scripts/eval_benchmark.py `
  --predictions-root experiments/outputs/benchmark_v1 `
  --benchmark data/eval/benchmark_v1.jsonl
```

Dieser Benchmark benötigt vorher einen separaten A–D-Inferenzlauf. Seine Ergebnisse gehören nach `experiments/results/benchmark_v1_eval.md` und `experiments/results/benchmark_v1_eval.json`. Coverage muss immer zusammen mit Accuracy berichtet werden.

### 7.6 Challenge: L1 und Human Evaluation erscheinen nicht automatisch im Tiny-Run

Die normale `metrics.json`-Datei des Tiny-Runs enthält die Statusfelder, aber nicht automatisch einen finalen unabhängigen L1-Score oder Human Ratings. Im Qwen3-1.7B-Lauf standen `l1_human_effectiveness` und `L4_human` deshalb auf `pending`. Das ist beabsichtigt: Der Tiny-Run prüft primär Pipeline-, Format- und Reproduzierbarkeitseigenschaften.

Die Human Evaluation ist eine getrennte, blinde Studie. Für das finale Dashboard-v4-Design sind vorgesehen:

- 40 feste Testitems aus `data/frozen/dashboard_v4/human_eval_test_items_40.csv`,
- vier Methoden A–D auf demselben Modell, Dataset und Seed,
- drei unabhängige Ratings pro Output,
- sechs Rater,
- insgesamt 480 Rating-Einheiten,
- sechs Likert-Dimensionen von 1 bis 5: Chart-Eignung, Layout, Styling/Accessibility, Interaktionen, Rationale und allgemeine Nützlichkeit.

Die Studie wird nach erfolgreichem Format-Gate erstellt:

```powershell
python experiments/scripts/build_human_eval.py `
  --dataset dashboard_v4 `
  --model <model_key> `
  --seed 42 `
  --n-items 40 `
  --n-raters 6 `
  --ratings-per-output 3
```

Die blinde Streamlit-App wird mit folgendem Befehl gestartet:

```powershell
python experiments/scripts/run_human_eval.py `
  --study-dir experiments/results/human_eval/dashboard_v4/<model_key>/seed_42
```

Nach vollständiger Bewertung werden die Qualitäts- und Übereinstimmungsstatistiken berechnet:

```powershell
python experiments/scripts/compute_irr.py `
  --study-dir experiments/results/human_eval/dashboard_v4/<model_key>/seed_42
```

Die Resultate liegen dann unter `experiments/results/human_eval/dashboard_v4/<model_key>/seed_42/analysis/`. Erwartet werden unter anderem Krippendorff-Alpha, Mittelwerte und Standardabweichungen, Friedman- und Wilcoxon-Tests mit Holm-Korrektur, Effektgrößen, Bootstrap-Konfidenzintervalle und `per_item_scores.csv`. Ein kleinerer Tiny-Versuch darf als `pilot` markiert werden und darf nicht ungekennzeichnet mit der finalen Studie zusammengelegt werden.

Die Trennung ist wissenschaftlich notwendig. Menschen sollen nicht überwiegend kaputte JSON-Antworten bewerten. Automatische Metriken werden der Rater-Gruppe nicht gezeigt, damit Methodennamen, Modellnamen, Seeds und automatische Scores keine Bewertung verzerren. Human Ratings sind die zentrale Evidenz für Usefulness und allgemeine Dashboard-Qualität; L1 kann diese Rolle nicht ersetzen.

### 7.7 Challenge: Modellgröße und Coding-Agent sind verschiedene Fragen

Das 0,5B-Modell kann die leeren oder unvollständigen Layout- und Styling-Felder durch begrenzte Modellkapazität verstärken. Der Qwen3-1.7B-Lauf zeigt jedoch, dass eine größere Quelle allein das Problem nicht automatisch löst: JSON-Parsing war zwar bei A–D 100%, aber C/D blieben bei Schema und Completeness unter den Gates. Prompt-Vertrag, Non-Empty-Validierung, Training und Modellkapazität müssen daher getrennt betrachtet werden.

Ein starker Coding-Agent kann den technischen Fix rational umsetzen: Prompt mit befülltem Beispiel, striktere semantische Validierung, Regressionstest mit dem bisherigen Fehleroutput, 10–20-Item-Pilot und anschließenden Tiny-Run. Der Agent kann Code prüfen und Tests ausführen. Er kann aber keine Human Usefulness garantieren und keine unabhängige wissenschaftliche Evidenz ersetzen. Modellwahl des Coding-Agents ist kein Forschungsfaktor; entscheidend sind reproduzierbare Änderungen, Tests, Run-Hashes und unabhängige Bewertung. Für diese Reparatur ist kein großer Architekturumbau erforderlich.

Der Prompt-Teil dieses Minimalfixes wurde umgesetzt (siehe Abschnitt 7.3). Der Non-Empty-Validator, das Schema und der Parser blieben unverändert, weil sie die geforderte Semantik bereits korrekt abbilden. Offen bleibt die empirische Überprüfung: Steigt die Completeness bei A–D messbar, oder bleibt ein Rest-Defizit, das auf Modellkapazität statt Prompt-Vertrag zurückgeht?

## 8. Was die neue Pipeline tatsächlich ermöglicht

Die neue Pipeline ermöglicht einen reproduzierbaren technischen Smoke-Test auf kleiner Hardware. Sie beantwortet, ob Dataset, Promptformat, RAG-Retrieval, Modell-Inferenz, QLoRA-Training, Adapterübergabe, Evaluation, Hashing und Ergebnisverpackung zusammenarbeiten. Sie macht Fehler außerdem lokalisierbar: Prozessfehler, Adapterfehler, Cache-Konflikte, Coverage-Fehler und Modellqualitätsfehler werden getrennt ausgewiesen.

Für größere Modelle und `dashboard_v4` ist die Architektur grundsätzlich vorbereitet. Die allgemeine Pipeline verwendet Hugging-Face-`AutoTokenizer` und `AutoModelForCausalLM`, PEFT/QLoRA und konfigurationsgesteuerte Modellprofile. Die finale Matrix enthält Qwen3-1.7B, Qwen3-8B, Qwen3-14B und Llama-3.1-8B. Ein Modellprofil allein garantiert jedoch keine erfolgreiche Ausführung. Vor jedem neuen Modell müssen Tokenizer, Chat Template, Quantisierung, LoRA-Zielmodule, VRAM, Zugriff auf das Modell und ein kleiner Pilot geprüft werden.

Ein OLMo-Modell ist daher keine automatische Thesis-Erweiterung. Das Modell benötigt ein eigenes Profil, die Aufnahme in die Modellprüfung und einen 10–20-Item-Pilot. Die Kernpipeline ist für Hugging-Face-Causal-Language-Models ausgelegt; ein konkretes OLMo-Instruct-Modell muss trotzdem empirisch validiert werden. Ein 7B- oder größeres Modell ist nicht für die lokale 4-GB-GPU eingeplant.

## 9. Reproduzierbarer Ablauf

Die Tiny-Daten und ihre Integrität werden mit folgendem Befehl geprüft:

```powershell
python experiments/scripts/build_dashboard_v4_tiny.py --verify
```

Ein kleiner Format- und Pipeline-Pilot verwendet 20 Testitems, trainiert aber weiterhin auf allen 100 Tiny-Train-Items:

```powershell
python experiments/scripts/run_tiny_v4_kaggle.py --eval-items 20 --force
```

Um gezielt den Prompt-Minimalfix aus Abschnitt 7.3 mit möglichst wenig Compute zu verifizieren, reicht ein 10-Item-Pilot über alle vier Methoden:

```powershell
python experiments/scripts/run_tiny_v4_kaggle.py --eval-items 10 --force
```

`--eval-items` begrenzt ausschließlich die Test-Evaluation (`data.eval_max_samples`); Training läuft unverändert auf allen 100 Items, damit C/D-Ergebnisse mit dem bisherigen 20-Item-Pilot aus Abschnitt 6 vergleichbar bleiben. Der Befehl verwendet dieselben 10 ersten Test-IDs wie jeder größere `--eval-items`-Lauf (deterministische Reihenfolge aus `test.jsonl`), sodass Vorher/Nachher-Zahlen für dieselben Items verglichen werden können. Dieser 10-Item-Lauf dient ausschließlich der technischen Verifikation des Prompt-Fixes (Completeness, Schema, JSON-Parse) und darf nicht als Grundlage für weitere Prompt- oder Threshold-Anpassungen dienen — jede weitere Iteration am Prompt oder Validator muss stattdessen gegen `val.jsonl` geprüft werden, damit `test.jsonl` für die abschließende Tiny- und Final-Bewertung unberührt bleibt.

Der vollständige Tiny-Entwicklungslauf verwendet standardmäßig 50 In-Domain-Testitems und danach den separaten Sports-Test:

```powershell
python experiments/scripts/run_tiny_v4_kaggle.py
```

Ein finaler Run muss zuerst mit einem einzelnen Finalmodell und Seed 42 geprüft werden. Erst nach bestandenem Pilot sollen die übrigen Seeds und Modelle gestartet werden. Die finale Matrix ist in `src/config/matrix/final.yaml` definiert; der vollständige Matrix-Aufruf lautet grundsätzlich:

```powershell
python experiments/scripts/run_final_matrix.py --profile final --all-models --all-methods --seeds 42 43 44
```

Dieser Befehl darf erst verwendet werden, wenn der aktuelle C/D-Pilot, die Format-Gates, die Validation-Metadaten und die Artefaktprüfung erfolgreich sind. Ein Gate-Fehler muss diagnostiziert werden; Gates dürfen nicht nach Sichtung des Final-Testes nachträglich so geändert werden, dass ein gewünschtes Ergebnis entsteht.

## 10. Formulierungsvorschlag für die Masterarbeit

Die folgende Passage kann nach Anpassung an die endgültigen Run-Zahlen in den Methodikteil übernommen werden:

> The experimental system was implemented as a configuration-driven pipeline that evaluates four controlled conditions: prompt-only generation, retrieval-augmented generation, QLoRA fine-tuning, and the combination of QLoRA with retrieval. Training, validation, in-domain testing, and cross-domain testing were kept separate. Each run recorded the model configuration, dataset hashes, knowledge-base hashes, random seed, cache identity, generated predictions, automatic metrics, and adapter provenance. This design prevented the accidental reuse of predictions or adapters produced under different experimental settings.
>
> During the small-GPU validation, malformed structured outputs were traced to input-budget overflow rather than to the retrieval mechanism itself. Retrieval contexts and instructional prompts could exceed the available input budget, causing truncation inside JSON examples. The pipeline was therefore changed to reserve output capacity explicitly, compress the fixed instruction, truncate retrieved passages deterministically while retaining the required top-k passages, and fail explicitly if the resulting prompt still exceeded the budget. Format acceptance was subsequently evaluated using coverage, JSON parse rate, strict schema validity, required-key rate, and field completeness. These checks were treated as engineering release criteria; semantic dashboard quality was assessed separately using independent evaluation layers and human ratings.

Diese Formulierung darf erst mit finalen, überprüften Zahlen ergänzt werden. Die Tiny-Ergebnisse dienen ausschließlich als Nachweis der Pipeline-Funktion und dürfen nicht als finaler Nachweis der Forschungsfragen präsentiert werden.

## 11. Verwandte Projektdokumente

- [`docs/evaluation/final_run_readiness_v4.md`](../evaluation/final_run_readiness_v4.md): Go/No-Go-Gates, Pilotplan und offene Thesis-Evaluation.
- [`docs/evaluation/evaluation_protocol.md`](../evaluation/evaluation_protocol.md): Layer L1–L4, Metrikinterpretation und Nicht-Zirkularität.
- [`docs/project/KAGGLE_TINY_V4_RUN.md`](KAGGLE_TINY_V4_RUN.md): Tiny-Dataset und Kaggle-Ausführung.
- [`src/config/matrix/final.yaml`](../../src/config/matrix/final.yaml): finale Modell-, Methoden- und Seed-Matrix.
- [`src/config/data/dashboard_v4.yaml`](../../src/config/data/dashboard_v4.yaml): vollständiges Frozen-Dataset v4.
- [`data/frozen/dashboard_v4_tiny/manifest.json`](../../data/frozen/dashboard_v4_tiny/manifest.json): Tiny-Lineage, Counts und Leakage-Prüfungen.

## 12. Literatur und methodische Anker

Die wissenschaftliche Begründung baut auf den bereits im Thesis-Referenzbestand geführten Quellen auf:

- Cleveland, W. S., & McGill, R. (1984). *Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods*. [DOI](https://doi.org/10.1080/01621459.1984.10478080).
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Saket, B., Endert, A., & Stasko, J. (2018). *Task-Based Effectiveness of Basic Visualizations*. [DOI](https://doi.org/10.1109/TVCG.2018.2865020).
- van der Lee, C., et al. (2019). *Best Practices for the Human Evaluation of Automatically Generated Text*. [ACL Anthology](https://aclanthology.org/W19-8643/).

Die vollständigen BibTeX-Einträge befinden sich in [`docs/thesis/references.bib`](../thesis/references.bib).
