[English](README.md) · [Deutsch](README_DE.md)

# OCR-Demo (Ollama + FastAPI)

Minimales OCR-Demo mit Ollama-Vision-Modell über ein FastAPI-Backend, inklusive schlanker Weboberfläche und Evaluations-Runner.

## Funktionen

- `POST /api/ocr` für Klartext- oder strukturierte Extraktion
- `POST /api/compare` für Side-by-Side-Vergleich gegen externe Engines (Azure, OCR-Demo-Peer, Google Vision, Plain-Text-Endpoint) inklusive Metriken-Panel und optionalem CER/WER gegen Referenztext
- `POST /api/benchmark` für Batch-Benchmarks (N Dateien × M Runner) mit Live-Progress, CSV-Export und optionalem MLflow-Tracking
- `GET /api/models` zum Auflisten verfügbarer Ollama-Modelle
- `GET /api/schemas` zum Anzeigen unterstützter strukturierter Schemata
- `GET /docs` (Swagger UI) und `GET /redoc` (ReDoc) für die interaktive API-Dokumentation
- Browser-UI unter `/` mit zentrierter Startkarte, Drag-and-Drop-Upload, Auto-Run bei Dateiauswahl, Expertenoptionen, schnellen JSON-Vorgaben (Rechnung, Beleg, Tabelle, Visitenkarte), Hell/Dunkel-Modus, Bild/PDF-Vorschau, JSON-Highlighting und CSV-Download für Tabellen
- Wort-Polygon-Overlay im Layout-Viewer (`OCR_WORD_DETECTOR=paddleocr|doctr`): wortgenaue Bounding-Polygone pro Layout-Region
- Evaluations-Runner mit CER/WER und Feldgenauigkeit

## Anforderungen

- Python 3.12+
- `uv` 0.10+
- Laufende Ollama-Instanz (Standard: `http://localhost:11434`)
- Ein vision-fähiges Modell in Ollama

## Setup

```bash
uv sync --all-groups
```

Optionale Umgebungsvariablen:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="glm-ocr:latest"
export OCR_BACKEND="direct" # direct | expert
export OCR_EXPERT_MODE="selfhosted"
export OCR_EXPERT_ENABLE_LAYOUT="true"
export OCR_EXPERT_LAYOUT_MODEL="PaddlePaddle/PP-DocLayoutV3_safetensors"
export OCR_WORD_DETECTOR="none"         # none | paddleocr | doctr
export OCR_EXPERT_OCR_API_HOST="localhost"
export OCR_EXPERT_OCR_API_PORT="11434"
export ANALYZE_STORE_DIR="/tmp/ocr-demo-analyze-results"
export DEFAULT_TOKEN_LIMIT="16384"
export MAX_UPLOAD_BYTES="8388608"
export MAX_IMAGE_DIM="2048"               # Obergrenze für OCR-Bildgröße
export OCR_EXPERT_LAYOUT_MAX_DIM="1800"   # Layout-Detektor sieht max. so groß
export OCR_BINARIZED_MIN_DIM="1800"       # 1-bit/L-Eingaben werden mind. so groß
export BENCHMARK_MAX_FILES="50"          # /api/benchmark Hard-Cap
export BENCHMARK_MAX_RUNNERS="5"         # /api/benchmark Hard-Cap
export MLFLOW_TRACKING_URI=""            # leer = kein Tracking; HTTP- oder file:-URI
export MLFLOW_EXPERIMENT_NAME="ocr-demo"
```

Eingabe-Preprocessing (in `app/services/ocr_pipeline.py`):

- RGBA/LA und transparente Palette-PNGs werden auf **weißem** Hintergrund komponiert (vermeidet schwarze Defaults, die schwarzen Text auf transparentem Hintergrund unsichtbar machen würden).
- Bitonale (`1`) und Graustufen (`L`) Eingaben werden auf mindestens
  `OCR_BINARIZED_MIN_DIM` Pixel (Standard 1800) hochskaliert, damit Modelle
  „l"/„I"/„1" zuverlässiger unterscheiden können.
- Im Expert-Backend wird das Bild vor dem Layout-Detektor zusätzlich auf
  `OCR_EXPERT_LAYOUT_MAX_DIM` (Standard 1800) heruntergeskaliert. Die
  Bounding-Boxes werden danach wieder auf die Originalauflösung skaliert,
  sodass die Per-Region-OCR weiterhin auf dem hochaufgelösten Bild arbeitet.

## Starten

```bash
uv run uvicorn app.main:app --reload
```

Öffnen: `http://127.0.0.1:8000`

## API

`POST /api/ocr` akzeptiert entweder `multipart/form-data` oder einen rohen Body mit
`Content-Type: application/octet-stream`.

Zusätzlich gibt es eine Azure-Read-kompatible Oberfläche für das Container-Swagger aus
`swagger.json`:

- `GET /status`
- `GET /ready`
- `GET /ContainerReadiness`
- `GET /ContainerLiveness`
- `POST /formrecognizer/documentModels/prebuilt-read:syncAnalyze`
- `POST /formrecognizer/documentModels/prebuilt-read:analyze`
- `GET /formrecognizer/documentModels/prebuilt-read/analyzeResults/{resultId}`

Kompatibilitäts-Hinweise:

- `api-version=2022-08-31` ist erforderlich.
- `application/octet-stream` und `application/json` mit `{"urlSource":"..."}` werden akzeptiert.
- `:analyze` liefert `202` plus `Operation-Location`; die Verarbeitung läuft im Hintergrund und wird im Analyze-Store für Polling bereitgestellt.
- Analyze-Ergebnisse werden zusätzlich im Dateisystem unter `ANALYZE_STORE_DIR` persistiert, damit Polling nach einem Prozessneustart auf demselben Volume weiter funktioniert.
- `pages` und `stringIndexType` werden akzeptiert; `pages` filtert aktuell nur das Antwort-Payload, nicht die eigentliche OCR-Ausführung.
- `modelId` ist auf `prebuilt-read` begrenzt.
- `pages`, `paragraphs`, `lines`, `words` und `spans` werden jetzt best-effort aus OCR-Text und Layoutdaten gefüllt. `textElements` bleibt dabei eine pragmatische Annäherung, keine vollständige Grapheme-Cluster-Implementierung.
- `pages[].words` nutzt — sobald `OCR_WORD_DETECTOR=doctr|paddleocr` aktiv ist und der Detektor Wort-Polygone geliefert hat — die echten Detektor-Boxen (gleiche Daten wie der „Wörter"-Tab im Browser) statt der synthetischen Wort-Wrapper aus den Layout-Regionen. Ohne Detektor bleibt der bisherige Fallback erhalten.

Multipart-Felder:

- `file`: Bild, PDF oder Word-Dokument (`image/png`, `image/jpeg`, `image/webp`, `image/gif`, `image/tif`, `image/tiff`, `image/x-tiff`, `application/pdf`, `application/msword`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`)
- `mode`: `plain` oder `structured`
- `schema_name`: erforderlich bei `mode=structured`
- `backend`: optional `direct` oder `expert` (UI: Direct/Dev, Standard aus `OCR_BACKEND`)
- `model`: optionale Modell-Überschreibung
- `token_limit`: optionale Token-/Kontextgrenze (`1..128000`), wird als Ollama-`num_ctx` gesetzt
- `gif_max_frames`: optionales Frame-Limit für animierte GIFs (`1..32`, Standard: `8`)
- `expert_enable_layout`: optionales Layout-Override für `backend=expert` (`true|false`)
- `task`: Klartext-Aufgabenpreset (`ocr_text`, `describe_image`, `read_scene_text`, `extract_table_markdown`, `summarize_document`)
- `custom_prompt`: optionaler Klartext-Prompt, hat Vorrang vor `task`

Raw-Upload (`application/octet-stream`):

- Der Request-Body enthält direkt die Datei-Bytes.
- `mode`, `schema_name`, `backend`, `model`, `token_limit`, `gif_max_frames`,
  `expert_enable_layout`, `task` und `custom_prompt` können als Query-Parameter
  übergeben werden.
- Der Server erkennt `png`, `jpeg`, `webp`, `gif`, `tiff`, `pdf`, `doc` und `docx` anhand der
  Dateisignatur automatisch. Word-Dokumente werden intern via LibreOffice in PDF konvertiert.

PowerShell-Beispiel:

```powershell
Invoke-RestMethod -Method POST `
  -Uri 'https://HOST/api/ocr?backend=direct&mode=plain' `
  -ContentType 'application/octet-stream' `
  -InFile 'C:\path\scan.tiff'
```

Beispiele für `schema_name`:

- `auto` (Schema wird automatisch erkannt)
- `invoice_basic`
- `receipt_basic`
- `table_basic`
- `business_card_basic`

Hinweis: Bei PDF-Dateien werden alle Seiten verarbeitet.
Hinweis: Animierte GIFs werden als Mehrseiten-Eingabe behandelt; bis zu 8 Frames werden gleichmäßig gesampelt verarbeitet.
Hinweis: Für `task=describe_image` bei animierten GIFs wird effizient ein Storyboard aus Sample-Frames in einem Einzelaufruf beschrieben.
Hinweis: `backend=expert` nutzt GLM-OCR primär für `mode=plain` + `task=ocr_text`; für andere Aufgaben fällt die App auf den Direct-Pfad zurück.
Hinweis: Expert/Dev läuft in dieser App nur im Self-Hosted-Modus (`OCR_EXPERT_MODE=selfhosted`).
Hinweis: Bei `backend=expert` kann die Antwort zusätzlich `markdown` enthalten; die UI rendert daraus eine sichere Vorschau, behält aber `text` als Rohausgabe bei.
Hinweis: Das Layout-Modell ist über `OCR_EXPERT_LAYOUT_MODEL` konfigurierbar (Standard: `PaddlePaddle/PP-DocLayoutV3_safetensors`) und kann pro Request via `expert_layout_model` überschrieben werden. PP-DocLayout-Modelle werden direkt von GLM-OCR geladen. Für andere HuggingFace-Object-Detection-Modelle wird automatisch ein generischer Detektor (`HFLayoutDetector`) verwendet, der `AutoModelForObjectDetection` nutzt. YOLO-basierte Modelle werden nicht unterstützt.

Verfügbare Layout-Modelle:

| Modell | Architektur | Polygone | Stärken | Einschränkungen |
|---|---|---|---|---|
| `PaddlePaddle/PP-DocLayoutV3_safetensors` (Standard) | PP-DocLayout V3 mit Instanz-Segmentierung (nativ in GLM-OCR) | Echte Polygone aus Segmentierungsmasken, variable Punktanzahl, konturgetreu | Beste Genauigkeit für nicht-planare Dokumente (schräg, gebogen, Handyfoto), viele Kategorien, Lesereihenfolge | Nur über GLM-OCR-Pipeline nutzbar |
| `pascalrai/Deformable-DETR-Document-Layout-Analysis` | Deformable DETR (reine Objekterkennung, keine Segmentierung) | Nur achsenparallele Bounding-Boxen (4-Punkt-Rechtecke) | Trainiert auf DocLayNet (mAP 0.61), gute Tabellen-/Texterkennung | Benötigt `timm`; keine echten Polygone möglich (architekturbedingt) |
| `Aryn/deformable-detr-DocLayNet` | Deformable DETR (reine Objekterkennung) | Nur achsenparallele Bounding-Boxen | Trainiert auf DocLayNet, alternative Gewichtung | Benötigt `timm`; keine echten Polygone möglich |
| `docling-project/docling-layout-heron` | RT-DETRv2 (reine Objekterkennung) | Nur achsenparallele Bounding-Boxen | Schnelle Inferenz | Erkennt gescannte Seiten oft als einzelne „Picture"-Region; keine echten Polygone möglich |
| `docling-project/docling-layout-heron-101` | RT-DETRv2 (reine Objekterkennung) | Nur achsenparallele Bounding-Boxen | Größere Variante von Heron | Gleiche Einschränkungen wie Heron |

Hinweis: Für erkannte Tabellenregionen wird automatisch eine Zellstruktur-Erkennung via Microsoft Table Transformer (`table-transformer-structure-recognition-v1.1-all`) durchgeführt. Die erkannten Zellen (Zeilen × Spalten, Header, Spanning Cells) werden als `cells`-Array in der jeweiligen Layout-Region zurückgegeben.

Response-Format:

```json
{
  "status": "succeeded",
  "createdDateTime": "2026-03-09T08:00:00+00:00",
  "lastUpdatedDateTime": "2026-03-09T08:00:01+00:00",
  "analyzeResult": {
    "apiVersion": "2026-03-09-preview",
    "modelId": "glm-ocr:latest",
    "stringIndexType": "textElements",
    "content": "...",
    "pages": [
      {
        "pageNumber": 1,
        "angle": 0.0,
        "width": 2480,
        "height": 3508,
        "unit": "pixel",
        "words": [],
        "lines": [],
        "spans": [],
        "kind": "document",
        "content": "..."
      }
    ],
    "paragraphs": [
      {
        "content": "...",
        "spans": []
      }
    ],
    "styles": [],
    "languages": []
  },
  "text": "...",
  "markdown": "# Titel\n\n...",
  "structured": null,
  "layout": [
    {
      "page_number": 1,
      "regions": [
        {
          "index": 0,
          "label": "text_block",
          "content": "...",
          "bbox_2d": [100.0, 120.0, 900.0, 260.0],
          "confidence": 0.96
        }
      ]
    }
  ],
  "model": "glm-ocr:latest",
  "backend": "direct",
  "mode": "plain",
  "schema_name": null,
  "latency_ms": 1234,
  "warnings": []
}
```

## Vergleich mit externer OCR-Engine

`POST /api/compare` führt unsere OCR-Pipeline parallel zu einer externen Engine
aus und liefert Diff, Side-by-Side-Metriken und (optional) CER/WER gegen einen
Referenztext. `GET /api/compare/engines` listet die unterstützten Engines.

**Unterstützte Engines** (`engine`-Form-Feld):

| `engine` | Konfigurations-Felder | Bemerkungen |
|---|---|---|
| `azure` | `azure_endpoint`, `azure_key` | Azure Form Recognizer / Document Intelligence prebuilt-read. Nutzt asynchrones Polling. |
| `self_peer` | `peer_base_url`, `peer_backend` | Postet die Datei an `<peer_base_url>/api/ocr` einer anderen Instanz dieser App — sinnvoll, um zwei Konfigurationen oder Modellversionen direkt zu vergleichen. |
| `google_vision` | `google_api_key` | Google Cloud Vision REST (`DOCUMENT_TEXT_DETECTION`). |
| `plain_text` | `plain_text_url`, optional `plain_text_method`, `plain_text_field`, `plain_text_auth_header`, `plain_text_auth_value` | Generischer Endpoint, der entweder reinen Text oder `{"text": "..."}` liefert. Liefert keine Bounding-Boxen, Diff bleibt textbasiert. |

**Optionale Parameter** (engine-unabhängig):

- `reference_text`: Ground-Truth-Text. Wenn gesetzt, ergänzt die Antwort einen `metrics.reference`-Block mit echten CER, WER und Token-F1 für beide Seiten.
- `expert_*` und `backend`: greifen für unsere eigene OCR-Seite (siehe oben).
- `expert_compare_include_detector_only`: bezieht zusätzlich Wort-Polygone des Detektors in den Diff ein, die kein Layout-Token getroffen haben.

**Metriken im Response** (`metrics`-Block):

- `intrinsic`: Tokens, Zeichen, Ø Konfidenz, Latenz pro Seite.
- `comparison`: paarweise Δ Zeichen, Δ Wörter (normalisierte Levenshtein-Distanz — bewusst _nicht_ CER/WER, da keine Seite Ground Truth ist), Token-Jaccard, Token-Precision/Recall/F1.
- `reference`: nur wenn `reference_text` mitgeliefert wurde — echte CER, WER, Token-F1 pro Seite.

**Azure-Preset** (Browser-Workflow):
Sind `AZURE_PRESET_LABEL`, `AZURE_PRESET_ENDPOINT` und `AZURE_PRESET_KEY` gesetzt,
erscheint im Compare-Formular ein Schnellbutton mit dem Label. Schickt der
Browser eine Anfrage an genau diese Endpoint-URL ohne API-Key, ergänzt der
Server den Schlüssel intern — der geheime Wert verlässt nie das Backend.

**Warum kein AWS Textract?**
AWS Textract ist bewusst _nicht_ enthalten, weil es eine SigV4-Signatur
braucht und damit entweder `boto3` (~10 MB) oder eine eigene Signatur-
Implementierung erfordert. Da keine konkrete AWS-Anforderung besteht,
bleibt die Abhängigkeit draußen. Bei Bedarf:

1. `boto3` als optionalen Extra in `pyproject.toml` ergänzen (analog zum bestehenden `paddle`-Extra).
2. `app/services/compare_engines/aws_textract.py` nach dem Muster der anderen Engines schreiben (Klasse mit `name`/`label`/`async def analyze`).
3. In `app/services/compare_engines/registry.py` registrieren.

## Batch-Benchmark

`/benchmark` (UI) bzw. `POST /api/benchmark` führen N Dateien gegen M Runner
aus — Runner sind entweder lokale Ollama-Modelle oder externe Engines aus
dem Compare-Flow. Pro Zeile werden Token-/Zeichen-Anzahl, Latenz und
(falls Referenztext mitgegeben wurde) CER/WER/Token-F1 berechnet. Aggregat
pro Runner liefert Durchschnitt + Standardabweichung.

### Wie es intern funktioniert

- **In-Memory-Store** (`BenchmarkJobStore` in `app/services/benchmark.py`):
  Ein einzelnes Python-Dict im FastAPI-Prozess hält alle Jobs. `asyncio.Lock`
  serialisiert Mutationen; bei Prozess-Neustart sind die Jobs weg. Pro Replica
  unabhängig — nicht für horizontales Scaling ausgelegt.
- **Worker** (`run_benchmark_job`): wird als `asyncio.create_task(...)` im
  POST-Handler gefeuert, läuft im Hintergrund und mutiert `job.rows` direkt.
  Der POST kommt sofort mit `{job_id}` zurück, der Worker arbeitet weiter.
- **Sequentiell, kein Parallelismus**: alle (Datei × Runner)-Paare werden
  nacheinander abgearbeitet. Grund: Ollama lädt Modelle exklusiv, parallele
  Calls thrashen den Speicher und verfälschen die Latenz-Messung. Externe
  Engines könnten parallel laufen — bisher nicht implementiert,
  weil Latenz-Vergleichbarkeit wichtiger ist als Wall-Clock-Zeit.
- **Tracking**: jede Zeile (`BenchmarkRow`) bekommt Status `pending → running
  → done/error`, plus Metriken sobald der Runner zurückkommt. Das Frontend
  pollt `GET /api/benchmark/{id}` alle 2 s — das Polling ist eine
  Browser-Convenience, der Backend pusht nichts. Bei MLflow wird zusätzlich
  ein verschachtelter Run pro Zeile geschrieben (siehe unten).
- **Persistenz**: drei Schichten, jede opt-in.
  - In-Memory: bis zum nächsten Restart.
  - CSV: `GET /api/benchmark/{id}/csv` als Anhang herunterladen.
  - MLflow: bei gesetztem `MLFLOW_TRACKING_URI` zusätzliches Logging mit
    Artefakten und Parent/Child-Run-Hierarchie.

### Job-Lifecycle

```
POST   /api/benchmark              → { job_id }
GET    /api/benchmark              → { jobs: [...] }
GET    /api/benchmark/{job_id}     → vollständiger Job-State + Live-Progress
GET    /api/benchmark/{job_id}/csv → CSV-Export
DELETE /api/benchmark/{job_id}     → aus dem In-Memory-Store löschen
```

Hard-Caps konfigurierbar via `BENCHMARK_MAX_FILES` (Standard 50) und
`BENCHMARK_MAX_RUNNERS` (Standard 5).

### Per REST steuerbar (curl-Beispiel)

Browser-UI ist optional — die API ist self-contained:

```bash
# 1. Job submitten (zwei Dateien, je ein Referenztext, zwei Modelle, plus Azure)
JOB=$(curl -s -X POST http://localhost:8000/api/benchmark \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "references=Hello world from doc1" \
  -F "references=" \
  -F "models=glm-ocr:latest,qwen2.5vl:7b" \
  -F "engines=azure" \
  -F "azure_endpoint=https://your-host" \
  -F "azure_key=…" \
  | jq -r .job_id)

# 2. Pollen bis fertig
while true; do
  status=$(curl -s "http://localhost:8000/api/benchmark/$JOB" | jq -r .status)
  echo "$status"
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 2
done

# 3. Ergebnis als CSV holen
curl -s "http://localhost:8000/api/benchmark/$JOB/csv" -o "benchmark-$JOB.csv"

# 4. Aus dem Server-Speicher entfernen (optional)
curl -s -X DELETE "http://localhost:8000/api/benchmark/$JOB"
```

`models` und `engines` sind kommagetrennte Listen. Mindestens einer der
beiden Werte muss nicht leer sein. `references` muss in derselben
Reihenfolge wie `files` mitgegeben werden — leerer String = keine Referenz
für diese Datei.

Engine-spezifische Felder (nur die ausgewählten Engines werden bedient):

| Engine | Erforderliche Felder |
|---|---|
| `azure` | `azure_endpoint`, `azure_key` |
| `self_peer` | `peer_base_url`, optional `peer_backend`, `peer_model` |
| `google_vision` | `google_api_key` |
| `plain_text` | `plain_text_url`, optional `plain_text_method`, `plain_text_field`, `plain_text_auth_header`, `plain_text_auth_value` |
| `local_models` | (nicht über Engines — siehe `models`) |

### Antwort-Schema (`GET /api/benchmark/{job_id}`)

```json
{
  "id": "abc123…",
  "created_at": "2026-05-…",
  "status": "running",
  "progress": {"done": 5, "total": 10},
  "options": {"files": [...], "models": [...], "engines": [...]},
  "rows": [
    {
      "file_index": 0,
      "file_name": "doc1.pdf",
      "runner_kind": "local_model",
      "runner_label": "glm-ocr:latest",
      "status": "done",
      "text_chars": 1234,
      "text_tokens": 192,
      "latency_ms": 4521,
      "cer": 0.082,
      "wer": 0.137,
      "token_f1": 0.882,
      "avg_confidence": 0.94,
      "warnings": [],
      "error": null
    }
  ],
  "aggregate": {
    "per_runner": {
      "glm-ocr:latest": {
        "doc_count": 2, "success_count": 2, "failure_count": 0,
        "mean_cer": 0.082, "stdev_cer": 0.041,
        "mean_wer": 0.137, "mean_token_f1": 0.882,
        "mean_latency_ms": 4231
      }
    }
  },
  "error": null,
  "mlflow": {"run_id": "…", "run_url": "https://mlflow…/#/experiments/…/runs/…"}
}
```

### MLflow-Tracking (optional)

Wenn `MLFLOW_TRACKING_URI` gesetzt ist UND `mlflow` installiert wurde
(`pip install '.[mlflow]'`), schreibt der Worker zusätzlich:

- einen Parent-Run pro Job mit Aggregat-Metriken (`<runner>.mean_cer`, …),
- pro (Datei, Runner) einen verschachtelten Child-Run mit Parametern,
  Metriken und `hypothesis.txt` / `reference.txt` als Artefakten.

Konfiguration:

```bash
export MLFLOW_TRACKING_URI=http://mlflow:5000   # oder file:./mlruns
export MLFLOW_EXPERIMENT_NAME=ocr-demo          # default: "ocr-demo"
```

Im Benchmark-UI erscheint ein „MLflow-Run öffnen"-Link, sobald der Job
einen HTTP/HTTPS-Tracking-Server nutzt; das `mlflow.run_url`-Feld in der
JSON-Antwort taugt fürs Verlinken aus eigenen Tools. Bei `file:`-URIs gibt
es keine sinnvolle Browser-URL, dann bleibt das Feld `null`.

## Evaluation

Beispielbilder nach `data/samples/` legen und `data/ground_truth/manifest.jsonl` aktualisieren.

```bash
uv run python -m eval.run --manifest data/ground_truth/manifest.jsonl --samples-dir data/samples --reports-dir eval/reports
```

Der Report wird unter `eval/reports/eval_report_<timestamp>.json` geschrieben.

## Qualitätsprüfungen

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app eval tests
uv run pytest
```

## Abhängigkeiten verwalten (uv)

Installieren/Aktualisieren und Lock-Datei erzeugen:

```bash
uv sync --all-groups
uv lock
```

Runtime-Abhängigkeit hinzufügen:

```bash
uv add <package>
```

Dev-Abhängigkeit hinzufügen:

```bash
uv add --dev <package>
```

## Wort-Polygon-Detektor (optional)

Der Layout-Viewer kann wortgenaue Bounding-Polygone anzeigen. Dafür wird `OCR_WORD_DETECTOR` gesetzt (Standard: `doctr`).

| Backend | Env-Wert | Installation |
|---|---|---|
| Kein Detektor | `none` | — |
| DocTR (Standard) | `doctr` | in den Haupt-Dependencies enthalten |
| PaddleOCR | `paddleocr` | `pip install ".[paddle]"` bzw. Docker-Extra `paddle` |

Im Docker-Image ist zusätzlich das `paddle`-Extra enthalten (`pip install ".[paddle]"`).

Hinweise:
- PaddleOCR erfordert Python 3.12 (keine Wheels für 3.13).
- PaddleOCR detektiert Fließtext auf Zeilenebene; die Polygone werden proportional in Wort-Teilboxen aufgeteilt.
- DocTR liefert nativ wortgenaue Polygone mit eigenem erkanntem Text.

## Docker (isoliertes Ausführen und Testen)

App + Ollama bauen und starten:

```bash
docker compose up --build
```

Öffnen: `http://127.0.0.1:8000`

GPU-Hinweise:

- Compose ist für den `ollama`-Service auf NVIDIA-GPUs konfiguriert.
- Erfordert NVIDIA-Treiber + NVIDIA Container Toolkit auf dem Host.
- Optionale Overrides:
  - `DEFAULT_TOKEN_LIMIT` (Standard: `16384`)
  - `OLLAMA_GPU_DEVICES` (Standard: `all`)
  - `NVIDIA_VISIBLE_DEVICES` (Standard: `all`)
  - `NVIDIA_DRIVER_CAPABILITIES` (Standard: `compute,utility`)

OCR-Modell in Ollama laden (einmalig):

```bash
docker compose exec ollama ollama pull glm-ocr:latest
```

Prüfen, ob nach einer Anfrage GPU genutzt wird:

```bash
docker compose exec ollama ollama ps
```

Bei `PROCESSOR` sollte `GPU` statt `CPU` stehen.

Isolierte Qualitätsprüfungen + Tests ausführen:

```bash
docker compose --profile test run --rm test
```

Container stoppen:

```bash
docker compose down
```

## Lizenz

Apache-2.0 — siehe [`LICENSE`](LICENSE).

## Autor

HN-Tran — <https://github.com/HN-Tran>
