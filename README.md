# OCR-Demo (Ollama + FastAPI)

Minimales OCR-Demo mit Ollama-Vision-Modell über ein FastAPI-Backend, inklusive schlanker Weboberfläche und Evaluations-Runner.

## Funktionen

- `POST /api/ocr` für Klartext- oder strukturierte Extraktion
- `GET /api/models` zum Auflisten verfügbarer Ollama-Modelle
- `GET /api/schemas` zum Anzeigen unterstützter strukturierter Schemata
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
export MAX_IMAGE_DIM="2048"
```

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

Multipart-Felder:

- `file`: Bild oder PDF (`image/png`, `image/jpeg`, `image/webp`, `image/gif`, `image/tif`, `image/tiff`, `image/x-tiff`, `application/pdf`)
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
- Der Server erkennt `png`, `jpeg`, `webp`, `gif`, `tiff` und `pdf` anhand der
  Dateisignatur automatisch.

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

Der Layout-Viewer kann wortgenaue Bounding-Polygone anzeigen. Dafür muss ein optionales Backend installiert und über `OCR_WORD_DETECTOR` aktiviert werden.

| Backend | Env-Wert | Installation |
|---|---|---|
| Kein Detektor (Standard) | `none` | — |
| PaddleOCR | `paddleocr` | `pip install ".[paddle]"` bzw. Docker-Extra `paddle` |
| DocTR | `doctr` | `pip install ".[doctr]"` bzw. Docker-Extra `doctr` |

Im Docker-Image sind beide Extras bereits enthalten (`pip install ".[paddle,doctr]"`).

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

Proprietär - nur für interne Nutzung.

## Autor

HN-Tran
