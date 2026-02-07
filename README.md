# OCR-Demo (Ollama + FastAPI)

Minimales OCR-Demo mit Ollama-Vision-Modell über ein FastAPI-Backend, inklusive schlanker Weboberfläche und Evaluations-Runner.

## Funktionen

- `POST /api/ocr` für Klartext- oder strukturierte Extraktion
- `GET /api/models` zum Auflisten verfügbarer Ollama-Modelle
- `GET /api/schemas` zum Anzeigen unterstützter strukturierter Schemata
- Browser-UI unter `/` mit zentrierter Startkarte, Drag-and-Drop-Upload, Auto-Run bei Dateiauswahl, Expertenoptionen, schnellen Rechnungs-/Beleg-Vorgaben, Hell/Dunkel-Modus, Bild/PDF-Vorschau und JSON-Highlighting
- Evaluations-Runner mit CER/WER und Feldgenauigkeit

## Anforderungen

- Python 3.10+
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
export DEFAULT_TOKEN_LIMIT="4096"
export MAX_UPLOAD_BYTES="8388608"
export MAX_IMAGE_DIM="2048"
```

## Starten

```bash
uv run uvicorn app.main:app --reload
```

Öffnen: `http://127.0.0.1:8000`

## API

`POST /api/ocr` (multipart/form-data) Felder:

- `file`: Bild (`image/png`, `image/jpeg`, `image/webp`)
- `mode`: `plain` oder `structured`
- `schema_name`: erforderlich bei `mode=structured`
- `model`: optionale Modell-Überschreibung
- `token_limit`: optionale Token-/Kontextgrenze, wird als Ollama-`num_ctx` gesetzt
- `task`: Klartext-Aufgabenpreset (`ocr_text`, `describe_image`, `read_scene_text`)
- `custom_prompt`: optionaler Klartext-Prompt, hat Vorrang vor `task`

Hinweis: Die UI kann PDFs clientseitig vorschauen, die OCR-API akzeptiert derzeit nur Bilder.

Response-Format:

```json
{
  "text": "...",
  "structured": null,
  "model": "glm-ocr:latest",
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
  - `DEFAULT_TOKEN_LIMIT` (Standard: `4096`)
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
