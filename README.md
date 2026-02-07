# OCR Demo (Ollama + FastAPI)

Minimal OCR demo that uses an Ollama vision model from a FastAPI backend, with a lightweight web UI and benchmark harness.

## Features

- `POST /api/ocr` for plain or structured extraction
- `GET /api/models` to list available Ollama models
- `GET /api/schemas` to inspect supported structured schemas
- Browser UI at `/` with dark mode, image/PDF preview, JSON highlighting, and prompt controls
- Evaluation runner with CER/WER and field accuracy metrics

## Requirements

- Python 3.10+
- `uv` 0.10+
- Running Ollama instance (default: `http://localhost:11434`)
- A vision-capable model pulled into Ollama

## Setup

```bash
uv sync --all-groups
```

Optional environment variables:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="glm-ocr:latest"
export MAX_UPLOAD_BYTES="8388608"
export MAX_IMAGE_DIM="2048"
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000`

## API

`POST /api/ocr` multipart form fields:

- `file`: image (`image/png`, `image/jpeg`, `image/webp`)
- `mode`: `plain` or `structured`
- `schema_name`: required when mode is `structured`
- `model`: optional model override
- `task`: plain-mode task preset (`ocr_text`, `describe_image`, `read_scene_text`)
- `custom_prompt`: optional plain-mode override prompt (takes precedence over `task`)

Note: the UI can preview PDFs client-side, but OCR API input is currently image-only.

For vision models, a common plain-mode setup is `task=describe_image` or a custom prompt such as
`Describe this image in concise detail.`.

Response shape:

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

Add sample images to `data/samples/` and update `data/ground_truth/manifest.jsonl`.

```bash
uv run python -m eval.run --manifest data/ground_truth/manifest.jsonl --samples-dir data/samples --reports-dir eval/reports
```

Report output is written to `eval/reports/eval_report_<timestamp>.json`.

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app eval tests
uv run pytest
```

## Dependency Management (uv)

Install/update dependencies and lock:

```bash
uv sync --all-groups
uv lock
```

Add a runtime dependency:

```bash
uv add <package>
```

Add a dev dependency:

```bash
uv add --dev <package>
```

## Docker (Isolated Run and Test)

Build and run app + Ollama:

```bash
docker compose up --build
```

Open: `http://127.0.0.1:8000`

GPU notes:

- Compose is configured to request NVIDIA GPUs for the `ollama` service.
- Requires NVIDIA driver + NVIDIA Container Toolkit installed on the host.
- Optional overrides:
  - `OLLAMA_GPU_DEVICES` (default: `all`)
  - `NVIDIA_VISIBLE_DEVICES` (default: `all`)
  - `NVIDIA_DRIVER_CAPABILITIES` (default: `compute,utility`)

Pull your OCR model inside Ollama (first-time setup):

```bash
docker compose exec ollama ollama pull glm-ocr:latest
```

Verify model is running on GPU after a request:

```bash
docker compose exec ollama ollama ps
```

Look for `PROCESSOR` showing `GPU` instead of `CPU`.

Run isolated quality checks + tests:

```bash
docker compose --profile test run --rm test
```

Stop containers:

```bash
docker compose down
```

## License

Proprietary - Internal use only.

## Author

HN-Tran
