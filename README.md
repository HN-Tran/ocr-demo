[English](README.md) · [Deutsch](README_DE.md)

# docread (FastAPI + pluggable vision LLMs)

Document OCR service using vision language models via a FastAPI backend. Inference defaults to Ollama and can use OpenAI-compatible servers (llama.cpp, vLLM, etc.). Includes a lightweight web UI and evaluation runner.

## Features

- `POST /api/ocr` for plain-text or structured extraction
- `POST /api/compare` for side-by-side comparison against external engines (Azure, docread-Peer, Google Vision, Plain-Text-Endpoint) including a metrics panel and optional CER/WER against reference text
- `POST /api/benchmark` for batch benchmarks (N files × M runners) with live progress, CSV export, and optional MLflow tracking
- `GET /api/models` to list models from the configured inference provider
- `GET /api/schemas` to display supported structured schemas
- `GET /docs` (Swagger UI) and `GET /redoc` (ReDoc) for interactive API documentation
- Browser UI at `/` with centered start card, drag-and-drop upload, auto-run on file selection, expert options, quick JSON presets (invoice, receipt, table, business card), light/dark mode, image/PDF preview, JSON highlighting, and CSV download for tables
- Word polygon overlay in the layout viewer (`OCR_WORD_DETECTOR=paddleocr|doctr`): word-precise bounding polygons per layout region
- Evaluation runner with CER/WER and field accuracy

## Requirements

- Python 3.12+
- `uv` 0.10+
- A vision-capable model on your inference backend (Ollama by default, or an OpenAI-compatible server)

## Setup

```bash
uv sync --all-groups
```

Optional environment variables:

```bash
export INFERENCE_PROVIDER="ollama"              # ollama | openai_compatible
export INFERENCE_BASE_URL="http://localhost:11434"
export INFERENCE_MODEL="glm-ocr:latest"
# OpenAI-compatible example (vLLM / llama.cpp server); GLM-OCR via Docker: docs/llamacpp-docker-glm-ocr.md
# export INFERENCE_PROVIDER="openai_compatible"
# export INFERENCE_BASE_URL="http://localhost:8000/v1"
# export INFERENCE_MODEL="your-vision-model"
# export INFERENCE_API_KEY=""                   # optional Bearer token
# export INFERENCE_VISION_MODELS="model-a"      # optional comma-separated allowlist for vision_only
# export INFERENCE_VISION_PROBE="true"          # probe OpenAI-compatible catalogs when allowlist empty
# export INFERENCE_EXTRA_PROVIDERS='{"openai_compatible":{"base_url":"http://localhost:8000/v1"}}'
# Legacy aliases (still supported): OLLAMA_BASE_URL, OLLAMA_MODEL
# API: inference_provider form field; model as provider/model (e.g. openai_compatible/my-vlm)
export OCR_BACKEND="direct" # direct | expert
export OCR_EXPERT_MODE="selfhosted"
export OCR_EXPERT_ENABLE_LAYOUT="true"
export OCR_EXPERT_LAYOUT_MODEL="PaddlePaddle/PP-DocLayoutV3_safetensors"
export OCR_WORD_DETECTOR="none"         # none | paddleocr | doctr
export OCR_EXPERT_OCR_API_HOST="localhost"
export OCR_EXPERT_OCR_API_PORT="11434"
export ANALYZE_STORE_DIR="/tmp/docread-analyze-results"
export DEFAULT_TOKEN_LIMIT="16384"
export MAX_UPLOAD_BYTES="8388608"
export MAX_IMAGE_DIM="2048"               # Upper limit for OCR image size
export OCR_EXPERT_LAYOUT_MAX_DIM="1800"   # Layout detector sees at most this size
export OCR_BINARIZED_MIN_DIM="1800"       # 1-bit/L inputs are upscaled to at least this size
export BENCHMARK_MAX_FILES="50"           # /api/benchmark hard cap
export BENCHMARK_MAX_RUNNERS="5"          # /api/benchmark hard cap
export MLFLOW_TRACKING_URI=""             # empty = no tracking; HTTP or file: URI
export MLFLOW_EXPERIMENT_NAME="docread"
```

Input preprocessing (in `app/services/ocr_pipeline.py`):

- RGBA/LA and transparent palette PNGs are composited onto a **white** background (avoids black defaults that would make black text on a transparent background invisible).
- Bitonal (`1`) and grayscale (`L`) inputs are upscaled to at least `OCR_BINARIZED_MIN_DIM` pixels (default 1800) so models can more reliably distinguish `l`/`I`/`1`.
- In the expert backend, the image is additionally downscaled to `OCR_EXPERT_LAYOUT_MAX_DIM` (default 1800) before the layout detector. Bounding boxes are then rescaled back to the original resolution so that per-region OCR continues to work on the high-resolution image.

## Start

```bash
uv run uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000`

## API

`POST /api/ocr` accepts either `multipart/form-data` or a raw body with `Content-Type: application/octet-stream`.

In addition, there is an Azure-Read-compatible interface for the container Swagger from `swagger.json`:

- `GET /status`
- `GET /ready`
- `GET /ContainerReadiness`
- `GET /ContainerLiveness`
- `POST /formrecognizer/documentModels/prebuilt-read:syncAnalyze`
- `POST /formrecognizer/documentModels/prebuilt-read:analyze`
- `GET /formrecognizer/documentModels/prebuilt-read/analyzeResults/{resultId}`

Compatibility notes:

- `api-version=2022-08-31` is required.
- `application/octet-stream` and `application/json` with `{"urlSource":"..."}` are accepted.
- `:analyze` returns `202` plus `Operation-Location`; processing runs in the background and is stored in the analyze store for polling.
- Analyze results are additionally persisted on the filesystem under `ANALYZE_STORE_DIR` so polling continues to work after a process restart on the same volume.
- `pages` and `stringIndexType` are accepted; `pages` currently only filters the response payload, not the actual OCR execution.
- `modelId` is limited to `prebuilt-read`.
- `pages`, `paragraphs`, `lines`, `words`, and `spans` are now best-effort populated from OCR text and layout data. `textElements` remains a pragmatic approximation, not a complete grapheme cluster implementation.
- `pages[].words` uses — when `OCR_WORD_DETECTOR=doctr|paddleocr` is active and the detector provided word polygons — the real detector boxes (same data as the "Words" tab in the browser) instead of the synthetic word wrappers from layout regions. Without a detector the previous fallback remains.

Multipart fields:

- `file`: image, PDF, or Word document (`image/png`, `image/jpeg`, `image/webp`, `image/gif`, `image/tif`, `image/tiff`, `image/x-tiff`, `application/pdf`, `application/msword`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`)
- `mode`: `plain` or `structured`
- `schema_name`: required for `mode=structured`
- `backend`: optional `direct` or `expert` (UI: Direct/Dev, default from `OCR_BACKEND`)
- `model`: optional model override
- `token_limit`: optional token/context limit (`1..128000`), set as Ollama `num_ctx`
- `gif_max_frames`: optional frame limit for animated GIFs (`1..32`, default: `8`)
- `expert_enable_layout`: optional layout override for `backend=expert` (`true|false`)
- `task`: plain-text task preset (`ocr_text`, `describe_image`, `read_scene_text`, `extract_table_markdown`, `summarize_document`)
- `custom_prompt`: optional plain-text prompt, takes precedence over `task`

Raw upload (`application/octet-stream`):

- The request body contains the file bytes directly.
- `mode`, `schema_name`, `backend`, `model`, `token_limit`, `gif_max_frames`, `expert_enable_layout`, `task`, and `custom_prompt` can be passed as query parameters.
- The server automatically detects `png`, `jpeg`, `webp`, `gif`, `tiff`, `pdf`, `doc`, and `docx` from the file signature. Word documents are internally converted to PDF via LibreOffice.

PowerShell example:

```powershell
Invoke-RestMethod -Method POST `
  -Uri 'https://HOST/api/ocr?backend=direct&mode=plain' `
  -ContentType 'application/octet-stream' `
  -InFile 'C:\path\scan.tiff'
```

Example `schema_name` values:

- `auto` (schema is detected automatically)
- `invoice_basic`
- `receipt_basic`
- `table_basic`
- `business_card_basic`

Note: For PDF files, all pages are processed.
Note: Animated GIFs are treated as multi-page input; up to 8 frames are uniformly sampled.
Note: For `task=describe_image` with animated GIFs, a storyboard from sample frames is efficiently described in a single call.
Note: `backend=expert` uses GLM-OCR primarily for `mode=plain` + `task=ocr_text`; for other tasks the app falls back to the direct path.
Note: Expert/Dev only runs in self-hosted mode in this app (`OCR_EXPERT_MODE=selfhosted`).
Note: With `backend=expert`, the response may additionally contain `markdown`; the UI renders a safe preview of it but keeps `text` as raw output.
Note: The layout model is configurable via `OCR_EXPERT_LAYOUT_MODEL` (default: `PaddlePaddle/PP-DocLayoutV3_safetensors`) and can be overridden per request via `expert_layout_model`. PP-DocLayout models are loaded directly by GLM-OCR. For other HuggingFace object detection models, a generic detector (`HFLayoutDetector`) is used automatically, which uses `AutoModelForObjectDetection`. YOLO-based models are not supported.

Available layout models:

| Model | Architecture | Polygons | Strengths | Limitations |
|---|---|---|---|---|
| `PaddlePaddle/PP-DocLayoutV3_safetensors` (default) | PP-DocLayout V3 with instance segmentation (native in GLM-OCR) | True polygons from segmentation masks, variable point count, contour-accurate | Best accuracy for non-planar documents (tilted, curved, phone photo), many categories, reading order | Only usable via GLM-OCR pipeline |
| `pascalrai/Deformable-DETR-Document-Layout-Analysis` | Deformable DETR (object detection only, no segmentation) | Axis-aligned bounding boxes only (4-point rectangles) | Trained on DocLayNet (mAP 0.61), good table/text detection | Requires `timm`; no true polygons possible (architecture limitation) |
| `Aryn/deformable-detr-DocLayNet` | Deformable DETR (object detection only) | Axis-aligned bounding boxes only | Trained on DocLayNet, alternative weights | Requires `timm`; no true polygons possible |
| `docling-project/docling-layout-heron` | RT-DETRv2 (object detection only) | Axis-aligned bounding boxes only | Fast inference | Often detects scanned pages as a single "Picture" region; no true polygons possible |
| `docling-project/docling-layout-heron-101` | RT-DETRv2 (object detection only) | Axis-aligned bounding boxes only | Larger variant of Heron | Same limitations as Heron |

Note: For detected table regions, cell structure detection is automatically performed via Microsoft Table Transformer (`table-transformer-structure-recognition-v1.1-all`). The detected cells (rows × columns, headers, spanning cells) are returned as a `cells` array in the respective layout region.

Response format:

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
  "markdown": "# Title\n\n...",
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

## Comparison with External OCR Engine

`POST /api/compare` runs our OCR pipeline in parallel with an external engine and returns a diff, side-by-side metrics, and (optionally) CER/WER against a reference text. `GET /api/compare/engines` lists the supported engines.

**Supported engines** (`engine` form field):

| `engine` | Configuration fields | Notes |
|---|---|---|
| `azure` | `azure_endpoint`, `azure_key` | Azure Form Recognizer / Document Intelligence prebuilt-read. Uses async polling. |
| `self_peer` | `peer_base_url`, `peer_backend` | Posts the file to `<peer_base_url>/api/ocr` of another instance of this app — useful for comparing two configurations or model versions directly. |
| `google_vision` | `google_api_key` | Google Cloud Vision REST (`DOCUMENT_TEXT_DETECTION`). |
| `plain_text` | `plain_text_url`, optional `plain_text_method`, `plain_text_field`, `plain_text_auth_header`, `plain_text_auth_value` | Generic endpoint that returns either plain text or `{"text": "..."}`. Provides no bounding boxes; diff remains text-based. |

**Optional parameters** (engine-independent):

- `reference_text`: Ground-truth text. When set, the response adds a `metrics.reference` block with true CER, WER, and token-F1 for both sides.
- `expert_*` and `backend`: apply to our own OCR side (see above).
- `expert_compare_include_detector_only`: additionally includes word polygons from the detector that did not hit a layout token in the diff.

**Metrics in response** (`metrics` block):

- `intrinsic`: tokens, characters, avg confidence, latency per page.
- `comparison`: pairwise Δ characters, Δ words (normalized Levenshtein distance — deliberately _not_ CER/WER since neither side is ground truth), token Jaccard, token precision/recall/F1.
- `reference`: only when `reference_text` was provided — true CER, WER, token-F1 per side.

**Azure preset** (browser workflow):
When `AZURE_PRESET_LABEL`, `AZURE_PRESET_ENDPOINT`, and `AZURE_PRESET_KEY` are set, the compare form shows a quick button with the label. If the browser sends a request to exactly this endpoint URL without an API key, the server adds the key internally — the secret never leaves the backend.

**Why no AWS Textract?**
AWS Textract is deliberately _not_ included because it requires SigV4 signing, which requires either `boto3` (~10 MB) or a custom signing implementation. Since there is no concrete AWS requirement, the dependency stays out. To add it:

1. Add `boto3` as an optional extra in `pyproject.toml` (analogous to the existing `paddle` extra).
2. Write `app/services/compare_engines/aws_textract.py` following the pattern of the other engines (class with `name`/`label`/`async def analyze`).
3. Register in `app/services/compare_engines/registry.py`.

## Batch Benchmark

`/benchmark` (UI) or `POST /api/benchmark` run N files against M runners — runners are either local Ollama models or external engines from the compare flow. Per row, token/character count, latency, and (if reference text was provided) CER/WER/token-F1 are calculated. The aggregate per runner delivers mean + standard deviation.

### How it works internally

- **In-memory store** (`BenchmarkJobStore` in `app/services/benchmark.py`): A single Python dict in the FastAPI process holds all jobs. `asyncio.Lock` serializes mutations; jobs are lost on process restart. Independent per replica — not designed for horizontal scaling.
- **Worker** (`run_benchmark_job`): fired as `asyncio.create_task(...)` in the POST handler, runs in the background, and mutates `job.rows` directly. The POST returns immediately with `{job_id}`; the worker continues in the background.
- **Sequential, no parallelism**: all (file × runner) pairs are processed one after another. Reason: Ollama loads models exclusively; parallel calls thrash memory and skew latency measurements. External engines _could_ run in parallel — not implemented yet because latency comparability is more important than wall-clock time.
- **Tracking**: each row (`BenchmarkRow`) gets status `pending → running → done/error`, plus metrics as soon as the runner responds. The frontend polls `GET /api/benchmark/{id}` every 2 s — polling is a browser convenience; the backend pushes nothing. With MLflow, an additional nested run per row is written (see below).
- **Persistence**: three layers, each opt-in.
  - In-memory: until next restart.
  - CSV: download via `GET /api/benchmark/{id}/csv`.
  - MLflow: with `MLFLOW_TRACKING_URI` set, additional logging with artifacts and parent/child run hierarchy.

### Job Lifecycle

```
POST   /api/benchmark              → { job_id }
GET    /api/benchmark              → { jobs: [...] }
GET    /api/benchmark/{job_id}     → full job state + live progress
GET    /api/benchmark/{job_id}/csv → CSV export
DELETE /api/benchmark/{job_id}     → remove from in-memory store
```

Hard caps configurable via `BENCHMARK_MAX_FILES` (default 50) and `BENCHMARK_MAX_RUNNERS` (default 5).

### Controllable via REST (curl example)

The browser UI is optional — the API is self-contained:

```bash
# 1. Submit job (two files, one reference text each, two models, plus Azure)
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

# 2. Poll until done
while true; do
  status=$(curl -s "http://localhost:8000/api/benchmark/$JOB" | jq -r .status)
  echo "$status"
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 2
done

# 3. Fetch result as CSV
curl -s "http://localhost:8000/api/benchmark/$JOB/csv" -o "benchmark-$JOB.csv"

# 4. Remove from server memory (optional)
curl -s -X DELETE "http://localhost:8000/api/benchmark/$JOB"
```

`models` and `engines` are comma-separated lists. At least one must be non-empty. `references` must be provided in the same order as `files` — an empty string means no reference for that file.

Engine-specific fields (only the selected engines are served):

| Engine | Required fields |
|---|---|
| `azure` | `azure_endpoint`, `azure_key` |
| `self_peer` | `peer_base_url`, optional `peer_backend`, `peer_model` |
| `google_vision` | `google_api_key` |
| `plain_text` | `plain_text_url`, optional `plain_text_method`, `plain_text_field`, `plain_text_auth_header`, `plain_text_auth_value` |
| `local_models` | (not via engines — see `models`) |

### Response schema (`GET /api/benchmark/{job_id}`)

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

### MLflow Tracking (optional)

When `MLFLOW_TRACKING_URI` is set AND `mlflow` is installed (`pip install '.[mlflow]'`), the worker additionally writes:

- a parent run per job with aggregate metrics (`<runner>.mean_cer`, …),
- per (file, runner) a nested child run with parameters, metrics, and `hypothesis.txt` / `reference.txt` as artifacts.

Configuration:

```bash
export MLFLOW_TRACKING_URI=http://mlflow:5000   # or file:./mlruns
export MLFLOW_EXPERIMENT_NAME=docread          # default: "docread"
```

In the benchmark UI, an "Open MLflow Run" link appears as soon as the job uses an HTTP/HTTPS tracking server; the `mlflow.run_url` field in the JSON response is useful for linking from custom tools. For `file:` URIs there is no useful browser URL, so the field remains `null`.

## Evaluation

Place sample images in `data/samples/` and update `data/ground_truth/manifest.jsonl`.

```bash
uv run python -m eval.run --manifest data/ground_truth/manifest.jsonl --samples-dir data/samples --reports-dir eval/reports
```

The report is written to `eval/reports/eval_report_<timestamp>.json`.

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app eval tests
uv run pytest
```

## Managing Dependencies (uv)

Install/update and generate lock file:

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

## Word Polygon Detector (optional)

The layout viewer can display word-precise bounding polygons. Set `OCR_WORD_DETECTOR` for this (default: `doctr`).

| Backend | Env value | Installation |
|---|---|---|
| No detector | `none` | — |
| DocTR (default) | `doctr` | included in main dependencies |
| PaddleOCR | `paddleocr` | `pip install ".[paddle]"` or Docker extra `paddle` |

The Docker image additionally includes the `paddle` extra (`pip install ".[paddle]"`).

Notes:
- PaddleOCR requires Python 3.12 (no wheels for 3.13).
- PaddleOCR detects running text at line level; polygons are proportionally split into word sub-boxes.
- DocTR natively delivers word-precise polygons with its own recognized text.

## Docker (isolated execution and testing)

Build and start the app:

```bash
docker compose up --build
```

Open: `http://127.0.0.1:8000`

UI language defaults to English (`APP_LOCALE=en`). Use the **EN / DE** toggle next to the theme control, or set `APP_LOCALE=de` in `.env`. Preference is stored in a cookie and `localStorage`.

Docker persists downloaded models in the `docread_model_cache` volume (`/home/appuser/.cache`). Default image uses **CPU** PyTorch (`Dockerfile`). GPU layout uses official base images (see [`docs/docker-pytorch.md`](docs/docker-pytorch.md)):

```bash
# NVIDIA — Dockerfile.cuda (pytorch/pytorch)
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build

# AMD — Dockerfile.rocm (rocm/pytorch)
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
```

GPU vision LLM uses **llama.cpp** ([`docs/llamacpp-docker-glm-ocr.md`](docs/llamacpp-docker-glm-ocr.md)), not PyTorch Vulkan.

Inference env vars (`INFERENCE_PROVIDER`, `INFERENCE_BASE_URL`, `INFERENCE_MODEL`, …) are wired in `docker-compose.yml` and read from `.env`. For docread + bundled GLM-OCR:

```bash
docker compose -f docker-compose.stack.yml up --build
```

GPU notes:

- Compose is configured for NVIDIA GPUs on the `ollama` service.
- Requires NVIDIA drivers + NVIDIA Container Toolkit on the host.
- Optional overrides:
  - `DEFAULT_TOKEN_LIMIT` (default: `16384`)
  - `OLLAMA_GPU_DEVICES` (default: `all`)
  - `NVIDIA_VISIBLE_DEVICES` (default: `all`)
  - `NVIDIA_DRIVER_CAPABILITIES` (default: `compute,utility`)

Load the OCR model in Ollama (once):

```bash
docker compose exec ollama ollama pull glm-ocr:latest
```

Check if GPU is being used after a request:

```bash
docker compose exec ollama ollama ps
```

`PROCESSOR` should show `GPU` instead of `CPU`.

Run isolated quality checks + tests:

```bash
docker compose --profile test run --rm test
```

Stop containers:

```bash
docker compose down
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

## Author

HN-Tran — <https://github.com/HN-Tran>
