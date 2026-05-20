# GLM-OCR via llama.cpp (Docker)

Run [ggml-org/GLM-OCR-GGUF](https://huggingface.co/ggml-org/GLM-OCR-GGUF) in Docker and point docread at its OpenAI-compatible API.

## 1. Start llama-server (Docker)

### AMD Radeon (RX 9070 XT, etc.)

The **`server-cuda` image is NVIDIA-only** and will not use a Radeon GPU.

**Try Vulkan first** (works on many Arch / Mesa setups):

```bash
cd /path/to/docread
docker compose -f docker-compose.llamacpp.yml up llamacpp-vulkan
# If /dev/dri permission denied: set DOCKER_GPU_GID in .env and uncomment group_add in docker-compose.llamacpp.yml
```

**ROCm** (if you have ROCm 6.3+ / 7.x and `/dev/kfd` on the host, e.g. Ubuntu 24.04):

```bash
docker compose -f docker-compose.llamacpp.yml up llamacpp-rocm
```

Host checks:

```bash
ls -l /dev/dri /dev/kfd 2>/dev/null
groups   # should include video or render
```

RX 9070 XT is listed in recent [ROCm Radeon compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html) matrices; on Arch, Vulkan is often easier than ROCm-in-Docker.

### NVIDIA GPU

```bash
docker compose -f docker-compose.llamacpp.yml --profile nvidia up llamacpp-cuda
```

### CPU only

```bash
docker compose -f docker-compose.llamacpp.yml --profile cpu up llamacpp-cpu
```

First run downloads ~1–2 GB of GGUF weights into the `llamacpp_cache` volume. Wait until:

```bash
curl -s http://127.0.0.1:8080/health
# {"status":"ok"} when ready
```

Optional env (`.env` in repo root or export):

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLAMACPP_PORT` | `8080` | Host port |
| `LLAMACPP_HF_REPO` | `ggml-org/GLM-OCR-GGUF` | Hugging Face repo (`:F16` for higher quality) |
| `LLAMACPP_CTX` | `8192` | Context size |
| `LLAMACPP_N_GPU_LAYERS` | `99` | Offload layers to GPU (CUDA / ROCm / Vulkan) |
| `DOCKER_GPU_GID` | — | Optional numeric GID for `/dev/dri`; uncomment `group_add` in compose when needed |

## 2. Resolve model id

```bash
curl -s http://127.0.0.1:8080/v1/models | jq '.data[].id'
```

Typical ids look like `GLM-OCR-Q8_0.gguf` (use the exact string from the API).

## 3. Configure docread

**docread on the host** (separate terminal):

```bash
export INFERENCE_PROVIDER=openai_compatible
export INFERENCE_BASE_URL=http://127.0.0.1:8080/v1
export INFERENCE_MODEL=GLM-OCR-Q8_0.gguf   # from /v1/models
export INFERENCE_VISION_MODELS=$INFERENCE_MODEL
export INFERENCE_VISION_PROBE=false

uv sync --all-groups
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 → Inference-Provider **openai_compatible** → pick the GLM-OCR model.

**docread in Docker** (llama.cpp on the host):

```bash
cp .env.example .env
# Uncomment the openai_compatible / INFERENCE_* block in .env (host.docker.internal:8080)
docker compose up --build
```

`docker-compose.yml` already passes `INFERENCE_*` from `.env`. `extra_hosts` adds `host.docker.internal` on Linux.

**docread + llama.cpp both in Docker** (one command, shared network):

```bash
docker compose -f docker-compose.stack.yml up --build
```

Defaults: `INFERENCE_BASE_URL=http://llamacpp-vulkan:8080/v1`, model `GLM-OCR-Q8_0.gguf`. Confirm the model id with `curl` after the server is up; override in `.env` if needed.

## 4. Quick API test

```bash
curl -s "http://127.0.0.1:8000/api/models?provider=openai_compatible&vision_only=true"

curl -X POST http://127.0.0.1:8000/api/ocr \
  -F "file=@your-scan.png" \
  -F "mode=plain" \
  -F "inference_provider=openai_compatible" \
  -F "model=GLM-OCR-Q8_0.gguf"
```

## 5. Direct llama.cpp smoke test (optional)

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-OCR-Q8_0.gguf",
    "temperature": 0,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,'$(base64 -w0 your-scan.png)'"}},
        {"type": "text", "text": "OCR"}
      ]
    }]
  }'
```

GLM-OCR is trained for short prompts like `OCR`; docread uses its own longer templates, which usually still work.

## Troubleshooting

- **503 / Loading model**: wait for first-time HF download; check `docker compose -f docker-compose.llamacpp.yml logs -f`.
- **Radeon / 9070 XT**: do not use `llamacpp-cuda`; use `llamacpp-vulkan` or `llamacpp-rocm`.
- **Permission denied on /dev/dri**: set `DOCKER_GPU_GID` from `stat -c '%g' /dev/dri/renderD128`; ensure your user is in that group on the host (`groups`).
- **CUDA image on AMD or CPU-only host**: use `llamacpp-vulkan`, `llamacpp-rocm`, or `--profile cpu`.
- **Port 8080 busy**: set `LLAMACPP_PORT=8081` and use `http://127.0.0.1:8081/v1` in docread.
- **Slow model list**: set `INFERENCE_VISION_MODELS` to your model id (see step 3).
