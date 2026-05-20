# PyTorch in Docker (layout detector)

Layout (PP-DocLayout) uses **PyTorch**. There are three Dockerfiles — pick the matching compose override:

| Dockerfile | Base image | Compose |
|------------|------------|---------|
| `Dockerfile` | `python:3.12-slim` + CPU wheels | `docker compose up --build` |
| `Dockerfile.cuda` | [`pytorch/pytorch`](https://hub.docker.com/r/pytorch/pytorch) CUDA runtime | `+ docker-compose.cuda.yml` |
| `Dockerfile.rocm` | [`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch) (AMD-validated) | `+ docker-compose.rocm.yml` |

PyTorch has **no Vulkan** build. Vulkan is only for the [vision LLM (llama.cpp)](llamacpp-docker-glm-ocr.md).

## CPU (default)

```bash
docker compose up --build
```

Includes the optional `paddle` extra for PaddleOCR in Docker.

## NVIDIA

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

Override the base tag if needed:

```bash
PYTORCH_CUDA_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

`OCR_EXPERT_LAYOUT_DEVICE=auto` should resolve to `cuda` when the GPU is visible in the container.

## AMD ROCm

Uses AMD’s prebuilt image per [PyTorch on ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html):

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
```

Override the tag from [Docker Hub](https://hub.docker.com/r/rocm/pytorch/tags) if your host ROCm version differs:

```bash
ROCM_PYTORCH_IMAGE=rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
```

If the container cannot access the GPU (`/dev/dri` permission denied), set a **numeric** supplementary GID in `.env` and uncomment `group_add` in `docker-compose.rocm.yml` (do not use a group name like `render` unless it exists in the host `/etc/group`):

```bash
echo "DOCKER_GPU_GID=$(stat -c '%g' /dev/dri/renderD128)" >> .env
# then uncomment group_add in docker-compose.rocm.yml
```

(Ollama on NVIDIA uses `gpus: all` instead; that path does not apply here.)

Some RDNA4 cards need in `.env`:

```bash
HSA_OVERRIDE_GFX_VERSION=12.0.0
```

CUDA/ROCm images install app dependencies **without** replacing the base image’s `torch`/`torchvision` (see `scripts/docker_install_app_deps.py`).

## Vision LLM (separate)

```bash
docker compose -f docker-compose.stack.yml up --build
```

See [llamacpp-docker-glm-ocr.md](llamacpp-docker-glm-ocr.md).

## Model cache

`docread_model_cache` → `/home/appuser/.cache` for Hugging Face and DocTR weights.
