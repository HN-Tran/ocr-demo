FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VERIFY_SSL=false

ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libreoffice-writer && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
COPY eval ./eval
COPY tests ./tests
COPY data ./data


FROM base AS runtime
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
        --index-url ${PYTORCH_CPU_INDEX_URL} \
        --extra-index-url https://pypi.org/simple \
        torch torchvision && \
    python -m pip install --no-cache-dir ".[paddle,doctr]"

RUN useradd -r -u 101 -m appuser && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser /app

USER 101
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS test
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
        --index-url ${PYTORCH_CPU_INDEX_URL} \
        --extra-index-url https://pypi.org/simple \
        torch torchvision && \
    python -m pip install --no-cache-dir . && \
    python -m pip install --no-cache-dir pytest mypy ruff
CMD ["pytest", "-q"]
