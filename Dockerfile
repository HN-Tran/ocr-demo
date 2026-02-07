FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/.uv-cache

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY app ./app
COPY eval ./eval
COPY tests ./tests
COPY data ./data


FROM base AS runtime
RUN uv sync --locked --no-dev
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS test
RUN uv sync --locked --all-groups
CMD ["uv", "run", "pytest", "-q"]

