import os
from dataclasses import dataclass


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str
    ollama_base_url: str
    ollama_model: str
    default_token_limit: int
    request_timeout_s: float
    max_upload_bytes: int
    max_image_dim: int
    host: str
    port: int


def get_settings() -> Settings:
    default_token_limit = _env_int("DEFAULT_TOKEN_LIMIT", 4096)
    if default_token_limit < 1:
        default_token_limit = 4096

    return Settings(
        app_name=os.getenv("APP_NAME", "OCR-Demo"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "glm-ocr:latest"),
        default_token_limit=default_token_limit,
        request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 120.0),
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024),
        max_image_dim=_env_int("MAX_IMAGE_DIM", 2048),
        host=os.getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
    )
