import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


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


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_base_path: str
    analyze_store_dir: str
    ollama_base_url: str
    ollama_model: str
    ocr_backend: str
    ocr_expert_mode: str
    ocr_expert_enable_layout: bool
    ocr_expert_layout_model: str
    ocr_expert_ocr_api_host: str
    ocr_expert_ocr_api_port: int
    default_token_limit: int
    request_timeout_s: float
    max_upload_bytes: int
    max_image_dim: int
    host: str
    port: int


def get_settings() -> Settings:
    default_token_limit = _env_int("DEFAULT_TOKEN_LIMIT", 16384)
    if default_token_limit < 1:
        default_token_limit = 16384
    if default_token_limit > 128000:
        default_token_limit = 128000

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    parsed_ollama_url = urlparse(ollama_base_url)
    default_expert_host = parsed_ollama_url.hostname or "localhost"
    default_expert_port = parsed_ollama_url.port or 11434

    ocr_backend = os.getenv("OCR_BACKEND", "expert").strip().lower()
    if ocr_backend not in {"direct", "expert"}:
        ocr_backend = "expert"

    ocr_expert_mode = os.getenv("OCR_EXPERT_MODE", "selfhosted").strip().lower()
    if ocr_expert_mode != "selfhosted":
        ocr_expert_mode = "selfhosted"

    ocr_expert_ocr_api_port = _env_int("OCR_EXPERT_OCR_API_PORT", default_expert_port)
    if ocr_expert_ocr_api_port <= 0:
        ocr_expert_ocr_api_port = default_expert_port

    return Settings(
        app_name=os.getenv("APP_NAME", "OCR-Demo"),
        app_base_path=os.getenv("APP_BASE_PATH", ""),
        analyze_store_dir=str(
            Path(os.getenv("ANALYZE_STORE_DIR", "/tmp/ocr-demo-analyze-results"))
        ),
        ollama_base_url=ollama_base_url,
        ollama_model=os.getenv("OLLAMA_MODEL", "glm-ocr:latest"),
        ocr_backend=ocr_backend,
        ocr_expert_mode=ocr_expert_mode,
        ocr_expert_enable_layout=_env_bool("OCR_EXPERT_ENABLE_LAYOUT", True),
        ocr_expert_layout_model=os.getenv(
            "OCR_EXPERT_LAYOUT_MODEL", "PaddlePaddle/PP-DocLayoutV3_safetensors"
        ),
        ocr_expert_ocr_api_host=os.getenv("OCR_EXPERT_OCR_API_HOST", default_expert_host),
        ocr_expert_ocr_api_port=ocr_expert_ocr_api_port,
        default_token_limit=default_token_limit,
        request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 120.0),
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024),
        max_image_dim=_env_int("MAX_IMAGE_DIM", 2048),
        host=os.getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
    )
