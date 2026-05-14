import os
from dataclasses import dataclass
from pathlib import Path


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
    ocr_expert_enable_layout: bool
    ocr_expert_layout_model: str
    ocr_expert_table_transformer: bool
    ocr_expert_per_region_ocr: bool
    ocr_expert_text_anchor: bool
    ocr_expert_text_anchor_threshold: float
    ocr_expert_compare_include_detector_only: bool
    ocr_expert_layout_max_dim: int
    ocr_binarized_min_dim: int
    azure_preset_label: str
    azure_preset_endpoint: str
    azure_preset_layout_endpoint: str
    azure_preset_key: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    benchmark_max_files: int
    benchmark_max_runners: int
    examples: tuple[tuple[str, str], ...]
    ocr_word_detector: str
    default_token_limit: int
    request_timeout_s: float
    max_upload_bytes: int
    max_image_dim: int
    verify_ssl: bool
    host: str
    port: int


def get_settings() -> Settings:
    default_token_limit = _env_int("DEFAULT_TOKEN_LIMIT", 16384)
    if default_token_limit < 1:
        default_token_limit = 16384
    if default_token_limit > 128000:
        default_token_limit = 128000

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    ocr_backend = os.getenv("OCR_BACKEND", "expert").strip().lower()
    if ocr_backend not in {"direct", "expert"}:
        ocr_backend = "expert"

    examples: list[tuple[str, str]] = []
    for slot in range(1, 4):
        label = os.getenv(f"EXAMPLE_{slot}_LABEL", "").strip()
        path = os.getenv(f"EXAMPLE_{slot}_PATH", "").strip()
        if label and path:
            examples.append((label, path))

    return Settings(
        app_name=os.getenv("APP_NAME", "OCR-Demo"),
        app_base_path=os.getenv("APP_BASE_PATH", ""),
        analyze_store_dir=str(
            Path(os.getenv("ANALYZE_STORE_DIR", "/tmp/ocr-demo-analyze-results"))
        ),
        ollama_base_url=ollama_base_url,
        ollama_model=os.getenv("OLLAMA_MODEL", "glm-ocr:latest"),
        ocr_backend=ocr_backend,
        ocr_expert_enable_layout=_env_bool("OCR_EXPERT_ENABLE_LAYOUT", True),
        ocr_expert_layout_model=os.getenv(
            "OCR_EXPERT_LAYOUT_MODEL", "PaddlePaddle/PP-DocLayoutV3_safetensors"
        ),
        ocr_expert_table_transformer=_env_bool("OCR_EXPERT_TABLE_TRANSFORMER", False),
        ocr_expert_per_region_ocr=_env_bool("OCR_EXPERT_PER_REGION_OCR", True),
        ocr_expert_text_anchor=_env_bool("OCR_EXPERT_TEXT_ANCHOR", True),
        ocr_expert_text_anchor_threshold=_env_float("OCR_EXPERT_TEXT_ANCHOR_THRESHOLD", 60.0),
        ocr_expert_compare_include_detector_only=_env_bool(
            "OCR_EXPERT_COMPARE_INCLUDE_DETECTOR_ONLY", False
        ),
        ocr_expert_layout_max_dim=max(256, _env_int("OCR_EXPERT_LAYOUT_MAX_DIM", 1800)),
        ocr_binarized_min_dim=max(0, _env_int("OCR_BINARIZED_MIN_DIM", 1800)),
        azure_preset_label=os.getenv("AZURE_PRESET_LABEL", "").strip(),
        azure_preset_endpoint=os.getenv("AZURE_PRESET_ENDPOINT", "").strip(),
        azure_preset_layout_endpoint=os.getenv("AZURE_PRESET_LAYOUT_ENDPOINT", "").strip(),
        azure_preset_key=os.getenv("AZURE_PRESET_KEY", "").strip(),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "").strip(),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "ocr-demo").strip(),
        benchmark_max_files=max(1, _env_int("BENCHMARK_MAX_FILES", 50)),
        benchmark_max_runners=max(1, _env_int("BENCHMARK_MAX_RUNNERS", 5)),
        examples=tuple(examples),
        ocr_word_detector=os.getenv("OCR_WORD_DETECTOR", "doctr").strip().lower(),
        default_token_limit=default_token_limit,
        verify_ssl=_env_bool("VERIFY_SSL", False),
        request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 120.0),
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024),
        max_image_dim=_env_int("MAX_IMAGE_DIM", 3600),
        host=os.getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
    )
