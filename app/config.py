import json
import os
from dataclasses import dataclass

from app.i18n import normalize_locale
from pathlib import Path
from typing import Any


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


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_inference_extra_providers(raw: str) -> dict[str, "InferenceProviderConfig"]:
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"INFERENCE_EXTRA_PROVIDERS ist kein gültiges JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("INFERENCE_EXTRA_PROVIDERS muss ein JSON-Objekt sein.")
    out: dict[str, InferenceProviderConfig] = {}
    for provider_id, entry in payload.items():
        key = str(provider_id).strip().lower()
        if key not in {"ollama", "openai_compatible"}:
            raise ValueError(
                f"Unbekannter Zusatz-Provider '{provider_id}' in INFERENCE_EXTRA_PROVIDERS."
            )
        if not isinstance(entry, dict):
            raise ValueError(f"Konfiguration für Provider '{provider_id}' muss ein Objekt sein.")
        base_url = str(entry.get("base_url", "")).strip()
        if not base_url:
            raise ValueError(f"base_url fehlt für Provider '{provider_id}'.")
        api_key = str(entry.get("api_key", "")).strip()
        vision_raw = entry.get("vision_models", [])
        if isinstance(vision_raw, str):
            vision_models = _parse_csv_tuple(vision_raw)
        elif isinstance(vision_raw, list):
            vision_models = tuple(
                str(item).strip() for item in vision_raw if str(item).strip()
            )
        else:
            vision_models = ()
        vision_probe_raw = entry.get("vision_probe")
        vision_probe = None
        if isinstance(vision_probe_raw, bool):
            vision_probe = vision_probe_raw
        elif isinstance(vision_probe_raw, str):
            normalized = vision_probe_raw.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                vision_probe = True
            elif normalized in {"0", "false", "no", "off"}:
                vision_probe = False
        out[key] = InferenceProviderConfig(
            base_url=base_url,
            api_key=api_key,
            vision_models=vision_models,
            vision_probe=vision_probe,
        )
    return out


@dataclass(frozen=True)
class InferenceProviderConfig:
    base_url: str
    api_key: str
    vision_models: tuple[str, ...]
    vision_probe: bool | None = None


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_base_path: str
    analyze_store_dir: str
    inference_provider: str
    inference_base_url: str
    inference_model: str
    inference_api_key: str
    inference_vision_models: tuple[str, ...]
    inference_vision_probe: bool
    inference_extra_providers: dict[str, InferenceProviderConfig]
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
    benchmark_job_ttl_s: float
    examples: tuple[tuple[str, str], ...]
    ocr_word_detector: str
    default_token_limit: int
    request_timeout_s: float
    max_upload_bytes: int
    max_image_dim: int
    verify_ssl: bool
    deskew_enabled: bool
    deskew_min_angle_deg: float
    host: str
    port: int
    app_locale: str


def get_settings() -> Settings:
    default_token_limit = _env_int("DEFAULT_TOKEN_LIMIT", 16384)
    if default_token_limit < 1:
        default_token_limit = 16384
    if default_token_limit > 128000:
        default_token_limit = 128000

    inference_provider = os.getenv("INFERENCE_PROVIDER", "ollama").strip().lower()
    if inference_provider not in {"ollama", "openai_compatible"}:
        inference_provider = "ollama"

    inference_base_url = (
        os.getenv("INFERENCE_BASE_URL", "").strip()
        or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    )
    if not inference_base_url:
        inference_base_url = "http://localhost:11434"
    if inference_provider == "openai_compatible" and inference_base_url == "http://localhost:11434":
        if not os.getenv("INFERENCE_BASE_URL", "").strip() and not os.getenv(
            "OLLAMA_BASE_URL", ""
        ).strip():
            inference_base_url = "http://localhost:8000/v1"

    inference_model = (
        os.getenv("INFERENCE_MODEL", "").strip()
        or os.getenv("OLLAMA_MODEL", "glm-ocr:latest").strip()
    )
    inference_api_key = os.getenv("INFERENCE_API_KEY", "").strip()
    inference_vision_models = _parse_csv_tuple(os.getenv("INFERENCE_VISION_MODELS", ""))
    inference_vision_probe = _env_bool("INFERENCE_VISION_PROBE", True)
    inference_extra_providers = _parse_inference_extra_providers(
        os.getenv("INFERENCE_EXTRA_PROVIDERS", "")
    )

    ollama_base_url = inference_base_url
    ollama_model = inference_model

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
        app_name=os.getenv("APP_NAME", "docread"),
        app_base_path=os.getenv("APP_BASE_PATH", ""),
        analyze_store_dir=str(
            Path(os.getenv("ANALYZE_STORE_DIR", "/tmp/docread-analyze-results"))
        ),
        inference_provider=inference_provider,
        inference_base_url=inference_base_url,
        inference_model=inference_model,
        inference_api_key=inference_api_key,
        inference_vision_models=inference_vision_models,
        inference_vision_probe=inference_vision_probe,
        inference_extra_providers=inference_extra_providers,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
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
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "docread").strip(),
        benchmark_max_files=max(1, _env_int("BENCHMARK_MAX_FILES", 50)),
        benchmark_max_runners=max(1, _env_int("BENCHMARK_MAX_RUNNERS", 5)),
        benchmark_job_ttl_s=max(0.0, _env_float("BENCHMARK_JOB_TTL_S", 3600.0)),
        examples=tuple(examples),
        ocr_word_detector=os.getenv("OCR_WORD_DETECTOR", "doctr").strip().lower(),
        default_token_limit=default_token_limit,
        verify_ssl=_env_bool("VERIFY_SSL", False),
        deskew_enabled=_env_bool("DESKEW_ENABLED", False),
        deskew_min_angle_deg=max(0.0, _env_float("DESKEW_MIN_ANGLE_DEG", 0.5)),
        request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 120.0),
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024),
        max_image_dim=_env_int("MAX_IMAGE_DIM", 3600),
        host=os.getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
        app_locale=normalize_locale(os.getenv("APP_LOCALE", "en")),
    )
