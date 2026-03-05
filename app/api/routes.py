from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.config import get_settings
from app.schemas import SCHEMA_REGISTRY
from app.services.backend_router import OCRBackendRouter
from app.services.ollama_client import OllamaClient, OllamaError

router = APIRouter(prefix="/api")
ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/tif",
    "image/tiff",
    "image/x-tiff",
    "application/pdf",
}


def _normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    normalized = content_type.split(";", 1)[0].strip().lower()
    return normalized or None


def _sniff_content_type(payload: bytes) -> str | None:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if payload.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return "image/tiff"
    if payload.startswith(b"%PDF-"):
        return "application/pdf"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return None


def _resolve_effective_content_type(content_type: str | None, payload: bytes) -> str:
    normalized = _normalize_content_type(content_type)
    if normalized in ALLOWED_MIME_TYPES:
        return cast(str, normalized)
    if normalized in {None, "application/octet-stream"}:
        sniffed = _sniff_content_type(payload)
        if sniffed is not None:
            return sniffed
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "application/octet-stream konnte keinem unterstützten Bild- oder PDF-Typ "
                "zugeordnet werden."
            ),
        )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Nicht unterstützter Datei-Inhaltstyp: {normalized}",
    )


def _query_param(request: Request, name: str) -> str | None:
    value = request.query_params.get(name)
    if value is None or value == "":
        return None
    return value


def _resolve_text_param(form_value: object, query_value: str | None, default: str | None) -> str | None:
    if isinstance(form_value, str) and form_value != "":
        return form_value
    if query_value is not None:
        return query_value
    return default


def _resolve_int_param(form_value: object, query_value: str | None, field_name: str) -> int | None:
    if isinstance(form_value, int) and not isinstance(form_value, bool):
        return form_value
    if query_value is None:
        return None
    try:
        return int(query_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ungültiger Integer-Parameter: {field_name}",
        ) from exc


def _resolve_bool_param(
    form_value: object, query_value: str | None, field_name: str
) -> bool | None:
    if isinstance(form_value, bool):
        return form_value
    if query_value is None:
        return None
    normalized = query_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Ungültiger Boolean-Parameter: {field_name}",
    )


def get_ocr_backend_router(request: Request) -> OCRBackendRouter:
    return cast(OCRBackendRouter, request.app.state.ocr_backend_router)


def get_ollama_client(request: Request) -> OllamaClient:
    return cast(OllamaClient, request.app.state.ollama_client)


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ollama_base_url": settings.ollama_base_url,
        "default_model": settings.ollama_model,
        "default_backend": settings.ocr_backend,
        "default_token_limit": settings.default_token_limit,
    }


@router.get("/models")
async def models(client: OllamaClient = Depends(get_ollama_client)) -> dict:
    try:
        model_names = await client.list_models()
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return {"models": model_names}


@router.get("/schemas")
async def schemas() -> dict:
    return {"schemas": SCHEMA_REGISTRY}


@router.post("/ocr")
async def ocr(
    request: Request,
    file: UploadFile | None = File(None),
    mode: str | None = Form(None),
    schema_name: str | None = Form(None),
    model: str | None = Form(None),
    task: str | None = Form(None),
    custom_prompt: str | None = Form(None),
    token_limit: int | None = Form(None),
    gif_max_frames: int | None = Form(None),
    expert_enable_layout: bool | None = Form(None),
    backend: str | None = Form(None),
    pipeline: OCRBackendRouter = Depends(get_ocr_backend_router),
) -> dict:
    settings = get_settings()

    mode = cast(str, _resolve_text_param(mode, _query_param(request, "mode"), "plain"))
    schema_name = _resolve_text_param(schema_name, _query_param(request, "schema_name"), None)
    model = _resolve_text_param(model, _query_param(request, "model"), None)
    task = _resolve_text_param(task, _query_param(request, "task"), None)
    custom_prompt = _resolve_text_param(custom_prompt, _query_param(request, "custom_prompt"), None)
    token_limit = _resolve_int_param(token_limit, _query_param(request, "token_limit"), "token_limit")
    gif_max_frames = _resolve_int_param(
        gif_max_frames, _query_param(request, "gif_max_frames"), "gif_max_frames"
    )
    expert_enable_layout = _resolve_bool_param(
        expert_enable_layout,
        _query_param(request, "expert_enable_layout"),
        "expert_enable_layout",
    )
    backend = _resolve_text_param(backend, _query_param(request, "backend"), None)

    if file is not None:
        image_bytes = await file.read()
        content_type = _resolve_effective_content_type(file.content_type, image_bytes)
    else:
        image_bytes = await request.body()
        content_type = _resolve_effective_content_type(request.headers.get("content-type"), image_bytes)

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Die hochgeladene Datei ist leer.",
        )
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Datei überschreitet {settings.max_upload_bytes} Bytes.",
        )

    try:
        result, selected_backend = await pipeline.run(
            backend=backend,
            image_bytes=image_bytes,
            content_type=content_type,
            mode=mode,
            schema_name=schema_name,
            model=model,
            task=task,
            custom_prompt=custom_prompt,
            token_limit=token_limit,
            gif_max_frames=gif_max_frames,
            expert_enable_layout=expert_enable_layout,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger = cast(logging.Logger, request.app.state.logger)
        logger.exception("Unerwarteter OCR-Fehler")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unerwarteter OCR-Fehler: {exc}",
        ) from exc

    return {
        "text": result.text,
        "structured": result.structured,
        "layout": result.layout,
        "layout_visualizations": result.layout_visualizations,
        "model": result.model,
        "mode": result.mode,
        "schema_name": result.schema_name,
        "backend": selected_backend,
        "latency_ms": result.latency_ms,
        "warnings": result.warnings,
    }


# Backward-compatible alias paths for external clients.
router.add_api_route("/health/", health, methods=["GET"], include_in_schema=False)
router.add_api_route("/models/", models, methods=["GET"], include_in_schema=False)
router.add_api_route("/schemas/", schemas, methods=["GET"], include_in_schema=False)
router.add_api_route("/ocr/", ocr, methods=["POST"], include_in_schema=False)

router.add_api_route("/v1/health", health, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/models", models, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/schemas", schemas, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/ocr", ocr, methods=["POST"], include_in_schema=False)
router.add_api_route("/v1/health/", health, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/models/", models, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/schemas/", schemas, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/ocr/", ocr, methods=["POST"], include_in_schema=False)
