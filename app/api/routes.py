from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.config import get_settings
from app.schemas import SCHEMA_REGISTRY
from app.services.ocr_pipeline import OCRPipeline
from app.services.ollama_client import OllamaClient, OllamaError

router = APIRouter(prefix="/api")
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}


def get_ocr_pipeline(request: Request) -> OCRPipeline:
    return cast(OCRPipeline, request.app.state.ocr_pipeline)


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
    file: UploadFile = File(...),
    mode: str = Form("plain"),
    schema_name: str | None = Form(None),
    model: str | None = Form(None),
    task: str | None = Form(None),
    custom_prompt: str | None = Form(None),
    pipeline: OCRPipeline = Depends(get_ocr_pipeline),
) -> dict:
    settings = get_settings()
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file content type: {file.content_type}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_bytes} bytes.",
        )

    try:
        result = await pipeline.run(
            image_bytes=image_bytes,
            mode=mode,
            schema_name=schema_name,
            model=model,
            task=task,
            custom_prompt=custom_prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger = cast(logging.Logger, request.app.state.logger)
        logger.exception("Unexpected OCR failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected OCR failure: {exc}",
        ) from exc

    return {
        "text": result.text,
        "structured": result.structured,
        "model": result.model,
        "mode": result.mode,
        "schema_name": result.schema_name,
        "latency_ms": result.latency_ms,
        "warnings": result.warnings,
    }
