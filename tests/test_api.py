from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request

from app.api.routes import health, ocr
from app.services.ocr_pipeline import OCRPipeline


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfe\xa7\xd6\xe9Q\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class FakeUploadFile:
    def __init__(self, *, content: bytes, content_type: str) -> None:
        self._content = content
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._content


def _upload_file(*, content: bytes, content_type: str) -> UploadFile:
    return cast(UploadFile, FakeUploadFile(content=content, content_type=content_type))


def _request() -> Request:
    logger = logging.getLogger("test")
    return cast(
        Request,
        SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(logger=logger))),
    )


class FakePipeline:
    def __init__(self) -> None:
        self.last_call: dict[str, Any] = {}

    async def run(
        self,
        *,
        image_bytes: bytes,
        content_type: str | None,
        mode: str,
        schema_name: str | None,
        model: str | None,
        task: str | None,
        custom_prompt: str | None,
        token_limit: int | None,
    ) -> Any:
        self.last_call = {
            "image_bytes": image_bytes,
            "content_type": content_type,
            "mode": mode,
            "schema_name": schema_name,
            "model": model,
            "task": task,
            "custom_prompt": custom_prompt,
            "token_limit": token_limit,
        }
        if mode == "structured" and not schema_name:
            raise ValueError("schema_name ist für den strukturierten Modus erforderlich")
        return type(
            "OCRResult",
            (),
            {
                "text": "hello world",
                "structured": {"vendor": "ACME"} if mode == "structured" else None,
                "model": model or "fake-model",
                "mode": mode,
                "schema_name": schema_name,
                "latency_ms": 12,
                "warnings": [],
            },
        )()


def _pipeline() -> OCRPipeline:
    return cast(OCRPipeline, FakePipeline())


def test_health() -> None:
    payload = asyncio.run(health())
    assert payload["status"] == "ok"


def test_ocr_rejects_bad_file_type() -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            ocr(
                request=_request(),
                file=_upload_file(content=b"hello", content_type="text/plain"),
                mode="plain",
                schema_name=None,
                model=None,
                task=None,
                custom_prompt=None,
                token_limit=None,
                pipeline=_pipeline(),
            )
        )
    assert exc_info.value.status_code == 400
    assert "Nicht unterstützter Datei-Inhaltstyp" in str(exc_info.value.detail)


def test_ocr_plain() -> None:
    response = asyncio.run(
        ocr(
            request=_request(),
            file=_upload_file(content=_png_bytes(), content_type="image/png"),
            mode="plain",
            schema_name=None,
            model=None,
            task=None,
            custom_prompt=None,
            token_limit=None,
            pipeline=_pipeline(),
        )
    )
    assert response["text"] == "hello world"
    assert response["structured"] is None


def test_ocr_structured_requires_schema() -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            ocr(
                request=_request(),
                file=_upload_file(content=_png_bytes(), content_type="image/png"),
                mode="structured",
                schema_name=None,
                model=None,
                task=None,
                custom_prompt=None,
                token_limit=None,
                pipeline=_pipeline(),
            )
        )
    assert exc_info.value.status_code == 400
    assert "schema_name ist für den strukturierten Modus erforderlich" in str(exc_info.value.detail)


def test_ocr_structured_with_schema() -> None:
    response = asyncio.run(
        ocr(
            request=_request(),
            file=_upload_file(content=_png_bytes(), content_type="image/png"),
            mode="structured",
            schema_name="invoice_basic",
            model=None,
            task=None,
            custom_prompt=None,
            token_limit=None,
            pipeline=_pipeline(),
        )
    )
    assert response["structured"] == {"vendor": "ACME"}


def test_ocr_plain_forwards_task_and_custom_prompt() -> None:
    fake_pipeline = FakePipeline()
    response = asyncio.run(
        ocr(
            request=_request(),
            file=_upload_file(content=_png_bytes(), content_type="image/png"),
            mode="plain",
            schema_name=None,
            model="llava:latest",
            task="describe_image",
            custom_prompt="Beschreibe dieses Bild.",
            token_limit=8192,
            pipeline=cast(OCRPipeline, fake_pipeline),
        )
    )
    assert response["model"] == "llava:latest"
    assert fake_pipeline.last_call["task"] == "describe_image"
    assert fake_pipeline.last_call["custom_prompt"] == "Beschreibe dieses Bild."
    assert fake_pipeline.last_call["token_limit"] == 8192


def test_ocr_accepts_pdf_content_type() -> None:
    fake_pipeline = FakePipeline()
    response = asyncio.run(
        ocr(
            request=_request(),
            file=_upload_file(content=b"%PDF-1.4\n%%EOF", content_type="application/pdf"),
            mode="plain",
            schema_name=None,
            model=None,
            task=None,
            custom_prompt=None,
            token_limit=None,
            pipeline=cast(OCRPipeline, fake_pipeline),
        )
    )
    assert response["text"] == "hello world"
    assert fake_pipeline.last_call["content_type"] == "application/pdf"
