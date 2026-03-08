from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request

from app.api.routes import health, ocr, router, schemas
from app.services.backend_router import OCRBackendRouter


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


class FakeRequest:
    def __init__(
        self,
        *,
        body: bytes = b"",
        content_type: str | None = None,
        query_params: dict[str, str] | None = None,
    ) -> None:
        logger = logging.getLogger("test")
        self.app = SimpleNamespace(state=SimpleNamespace(logger=logger))
        self.headers = {} if content_type is None else {"content-type": content_type}
        self.query_params = query_params or {}
        self._body = body

    async def body(self) -> bytes:
        return self._body


def _request(
    *, body: bytes = b"", content_type: str | None = None, query_params: dict[str, str] | None = None
) -> Request:
    return cast(
        Request,
        FakeRequest(body=body, content_type=content_type, query_params=query_params),
    )


class FakeBackendRouter:
    def __init__(self) -> None:
        self.last_call: dict[str, Any] = {}
        self.default_backend = "direct"

    async def run(
        self,
        *,
        backend: str | None,
        image_bytes: bytes,
        content_type: str | None,
        mode: str,
        schema_name: str | None,
        model: str | None,
        task: str | None,
        custom_prompt: str | None,
        token_limit: int | None,
        gif_max_frames: int | None,
        expert_enable_layout: bool | None,
    ) -> Any:
        selected_backend = backend or self.default_backend
        self.last_call = {
            "backend": selected_backend,
            "image_bytes": image_bytes,
            "content_type": content_type,
            "mode": mode,
            "schema_name": schema_name,
            "model": model,
            "task": task,
            "custom_prompt": custom_prompt,
            "token_limit": token_limit,
            "gif_max_frames": gif_max_frames,
            "expert_enable_layout": expert_enable_layout,
        }
        if mode == "structured" and not schema_name:
            raise ValueError("schema_name ist für den strukturierten Modus erforderlich")
        result = type(
            "OCRResult",
            (),
            {
                "text": "hello world",
                "markdown": (
                    "# Dokument\n\nhello world" if selected_backend == "expert" and mode == "plain" else None
                ),
                "structured": {"vendor": "ACME"} if mode == "structured" else None,
                "page_infos": [
                    {
                        "page_number": 1,
                        "angle": 0.0,
                        "width": 1000,
                        "height": 1200,
                        "unit": "pixel",
                        "words": [],
                        "lines": [],
                        "spans": [],
                        "kind": "document",
                    }
                ],
                "page_texts": ["hello world"],
                "layout": (
                    [
                        {
                            "page_number": 1,
                            "regions": [
                                {
                                    "index": 0,
                                    "label": "text_block",
                                    "content": "hello world",
                                    "bbox_2d": [100.0, 120.0, 900.0, 260.0],
                                    "confidence": 0.96,
                                }
                            ],
                        }
                    ]
                    if selected_backend == "expert"
                    else None
                ),
                "layout_visualizations": (
                    ["data:image/png;base64,ZmFrZQ=="] if selected_backend == "expert" else None
                ),
                "model": model or "fake-model",
                "mode": mode,
                "schema_name": schema_name,
                "latency_ms": 12,
                "warnings": [],
            },
        )()
        return result, selected_backend


def _pipeline() -> OCRBackendRouter:
    return cast(OCRBackendRouter, FakeBackendRouter())


def test_health() -> None:
    payload = asyncio.run(health())
    assert payload["status"] == "ok"


def test_schemas_contains_new_presets() -> None:
    payload = asyncio.run(schemas())
    schema_map = payload["schemas"]
    assert "table_basic" in schema_map
    assert "business_card_basic" in schema_map


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
    assert response["status"] == "succeeded"
    assert response["createdDateTime"]
    assert response["lastUpdatedDateTime"]
    assert response["analyzeResult"]["apiVersion"] == "2026-03-09-preview"
    assert response["analyzeResult"]["modelId"] == "fake-model"
    assert response["analyzeResult"]["stringIndexType"] == "textElements"
    assert response["analyzeResult"]["content"] == "hello world"
    assert response["analyzeResult"]["pages"] == [
        {
            "pageNumber": 1,
            "angle": 0.0,
            "width": 1000,
            "height": 1200,
            "unit": "pixel",
            "words": [
                {
                    "content": "hello",
                    "span": {"offset": 0, "length": 5},
                },
                {
                    "content": "world",
                    "span": {"offset": 6, "length": 5},
                },
            ],
            "lines": [
                {
                    "content": "hello world",
                    "spans": [{"offset": 0, "length": 11}],
                }
            ],
            "spans": [{"offset": 0, "length": 11}],
            "kind": "document",
            "content": "hello world",
        }
    ]
    assert response["analyzeResult"]["paragraphs"] == [
        {"content": "hello world", "spans": [{"offset": 0, "length": 11}]}
    ]
    assert response["analyzeResult"]["styles"] == []
    assert response["analyzeResult"]["languages"] == []
    assert response["text"] == "hello world"
    assert response["markdown"] is None
    assert response["structured"] is None
    assert response["layout"] is None
    assert response["layout_visualizations"] is None


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
    assert response["analyzeResult"]["content"] == "hello world"


def test_ocr_plain_forwards_task_and_custom_prompt() -> None:
    fake_pipeline = FakeBackendRouter()
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
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    assert response["model"] == "llava:latest"
    assert fake_pipeline.last_call["task"] == "describe_image"
    assert fake_pipeline.last_call["custom_prompt"] == "Beschreibe dieses Bild."
    assert fake_pipeline.last_call["token_limit"] == 8192


def test_ocr_accepts_pdf_content_type() -> None:
    fake_pipeline = FakeBackendRouter()
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
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    assert response["text"] == "hello world"
    assert fake_pipeline.last_call["content_type"] == "application/pdf"


@pytest.mark.parametrize("content_type", ["image/tif", "image/tiff", "image/x-tiff"])
def test_ocr_accepts_tiff_content_type(content_type: str) -> None:
    fake_pipeline = FakeBackendRouter()
    response = asyncio.run(
        ocr(
            request=_request(),
            file=_upload_file(content=b"II*\x00", content_type=content_type),
            mode="plain",
            schema_name=None,
            model=None,
            task=None,
            custom_prompt=None,
            token_limit=None,
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    assert response["text"] == "hello world"
    assert fake_pipeline.last_call["content_type"] == content_type


def test_ocr_accepts_gif_content_type() -> None:
    fake_pipeline = FakeBackendRouter()
    response = asyncio.run(
        ocr(
            request=_request(),
            file=_upload_file(content=b"GIF89a", content_type="image/gif"),
            mode="plain",
            schema_name=None,
            model=None,
            task=None,
            custom_prompt=None,
            token_limit=None,
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    assert response["text"] == "hello world"
    assert fake_pipeline.last_call["content_type"] == "image/gif"


def test_ocr_accepts_octet_stream_body() -> None:
    fake_pipeline = FakeBackendRouter()
    response = asyncio.run(
        ocr(
            request=_request(body=_png_bytes(), content_type="application/octet-stream"),
            file=None,
            mode=None,
            schema_name=None,
            model=None,
            task=None,
            custom_prompt=None,
            token_limit=None,
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    assert response["text"] == "hello world"
    assert fake_pipeline.last_call["content_type"] == "image/png"
    assert fake_pipeline.last_call["mode"] == "plain"


def test_ocr_octet_stream_reads_query_parameters() -> None:
    fake_pipeline = FakeBackendRouter()
    asyncio.run(
        ocr(
            request=_request(
                body=b"%PDF-1.4\n%%EOF",
                content_type="application/octet-stream",
                query_params={
                    "mode": "plain",
                    "backend": "expert",
                    "task": "describe_image",
                    "token_limit": "8192",
                    "gif_max_frames": "6",
                    "expert_enable_layout": "false",
                },
            ),
            file=None,
            mode=None,
            schema_name=None,
            model=None,
            task=None,
            custom_prompt=None,
            token_limit=None,
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    assert fake_pipeline.last_call["content_type"] == "application/pdf"
    assert fake_pipeline.last_call["backend"] == "expert"
    assert fake_pipeline.last_call["task"] == "describe_image"
    assert fake_pipeline.last_call["token_limit"] == 8192
    assert fake_pipeline.last_call["gif_max_frames"] == 6
    assert fake_pipeline.last_call["expert_enable_layout"] is False


def test_ocr_rejects_unknown_octet_stream_body() -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            ocr(
                request=_request(body=b"not-an-image", content_type="application/octet-stream"),
                file=None,
                mode=None,
                schema_name=None,
                model=None,
                task=None,
                custom_prompt=None,
                token_limit=None,
                pipeline=_pipeline(),
            )
        )
    assert exc_info.value.status_code == 400
    assert "application/octet-stream konnte keinem unterstützten" in str(exc_info.value.detail)


def test_ocr_forwards_gif_max_frames() -> None:
    fake_pipeline = FakeBackendRouter()
    asyncio.run(
        ocr(
            request=_request(),
            file=_upload_file(content=b"GIF89a", content_type="image/gif"),
            mode="plain",
            schema_name=None,
            model=None,
            task="describe_image",
            custom_prompt=None,
            token_limit=None,
            gif_max_frames=5,
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    assert fake_pipeline.last_call["gif_max_frames"] == 5


def test_ocr_forwards_backend_choice() -> None:
    fake_pipeline = FakeBackendRouter()
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
            backend="expert",
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    assert fake_pipeline.last_call["backend"] == "expert"
    assert response["backend"] == "expert"
    assert response["layout"] == [
        {
            "page_number": 1,
            "regions": [
                {
                    "index": 0,
                    "label": "text_block",
                    "content": "hello world",
                    "bbox_2d": [100.0, 120.0, 900.0, 260.0],
                    "confidence": 0.96,
                }
            ],
        }
    ]
    assert response["layout_visualizations"] == ["data:image/png;base64,ZmFrZQ=="]
    assert response["markdown"] == "# Dokument\n\nhello world"
    assert response["analyzeResult"]["pages"] == [
        {
            "pageNumber": 1,
            "angle": 0.0,
            "width": 1000,
            "height": 1200,
            "unit": "pixel",
            "words": [
                {
                    "content": "hello",
                    "span": {"offset": 0, "length": 5},
                },
                {
                    "content": "world",
                    "span": {"offset": 6, "length": 5},
                },
            ],
            "lines": [
                {
                    "content": "hello world",
                    "spans": [{"offset": 0, "length": 11}],
                    "polygon": [100.0, 120.0, 900.0, 120.0, 900.0, 260.0, 100.0, 260.0],
                }
            ],
            "spans": [{"offset": 0, "length": 11}],
            "kind": "document",
            "content": "hello world",
        }
    ]
    assert response["analyzeResult"]["paragraphs"] == [
        {
            "content": "hello world",
            "spans": [{"offset": 0, "length": 11}],
            "boundingRegions": [
                {
                    "pageNumber": 1,
                    "polygon": [100.0, 120.0, 900.0, 120.0, 900.0, 260.0, 100.0, 260.0],
                }
            ],
        }
    ]


def test_ocr_forwards_expert_enable_layout() -> None:
    fake_pipeline = FakeBackendRouter()
    asyncio.run(
        ocr(
            request=_request(),
            file=_upload_file(content=_png_bytes(), content_type="image/png"),
            mode="plain",
            schema_name=None,
            model=None,
            task=None,
            custom_prompt=None,
            token_limit=None,
            backend="expert",
            expert_enable_layout=False,
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    assert fake_pipeline.last_call["expert_enable_layout"] is False


def test_api_v1_alias_routes_exist() -> None:
    route_paths = {route.path for route in router.routes}
    assert "/api/health/" in route_paths
    assert "/api/models/" in route_paths
    assert "/api/schemas/" in route_paths
    assert "/api/ocr/" in route_paths
    assert "/api/v1/health" in route_paths
    assert "/api/v1/models" in route_paths
    assert "/api/v1/schemas" in route_paths
    assert "/api/v1/ocr" in route_paths
    assert "/api/v1/health/" in route_paths
    assert "/api/v1/models/" in route_paths
    assert "/api/v1/schemas/" in route_paths
    assert "/api/v1/ocr/" in route_paths
