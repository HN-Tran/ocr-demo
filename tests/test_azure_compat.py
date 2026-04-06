from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api.routes import (
    compat_analyze,
    compat_authentication_renew,
    compat_get_analyze_result,
    compat_service_ready,
    compat_sync_analyze,
    compat_usage_logs,
)
from app.services.analyze_operation_store import AnalyzeOperationStore
from app.services.backend_router import OCRBackendRouter


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfe\xa7\xd6\xe9Q\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class FakeCompatRequest:
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

    def url_for(self, name: str, /, **path_params: str) -> str:
        if name != "compat_get_analyze_result":
            raise AssertionError(f"Unerwarteter Routenname: {name}")
        return (
            "http://testserver/formrecognizer/documentModels/"
            f"{path_params['modelId']}/analyzeResults/{path_params['rId']}"
        )


def _request(
    *,
    body: bytes = b"",
    content_type: str | None = None,
    query_params: dict[str, str] | None = None,
) -> Request:
    return cast(
        Request,
        FakeCompatRequest(body=body, content_type=content_type, query_params=query_params),
    )


class FakeBackendRouter:
    def __init__(self) -> None:
        self.last_call: dict[str, Any] = {}

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
        expert_layout_model: str | None = None,
        expert_layout_threshold: float | None = None,
        expert_table_transformer: bool | None = None,
        expert_word_detector: str | None = None,
    ) -> Any:
        self.last_call = {
            "backend": backend,
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
            "expert_layout_model": expert_layout_model,
            "expert_table_transformer": expert_table_transformer,
            "expert_word_detector": expert_word_detector,
        }
        result = type(
            "OCRResult",
            (),
            {
                "text": "page one\n\npage two",
                "layout": [
                    {
                        "page_number": 1,
                        "regions": [
                            {
                                "index": 0,
                                "label": "text_block",
                                "content": "page one",
                                "bbox_2d": [10.0, 10.0, 50.0, 50.0],
                                "polygon": [10.0, 10.0, 48.0, 12.0, 50.0, 50.0, 12.0, 48.0],
                                "confidence": 0.93,
                            }
                        ],
                    },
                    {
                        "page_number": 2,
                        "regions": [
                            {
                                "index": 0,
                                "label": "text_block",
                                "content": "page two",
                                "bbox_2d": [10.0, 10.0, 50.0, 50.0],
                                "polygon": [10.0, 10.0, 48.0, 12.0, 50.0, 50.0, 12.0, 48.0],
                                "confidence": 0.87,
                            }
                        ],
                    },
                ],
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
                    },
                    {
                        "page_number": 2,
                        "angle": 0.0,
                        "width": 1000,
                        "height": 1200,
                        "unit": "pixel",
                        "words": [],
                        "lines": [],
                        "spans": [],
                        "kind": "document",
                    },
                ],
                "page_texts": ["page one", "page two"],
            },
        )()
        return result, "direct"


class FakeBackendRouterWithoutLayout(FakeBackendRouter):
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
        expert_layout_model: str | None = None,
        expert_layout_threshold: float | None = None,
        expert_table_transformer: bool | None = None,
        expert_word_detector: str | None = None,
    ) -> Any:
        result, selected_backend = await super().run(
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
            expert_layout_model=expert_layout_model,
            expert_table_transformer=expert_table_transformer,
            expert_word_detector=expert_word_detector,
        )
        result.layout = None
        return result, selected_backend


def _pipeline() -> OCRBackendRouter:
    return cast(OCRBackendRouter, FakeBackendRouter())


def test_compat_service_ready_payload() -> None:
    response = asyncio.run(compat_service_ready())
    payload = json.loads(response.body.decode("utf-8"))
    assert payload["status"] == "ok"
    assert payload["service"] == "prebuilt-read"
    assert payload["apiStatus"] == "Healthy"
    assert response.headers["apim-request-id"]


def test_compat_usage_and_auth_stubs() -> None:
    usage_response = asyncio.run(compat_usage_logs(month="03", year="2026"))
    renew_response = asyncio.run(compat_authentication_renew(token="abc"))
    usage_payload = json.loads(usage_response.body.decode("utf-8"))
    renew_payload = json.loads(renew_response.body.decode("utf-8"))
    assert usage_payload["meters"] == []
    assert usage_payload["month"] == "03"
    assert renew_payload == {"status": "ok", "token": "abc"}
    assert usage_response.headers["apim-request-id"]
    assert renew_response.headers["apim-request-id"]


def test_sync_analyze_returns_azure_shape_and_filters_pages() -> None:
    fake_pipeline = FakeBackendRouter()
    response = asyncio.run(
        compat_sync_analyze(
            request=_request(body=_png_bytes(), content_type="application/octet-stream"),
            modelId="prebuilt-read",
            api_version="2022-08-31",
            pages="2",
            locale=None,
            string_index_type="unicodeCodePoint",
            backend="expert",
            expert_enable_layout=True,
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    payload = json.loads(response.body.decode("utf-8"))

    assert fake_pipeline.last_call["backend"] == "expert"
    assert fake_pipeline.last_call["expert_enable_layout"] is True
    assert fake_pipeline.last_call["mode"] == "plain"
    assert fake_pipeline.last_call["task"] == "ocr_text"
    assert response.headers["apim-request-id"]
    assert payload["status"] == "succeeded"
    assert payload["analyzeResult"]["apiVersion"] == "2022-08-31"
    assert payload["analyzeResult"]["modelId"] == "prebuilt-read"
    assert payload["analyzeResult"]["stringIndexType"] == "unicodeCodePoint"
    assert payload["analyzeResult"]["content"] == "page two"
    assert payload["analyzeResult"]["pages"] == [
        {
            "pageNumber": 2,
            "angle": 0.0,
            "width": 1000,
            "height": 1200,
            "unit": "pixel",
            "words": [
                {
                    "content": "page",
                    "span": {"offset": 0, "length": 4},
                    "confidence": 0.87,
                    "polygon": [10.0, 10.0, 29.0, 11.0, 31.0, 49.0, 12.0, 48.0],
                },
                {
                    "content": "two",
                    "span": {"offset": 5, "length": 3},
                    "confidence": 0.87,
                    "polygon": [33.75, 11.25, 48.0, 12.0, 50.0, 50.0, 35.75, 49.25],
                },
            ],
            "lines": [
                {
                    "content": "page two",
                    "spans": [{"offset": 0, "length": 8}],
                    "confidence": 0.87,
                    "polygon": [10.0, 10.0, 48.0, 12.0, 50.0, 50.0, 12.0, 48.0],
                }
            ],
            "spans": [{"offset": 0, "length": 8}],
            "kind": "document",
            "content": "page two",
        }
    ]
    assert payload["analyzeResult"]["paragraphs"] == [
        {
            "content": "page two",
            "spans": [{"offset": 0, "length": 8}],
            "boundingRegions": [
                {
                    "pageNumber": 2,
                    "polygon": [10.0, 10.0, 48.0, 12.0, 50.0, 50.0, 12.0, 48.0],
                }
            ],
        }
    ]


def test_sync_analyze_without_layout_keeps_word_shape_stable() -> None:
    fake_pipeline = FakeBackendRouterWithoutLayout()
    response = asyncio.run(
        compat_sync_analyze(
            request=_request(body=_png_bytes(), content_type="application/octet-stream"),
            modelId="prebuilt-read",
            api_version="2022-08-31",
            pages="1",
            locale=None,
            string_index_type="textElements",
            pipeline=cast(OCRBackendRouter, fake_pipeline),
        )
    )
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["analyzeResult"]["pages"] == [
        {
            "pageNumber": 1,
            "angle": 0.0,
            "width": 1000,
            "height": 1200,
            "unit": "pixel",
            "words": [
                {
                    "content": "page",
                    "span": {"offset": 0, "length": 4},
                    "confidence": 0.0,
                    "polygon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                {
                    "content": "one",
                    "span": {"offset": 5, "length": 3},
                    "confidence": 0.0,
                    "polygon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
            ],
            "lines": [
                {
                    "content": "page one",
                    "spans": [{"offset": 0, "length": 8}],
                    "confidence": 0.0,
                    "polygon": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ],
            "spans": [{"offset": 0, "length": 8}],
            "kind": "document",
            "content": "page one",
        }
    ]
    assert payload["analyzeResult"]["paragraphs"] == [
        {
            "content": "page one",
            "spans": [{"offset": 0, "length": 8}],
            "boundingRegions": [],
        }
    ]


@pytest.mark.anyio
async def test_async_analyze_returns_operation_location_and_result_can_be_polled() -> None:
    fake_pipeline = FakeBackendRouter()
    store = AnalyzeOperationStore()
    analyze_response = await compat_analyze(
        request=_request(body=_png_bytes(), content_type="application/octet-stream"),
        modelId="prebuilt-read",
        api_version="2022-08-31",
        pages=None,
        locale=None,
        string_index_type=None,
        backend="expert",
        expert_enable_layout=True,
        pipeline=cast(OCRBackendRouter, fake_pipeline),
        store=store,
    )

    assert analyze_response.status_code == 202
    operation_location = analyze_response.headers["Operation-Location"]
    assert operation_location.endswith("?api-version=2022-08-31")
    assert analyze_response.headers["apim-request-id"]
    assert analyze_response.headers["Retry-After"] == "1"

    operation_id = operation_location.rsplit("/", 1)[-1].split("?", 1)[0]
    initial_poll = await compat_get_analyze_result(
        modelId="prebuilt-read",
        rId=operation_id,
        api_version="2022-08-31",
        store=store,
    )
    initial_payload = json.loads(initial_poll.body.decode("utf-8"))
    assert initial_payload["status"] in {"notStarted", "running", "succeeded"}
    assert initial_poll.headers["apim-request-id"] == analyze_response.headers["apim-request-id"]

    await asyncio.sleep(0)
    assert fake_pipeline.last_call["backend"] == "expert"
    assert fake_pipeline.last_call["expert_enable_layout"] is True
    poll_response = await compat_get_analyze_result(
        modelId="prebuilt-read",
        rId=operation_id,
        api_version="2022-08-31",
        store=store,
    )
    payload = json.loads(poll_response.body.decode("utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["analyzeResult"]["modelId"] == "prebuilt-read"
    assert payload["analyzeResult"]["content"] == "page one\n\npage two"
    assert payload["analyzeResult"]["pages"][0]["spans"] == [{"offset": 0, "length": 8}]
    assert payload["analyzeResult"]["pages"][1]["spans"] == [{"offset": 10, "length": 8}]
    assert poll_response.headers["apim-request-id"] == analyze_response.headers["apim-request-id"]


def test_sync_analyze_rejects_unknown_model() -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            compat_sync_analyze(
                request=_request(body=_png_bytes(), content_type="application/octet-stream"),
                modelId="unknown-model",
                api_version="2022-08-31",
                pages=None,
                locale=None,
                string_index_type=None,
                pipeline=_pipeline(),
            )
        )

    assert exc_info.value.status_code == 404
