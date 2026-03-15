from __future__ import annotations

import asyncio
from typing import cast

import pytest

from app.services.backend_router import OCRBackendRouter
from app.services.ocr_pipeline import OCRResult
from app.services.ocr_service import OCRService


class _FakeService:
    def __init__(self, *, text: str) -> None:
        self.text = text
        self.calls = 0

    async def run(
        self,
        *,
        image_bytes: bytes,
        content_type: str | None = None,
        mode: str,
        schema_name: str | None,
        model: str | None = None,
        task: str | None = None,
        custom_prompt: str | None = None,
        token_limit: int | None = None,
        gif_max_frames: int | None = None,
        expert_enable_layout: bool | None = None,
        expert_layout_model: str | None = None,
    ) -> OCRResult:
        self.calls += 1
        return OCRResult(
            text=self.text,
            structured=None,
            model=model or "m",
            mode=mode,
            schema_name=schema_name,
            latency_ms=1,
            warnings=[],
        )


def test_router_uses_default_backend() -> None:
    direct = _FakeService(text="direct")
    expert = _FakeService(text="expert")
    router = OCRBackendRouter(
        default_backend="direct",
        backends={
            "direct": cast(OCRService, direct),
            "expert": cast(OCRService, expert),
        },
    )

    result, selected = asyncio.run(
        router.run(
            backend=None,
            image_bytes=b"img",
            content_type="image/png",
            mode="plain",
            schema_name=None,
        )
    )
    assert selected == "direct"
    assert result.text == "direct"
    assert direct.calls == 1
    assert expert.calls == 0


def test_router_uses_requested_backend() -> None:
    direct = _FakeService(text="direct")
    expert = _FakeService(text="expert")
    router = OCRBackendRouter(
        default_backend="direct",
        backends={
            "direct": cast(OCRService, direct),
            "expert": cast(OCRService, expert),
        },
    )

    result, selected = asyncio.run(
        router.run(
            backend="expert",
            image_bytes=b"img",
            content_type="image/png",
            mode="plain",
            schema_name=None,
        )
    )
    assert selected == "expert"
    assert result.text == "expert"
    assert expert.calls == 1
    assert direct.calls == 0


def test_router_rejects_unknown_backend() -> None:
    direct = _FakeService(text="direct")
    router = OCRBackendRouter(
        default_backend="direct",
        backends={"direct": cast(OCRService, direct)},
    )
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            router.run(
                backend="unknown",
                image_bytes=b"img",
                content_type="image/png",
                mode="plain",
                schema_name=None,
            )
        )
    assert "Unbekanntes Backend" in str(exc_info.value)
