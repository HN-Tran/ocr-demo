from __future__ import annotations

import asyncio
from typing import cast

import pytest

from app.services.ocr_pipeline import OCRPipeline
from app.services.ollama_client import OllamaClient


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfe\xa7\xd6\xe9Q\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class FakeOllamaClient:
    def __init__(self) -> None:
        self.last_prompt = ""
        self.last_model = ""

    async def run_ocr(self, *, image_bytes: bytes, prompt: str, model: str) -> str:
        self.last_prompt = prompt
        self.last_model = model
        return "ok"


def _pipeline(fake_client: FakeOllamaClient) -> OCRPipeline:
    return OCRPipeline(
        ollama_client=cast(OllamaClient, fake_client),
        default_model="glm-ocr:latest",
        max_image_dim=2048,
    )


def test_plain_describe_image_task_uses_describe_prompt() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="describe_image",
            custom_prompt=None,
        )
    )
    assert "Describe this image" in fake_client.last_prompt


def test_plain_custom_prompt_overrides_task_prompt() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="describe_image",
            custom_prompt="Describe this image in one sentence.",
        )
    )
    assert fake_client.last_prompt == "Describe this image in one sentence."


def test_plain_unknown_task_raises() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            pipeline.run(
                image_bytes=_png_bytes(),
                mode="plain",
                schema_name=None,
                task="unknown_task",
                custom_prompt=None,
            )
        )
    assert "Unknown task" in str(exc_info.value)


def test_structured_rejects_custom_prompt() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            pipeline.run(
                image_bytes=_png_bytes(),
                mode="structured",
                schema_name="invoice_basic",
                task=None,
                custom_prompt="Return JSON",
            )
        )
    assert "custom_prompt is only supported for plain mode" in str(exc_info.value)
