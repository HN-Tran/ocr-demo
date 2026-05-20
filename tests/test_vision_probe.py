from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.inference.openai_compatible import OpenAICompatibleClient
from app.services.inference.vision_probe import (
    guess_vision_from_name,
    response_indicates_no_vision,
)


def test_guess_vision_from_name_positive() -> None:
    assert guess_vision_from_name("qwen2-vl-7b") is True
    assert guess_vision_from_name("glm-ocr:latest") is True


def test_guess_vision_from_name_negative() -> None:
    assert guess_vision_from_name("text-embedding-3-small") is False


def test_guess_vision_from_name_unknown() -> None:
    assert guess_vision_from_name("some-random-llm") is None


def test_response_indicates_no_vision() -> None:
    assert response_indicates_no_vision(400, "model does not support image_url input")
    assert not response_indicates_no_vision(500, "internal error")


def test_openai_supports_vision_uses_name_heuristic_without_http() -> None:
    client = OpenAICompatibleClient(
        base_url="http://vllm.test/v1",
        timeout_s=5.0,
        vision_probe=False,
    )
    assert asyncio.run(client.supports_vision("glm-ocr:latest")) is True
    assert asyncio.run(client.supports_vision("text-embedding-3-small")) is False


def test_openai_supports_vision_probes_when_name_unknown() -> None:
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    post_response = MagicMock()
    post_response.is_success = True
    post_response.json.return_value = {
        "choices": [{"message": {"content": "x"}}],
    }
    mock_client.post = AsyncMock(return_value=post_response)
    client = OpenAICompatibleClient(
        base_url="http://vllm.test/v1",
        timeout_s=5.0,
        vision_probe=True,
    )
    with patch(
        "app.services.inference.openai_compatible.httpx.AsyncClient",
        return_value=mock_client,
    ):
        assert asyncio.run(client.supports_vision("custom-server-model")) is True
    assert mock_client.post.called
