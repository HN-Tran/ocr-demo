from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.services.inference.factory import create_vision_client, create_vision_registry
from app.services.inference.model_ref import parse_model_ref
from app.services.inference.registry import _client_for_provider
from app.services.inference.ollama import OllamaClient, OllamaError
from app.services.inference.openai_compatible import OpenAICompatibleClient, OpenAICompatibleError


def _base_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "app_name": "docread",
        "app_base_path": "",
        "analyze_store_dir": "/tmp/docread-tests",
        "inference_provider": "ollama",
        "inference_base_url": "http://ollama.test",
        "inference_model": "glm-ocr:latest",
        "inference_api_key": "",
        "inference_vision_models": (),
        "inference_extra_providers": {},
        "ollama_base_url": "http://ollama.test",
        "ollama_model": "glm-ocr:latest",
        "ocr_backend": "direct",
        "ocr_expert_enable_layout": True,
        "ocr_expert_layout_model": "layout",
        "ocr_expert_table_transformer": False,
        "ocr_expert_per_region_ocr": True,
        "ocr_expert_text_anchor": True,
        "ocr_expert_text_anchor_threshold": 60.0,
        "ocr_expert_compare_include_detector_only": False,
        "ocr_expert_layout_max_dim": 1800,
        "ocr_binarized_min_dim": 1800,
        "azure_preset_label": "",
        "azure_preset_endpoint": "",
        "azure_preset_layout_endpoint": "",
        "azure_preset_key": "",
        "mlflow_tracking_uri": "",
        "mlflow_experiment_name": "docread",
        "benchmark_max_files": 50,
        "benchmark_max_runners": 5,
        "benchmark_job_ttl_s": 3600.0,
        "examples": (),
        "ocr_word_detector": "none",
        "default_token_limit": 4096,
        "request_timeout_s": 30.0,
        "max_upload_bytes": 8 * 1024 * 1024,
        "max_image_dim": 2048,
        "verify_ssl": False,
        "deskew_enabled": False,
        "deskew_min_angle_deg": 0.5,
        "host": "127.0.0.1",
        "port": 8000,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _mock_http_client(*, get_json: object | None = None, post_json: object | None = None) -> MagicMock:
    instance = MagicMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=None)
    if get_json is not None:
        get_response = MagicMock()
        get_response.raise_for_status = MagicMock()
        get_response.json.return_value = get_json
        instance.get = AsyncMock(return_value=get_response)
    if post_json is not None:
        post_response = MagicMock()
        post_response.raise_for_status = MagicMock()
        post_response.json.return_value = post_json
        instance.post = AsyncMock(return_value=post_response)
    return instance


def test_ollama_run_vision_chat_parses_message() -> None:
    mock_client = _mock_http_client(post_json={"message": {"content": "hello ocr"}})
    with patch(
        "app.services.inference.ollama.httpx.AsyncClient",
        return_value=mock_client,
    ):
        client = OllamaClient(base_url="http://ollama.test", timeout_s=5.0)
        text = asyncio.run(
            client.run_vision_chat(
                image_bytes=b"\x89PNG",
                prompt="read",
                model="glm-ocr:latest",
                max_tokens=1024,
            )
        )
    assert text == "hello ocr"


def test_ollama_supports_vision_from_capabilities() -> None:
    mock_client = _mock_http_client(
        post_json={"capabilities": ["completion", "vision"]},
    )
    with patch(
        "app.services.inference.ollama.httpx.AsyncClient",
        return_value=mock_client,
    ):
        client = OllamaClient(base_url="http://ollama.test", timeout_s=5.0)
        assert asyncio.run(client.supports_vision("glm-ocr:latest")) is True


def test_openai_compatible_run_vision_chat() -> None:
    mock_client = _mock_http_client(
        post_json={"choices": [{"message": {"content": "openai ocr"}}]},
    )
    with patch(
        "app.services.inference.openai_compatible.httpx.AsyncClient",
        return_value=mock_client,
    ):
        client = OpenAICompatibleClient(
            base_url="http://vllm.test/v1",
            timeout_s=5.0,
            api_key="secret",
            vision_models=("vlm",),
        )
        text = asyncio.run(
            client.run_vision_chat(
                image_bytes=b"\x89PNG",
                prompt="read",
                model="vlm",
            )
        )
    assert text == "openai ocr"
    headers = mock_client.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret"


def test_openai_compatible_vision_allowlist() -> None:
    client = OpenAICompatibleClient(
        base_url="http://vllm.test/v1",
        timeout_s=5.0,
        vision_models=("vlm-a",),
    )
    assert asyncio.run(client.supports_vision("vlm-a")) is True
    assert asyncio.run(client.supports_vision("other")) is False


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unbekannter Inference-Provider"):
        _client_for_provider(
            "unknown",
            base_url="http://x",
            timeout_s=5.0,
            api_key="",
            vision_models=(),
        )


def test_parse_model_ref_qualified() -> None:
    provider, model = parse_model_ref(
        "openai_compatible/vlm",
        inference_provider=None,
        default_provider="ollama",
        known_providers={"ollama", "openai_compatible"},
    )
    assert provider == "openai_compatible"
    assert model == "vlm"


def test_registry_lists_extra_providers() -> None:
    from app.config import InferenceProviderConfig

    settings = _base_settings(
        inference_extra_providers={
            "openai_compatible": InferenceProviderConfig(
                base_url="http://vllm.test/v1",
                api_key="",
                vision_models=("vlm",),
            )
        }
    )
    registry = create_vision_registry(settings)
    assert registry.provider_ids == ("ollama", "openai_compatible")


def test_factory_selects_openai_compatible() -> None:
    settings = _base_settings(
        inference_provider="openai_compatible",
        inference_base_url="http://vllm.test/v1",
        inference_api_key="k",
        inference_vision_models=("m1",),
    )
    client = create_vision_client(settings)
    assert client.provider_id == "openai_compatible"
    assert isinstance(client, OpenAICompatibleClient)


def test_openai_compatible_empty_content_raises() -> None:
    mock_client = _mock_http_client(
        post_json={"choices": [{"message": {"content": "   "}}]},
    )
    with patch(
        "app.services.inference.openai_compatible.httpx.AsyncClient",
        return_value=mock_client,
    ):
        client = OpenAICompatibleClient(base_url="http://vllm.test/v1", timeout_s=5.0)
        with pytest.raises(OpenAICompatibleError):
            asyncio.run(
                client.run_vision_chat(
                    image_bytes=b"x",
                    prompt="p",
                    model="m",
                )
            )


def test_ollama_empty_content_raises() -> None:
    mock_client = _mock_http_client(post_json={"message": {"content": ""}})
    with patch(
        "app.services.inference.ollama.httpx.AsyncClient",
        return_value=mock_client,
    ):
        client = OllamaClient(base_url="http://ollama.test", timeout_s=5.0)
        with pytest.raises(OllamaError):
            asyncio.run(
                client.run_vision_chat(
                    image_bytes=b"x",
                    prompt="p",
                    model="m",
                )
            )
