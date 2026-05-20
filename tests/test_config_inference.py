from __future__ import annotations

import pytest

from app.config import get_settings


def test_get_settings_defaults_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("INFERENCE_BASE_URL", raising=False)
    monkeypatch.delenv("INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    settings = get_settings()
    assert settings.inference_provider == "ollama"
    assert settings.inference_base_url == "http://localhost:11434"
    assert settings.inference_model == "glm-ocr:latest"


def test_get_settings_extra_providers_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "INFERENCE_EXTRA_PROVIDERS",
        '{"openai_compatible":{"base_url":"http://vllm.local/v1","vision_models":["vlm"]}}',
    )
    settings = get_settings()
    assert "openai_compatible" in settings.inference_extra_providers
    assert (
        settings.inference_extra_providers["openai_compatible"].base_url == "http://vllm.local/v1"
    )


def test_get_settings_openai_compatible_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai_compatible")
    monkeypatch.setenv("INFERENCE_BASE_URL", "http://vllm.local/v1")
    monkeypatch.setenv("INFERENCE_MODEL", "vlm")
    monkeypatch.setenv("INFERENCE_VISION_MODELS", "vlm,other")
    settings = get_settings()
    assert settings.inference_provider == "openai_compatible"
    assert settings.inference_base_url == "http://vllm.local/v1"
    assert settings.inference_vision_models == ("vlm", "other")
