from __future__ import annotations

import pytest

from app.services.inference.model_ref import format_model_ref, parse_model_ref


def test_parse_model_ref_defaults_to_primary_provider() -> None:
    provider, model = parse_model_ref(
        "glm-ocr:latest",
        inference_provider=None,
        default_provider="ollama",
        known_providers={"ollama", "openai_compatible"},
    )
    assert provider == "ollama"
    assert model == "glm-ocr:latest"


def test_parse_model_ref_explicit_provider_param() -> None:
    provider, model = parse_model_ref(
        "vlm",
        inference_provider="openai_compatible",
        default_provider="ollama",
        known_providers={"ollama", "openai_compatible"},
    )
    assert provider == "openai_compatible"
    assert model == "vlm"


def test_format_model_ref() -> None:
    assert format_model_ref("ollama", "glm-ocr:latest") == "ollama/glm-ocr:latest"


def test_parse_model_ref_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unbekannter Inference-Provider"):
        parse_model_ref(
            None,
            inference_provider="unknown",
            default_provider="ollama",
            known_providers={"ollama"},
        )
