from __future__ import annotations

from typing import TYPE_CHECKING

from app.services.inference.ollama import OllamaClient
from app.services.inference.openai_compatible import OpenAICompatibleClient
from app.services.inference.protocol import VisionLlmClient
from app.services.inference.registry import VisionClientRegistry, create_vision_registry

if TYPE_CHECKING:
    from app.config import Settings

SUPPORTED_INFERENCE_PROVIDERS = ("ollama", "openai_compatible")


def create_vision_client(settings: Settings) -> VisionLlmClient:
    """Return the default provider client (backward compatible)."""
    return create_vision_registry(settings).get(settings.inference_provider)


__all__ = [
    "SUPPORTED_INFERENCE_PROVIDERS",
    "OllamaClient",
    "OpenAICompatibleClient",
    "VisionLlmClient",
    "VisionClientRegistry",
    "create_vision_client",
    "create_vision_registry",
]
