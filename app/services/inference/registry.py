from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.services.inference.model_ref import parse_model_ref
from app.services.inference.protocol import VisionLlmClient

if TYPE_CHECKING:
    from app.config import InferenceProviderConfig, Settings


@dataclass(frozen=True)
class ResolvedInference:
    provider_id: str
    model_id: str
    client: VisionLlmClient


class VisionClientRegistry:
    def __init__(
        self,
        *,
        clients: dict[str, VisionLlmClient],
        default_provider: str,
        default_model: str,
    ) -> None:
        if not clients:
            raise ValueError("Mindestens ein Inference-Provider muss konfiguriert sein.")
        if default_provider not in clients:
            raise ValueError(
                f"Standard-Provider '{default_provider}' ist nicht in der Registry registriert."
            )
        self._clients = dict(clients)
        self.default_provider = default_provider
        self.default_model = default_model

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._clients.keys()))

    def get(self, provider_id: str) -> VisionLlmClient:
        key = provider_id.strip().lower()
        client = self._clients.get(key)
        if client is None:
            raise ValueError(
                f"Inference-Provider '{provider_id}' ist nicht konfiguriert. "
                f"Verfügbar: {', '.join(self.provider_ids)}"
            )
        return client

    def resolve(
        self,
        *,
        inference_provider: str | None = None,
        model: str | None = None,
    ) -> ResolvedInference:
        provider_id, model_id = parse_model_ref(
            model,
            inference_provider=inference_provider,
            default_provider=self.default_provider,
            known_providers=set(self._clients.keys()),
        )
        selected_model = (model_id or "").strip() or self.default_model
        return ResolvedInference(
            provider_id=provider_id,
            model_id=selected_model,
            client=self.get(provider_id),
        )

    def describe(self) -> list[dict[str, str]]:
        return [
            {
                "id": provider_id,
                "default": "true" if provider_id == self.default_provider else "false",
            }
            for provider_id in self.provider_ids
        ]


def _client_for_provider(
    provider_id: str,
    *,
    base_url: str,
    timeout_s: float,
    api_key: str,
    vision_models: tuple[str, ...],
) -> VisionLlmClient:
    if provider_id == "ollama":
        from app.services.inference.ollama import OllamaClient

        return OllamaClient(base_url=base_url, timeout_s=timeout_s)
    if provider_id == "openai_compatible":
        from app.services.inference.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(
            base_url=base_url,
            timeout_s=timeout_s,
            api_key=api_key,
            vision_models=vision_models,
        )
    raise ValueError(f"Unbekannter Inference-Provider '{provider_id}'.")


def create_vision_registry(settings: Settings) -> VisionClientRegistry:
    clients: dict[str, VisionLlmClient] = {
        settings.inference_provider: _client_for_provider(
            settings.inference_provider,
            base_url=settings.inference_base_url,
            timeout_s=settings.request_timeout_s,
            api_key=settings.inference_api_key,
            vision_models=settings.inference_vision_models,
        )
    }
    for provider_id, cfg in settings.inference_extra_providers.items():
        if provider_id in clients:
            continue
        clients[provider_id] = _client_for_provider(
            provider_id,
            base_url=cfg.base_url,
            timeout_s=settings.request_timeout_s,
            api_key=cfg.api_key,
            vision_models=cfg.vision_models,
        )
    return VisionClientRegistry(
        clients=clients,
        default_provider=settings.inference_provider,
        default_model=settings.inference_model,
    )
