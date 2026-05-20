from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionLlmClient(Protocol):
    """Multimodal vision+language backend for OCR prompts."""

    @property
    def provider_id(self) -> str: ...

    async def list_models(self) -> list[str]: ...

    async def supports_vision(self, model: str) -> bool: ...

    async def run_vision_chat(
        self,
        *,
        image_bytes: bytes,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
    ) -> str: ...
