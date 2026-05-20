from __future__ import annotations

import base64
from collections.abc import Sequence

import httpx

from app.services.inference.errors import InferenceError
from app.services.inference.vision_probe import (
    get_cached_vision_probe,
    guess_vision_from_name,
    probe_request_body,
    response_indicates_no_vision,
    set_cached_vision_probe,
)


class OpenAICompatibleError(InferenceError):
    """Raised when an OpenAI-compatible inference request fails."""


class OpenAICompatibleClient:
    """Vision chat via OpenAI-compatible HTTP APIs (vLLM, llama.cpp server, etc.)."""

    provider_id = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float,
        api_key: str = "",
        vision_models: Sequence[str] = (),
        vision_probe: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.api_key = api_key.strip()
        self._vision_models = tuple(m.strip() for m in vision_models if m.strip())
        self._vision_probe = vision_probe

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def list_models(self) -> list[str]:
        url = f"{self.base_url}/models"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            try:
                response = await client.get(url, headers=self._headers())
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise OpenAICompatibleError(
                    f"Modellliste konnte nicht geladen werden: {exc}"
                ) from exc
        payload = response.json()
        data = payload.get("data", [])
        if not isinstance(data, list):
            return []
        names: list[str] = []
        for entry in data:
            if isinstance(entry, dict):
                model_id = entry.get("id")
                if isinstance(model_id, str) and model_id:
                    names.append(model_id)
        return names

    async def supports_vision(self, model: str) -> bool:
        if self._vision_models:
            return model in self._vision_models

        cached = get_cached_vision_probe(self.base_url, model)
        if cached is not None:
            return cached

        guessed = guess_vision_from_name(model)
        if guessed is not None:
            set_cached_vision_probe(base_url=self.base_url, model=model, supports=guessed)
            return guessed

        if not self._vision_probe:
            return True

        supports = await self._probe_vision_support(model)
        set_cached_vision_probe(base_url=self.base_url, model=model, supports=supports)
        return supports

    async def _probe_vision_support(self, model: str) -> bool:
        url = f"{self.base_url}/chat/completions"
        body = probe_request_body(model=model)
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            try:
                response = await client.post(url, json=body, headers=self._headers())
            except httpx.HTTPError:
                return False
        if response.is_success:
            payload = response.json()
            choices = payload.get("choices", [])
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if isinstance(content, str) and content.strip():
                            return True
            return False
        return not response_indicates_no_vision(response.status_code, response.text)

    async def run_vision_chat(
        self,
        *,
        image_bytes: bytes,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        body: dict[str, object] = {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            try:
                response = await client.post(url, json=body, headers=self._headers())
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise OpenAICompatibleError(f"OCR-Anfrage fehlgeschlagen: {exc}") from exc
        payload = response.json()
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise OpenAICompatibleError("Leere choices-Antwort vom Server")
        first = choices[0]
        if not isinstance(first, dict):
            raise OpenAICompatibleError("Ungültige choices-Antwort vom Server")
        message = first.get("message", {})
        if not isinstance(message, dict):
            raise OpenAICompatibleError("Ungültige message-Antwort vom Server")
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise OpenAICompatibleError("Server hat leeren Inhalt zurückgegeben")
        return content
