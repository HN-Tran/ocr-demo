from __future__ import annotations

import base64

import httpx


class OllamaError(RuntimeError):
    """Wird ausgelöst, wenn eine Ollama-Anfrage fehlschlägt."""


class OllamaClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise OllamaError(f"Modellliste konnte nicht geladen werden: {exc}") from exc
        payload = response.json()
        models = payload.get("models", [])
        return [entry.get("name", "") for entry in models if entry.get("name")]

    async def run_ocr(
        self,
        *,
        image_bytes: bytes,
        prompt: str,
        model: str,
        num_ctx: int | None = None,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        options: dict[str, int] = {
            "temperature": 0,
        }
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        request_payload = {
            "model": model,
            "stream": False,
            "options": options,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded],
                }
            ],
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            try:
                response = await client.post(url, json=request_payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise OllamaError(f"OCR-Anfrage fehlgeschlagen: {exc}") from exc
        payload = response.json()
        message = payload.get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama hat leeren Inhalt zurückgegeben")
        return content
