"""Vergleich gegen eine andere Instanz dieser App (`/api/ocr`)."""

from __future__ import annotations

from typing import Any

import httpx

from .base import EngineResult


def _extract_words_per_page(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Aus dem analyzeResult unserer App die Wörter pro Seite extrahieren.

    Greift bevorzugt auf den DocTR/Paddle-Detector im Layout-Block zu;
    fällt auf das ``words``-Array pro Seite zurück, falls Layout fehlt.
    """
    layout = payload.get("layout")
    if isinstance(layout, list) and layout:
        pages_words: list[list[dict[str, Any]]] = []
        for page_layout in layout:
            wp = page_layout.get("word_polys") if isinstance(page_layout, dict) else None
            if isinstance(wp, list):
                pages_words.append([w for w in wp if isinstance(w, dict)])
            else:
                pages_words.append([])
        return pages_words

    analyze = payload.get("analyzeResult")
    pages = (analyze.get("pages") if isinstance(analyze, dict) else None) or []
    if not isinstance(pages, list):
        return []
    pages_words = []
    for page in pages:
        words = page.get("words") if isinstance(page, dict) else None
        if isinstance(words, list):
            pages_words.append([w for w in words if isinstance(w, dict)])
        else:
            pages_words.append([])
    return pages_words


class SelfPeerEngine:
    name = "self_peer"
    label = "OCR-Demo (anderer Endpunkt)"

    def __init__(
        self,
        *,
        base_url: str,
        backend: str | None = None,
        model: str | None = None,
        verify_ssl: bool = True,
        timeout_s: float = 120.0,
    ) -> None:
        if not base_url:
            raise ValueError("Peer-URL fehlt.")
        self._base_url = base_url.rstrip("/")
        self._backend = backend
        self._model = model
        self._verify_ssl = verify_ssl
        self._timeout_s = timeout_s

    async def analyze(self, image_bytes: bytes, content_type: str) -> EngineResult:
        url = f"{self._base_url}/api/ocr"
        files = {"file": ("upload.bin", image_bytes, content_type or "application/octet-stream")}
        data: dict[str, str] = {
            "mode": "plain",
            "task": "ocr_text",
            "expert_enable_layout": "true",
        }
        if self._backend:
            data["backend"] = self._backend
        if self._model:
            data["model"] = self._model
        async with httpx.AsyncClient(timeout=self._timeout_s, verify=self._verify_ssl) as client:
            resp = await client.post(url, files=files, data=data)
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()

        text = str(payload.get("text") or "")
        return EngineResult(
            text=text,
            words_per_page=_extract_words_per_page(payload),
            raw=payload,
        )
