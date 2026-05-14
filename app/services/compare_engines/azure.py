"""Azure Form Recognizer / Document Intelligence prebuilt-read-Adapter."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .base import EngineResult

AZURE_API_VERSION = "2022-08-31"


def _normalize_words_per_page(pages: list[Any]) -> list[list[dict[str, Any]]]:
    """Normalize Azure word polygon coords from pixels to 0-1000 scale, per page."""
    result_pages: list[list[dict[str, Any]]] = []
    for page in pages:
        if not isinstance(page, dict):
            result_pages.append([])
            continue
        pw = float(page.get("width") or 1000)
        ph = float(page.get("height") or 1000)
        page_words: list[dict[str, Any]] = []
        for word in page.get("words") or []:
            if not isinstance(word, dict):
                continue
            polygon = word.get("polygon", [])
            if isinstance(polygon, list) and len(polygon) >= 8:
                norm: list[float] = [
                    float(v) / (pw if i % 2 == 0 else ph) * 1000 for i, v in enumerate(polygon)
                ]
            else:
                norm = []
            page_words.append(
                {
                    "content": word.get("content", ""),
                    "polygon": norm,
                    "confidence": word.get("confidence", 0.0),
                }
            )
        result_pages.append(page_words)
    return result_pages


class AzureEngine:
    name = "azure"
    label = "Azure Form Recognizer"

    def __init__(
        self,
        *,
        endpoint: str,
        key: str,
        verify_ssl: bool = True,
        timeout_s: float = 60.0,
        full_analyze_url: str | None = None,
    ) -> None:
        if not full_analyze_url and not endpoint:
            raise ValueError("Azure-Endpunkt fehlt.")
        self._endpoint = endpoint
        self._full_analyze_url = full_analyze_url
        self._key = key
        self._verify_ssl = verify_ssl
        self._timeout_s = timeout_s

    async def analyze(self, image_bytes: bytes, content_type: str) -> EngineResult:
        if self._full_analyze_url:
            url = self._full_analyze_url
            params: dict[str, str] = {}
        else:
            url = f"{self._endpoint.rstrip('/')}/formrecognizer/documentModels/prebuilt-read:analyze"
            params = {"api-version": AZURE_API_VERSION}
        headers = {
            "Ocp-Apim-Subscription-Key": self._key,
            "Content-Type": content_type or "application/octet-stream",
        }
        async with httpx.AsyncClient(timeout=self._timeout_s, verify=self._verify_ssl) as client:
            resp = await client.post(url, content=image_bytes, headers=headers, params=params)
            resp.raise_for_status()
            if resp.status_code == 200:
                payload: dict[str, Any] = resp.json()
            else:
                op_url = resp.headers.get("Operation-Location", "")
                if not op_url:
                    raise ValueError("Azure returned 202 ohne Operation-Location-Header")
                poll_headers = {"Ocp-Apim-Subscription-Key": self._key}
                deadline = time.monotonic() + self._timeout_s
                payload = {}
                while time.monotonic() < deadline:
                    await asyncio.sleep(1.5)
                    poll = await client.get(op_url, headers=poll_headers)
                    poll.raise_for_status()
                    payload = poll.json()
                    st = str(payload.get("status", ""))
                    if st == "succeeded":
                        break
                    if st in ("failed", "canceled"):
                        err = payload.get("error", st)
                        raise ValueError(f"Azure OCR fehlgeschlagen: {err}")
                else:
                    raise TimeoutError("Azure OCR Timeout")

        analyze = payload.get("analyzeResult")
        analyze_dict: dict[str, Any] = analyze if isinstance(analyze, dict) else payload
        raw_pages = analyze_dict.get("pages") or []
        pages = raw_pages if isinstance(raw_pages, list) else []
        text = str(analyze_dict.get("content") or "")
        return EngineResult(
            text=text,
            words_per_page=_normalize_words_per_page(pages),
            raw=payload,
        )
