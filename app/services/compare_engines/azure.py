"""Azure Form Recognizer / Document Intelligence prebuilt-read-Adapter."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .base import EngineResult

AZURE_API_VERSION = "2022-08-31"


def _normalize_words_per_page(pages: list[Any]) -> list[list[dict[str, Any]]]:
    """Normalize Azure word polygon coords to 0-1000 scale, per page.

    Real Azure returns pixel-space polygons; our own compat endpoint returns
    polygons already in 0-1000 space (from the doctr word detector). Detect
    the coordinate space per-page: if the max polygon value exceeds 1.5 and
    the page dimensions are >> 1000, the coords are in pixel space and need
    scaling. If they're already ≤ 1000 (or page dims equal 1000), pass through.
    """
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
                floats = [float(v) for v in polygon]
                max_val = max(abs(v) for v in floats) if floats else 0.0
                # Already in 0-1000 space (our compat endpoint) — pass through.
                # Pixel-space polygons from real Azure have max_val scaled to pw/ph.
                if max_val <= 1000 and (pw > 1000 or ph > 1000):
                    norm = floats
                else:
                    norm = [
                        float(v) / (pw if i % 2 == 0 else ph) * 1000
                        for i, v in enumerate(floats)
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
        headers = {
            "Ocp-Apim-Subscription-Key": self._key,
            "Content-Type": content_type or "application/octet-stream",
        }
        async with httpx.AsyncClient(timeout=self._timeout_s, verify=self._verify_ssl) as client:
            if self._full_analyze_url:
                resp = await client.post(
                    self._full_analyze_url, content=image_bytes, headers=headers
                )
            else:
                url = f"{self._endpoint.rstrip('/')}/formrecognizer/documentModels/prebuilt-read:analyze"
                resp = await client.post(
                    url, content=image_bytes, headers=headers,
                    params={"api-version": AZURE_API_VERSION},
                )
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
