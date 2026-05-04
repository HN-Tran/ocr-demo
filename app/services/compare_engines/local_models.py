"""Vergleich gegen ein zweites Modell auf der LOKALEN Pipeline.

Beide Seiten des Compare-Flows hängen am selben Backend-Router; wir rufen
ihn nur mit zwei verschiedenen ``model``-Werten auf. Kein HTTP-Hop, kein
Peer — sinnvoll, um zwei Ollama-Modelle direkt nebeneinander zu prüfen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import EngineResult

if TYPE_CHECKING:
    from app.services.backend_router import OCRBackendRouter


def _extract_words_per_page(layout: list[dict[str, Any]] | None) -> list[list[dict[str, Any]]]:
    if not layout:
        return []
    out: list[list[dict[str, Any]]] = []
    for page_layout in layout:
        wp = page_layout.get("word_polys") if isinstance(page_layout, dict) else None
        out.append([w for w in wp if isinstance(w, dict)] if isinstance(wp, list) else [])
    return out


class LocalModelsEngine:
    name = "local_models"
    label = "Lokal — anderes Modell"

    def __init__(
        self,
        *,
        pipeline: OCRBackendRouter,
        model: str,
        backend: str | None = None,
    ) -> None:
        if not model:
            raise ValueError("Vergleichsmodell fehlt.")
        self._pipeline = pipeline
        self._model = model
        self._backend = backend

    async def analyze(self, image_bytes: bytes, content_type: str) -> EngineResult:
        result, _selected = await self._pipeline.run(
            backend=self._backend,
            image_bytes=image_bytes,
            content_type=content_type,
            mode="plain",
            schema_name=None,
            model=self._model,
            task="ocr_text",
            custom_prompt=None,
            token_limit=None,
            gif_max_frames=None,
        )
        return EngineResult(
            text=result.text,
            words_per_page=_extract_words_per_page(result.layout),
            warnings=list(result.warnings or []),
            raw=None,
        )
