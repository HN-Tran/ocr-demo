from __future__ import annotations

from app.services.ocr_pipeline import OCRResult
from app.services.ocr_service import OCRService

SUPPORTED_BACKENDS = ("direct", "expert")


class OCRBackendRouter:
    def __init__(self, *, default_backend: str, backends: dict[str, OCRService]) -> None:
        self.backends = backends
        self.default_backend = self.normalize_backend(default_backend)

    @staticmethod
    def normalize_backend(backend: str | None) -> str:
        selected_backend = (backend or "").strip().lower() or "direct"
        if selected_backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unbekanntes Backend '{selected_backend}'. Unterstützte Backends: {', '.join(SUPPORTED_BACKENDS)}"
            )
        return selected_backend

    async def run(
        self,
        *,
        backend: str | None,
        image_bytes: bytes,
        content_type: str | None = None,
        mode: str,
        schema_name: str | None,
        model: str | None = None,
        task: str | None = None,
        custom_prompt: str | None = None,
        token_limit: int | None = None,
        gif_max_frames: int | None = None,
        expert_enable_layout: bool | None = None,
        expert_layout_model: str | None = None,
        expert_layout_threshold: float | None = None,
        expert_table_transformer: bool | None = None,
        expert_per_region_ocr: bool | None = None,
        expert_word_detector: str | None = None,
    ) -> tuple[OCRResult, str]:
        requested_backend = (
            self.default_backend if backend is None else self.normalize_backend(backend)
        )
        service = self.backends.get(requested_backend)
        if service is None:
            raise ValueError(
                f"Backend '{requested_backend}' ist nicht verfügbar. Verfügbare Backends: {', '.join(sorted(self.backends.keys()))}"
            )
        result = await service.run(
            image_bytes=image_bytes,
            content_type=content_type,
            mode=mode,
            schema_name=schema_name,
            model=model,
            task=task,
            custom_prompt=custom_prompt,
            token_limit=token_limit,
            gif_max_frames=gif_max_frames,
            expert_enable_layout=expert_enable_layout,
            expert_layout_model=expert_layout_model,
            expert_layout_threshold=expert_layout_threshold,
            expert_table_transformer=expert_table_transformer,
            expert_per_region_ocr=expert_per_region_ocr,
            expert_word_detector=expert_word_detector,
        )
        return result, requested_backend
