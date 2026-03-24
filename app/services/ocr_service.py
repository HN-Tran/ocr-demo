from __future__ import annotations

from typing import Protocol

from app.services.ocr_pipeline import OCRResult


class OCRService(Protocol):
    async def run(
        self,
        *,
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
    ) -> OCRResult: ...
