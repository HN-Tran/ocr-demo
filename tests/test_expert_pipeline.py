from __future__ import annotations

import asyncio
from typing import Any, cast

from app.services.document_pipeline import (
    DocumentPipeline,
    _build_table_cells,
    _classify_label,
    _parse_table_html,
    _sort_reading_order,
    _strip_table_markup,
)
from app.services.inference.registry import VisionClientRegistry
from app.services.ocr_pipeline import OCRPipeline, OCRResult


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfe\xa7\xd6\xe9Q\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class _FakeDirectPipeline:
    def __init__(self) -> None:
        self.calls = 0
        self.plain_prompt_template = "Text Recognition:"

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
        **kwargs: object,
    ) -> OCRResult:
        self.calls += 1
        return OCRResult(
            text="direct-fallback",
            structured=None,
            model=model or "m",
            mode=mode,
            schema_name=schema_name,
            latency_ms=1,
            warnings=[],
        )


class _FakeOllamaClient:
    provider_id = "ollama"

    def __init__(self, response: str = "OCR text") -> None:
        self.response = response
        self.calls = 0

    async def list_models(self) -> list[str]:
        return ["test-model"]

    async def supports_vision(self, model: str) -> bool:
        return True

    async def run_vision_chat(
        self,
        *,
        image_bytes: bytes,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls += 1
        return self.response


def _make_pipeline(
    direct: Any = None,
    ollama: Any = None,
) -> DocumentPipeline:
    return DocumentPipeline(
        direct_pipeline=cast(OCRPipeline, direct or _FakeDirectPipeline()),
        vision_registry=VisionClientRegistry(
            clients={"ollama": ollama or _FakeOllamaClient()},
            default_provider="ollama",
            default_model="test-model",
        ),
        default_model="test-model",
        enable_layout=True,
        layout_model="test/layout-model",
        timeout_s=60.0,
    )


def test_falls_back_for_structured_mode() -> None:
    direct = _FakeDirectPipeline()
    pipeline = _make_pipeline(direct=direct)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            content_type="image/png",
            mode="structured",
            schema_name="invoice_basic",
        )
    )
    assert result.text == "direct-fallback"
    assert direct.calls == 1


def test_falls_back_for_custom_prompt() -> None:
    direct = _FakeDirectPipeline()
    pipeline = _make_pipeline(direct=direct)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            content_type="image/png",
            mode="plain",
            schema_name=None,
            custom_prompt="Describe this",
        )
    )
    assert result.text == "direct-fallback"
    assert direct.calls == 1


def test_falls_back_for_non_ocr_text_task() -> None:
    direct = _FakeDirectPipeline()
    pipeline = _make_pipeline(direct=direct)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            content_type="image/png",
            mode="plain",
            schema_name=None,
            task="describe_image",
        )
    )
    assert result.text == "direct-fallback"
    assert direct.calls == 1
    assert any("direkte Pipeline wurde verwendet" in w for w in result.warnings)


def test_falls_back_when_layout_disabled() -> None:
    direct = _FakeDirectPipeline()
    pipeline = _make_pipeline(direct=direct)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            content_type="image/png",
            mode="plain",
            schema_name=None,
            expert_enable_layout=False,
        )
    )
    assert result.text == "direct-fallback"
    assert direct.calls == 1


# ------------------------------------------------------------------
# Label classification
# ------------------------------------------------------------------


def test_classify_label_text() -> None:
    assert _classify_label("text") == "text"
    assert _classify_label("paragraph_title") == "text"
    assert _classify_label("Title") == "text"


def test_classify_label_table() -> None:
    assert _classify_label("table") == "table"
    assert _classify_label("Table") == "table"
    assert _classify_label("table_title") == "table"


def test_classify_label_formula() -> None:
    assert _classify_label("formula") == "formula"
    assert _classify_label("Formula") == "formula"
    assert _classify_label("isolate_formula") == "formula"


def test_classify_label_skip() -> None:
    assert _classify_label("image") == "skip"
    assert _classify_label("figure") == "skip"
    assert _classify_label("Picture") == "skip"


# ------------------------------------------------------------------
# Table parsing
# ------------------------------------------------------------------


def test_parse_table_html() -> None:
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    rows = _parse_table_html(html)
    assert rows == [["A", "B"], ["1", "2"]]


def test_parse_table_markdown() -> None:
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    rows = _parse_table_html(md)
    assert rows == [["A", "B"], ["1", "2"]]


def test_strip_table_markup() -> None:
    html = "<table><tr><td>Hello</td><td>World</td></tr></table>"
    assert "Hello" in _strip_table_markup(html)
    assert "World" in _strip_table_markup(html)
    assert "<" not in _strip_table_markup(html)


def test_build_table_cells() -> None:
    rows = [["A", "B"], ["1", "2"]]
    cells = _build_table_cells(rows, [100, 200, 900, 400])
    assert len(cells) == 4
    assert cells[0]["row"] == 0
    assert cells[0]["column"] == 0
    assert cells[0]["content"] == "A"
    assert cells[0]["is_header"] is True
    assert cells[2]["row"] == 1
    assert cells[2]["is_header"] is False


# ------------------------------------------------------------------
# Reading order
# ------------------------------------------------------------------


def test_sort_reading_order() -> None:
    regions = [
        {"bbox_2d": [500, 0, 900, 100], "label": "right"},
        {"bbox_2d": [0, 0, 400, 100], "label": "left"},
        {"bbox_2d": [0, 200, 900, 300], "label": "bottom"},
    ]
    sorted_regions = _sort_reading_order(regions)
    assert [r["label"] for r in sorted_regions] == ["left", "right", "bottom"]
