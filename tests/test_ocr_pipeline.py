from __future__ import annotations

import asyncio
from io import BytesIO
from typing import cast

import pytest
from PIL import Image

from app.services.ocr_pipeline import (
    PLAIN_TASK_DESCRIBE_IMAGE,
    PLAIN_TASK_PROMPTS,
    OCRPipeline,
)
from app.services.ollama_client import OllamaClient


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfe\xa7\xd6\xe9Q\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class FakeOllamaClient:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.last_prompt = ""
        self.prompts: list[str] = []
        self.last_model = ""
        self.last_num_ctx: int | None = None
        self.last_image_bytes = b""
        self.responses = list(responses) if responses is not None else []

    async def run_ocr(
        self, *, image_bytes: bytes, prompt: str, model: str, num_ctx: int | None = None
    ) -> str:
        self.last_image_bytes = image_bytes
        self.last_prompt = prompt
        self.prompts.append(prompt)
        self.last_model = model
        self.last_num_ctx = num_ctx
        if self.responses:
            return self.responses.pop(0)
        return "ok"


def _pdf_bytes(page_count: int = 1) -> bytes:
    first_image = Image.new("RGB", (12, 12), color=(255, 255, 255))
    extra_images = [
        Image.new("RGB", (12, 12), color=(255, 255, 255)) for _ in range(page_count - 1)
    ]
    output = BytesIO()
    if extra_images:
        first_image.save(output, format="PDF", save_all=True, append_images=extra_images)
    else:
        first_image.save(output, format="PDF")
    return output.getvalue()


def _tiff_bytes() -> bytes:
    image = Image.new("RGB", (12, 12), color=(255, 255, 255))
    output = BytesIO()
    image.save(output, format="TIFF")
    return output.getvalue()


def _gif_bytes(frame_count: int = 1) -> bytes:
    first_image = Image.new("RGB", (12, 12), color=(255, 255, 255))
    extra_images = [
        Image.new("RGB", (12, 12), color=(255, 255, max(0, 255 - (index + 1))))
        for index in range(frame_count - 1)
    ]
    output = BytesIO()
    if extra_images:
        first_image.save(
            output,
            format="GIF",
            save_all=True,
            append_images=extra_images,
            duration=80,
            loop=0,
        )
    else:
        first_image.save(output, format="GIF")
    return output.getvalue()


def _pipeline(fake_client: FakeOllamaClient) -> OCRPipeline:
    return OCRPipeline(
        ollama_client=cast(OllamaClient, fake_client),
        default_model="glm-ocr:latest",
        default_token_limit=4096,
        max_image_dim=2048,
    )


def test_plain_describe_image_task_uses_describe_prompt() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="describe_image",
            custom_prompt=None,
        )
    )
    assert "Describe this image briefly." == fake_client.last_prompt


def test_plain_custom_prompt_overrides_task_prompt() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="describe_image",
            custom_prompt="Beschreibe dieses Bild in einem Satz.",
        )
    )
    assert fake_client.last_prompt == "Beschreibe dieses Bild in einem Satz."


def test_plain_unknown_task_raises() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            pipeline.run(
                image_bytes=_png_bytes(),
                mode="plain",
                schema_name=None,
                task="unknown_task",
                custom_prompt=None,
            )
        )
    assert "Unbekannte Aufgabe" in str(exc_info.value)


def test_plain_extract_table_markdown_task_uses_table_prompt() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="extract_table_markdown",
            custom_prompt=None,
        )
    )
    assert "Markdown" in fake_client.last_prompt
    assert "No table detected." in fake_client.last_prompt


def test_plain_summarize_document_task_uses_summary_prompt() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="summarize_document",
            custom_prompt=None,
        )
    )
    assert "Summarize the document." in fake_client.last_prompt
    assert "bullet points in German" in fake_client.last_prompt


def test_structured_rejects_custom_prompt() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            pipeline.run(
                image_bytes=_png_bytes(),
                mode="structured",
                schema_name="invoice_basic",
                task=None,
                custom_prompt="Return JSON",
            )
        )
    assert "custom_prompt wird nur im Klartextmodus unterstützt" in str(exc_info.value)


def test_structured_auto_schema_detects_table_basic() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            "table_basic",
            (
                '{"title":"Preisliste","columns":["Artikel","Preis"],'
                '"rows":[["A","10.00"]],"notes":null}'
            ),
        ]
    )
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="structured",
            schema_name="auto",
        )
    )
    assert result.schema_name == "table_basic"
    assert result.structured is not None
    assert result.structured["title"] == "Preisliste"
    assert any("Automatisch erkanntes Schema: table_basic" in w for w in result.warnings)
    assert "Available schema_name" in fake_client.prompts[0]


def test_plain_prompt_echo_retries_with_short_prompt() -> None:
    describe_prompt = PLAIN_TASK_PROMPTS[PLAIN_TASK_DESCRIBE_IMAGE]
    fake_client = FakeOllamaClient(
        responses=[
            describe_prompt,
            "Ein Schreibtisch mit Laptop und Notizbuch.",
        ]
    )
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="describe_image",
            custom_prompt=None,
        )
    )
    assert result.text == "Ein Schreibtisch mit Laptop und Notizbuch."
    assert len(fake_client.prompts) == 2
    assert "Prompt-Echo erkannt" in " | ".join(result.warnings)


def test_plain_ocr_text_retries_when_output_looks_like_image_description() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            "Das Bild zeigt einen Schreibtisch mit Laptop und Notizbuch.",
            "OCR DEMO\nInvoice # INV-2026-001",
        ]
    )
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="ocr_text",
            custom_prompt=None,
        )
    )
    assert result.text == "OCR DEMO\nInvoice # INV-2026-001"
    assert len(fake_client.prompts) == 2
    assert "Bildbeschreibung statt OCR-Text" in " | ".join(result.warnings)


def test_plain_ocr_text_retries_when_output_is_empty_markdown_wrapper() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            "```markdown\n\n```",
            "RECHNUNG\nINV-2026-004\nGesamt: 199,00 EUR",
        ]
    )
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="ocr_text",
            custom_prompt=None,
        )
    )
    assert result.text == "RECHNUNG\nINV-2026-004\nGesamt: 199,00 EUR"
    assert len(fake_client.prompts) == 2
    assert "Leere Markdown-Hülle erkannt" in " | ".join(result.warnings)


def test_plain_extract_table_markdown_unwraps_markdown_code_fence() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            "```markdown\n| Artikel | Preis |\n| --- | --- |\n| A | 10,00 |\n```",
        ]
    )
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="extract_table_markdown",
            custom_prompt=None,
        )
    )
    assert result.text == "| Artikel | Preis |\n| --- | --- |\n| A | 10,00 |"


def test_plain_ocr_text_recovers_from_instruction_fragment_echo() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            "Das Bild nicht beschreiben. Wenn kein Text sichtbar ist, exactly ausgeben: Kein sichtbarer Text.",
            "Das Bild nicht beschreiben. Wenn kein Text sichtbar ist, gib exakt aus: Kein sichtbarer Text.",
            "RECHNUNG\nINV-2026-002\nGesamt: 99,50 EUR",
        ]
    )
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="ocr_text",
            custom_prompt=None,
        )
    )
    assert result.text == "RECHNUNG\nINV-2026-002\nGesamt: 99,50 EUR"
    assert len(fake_client.prompts) == 3
    assert "promptähnlichen" in " | ".join(result.warnings).lower()


def test_plain_describe_image_recovers_from_instruction_fragment_echo() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            "Nur Klartext zurückgeben.",
            "Only plain text.",
            "Das Bild zeigt einen Schreibtisch mit Laptop.",
        ]
    )
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="describe_image",
            custom_prompt=None,
        )
    )
    assert result.text == "Das Bild zeigt einen Schreibtisch mit Laptop."
    assert len(fake_client.prompts) == 3
    joined = " | ".join(result.warnings)
    assert "Prompt-Echo erkannt" in joined


def test_plain_describe_image_retries_when_output_looks_like_ocr_transcript() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            "INVOICE\nInvoice # INV-2026-001\nTotal: 42.75 USD\nDue Date: 2026-01-31",
            "Ein Kassenbeleg liegt auf einem Schreibtisch.",
        ]
    )
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="describe_image",
            custom_prompt=None,
        )
    )
    assert result.text == "Ein Kassenbeleg liegt auf einem Schreibtisch."
    assert len(fake_client.prompts) == 2
    assert "OCR-Transkript" in " | ".join(result.warnings)


def test_plain_describe_image_keeps_last_output_when_still_transcript_after_retries() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            "INVOICE\nInvoice # INV-2026-001\nTotal: 42.75 USD\nDue Date: 2026-01-31",
            "Invoice # INV-2026-001\nTotal: 42.75 USD",
            "Invoice # INV-2026-001\nTotal: 42.75 USD",
        ]
    )
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            task="describe_image",
            custom_prompt=None,
        )
    )
    assert result.text == "Invoice # INV-2026-001\nTotal: 42.75 USD"
    assert len(fake_client.prompts) == 3
    assert "OCR-Transkript" in " | ".join(result.warnings)


def test_structured_auto_schema_unknown_raises() -> None:
    fake_client = FakeOllamaClient(responses=["something_else"])
    pipeline = _pipeline(fake_client)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            pipeline.run(
                image_bytes=_png_bytes(),
                mode="structured",
                schema_name=None,
            )
        )
    assert "Schema konnte nicht automatisch erkannt werden" in str(exc_info.value)


def test_structured_sets_fields_empty_when_evidence_reports_no_visible_text() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            (
                '{"vendor":"Demo GmbH","invoice_number":"INV-42","invoice_date":"2026-01-01",'
                '"due_date":"2026-01-31","total":"42.75","currency":"EUR"}'
            ),
            "Kein sichtbarer Text.",
        ]
    )
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="structured",
            schema_name="invoice_basic",
        )
    )
    assert result.structured == {
        "vendor": None,
        "invoice_number": None,
        "invoice_date": None,
        "due_date": None,
        "total": None,
        "currency": None,
    }
    assert "Kein sichtbarer Text erkannt" in " | ".join(result.warnings)


def test_default_token_limit_is_forwarded() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            token_limit=None,
        )
    )
    assert fake_client.last_num_ctx == 4096


def test_token_limit_override_is_forwarded() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            mode="plain",
            schema_name=None,
            token_limit=8192,
        )
    )
    assert fake_client.last_num_ctx == 8192


def test_token_limit_must_be_positive() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            pipeline.run(
                image_bytes=_png_bytes(),
                mode="plain",
                schema_name=None,
                token_limit=0,
            )
        )
    assert "token_limit muss eine positive ganze Zahl sein" in str(exc_info.value)


def test_token_limit_must_not_exceed_max() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            pipeline.run(
                image_bytes=_png_bytes(),
                mode="plain",
                schema_name=None,
                token_limit=128001,
            )
        )
    assert "token_limit darf 128000 nicht überschreiten" in str(exc_info.value)


def test_gif_max_frames_must_be_positive() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            pipeline.run(
                image_bytes=_gif_bytes(frame_count=2),
                content_type="image/gif",
                mode="plain",
                schema_name=None,
                gif_max_frames=0,
            )
        )
    assert "gif_max_frames muss eine positive ganze Zahl sein" in str(exc_info.value)


def test_gif_max_frames_must_not_exceed_max() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            pipeline.run(
                image_bytes=_gif_bytes(frame_count=2),
                content_type="image/gif",
                mode="plain",
                schema_name=None,
                gif_max_frames=33,
            )
        )
    assert "gif_max_frames darf 32 nicht überschreiten" in str(exc_info.value)


def test_pdf_input_is_rendered_to_png_before_ollama_call() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_pdf_bytes(),
            content_type="application/pdf",
            mode="plain",
            schema_name=None,
        )
    )
    assert fake_client.last_image_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_tiff_input_is_rendered_to_png_before_ollama_call() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_tiff_bytes(),
            content_type="image/tiff",
            mode="plain",
            schema_name=None,
        )
    )
    assert fake_client.last_image_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_gif_input_is_rendered_to_png_before_ollama_call() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    asyncio.run(
        pipeline.run(
            image_bytes=_gif_bytes(),
            content_type="image/gif",
            mode="plain",
            schema_name=None,
        )
    )
    assert fake_client.last_image_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_plain_returns_page_infos_and_page_texts() -> None:
    fake_client = FakeOllamaClient(responses=["Zeile 1\nZeile 2"])
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_png_bytes(),
            content_type="image/png",
            mode="plain",
            schema_name=None,
        )
    )
    assert result.page_texts == ["Zeile 1\nZeile 2"]
    assert result.page_infos == [
        {
            "page_number": 1,
            "angle": 0.0,
            "width": 1,
            "height": 1,
            "unit": "pixel",
            "kind": "document",
            "words": [],
            "lines": [],
            "spans": [],
        }
    ]


def test_animated_gif_plain_samples_frames_with_warning() -> None:
    fake_client = FakeOllamaClient(responses=[f"Frame {index + 1}" for index in range(8)])
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_gif_bytes(frame_count=10),
            content_type="image/gif",
            mode="plain",
            schema_name=None,
        )
    )

    assert len(fake_client.prompts) == 8
    assert "--- Seite 1 ---" in result.text
    assert "--- Seite 8 ---" in result.text
    assert any("Animiertes GIF mit 10 Frames; 8 Frames" in warning for warning in result.warnings)


def test_animated_gif_respects_custom_gif_max_frames() -> None:
    fake_client = FakeOllamaClient(responses=["Frame 1", "Frame 2", "Frame 3"])
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_gif_bytes(frame_count=10),
            content_type="image/gif",
            mode="plain",
            schema_name=None,
            gif_max_frames=3,
        )
    )
    assert len(fake_client.prompts) == 3
    assert any("Animiertes GIF mit 10 Frames; 3 Frames" in warning for warning in result.warnings)


def test_animated_gif_describe_image_uses_single_storyboard_call() -> None:
    fake_client = FakeOllamaClient(responses=["Ein Hund springt über ein Sofa."])
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_gif_bytes(frame_count=10),
            content_type="image/gif",
            mode="plain",
            schema_name=None,
            task="describe_image",
        )
    )
    assert len(fake_client.prompts) == 1
    assert "chronological storyboard" in fake_client.last_prompt
    assert "Ein Hund springt über ein Sofa." == result.text
    assert any("Storyboard" in warning for warning in result.warnings)


def test_multi_page_pdf_plain_processes_all_pages() -> None:
    fake_client = FakeOllamaClient(responses=["Seite 1 Text", "Seite 2 Text"])
    pipeline = _pipeline(fake_client)
    result = asyncio.run(
        pipeline.run(
            image_bytes=_pdf_bytes(page_count=2),
            content_type="application/pdf",
            mode="plain",
            schema_name=None,
        )
    )

    assert len(fake_client.prompts) == 2
    assert "--- Seite 1 ---" in result.text
    assert "--- Seite 2 ---" in result.text
    assert "Seite 1 Text" in result.text
    assert "Seite 2 Text" in result.text
    assert any("alle Seiten wurden verarbeitet" in warning for warning in result.warnings)


def test_invalid_pdf_raises_validation_error() -> None:
    fake_client = FakeOllamaClient()
    pipeline = _pipeline(fake_client)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            pipeline.run(
                image_bytes=b"not-a-pdf",
                content_type="application/pdf",
                mode="plain",
                schema_name=None,
            )
        )
    assert "PDF konnte nicht verarbeitet werden" in str(exc_info.value)
