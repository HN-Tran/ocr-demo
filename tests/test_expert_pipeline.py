from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import pytest

from app.services.expert_pipeline import GLMOCRExpertPipeline
from app.services.ocr_pipeline import OCRPipeline, OCRResult
from app.services.ollama_client import OllamaError


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfe\xa7\xd6\xe9Q\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class _FakeDirectPipeline:
    def __init__(self) -> None:
        self.calls = 0

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


class _FakeParser:
    def __init__(
        self,
        *,
        layout: list[dict[str, object]] | None = None,
        layout_visualizations: list[Any] | None = None,
        extra_attributes: dict[str, Any] | None = None,
    ) -> None:
        self.calls = 0
        self.layout = layout
        self.layout_visualizations = layout_visualizations
        self.last_save_layout_visualization: bool | None = None
        self.markdown_result = "Expert OCR text"
        self.extra_attributes = extra_attributes or {}

    def parse(
        self,
        input_source: str,
        *,
        save_results: bool = False,
        save_layout_visualization: bool = False,
    ) -> Any:
        self.calls += 1
        self.last_save_layout_visualization = save_layout_visualization
        return type(
            "ParseResult",
            (),
            {
                "markdown_result": self.markdown_result,
                "_error": None,
                "json_result": self.layout,
                "_layout_visualization": self.layout_visualizations,
                **self.extra_attributes,
            },
        )()


def test_expert_falls_back_for_non_ocr_text_task() -> None:
    direct = _FakeDirectPipeline()
    expert = GLMOCRExpertPipeline(
        direct_pipeline=cast(OCRPipeline, direct),
        default_model="glm-ocr:latest",
        mode="selfhosted",
        ocr_api_host="localhost",
        ocr_api_port=11434,
        timeout_s=60.0,
        enable_layout=True,
    )

    result = asyncio.run(
        expert.run(
            image_bytes=_png_bytes(),
            content_type="image/png",
            mode="plain",
            schema_name=None,
            task="describe_image",
        )
    )
    assert result.text == "direct-fallback"
    assert direct.calls == 1
    assert any("direkte Pipeline wurde verwendet" in warning for warning in result.warnings)


def test_expert_uses_glm_parser_for_plain_ocr_text() -> None:
    direct = _FakeDirectPipeline()
    parser = _FakeParser()
    expert = GLMOCRExpertPipeline(
        direct_pipeline=cast(OCRPipeline, direct),
        default_model="glm-ocr:latest",
        mode="selfhosted",
        ocr_api_host="localhost",
        ocr_api_port=11434,
        timeout_s=60.0,
        enable_layout=True,
    )
    expert._get_parser = lambda *, model, enable_layout, layout_model: parser  # type: ignore[method-assign]

    result = asyncio.run(
        expert.run(
            image_bytes=_png_bytes(),
            content_type="image/png",
            mode="plain",
            schema_name=None,
        )
    )
    assert result.text == "Expert OCR text"
    assert parser.calls == 1
    assert direct.calls == 0
    assert result.layout is None
    assert result.markdown == "Expert OCR text"


def test_expert_respects_layout_override() -> None:
    direct = _FakeDirectPipeline()
    parser = _FakeParser()
    expert = GLMOCRExpertPipeline(
        direct_pipeline=cast(OCRPipeline, direct),
        default_model="glm-ocr:latest",
        mode="selfhosted",
        ocr_api_host="localhost",
        ocr_api_port=11434,
        timeout_s=60.0,
        enable_layout=True,
    )
    selected_layout_values: list[bool] = []

    def _fake_get_parser(*, model: str, enable_layout: bool, layout_model: str) -> _FakeParser:
        selected_layout_values.append(enable_layout)
        return parser

    expert._get_parser = _fake_get_parser  # type: ignore[method-assign]

    result = asyncio.run(
        expert.run(
            image_bytes=_png_bytes(),
            content_type="image/png",
            mode="plain",
            schema_name=None,
            expert_enable_layout=False,
        )
    )
    assert result.text == "Expert OCR text"
    assert selected_layout_values == [False]
    assert parser.last_save_layout_visualization is False


def test_expert_returns_layout_pages_and_visualizations() -> None:
    direct = _FakeDirectPipeline()
    with NamedTemporaryFile(suffix=".png", delete=False) as vis_file:
        vis_file.write(_png_bytes())
        vis_path = Path(vis_file.name)

    parser = _FakeParser(
        layout=[
            {
                "page_number": 1,
                "angle": 0.031004199758172,
                "width": 3000,
                "height": 4000,
                "unit": "pixel",
                "words": [{"content": "Expert"}],
                "lines": [{"content": "Expert OCR text"}],
                "spans": [{"offset": 0, "length": 15}],
                "kind": "document",
                "regions": [
                    {
                        "index": 0,
                        "label": "text_block",
                        "content": "Expert OCR text",
                        "bbox_2d": [100, 120, 900, 260],
                        "polygon": [[100, 120], [880, 140], [900, 260], [120, 240]],
                        "score": 0.9834,
                    }
                ],
            }
        ],
        layout_visualizations=[vis_path],
    )
    expert = GLMOCRExpertPipeline(
        direct_pipeline=cast(OCRPipeline, direct),
        default_model="glm-ocr:latest",
        mode="selfhosted",
        ocr_api_host="localhost",
        ocr_api_port=11434,
        timeout_s=60.0,
        enable_layout=True,
    )
    expert._get_parser = lambda *, model, enable_layout, layout_model: parser  # type: ignore[method-assign]

    try:
        result = asyncio.run(
            expert.run(
                image_bytes=_png_bytes(),
                content_type="image/png",
                mode="plain",
                schema_name=None,
            )
        )
    finally:
        vis_path.unlink(missing_ok=True)

    assert result.layout == [
        {
            "page_number": 1,
            "regions": [
                {
                    "index": 0,
                    "label": "text_block",
                    "content": "Expert OCR text",
                    "bbox_2d": [100.0, 120.0, 900.0, 260.0],
                    "polygon": [100.0, 120.0, 880.0, 140.0, 900.0, 260.0, 120.0, 240.0],
                    "confidence": 0.9834,
                }
            ],
        }
    ]
    assert result.layout_visualizations is not None
    assert len(result.layout_visualizations) == 1
    assert result.layout_visualizations[0].startswith("data:image/png;base64,")
    assert parser.last_save_layout_visualization is False
    assert result.page_texts == ["Expert OCR text"]
    assert result.page_infos == [
        {
            "page_number": 1,
            "angle": 0.031004199758172,
            "width": 3000,
            "height": 4000,
            "unit": "pixel",
            "words": [{"content": "Expert"}],
            "lines": [{"content": "Expert OCR text"}],
            "spans": [{"offset": 0, "length": 15}],
            "kind": "document",
        }
    ]
    assert any("Expert-Layout: 1 Regionen auf 1 Seite(n) erkannt." in w for w in result.warnings)


def test_expert_recovers_region_scores_from_nested_parse_result_payload() -> None:
    direct = _FakeDirectPipeline()
    parser = _FakeParser(
        layout=[
            {
                "page_number": 1,
                "regions": [
                    {
                        "index": 0,
                        "label": "text_block",
                        "content": "Nested score region",
                        "bbox_2d": [100, 120, 900, 260],
                        "polygon": [[100, 120], [880, 140], [900, 260], [120, 240]],
                        "confidence": 0.0,
                    }
                ],
            }
        ],
        extra_attributes={
            "raw_pipeline_result": {
                "pages": [
                    {
                        "regions": [
                            {
                                "index": 0,
                                "label": "text_block",
                                "content": "Nested score region",
                                "bbox_2d": [100, 120, 900, 260],
                                "polygon": [[100, 120], [880, 140], [900, 260], [120, 240]],
                                "score": 0.947,
                            }
                        ]
                    }
                ]
            }
        },
    )
    expert = GLMOCRExpertPipeline(
        direct_pipeline=cast(OCRPipeline, direct),
        default_model="glm-ocr:latest",
        mode="selfhosted",
        ocr_api_host="localhost",
        ocr_api_port=11434,
        timeout_s=60.0,
        enable_layout=True,
    )
    expert._get_parser = lambda *, model, enable_layout, layout_model: parser  # type: ignore[method-assign]

    result = asyncio.run(
        expert.run(
            image_bytes=_png_bytes(),
            content_type="image/png",
            mode="plain",
            schema_name=None,
        )
    )

    assert result.layout == [
        {
            "page_number": 1,
            "regions": [
                {
                    "index": 0,
                    "label": "text_block",
                    "content": "Nested score region",
                    "bbox_2d": [100.0, 120.0, 900.0, 260.0],
                    "polygon": [100.0, 120.0, 880.0, 140.0, 900.0, 260.0, 120.0, 240.0],
                    "confidence": 0.947,
                }
            ],
        }
    ]


def test_expert_rebuilds_text_from_layout_when_markdown_is_empty_wrapper() -> None:
    direct = _FakeDirectPipeline()
    parser = _FakeParser(
        layout=[
            {
                "regions": [
                    {
                        "index": 0,
                        "label": "text_block",
                        "content": "RECHNUNG",
                        "bbox_2d": [100, 120, 900, 200],
                    },
                    {
                        "index": 1,
                        "label": "text_block",
                        "content": "INV-2026-005",
                        "bbox_2d": [100, 210, 900, 280],
                    },
                ]
            }
        ]
    )
    parser.markdown_result = "```markdown\n\n```"

    expert = GLMOCRExpertPipeline(
        direct_pipeline=cast(OCRPipeline, direct),
        default_model="glm-ocr:latest",
        mode="selfhosted",
        ocr_api_host="localhost",
        ocr_api_port=11434,
        timeout_s=60.0,
        enable_layout=True,
    )
    expert._get_parser = lambda *, model, enable_layout, layout_model: parser  # type: ignore[method-assign]

    result = asyncio.run(
        expert.run(
            image_bytes=_png_bytes(),
            content_type="image/png",
            mode="plain",
            schema_name=None,
        )
    )

    assert result.text == "RECHNUNG\nINV-2026-005"
    assert result.markdown == "RECHNUNG\nINV-2026-005"
    assert any("Text aus Layout-Regionen rekonstruiert" in w for w in result.warnings)


def test_enable_layout_score_preservation_wraps_result_formatter() -> None:
    class _Formatter:
        def process(self, raw_layout: list[dict[str, object]]) -> tuple[str, str]:
            formatted_pages: list[list[dict[str, object]]] = []
            for page in raw_layout:
                formatted_pages.append(
                    [
                        {
                            "index": region.get("index"),
                            "label": region.get("label"),
                            "content": region.get("content"),
                            "bbox_2d": region.get("bbox_2d"),
                            "confidence": 0.0,
                        }
                        for region in page.get("regions", [])
                        if isinstance(region, dict)
                    ]
                )
            return json.dumps(formatted_pages), "markdown"

    parser = type(
        "Parser",
        (),
        {
            "_pipeline": type("Pipeline", (), {"result_formatter": _Formatter()})(),
        },
    )()

    GLMOCRExpertPipeline._enable_layout_score_preservation(parser)

    json_result, markdown_result = parser._pipeline.result_formatter.process(
        [
            {
                "page_number": 1,
                "regions": [
                    {
                        "index": 7,
                        "label": "table",
                        "content": "A | B",
                        "bbox_2d": [10, 20, 200, 100],
                        "score": 0.88,
                    }
                ],
            }
        ]
    )

    formatted = json.loads(json_result)
    assert markdown_result == "markdown"
    assert formatted[0][0]["score"] == 0.88
    assert formatted[0][0]["confidence"] == 0.88


def test_build_parser_uses_generated_selfhosted_config() -> None:
    captured_kwargs: dict[str, Any] = {}

    class _FakeGlmOcr:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            self._pipeline = type(
                "Pipeline",
                (),
                {
                    "ocr_client": type(
                        "OCRClient",
                        (),
                        {
                            "config": type("Config", (), {})(),
                            "api_mode": "openai",
                            "api_path": "/v1/chat/completions",
                            "api_scheme": "http",
                            "api_host": "localhost",
                            "api_port": 5002,
                            "api_url": "http://localhost:5002/v1/chat/completions",
                            "model": None,
                        },
                    )(),
                    "layout_detector": type(
                        "LayoutDetector",
                        (),
                        {
                            "id2label": {0: "text_block", 1: "table", 2: "figure"},
                            "label_task_mapping": {},
                        },
                    )(),
                    "result_formatter": type(
                        "Formatter", (), {"process": lambda self, value: value}
                    )(),
                },
            )()

    direct = _FakeDirectPipeline()
    expert = GLMOCRExpertPipeline(
        direct_pipeline=cast(OCRPipeline, direct),
        default_model="glm-ocr:latest",
        mode="selfhosted",
        ocr_api_host="ollama",
        ocr_api_port=11434,
        timeout_s=60.0,
        enable_layout=True,
    )
    expert._load_glmocr_class = staticmethod(lambda: _FakeGlmOcr)  # type: ignore[method-assign]

    parser = expert._build_parser(
        model="glm-ocr:latest",
        enable_layout=True,
        layout_model="PaddlePaddle/PP-DocLayoutV3_safetensors",
    )

    config_path = Path(captured_kwargs["config_path"])
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert captured_kwargs["mode"] == "selfhosted"
    assert "model" not in captured_kwargs
    assert payload["pipeline"]["ocr_api"]["model"] == "glm-ocr:latest"
    assert payload["pipeline"]["ocr_api"]["api_mode"] == "ollama_generate"
    assert payload["pipeline"]["ocr_api"]["api_path"] == "/api/generate"
    assert payload["pipeline"]["layout"]["model_dir"] == "PaddlePaddle/PP-DocLayoutV3_safetensors"
    assert payload["pipeline"]["layout"]["id2label"] is None
    assert parser._pipeline.ocr_client.api_mode == "ollama_generate"
    assert parser._pipeline.ocr_client.api_url == "http://ollama:11434/api/generate"
    assert parser._pipeline.ocr_client.model == "glm-ocr:latest"
    assert parser._pipeline.layout_detector.label_task_mapping == {
        "text": ["text_block"],
        "table": ["table"],
        "skip": ["figure"],
    }

    config_path.unlink(missing_ok=True)


def test_load_glmocr_class_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.expert_pipeline.find_spec", lambda name: None)

    with pytest.raises(OllamaError, match="erfordert das Paket 'glmocr'"):
        GLMOCRExpertPipeline._load_glmocr_class()


def test_load_glmocr_class_reports_internal_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.expert_pipeline.find_spec", lambda name: object())

    def _raise_import_error(name: str) -> Any:
        raise ImportError("No module named 'torchvision'")

    monkeypatch.setattr("app.services.expert_pipeline.importlib.import_module", _raise_import_error)

    with pytest.raises(OllamaError) as exc_info:
        GLMOCRExpertPipeline._load_glmocr_class()

    assert "konnte 'glmocr.GlmOcr' nicht laden" in str(exc_info.value)
    assert "torchvision" in str(exc_info.value)
