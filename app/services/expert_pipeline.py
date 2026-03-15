from __future__ import annotations

import base64
import copy
import importlib
import json
import logging
import mimetypes
import time
from importlib.util import find_spec
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from PIL import Image

from app.services.ocr_pipeline import (
    PLAIN_TASK_OCR_TEXT,
    OCRPipeline,
    OCRResult,
    normalize_ocr_text_output,
)
from app.services.ollama_client import OllamaError

logger = logging.getLogger(__name__)

_CONTENT_TYPE_SUFFIX_MAP = {
    "application/pdf": ".pdf",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tif": ".tif",
    "image/tiff": ".tif",
    "image/webp": ".webp",
    "image/x-tiff": ".tif",
}
_LAYOUT_VISUALIZATION_KEYS = (
    "_layout_visualization",
    "layout_visualization",
    "layout_visualizations",
    "layout_visualization_paths",
)
_DEFAULT_LAYOUT_MODEL = "PaddlePaddle/PP-DocLayoutV3_safetensors"
_GLMOCR_IMAGE_LABELS = {
    "figure",
    "figure_caption",
    "image",
    "picture",
    "illustration",
    "chart",
    "seal",
    "signature",
    "stamp",
    "barcode",
    "qr_code",
    "Picture",
}
_GLMOCR_TABLE_LABELS = {
    "table",
    "table_title",
    "table_caption",
    "table_footnote",
    "Table",
}
_GLMOCR_FORMULA_LABELS = {
    "formula",
    "formula_caption",
    "formula_number",
    "equation_footnote",
    "isolate_formula",
    "Formula",
}
_GLMOCR_RESULT_LABEL_MAPPING = {
    "image": sorted(_GLMOCR_IMAGE_LABELS),
    "table": sorted(_GLMOCR_TABLE_LABELS),
    "formula": sorted(_GLMOCR_FORMULA_LABELS),
    "text": [
        "abstract",
        "algorithm",
        "aside_text",
        "code",
        "content",
        "doc_title",
        "footer",
        "footnote",
        "header",
        "list",
        "number",
        "opara",
        "paragraph_title",
        "reference",
        "reference_content",
        "text",
        "text_block",
        "title",
        "vision_footnote",
        # Docling Heron labels
        "Caption",
        "Checkbox-Selected",
        "Checkbox-Unselected",
        "Code",
        "Document Index",
        "Footnote",
        "Form",
        "Key-Value Region",
        "List-item",
        "Page-footer",
        "Page-header",
        "Section-header",
        "Text",
        "Title",
    ],
}
_GLMOCR_TASK_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


class GLMOCRExpertPipeline:
    def __init__(
        self,
        *,
        direct_pipeline: OCRPipeline,
        default_model: str,
        mode: str,
        ocr_api_host: str,
        ocr_api_port: int,
        timeout_s: float,
        enable_layout: bool,
        layout_model: str = "",
    ) -> None:
        self.direct_pipeline = direct_pipeline
        self.default_model = default_model
        self.mode = mode
        self.ocr_api_host = ocr_api_host
        self.ocr_api_port = ocr_api_port
        self.timeout_s = timeout_s
        self.enable_layout = enable_layout
        self.layout_model = layout_model.strip() if layout_model else _DEFAULT_LAYOUT_MODEL
        self._parser_cache: dict[tuple[str, bool, str], Any] = {}
        self._config_path_cache: dict[tuple[str, bool, str], Path] = {}
        self._table_recognizer: Any = None

    @staticmethod
    def _load_glmocr_class() -> type[Any]:
        if find_spec("glmocr") is None:
            raise OllamaError(
                "Expert-Backend erfordert das Paket 'glmocr'. Bitte Abhängigkeit installieren."
            )

        try:
            module = importlib.import_module("glmocr")
            parser_class = module.GlmOcr
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(
                f"Expert-Backend konnte 'glmocr.GlmOcr' nicht laden: {type(exc).__name__}: {exc}"
            ) from exc

        return parser_class

    def _build_parser_config_payload(
        self, *, model: str, enable_layout: bool, layout_model: str
    ) -> dict[str, object]:
        return {
            "pipeline": {
                "enable_layout": enable_layout,
                "max_workers": 16,
                "page_maxsize": 100,
                "region_maxsize": 800,
                "page_loader": {
                    "max_tokens": 16384,
                    "temperature": 0.01,
                    "top_p": 0.00001,
                    "top_k": 1,
                    "repetition_penalty": 1.1,
                    "default_prompt": (
                        "Recognize the text in the image and output in Markdown format. "
                        "Preserve the original layout (headings/paragraphs/tables/formulas). "
                        "Do not fabricate content that does not exist in the image."
                    ),
                    "task_prompt_mapping": _GLMOCR_TASK_PROMPTS,
                },
                "ocr_api": {
                    "api_host": self.ocr_api_host,
                    "api_port": self.ocr_api_port,
                    "api_scheme": "http",
                    "api_path": "/api/generate",
                    "api_mode": "ollama_generate",
                    "model": model,
                    "verify_ssl": False,
                    "connect_timeout": max(1, int(self.timeout_s)),
                    "request_timeout": max(1, int(self.timeout_s)),
                },
                "result_formatter": {
                    "filter_nested": True,
                    "min_overlap_ratio": 0.8,
                    "output_format": "both",
                    "label_visualization_mapping": _GLMOCR_RESULT_LABEL_MAPPING,
                },
                "layout": {
                    "model_dir": layout_model,
                    "threshold": 0.4,
                    "batch_size": 8,
                    "workers": 1,
                    "cuda_visible_devices": "0",
                    "img_size": None,
                    "layout_nms": True,
                    "layout_unclip_ratio": [1.0, 1.0],
                    "layout_merge_bboxes_mode": "large",
                    "label_task_mapping": {},
                    "id2label": None,
                },
            }
        }

    def _get_parser_config_path(
        self, *, model: str, enable_layout: bool, layout_model: str
    ) -> Path:
        cache_key = (model, enable_layout, layout_model)
        cached_path = self._config_path_cache.get(cache_key)
        if cached_path is not None and cached_path.exists():
            return cached_path

        payload = self._build_parser_config_payload(
            model=model, enable_layout=enable_layout, layout_model=layout_model
        )
        with NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as temp_file:
            json.dump(payload, temp_file, ensure_ascii=True)
            config_path = Path(temp_file.name)

        self._config_path_cache[cache_key] = config_path
        return config_path

    @staticmethod
    def _normalize_id2label(raw_id2label: Any) -> dict[int, str] | None:
        if not isinstance(raw_id2label, dict):
            return None

        normalized: dict[int, str] = {}
        for raw_key, raw_value in raw_id2label.items():
            if raw_value is None or not str(raw_value).strip():
                continue
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                continue
            normalized[key] = str(raw_value).strip()
        return normalized or None

    @staticmethod
    def _classify_layout_label(label: str) -> str:
        normalized = label.strip().lower()
        if not normalized:
            return "text"
        if label in _GLMOCR_IMAGE_LABELS or normalized in _GLMOCR_IMAGE_LABELS:
            return "skip"
        if (
            label in _GLMOCR_TABLE_LABELS
            or normalized in _GLMOCR_TABLE_LABELS
            or "table" in normalized
        ):
            return "table"
        if (
            label in _GLMOCR_FORMULA_LABELS
            or normalized in _GLMOCR_FORMULA_LABELS
            or "formula" in normalized
            or "equation" in normalized
        ):
            return "formula"
        return "text"

    @classmethod
    def _build_layout_task_mapping(cls, id2label: dict[int, str] | None) -> dict[str, list[str]]:
        if not id2label:
            return {}

        mapping: dict[str, list[str]] = {"text": [], "table": [], "formula": [], "skip": []}
        for _, label in sorted(id2label.items()):
            task_type = cls._classify_layout_label(label)
            if label not in mapping[task_type]:
                mapping[task_type].append(label)
        return {task_type: labels for task_type, labels in mapping.items() if labels}

    def _configure_parser_runtime(self, *, parser: Any, model: str, enable_layout: bool) -> None:
        pipeline = getattr(parser, "_pipeline", None)
        if pipeline is None:
            return

        ocr_client = getattr(pipeline, "ocr_client", None)
        if ocr_client is not None:
            api_url = f"http://{self.ocr_api_host}:{self.ocr_api_port}/api/generate"
            for attr, value in (
                ("api_host", self.ocr_api_host),
                ("api_port", self.ocr_api_port),
                ("api_scheme", "http"),
                ("api_path", "/api/generate"),
                ("api_url", api_url),
                ("api_mode", "ollama_generate"),
                ("model", model),
            ):
                setattr(ocr_client, attr, value)
                if hasattr(getattr(ocr_client, "config", None), attr):
                    setattr(ocr_client.config, attr, value)

        if not enable_layout:
            return

        layout_detector = getattr(pipeline, "layout_detector", None)
        if layout_detector is None:
            return

        id2label = self._normalize_id2label(getattr(layout_detector, "id2label", None))
        if id2label is not None:
            layout_detector.id2label = id2label

        label_task_mapping = getattr(layout_detector, "label_task_mapping", None)
        if not isinstance(label_task_mapping, dict) or not label_task_mapping:
            layout_detector.label_task_mapping = self._build_layout_task_mapping(id2label)

    @staticmethod
    def _needs_custom_layout_detector(layout_model: str) -> bool:
        """Check if the model requires our generic HF detector instead of PP-DocLayout."""
        normalized = layout_model.strip().lower()
        return "pp-doclayout" not in normalized and "ppdoclayout" not in normalized

    def _build_parser(self, *, model: str, enable_layout: bool, layout_model: str) -> Any:
        parser_class = self._load_glmocr_class()
        use_custom_detector = enable_layout and self._needs_custom_layout_detector(layout_model)

        if use_custom_detector:
            # Build parser with layout disabled to prevent PPDocLayoutDetector from
            # loading (it would fail with an architecture mismatch). We inject our
            # HFLayoutDetector afterwards and re-enable layout on the pipeline.
            config_path = self._get_parser_config_path(
                model=model, enable_layout=False, layout_model=layout_model
            )
        else:
            config_path = self._get_parser_config_path(
                model=model, enable_layout=enable_layout, layout_model=layout_model
            )

        try:
            parser = parser_class(
                config_path=str(config_path),
                mode=self.mode,
                ocr_api_host=self.ocr_api_host,
                ocr_api_port=self.ocr_api_port,
                timeout=max(1, int(self.timeout_s)),
                enable_layout=not use_custom_detector and enable_layout,
            )

            if use_custom_detector:
                self._inject_custom_layout_detector(parser, layout_model=layout_model)

            self._configure_parser_runtime(parser=parser, model=model, enable_layout=enable_layout)
            self._enable_layout_score_preservation(parser)
            return parser
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"Expert-Backend konnte nicht initialisiert werden: {exc}") from exc

    def _inject_custom_layout_detector(self, parser: Any, *, layout_model: str) -> None:
        """Replace the pipeline's layout detector with our generic HF detector."""
        from app.services.layout_detector import HFLayoutDetector

        pipeline = getattr(parser, "_pipeline", None)
        if pipeline is None:
            raise OllamaError(
                "Expert-Backend Pipeline nicht gefunden; "
                "benutzerdefinierter Layout-Detektor kann nicht injiziert werden."
            )

        layout_config = getattr(pipeline, "config", None)
        layout_config = getattr(layout_config, "layout", None)
        if layout_config is None:
            from glmocr.config import LayoutConfig

            layout_config = LayoutConfig(model_dir=layout_model)
        else:
            layout_config.model_dir = layout_model

        detector = HFLayoutDetector(layout_config)
        detector.start()

        pipeline.layout_detector = detector
        pipeline.enable_layout = True
        parser.enable_layout = True

        if not hasattr(pipeline, "max_workers") or not pipeline.max_workers:
            pipeline.max_workers = 16

    def _get_parser(self, *, model: str, enable_layout: bool, layout_model: str) -> Any:
        cache_key = (model, enable_layout, layout_model)
        parser = self._parser_cache.get(cache_key)
        if parser is None:
            parser = self._build_parser(
                model=model, enable_layout=enable_layout, layout_model=layout_model
            )
            self._parser_cache[cache_key] = parser
        return parser

    @staticmethod
    def _extract_markdown(parse_result: Any) -> str:
        markdown = GLMOCRExpertPipeline._get_result_value(parse_result, "markdown_result", "")
        if isinstance(markdown, str):
            return normalize_ocr_text_output(markdown)
        if markdown is None:
            return ""
        return normalize_ocr_text_output(str(markdown))

    @staticmethod
    def _get_result_value(parse_result: Any, key: str, default: Any = None) -> Any:
        if isinstance(parse_result, dict):
            return parse_result.get(key, default)
        return getattr(parse_result, key, default)

    @staticmethod
    def _extract_error_message(parse_result: Any) -> str | None:
        error_message = GLMOCRExpertPipeline._get_result_value(parse_result, "_error")
        if error_message:
            return str(error_message).strip()
        return None

    @staticmethod
    def _coerce_confidence(value: Any) -> float | None:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if numeric_value < 0:
            return None
        return numeric_value

    @staticmethod
    def _region_signature(region: dict[str, object]) -> tuple[object, ...]:
        bbox = region.get("bbox_2d")
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox_value: tuple[object, ...] = tuple(
                round(float(value), 4) if isinstance(value, (int, float)) else value
                for value in bbox
            )
        else:
            bbox_value = ()

        return (
            region.get("index"),
            str(region.get("label") or "").strip(),
            str(region.get("content") or "").strip(),
            bbox_value,
        )

    @classmethod
    def _extract_page_region_scores(
        cls, raw_layout: Any
    ) -> list[list[tuple[tuple[object, ...], float | None]]]:
        if isinstance(raw_layout, dict):
            raw_pages = raw_layout.get("pages") or raw_layout.get("layout")
        else:
            raw_pages = raw_layout
        if raw_pages is None:
            return []
        if not isinstance(raw_pages, list):
            return []

        page_scores: list[list[tuple[tuple[object, ...], float | None]]] = []
        for raw_page in raw_pages:
            if isinstance(raw_page, dict):
                raw_regions = raw_page.get("regions")
            else:
                raw_regions = raw_page
            if not isinstance(raw_regions, list):
                page_scores.append([])
                continue

            region_scores: list[tuple[tuple[object, ...], float | None]] = []
            for raw_region in raw_regions:
                if not isinstance(raw_region, dict):
                    continue
                raw_confidence = raw_region.get("score")
                if raw_confidence is None:
                    raw_confidence = raw_region.get("confidence")
                region_scores.append(
                    (
                        cls._region_signature(raw_region),
                        cls._coerce_confidence(raw_confidence),
                    )
                )
            page_scores.append(region_scores)
        return page_scores

    @classmethod
    def _collect_layout_payloads(
        cls,
        value: Any,
        *,
        max_depth: int = 6,
        seen: set[int] | None = None,
    ) -> list[Any]:
        if max_depth < 0:
            return []

        if seen is None:
            seen = set()

        value_id = id(value)
        if value_id in seen:
            return []
        seen.add(value_id)

        payloads: list[Any] = []
        extracted_scores = cls._extract_page_region_scores(value)
        if extracted_scores and any(page_scores for page_scores in extracted_scores):
            payloads.append(value)

        children: list[Any] = []
        if isinstance(value, dict):
            children.extend(value.values())
        elif isinstance(value, (list, tuple, set)):
            children.extend(value)
        else:
            attributes = getattr(value, "__dict__", None)
            if isinstance(attributes, dict):
                children.extend(attributes.values())
            if not children:
                for name in dir(value):
                    if name.startswith("__"):
                        continue
                    try:
                        attr_value = getattr(value, name)
                    except Exception:  # noqa: BLE001
                        continue
                    if callable(attr_value):
                        continue
                    children.append(attr_value)

        for child in children:
            payloads.extend(cls._collect_layout_payloads(child, max_depth=max_depth - 1, seen=seen))
        return payloads

    @classmethod
    def _merge_page_score_sets(
        cls,
        base_scores: list[list[tuple[tuple[object, ...], float | None]]],
        candidate_scores: list[list[tuple[tuple[object, ...], float | None]]],
    ) -> list[list[tuple[tuple[object, ...], float | None]]]:
        page_count = max(len(base_scores), len(candidate_scores))
        merged_pages: list[list[tuple[tuple[object, ...], float | None]]] = []
        for page_index in range(page_count):
            merged_by_signature: dict[tuple[object, ...], float | None] = {}
            ordered_signatures: list[tuple[object, ...]] = []
            for page_scores in (base_scores, candidate_scores):
                if page_index >= len(page_scores):
                    continue
                for signature, confidence in page_scores[page_index]:
                    if signature not in merged_by_signature:
                        ordered_signatures.append(signature)
                        merged_by_signature[signature] = confidence
                        continue

                    existing = merged_by_signature[signature]
                    if confidence is None:
                        continue
                    if existing is None or confidence > existing:
                        merged_by_signature[signature] = confidence

            merged_pages.append(
                [(signature, merged_by_signature[signature]) for signature in ordered_signatures]
            )
        return merged_pages

    @classmethod
    def _extract_page_region_scores_from_parse_result(
        cls, parse_result: Any
    ) -> list[list[tuple[tuple[object, ...], float | None]]]:
        combined_scores: list[list[tuple[tuple[object, ...], float | None]]] = []
        for payload in cls._collect_layout_payloads(parse_result):
            payload_scores = cls._extract_page_region_scores(payload)
            if not payload_scores:
                continue
            combined_scores = cls._merge_page_score_sets(combined_scores, payload_scores)
        return combined_scores

    @classmethod
    def _apply_page_region_scores_to_layout(
        cls,
        layout: list[dict[str, object]] | None,
        page_scores: list[list[tuple[tuple[object, ...], float | None]]],
    ) -> list[dict[str, object]] | None:
        if not layout:
            return layout

        for page_index, page in enumerate(layout):
            if not isinstance(page, dict):
                continue
            regions = page.get("regions")
            if not isinstance(regions, list):
                continue

            candidate_scores = page_scores[page_index] if page_index < len(page_scores) else []
            scores_by_signature = {
                signature: confidence
                for signature, confidence in candidate_scores
                if confidence is not None
            }
            positional_scores = [
                confidence for _, confidence in candidate_scores if confidence is not None
            ]

            for region_index, region in enumerate(regions):
                if not isinstance(region, dict):
                    continue
                confidence = scores_by_signature.get(cls._region_signature(region))
                if confidence is None and region_index < len(positional_scores):
                    confidence = positional_scores[region_index]
                if confidence is not None:
                    region["confidence"] = confidence
        return layout

    @classmethod
    def _merge_region_scores(
        cls,
        formatted_layout: Any,
        page_scores: list[list[tuple[tuple[object, ...], float | None]]],
    ) -> Any:
        if isinstance(formatted_layout, dict):
            formatted_pages = formatted_layout.get("pages") or formatted_layout.get("layout")
        else:
            formatted_pages = formatted_layout
        if formatted_pages is None:
            return formatted_layout
        if not isinstance(formatted_pages, list):
            return formatted_layout

        for page_index, formatted_page in enumerate(formatted_pages):
            if isinstance(formatted_page, dict):
                formatted_regions = formatted_page.get("regions")
            else:
                formatted_regions = formatted_page
            if not isinstance(formatted_regions, list):
                continue

            candidate_scores = page_scores[page_index] if page_index < len(page_scores) else []
            scores_by_signature = {
                signature: confidence
                for signature, confidence in candidate_scores
                if confidence is not None
            }
            positional_scores = [confidence for _, confidence in candidate_scores]

            for region_index, formatted_region in enumerate(formatted_regions):
                if not isinstance(formatted_region, dict):
                    continue

                confidence = scores_by_signature.get(cls._region_signature(formatted_region))
                if confidence is None and region_index < len(positional_scores):
                    confidence = positional_scores[region_index]
                if confidence is not None:
                    formatted_region["score"] = confidence
                    formatted_region["confidence"] = confidence

        return formatted_layout

    @classmethod
    def _enable_layout_score_preservation(cls, parser: Any) -> None:
        pipeline = getattr(parser, "_pipeline", None)
        formatter_owner = None
        if pipeline is not None and hasattr(pipeline, "result_formatter"):
            formatter_owner = pipeline
        elif hasattr(parser, "result_formatter"):
            formatter_owner = parser
        if formatter_owner is None:
            return

        formatter = getattr(formatter_owner, "result_formatter", None)
        if formatter is None or getattr(formatter, "_ocr_demo_preserve_score", False):
            return
        if not callable(getattr(formatter, "process", None)):
            return

        class ScorePreservingFormatter:
            _ocr_demo_preserve_score = True

            def __init__(self, base_formatter: Any) -> None:
                self._base_formatter = base_formatter

            def __getattr__(self, name: str) -> Any:
                return getattr(self._base_formatter, name)

            def process(self, *args: Any, **kwargs: Any) -> Any:
                raw_payload = args[0] if args else None
                try:
                    score_source = copy.deepcopy(raw_payload)
                except Exception:  # noqa: BLE001
                    score_source = raw_payload
                page_scores = cls._extract_page_region_scores(score_source)
                formatted_layout = self._base_formatter.process(*args, **kwargs)

                if (
                    isinstance(formatted_layout, tuple)
                    and len(formatted_layout) == 2
                    and isinstance(formatted_layout[0], str)
                ):
                    json_result, markdown_result = formatted_layout
                    try:
                        parsed_layout = json.loads(json_result)
                    except Exception:  # noqa: BLE001
                        return formatted_layout
                    merged_layout = cls._merge_region_scores(parsed_layout, page_scores)
                    return json.dumps(merged_layout, ensure_ascii=False), markdown_result

                return cls._merge_region_scores(formatted_layout, page_scores)

        formatter_owner.result_formatter = ScorePreservingFormatter(formatter)

    @staticmethod
    def _normalize_layout_region(region: Any) -> dict[str, object] | None:
        if not isinstance(region, dict):
            return None

        normalized_region: dict[str, object] = {}

        index = region.get("index")
        if index is not None:
            try:
                normalized_region["index"] = int(index)
            except (TypeError, ValueError):
                normalized_region["index"] = str(index)

        label = region.get("label")
        if label is not None and str(label).strip():
            normalized_region["label"] = str(label).strip()

        content = region.get("content")
        if content is not None and str(content).strip():
            normalized_region["content"] = str(content).strip()

        bbox = region.get("bbox_2d")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                normalized_region["bbox_2d"] = [float(value) for value in bbox]
            except (TypeError, ValueError):
                pass

        polygon = GLMOCRExpertPipeline._normalize_polygon(region.get("polygon"))
        if polygon is not None:
            normalized_region["polygon"] = polygon
            if "bbox_2d" not in normalized_region:
                xs = polygon[0::2]
                ys = polygon[1::2]
                normalized_region["bbox_2d"] = [
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                ]

        confidence_raw = region.get("confidence")
        if confidence_raw is None:
            confidence_raw = region.get("score")
        confidence = GLMOCRExpertPipeline._coerce_confidence(confidence_raw)
        if confidence is not None:
            normalized_region["confidence"] = confidence

        if not normalized_region:
            return None
        return normalized_region

    @staticmethod
    def _normalize_polygon(value: Any) -> list[float] | None:
        if not isinstance(value, (list, tuple)):
            return None

        polygon: list[float] = []
        if value and all(isinstance(point, (int, float)) for point in value):
            if len(value) < 8 or len(value) % 2 != 0:
                return None
            try:
                polygon = [float(point) for point in value]
            except (TypeError, ValueError):
                return None
            return polygon

        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None
            if not all(isinstance(coordinate, (int, float)) for coordinate in point):
                return None
            polygon.extend((float(point[0]), float(point[1])))

        if len(polygon) < 8 or len(polygon) % 2 != 0:
            return None
        return polygon

    @classmethod
    def _extract_raw_layout_pages(cls, parse_result: Any) -> list[Any] | None:
        raw_layout = cls._get_result_value(parse_result, "json_result")
        if raw_layout is None:
            return None

        if isinstance(raw_layout, dict):
            candidate_pages = raw_layout.get("pages") or raw_layout.get("layout")
        else:
            candidate_pages = raw_layout

        if not isinstance(candidate_pages, list):
            return None
        return candidate_pages

    @classmethod
    def _extract_layout(cls, parse_result: Any) -> list[dict[str, object]] | None:
        candidate_pages = cls._extract_raw_layout_pages(parse_result)
        if candidate_pages is None:
            return None

        pages: list[dict[str, object]] = []
        for page_index, page in enumerate(candidate_pages, start=1):
            raw_regions: Any
            page_number = page_index
            if isinstance(page, dict):
                raw_regions = page.get("regions")
                raw_page_number = page.get("page_number") or page.get("page") or page_index
                try:
                    page_number = int(raw_page_number)
                except (TypeError, ValueError):
                    page_number = page_index
            else:
                raw_regions = page

            if not isinstance(raw_regions, list):
                continue

            regions = [
                normalized_region
                for region in raw_regions
                if (normalized_region := cls._normalize_layout_region(region)) is not None
            ]
            pages.append({"page_number": page_number, "regions": regions})

        return pages or None

    @staticmethod
    def _infer_page_size_from_regions(
        regions: list[dict[str, object]],
    ) -> tuple[float, float] | None:
        max_x = 0.0
        max_y = 0.0
        found_bbox = False
        for region in regions:
            bbox = region.get("bbox_2d")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                _, _, x2, y2 = [float(value) for value in bbox]
            except (TypeError, ValueError):
                continue
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            found_bbox = True
        if not found_bbox:
            return None
        return max_x, max_y

    @classmethod
    def _extract_page_infos(
        cls,
        parse_result: Any,
        *,
        layout: list[dict[str, object]] | None,
    ) -> list[dict[str, object]] | None:
        candidate_pages = cls._extract_raw_layout_pages(parse_result)
        if candidate_pages is None and not layout:
            return None

        layout_by_page = {
            int(page.get("page_number", index)): page
            for index, page in enumerate(layout or [], start=1)
            if isinstance(page, dict)
        }
        page_infos: list[dict[str, object]] = []
        page_count = max(len(candidate_pages or []), len(layout or []))
        for page_index in range(1, page_count + 1):
            raw_page = None
            if candidate_pages and page_index - 1 < len(candidate_pages):
                raw_page = candidate_pages[page_index - 1]
            raw_page_dict = raw_page if isinstance(raw_page, dict) else {}
            page_number = (
                raw_page_dict.get("page_number") or raw_page_dict.get("page") or page_index
            )
            try:
                normalized_page_number = int(page_number)
            except (TypeError, ValueError):
                normalized_page_number = page_index

            page_info: dict[str, object] = {
                "page_number": normalized_page_number,
                "angle": 0.0,
                "unit": "pixel",
                "kind": "document",
                "words": [],
                "lines": [],
                "spans": [],
            }

            for key in ("angle", "width", "height"):
                value = raw_page_dict.get(key)
                if isinstance(value, (int, float)):
                    page_info[key] = float(value) if key == "angle" else value
            for key in ("unit", "kind"):
                value = raw_page_dict.get(key)
                if isinstance(value, str) and value.strip():
                    page_info[key] = value.strip()
            for key in ("words", "lines", "spans"):
                value = raw_page_dict.get(key)
                if isinstance(value, list):
                    page_info[key] = value

            if "width" not in page_info or "height" not in page_info:
                page_layout = layout_by_page.get(normalized_page_number)
                regions = page_layout.get("regions") if isinstance(page_layout, dict) else None
                if isinstance(regions, list):
                    inferred_size = cls._infer_page_size_from_regions(
                        [region for region in regions if isinstance(region, dict)]
                    )
                    if inferred_size is not None:
                        inferred_width, inferred_height = inferred_size
                        page_info.setdefault("width", inferred_width)
                        page_info.setdefault("height", inferred_height)

            page_infos.append(page_info)

        return page_infos or None

    @staticmethod
    def _path_to_data_url(path: Path) -> str | None:
        if not path.is_file():
            return None

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if not mime_type.startswith("image/"):
            return None

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @classmethod
    def _normalize_layout_visualization_value(cls, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, dict):
            urls: list[str] = []
            for key in ("path", "paths", "url", "urls", "src", "image"):
                urls.extend(cls._normalize_layout_visualization_value(value.get(key)))
            return urls

        if isinstance(value, (list, tuple, set)):
            urls: list[str] = []
            for item in value:
                urls.extend(cls._normalize_layout_visualization_value(item))
            return urls

        if isinstance(value, (bytes, bytearray)):
            encoded = base64.b64encode(bytes(value)).decode("ascii")
            return [f"data:image/png;base64,{encoded}"]

        if isinstance(value, Path):
            data_url = cls._path_to_data_url(value)
            return [data_url] if data_url else []

        if isinstance(value, str):
            stripped_value = value.strip()
            if not stripped_value:
                return []
            if stripped_value.startswith(("data:image/", "http://", "https://")):
                return [stripped_value]

            data_url = cls._path_to_data_url(Path(stripped_value))
            return [data_url] if data_url else []

        return []

    @classmethod
    def _extract_layout_visualizations(cls, parse_result: Any) -> list[str] | None:
        visualizations: list[str] = []
        seen: set[str] = set()

        for key in _LAYOUT_VISUALIZATION_KEYS:
            for item in cls._normalize_layout_visualization_value(
                cls._get_result_value(parse_result, key)
            ):
                if item not in seen:
                    seen.add(item)
                    visualizations.append(item)

        raw_mapping = (
            parse_result
            if isinstance(parse_result, dict)
            else getattr(parse_result, "__dict__", {})
        )
        if isinstance(raw_mapping, dict):
            for key, value in raw_mapping.items():
                lowered_key = str(key).lower()
                if "layout" not in lowered_key or "visual" not in lowered_key:
                    continue
                for item in cls._normalize_layout_visualization_value(value):
                    if item not in seen:
                        seen.add(item)
                        visualizations.append(item)

        return visualizations or None

    @staticmethod
    def _build_text_from_layout(layout: list[dict[str, object]] | None) -> str:
        if not layout:
            return ""

        page_texts: list[str] = []
        for page_index, page in enumerate(layout, start=1):
            regions = page.get("regions")
            if not isinstance(regions, list):
                continue

            region_texts = [
                str(region.get("content", "")).strip()
                for region in regions
                if isinstance(region, dict) and str(region.get("content", "")).strip()
            ]
            if not region_texts:
                continue

            page_text = "\n".join(region_texts)
            if len(layout) > 1:
                page_number = page.get("page_number") or page_index
                page_texts.append(f"--- Seite {page_number} ---\n{page_text}")
            else:
                page_texts.append(page_text)

        return "\n\n".join(page_texts).strip()

    @staticmethod
    def _build_page_texts_from_layout(layout: list[dict[str, object]] | None) -> list[str] | None:
        if not layout:
            return None

        page_texts: list[str] = []
        for page in layout:
            if not isinstance(page, dict):
                continue
            regions = page.get("regions")
            if not isinstance(regions, list):
                page_texts.append("")
                continue
            region_texts = [
                str(region.get("content", "")).strip()
                for region in regions
                if isinstance(region, dict) and str(region.get("content", "")).strip()
            ]
            page_texts.append("\n".join(region_texts).strip())
        return page_texts

    def _get_table_recognizer(self) -> Any:
        if self._table_recognizer is None:
            from app.services.table_structure_recognizer import (
                TableStructureRecognizer,
            )

            self._table_recognizer = TableStructureRecognizer()
            self._table_recognizer.start()
        return self._table_recognizer

    @classmethod
    def _is_table_label(cls, label: str) -> bool:
        return cls._classify_layout_label(label) == "table"

    @staticmethod
    def _parse_table_content(text: str) -> list[list[str]]:
        """Parse a table (HTML or markdown) into a list of rows, each a list of cell strings."""
        import re
        from html import unescape

        # Try HTML table parse: look for <tr>...</tr> rows with <td>/<th> cells.
        if "<tr" in text.lower():
            rows: list[list[str]] = []
            for tr_match in re.finditer(
                r"<tr[^>]*>(.*?)</tr>",
                text,
                re.IGNORECASE | re.DOTALL,
            ):
                tr_html = tr_match.group(1)
                cells_text: list[str] = []
                for cell_match in re.finditer(
                    r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>",
                    tr_html,
                    re.IGNORECASE | re.DOTALL,
                ):
                    cell_val = re.sub(r"<[^>]+>", "", cell_match.group(1))
                    cells_text.append(unescape(cell_val).strip())
                if cells_text:
                    rows.append(cells_text)
            if rows:
                return rows

        # Try markdown table parse: lines starting with |.
        rows = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= {"-", ":", " "} for c in cells if c):
                continue
            rows.append(cells)
        return rows

    @staticmethod
    def _fill_cell_texts(
        cells: list[dict[str, object]],
        content: str,
    ) -> None:
        """Assign OCR text from *content* to cells by matching row/column indices."""
        parsed_rows = GLMOCRExpertPipeline._parse_table_content(content)
        if not parsed_rows:
            # Fallback: split by newlines, treat each line as a row.
            parsed_rows = [[line.strip()] for line in content.splitlines() if line.strip()]
        if not parsed_rows:
            return
        logger.info(
            "Parsed %d rows from table content (first row: %s)",
            len(parsed_rows),
            parsed_rows[0] if parsed_rows else "[]",
        )
        for cell in cells:
            row_idx = cell.get("row", -1)
            col_idx = cell.get("column", -1)
            if not isinstance(row_idx, int) or not isinstance(col_idx, int):
                continue
            if 0 <= row_idx < len(parsed_rows):
                row_cells = parsed_rows[row_idx]
                if 0 <= col_idx < len(row_cells):
                    cell["content"] = row_cells[col_idx]

    def _enrich_table_regions(
        self,
        layout: list[dict[str, object]],
        image: Image.Image,
    ) -> list[dict[str, object]]:
        """Add cell-level structure to table regions in *layout*."""
        img_w, img_h = image.size
        if img_w == 0 or img_h == 0:
            return layout

        recognizer = self._get_table_recognizer()
        for page in layout:
            if not isinstance(page, dict):
                continue
            regions = page.get("regions")
            if not isinstance(regions, list):
                continue
            for region in regions:
                if not isinstance(region, dict):
                    continue
                label = str(region.get("label") or "")
                is_table = self._is_table_label(label)
                logger.info(
                    "Region label=%r is_table=%s",
                    label,
                    is_table,
                )
                if not is_table:
                    continue
                bbox = region.get("bbox_2d")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    logger.warning("Table region has invalid bbox: %r", bbox)
                    continue
                try:
                    # Denormalize from 0-1000 to pixel coords.
                    x1 = float(bbox[0]) / 1000 * img_w
                    y1 = float(bbox[1]) / 1000 * img_h
                    x2 = float(bbox[2]) / 1000 * img_w
                    y2 = float(bbox[3]) / 1000 * img_h
                    x1 = max(0, min(x1, img_w))
                    y1 = max(0, min(y1, img_h))
                    x2 = max(0, min(x2, img_w))
                    y2 = max(0, min(y2, img_h))
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        logger.warning(
                            "Table crop too small: %.1fx%.1f",
                            x2 - x1,
                            y2 - y1,
                        )
                        continue
                    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
                    logger.info(
                        "Running table structure recognition on %dx%d crop",
                        crop.width,
                        crop.height,
                    )
                    cells = recognizer.recognize(crop)
                    logger.info(
                        "Table recognizer returned %d cells",
                        len(cells),
                    )
                    if not cells:
                        continue
                    # Convert crop-relative pixel coords back to 0-1000.
                    for cell in cells:
                        cb = cell.get("bbox_2d")
                        if isinstance(cb, list) and len(cb) == 4:
                            abs_x1 = x1 + cb[0]
                            abs_y1 = y1 + cb[1]
                            abs_x2 = x1 + cb[2]
                            abs_y2 = y1 + cb[3]
                            cell["bbox_2d"] = [
                                abs_x1 / img_w * 1000,
                                abs_y1 / img_h * 1000,
                                abs_x2 / img_w * 1000,
                                abs_y2 / img_h * 1000,
                            ]
                            cell["polygon"] = [
                                cell["bbox_2d"][0],
                                cell["bbox_2d"][1],
                                cell["bbox_2d"][2],
                                cell["bbox_2d"][1],
                                cell["bbox_2d"][2],
                                cell["bbox_2d"][3],
                                cell["bbox_2d"][0],
                                cell["bbox_2d"][3],
                            ]
                    # Map OCR text from region content into cells.
                    content = str(region.get("content") or "")
                    if content:
                        self._fill_cell_texts(cells, content)
                    region["cells"] = cells
                except Exception as _exc:  # noqa: BLE001
                    logger.warning(
                        "Table structure recognition failed for region: %s",
                        _exc,
                    )
        return layout

    async def _fallback_to_direct(
        self,
        *,
        reason: str,
        image_bytes: bytes,
        content_type: str | None,
        mode: str,
        schema_name: str | None,
        model: str | None,
        task: str | None,
        custom_prompt: str | None,
        token_limit: int | None,
        gif_max_frames: int | None,
        expert_enable_layout: bool | None,
        expert_layout_model: str | None = None,
    ) -> OCRResult:
        result = await self.direct_pipeline.run(
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
        )
        result.warnings.append(reason)
        return result

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
    ) -> OCRResult:
        selected_task = (task or PLAIN_TASK_OCR_TEXT).strip()
        selected_model = (model or "").strip() or self.default_model
        selected_enable_layout = (
            self.enable_layout if expert_enable_layout is None else expert_enable_layout
        )
        selected_layout_model = (expert_layout_model or "").strip() or self.layout_model

        if mode != "plain":
            return await self._fallback_to_direct(
                reason="Expert-Backend unterstützt derzeit nur mode=plain; direkte Pipeline wurde verwendet.",
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
            )

        if selected_task != PLAIN_TASK_OCR_TEXT or (custom_prompt and custom_prompt.strip()):
            return await self._fallback_to_direct(
                reason=(
                    "Expert-Backend unterstützt derzeit nur ocr_text ohne custom_prompt; "
                    "direkte Pipeline wurde verwendet."
                ),
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
            )

        if content_type == "image/gif":
            return await self._fallback_to_direct(
                reason=(
                    "Animierte GIF-Verarbeitung bleibt in der direkten Pipeline, "
                    "um Storyboard/GIF-Frame-Steuerung zu nutzen."
                ),
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
            )

        suffix = _CONTENT_TYPE_SUFFIX_MAP.get(content_type or "", ".bin")
        temp_path: Path | None = None
        start = time.perf_counter()
        try:
            with NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_path = Path(temp_file.name)

            parser = self._get_parser(
                model=selected_model,
                enable_layout=selected_enable_layout,
                layout_model=selected_layout_model,
            )
            parse_result = parser.parse(
                str(temp_path),
                save_results=False,
                save_layout_visualization=False,
            )

            layout = self._extract_layout(parse_result)
            layout = self._apply_page_region_scores_to_layout(
                layout, self._extract_page_region_scores_from_parse_result(parse_result)
            )
            table_enrich_warning: str | None = None
            if selected_enable_layout and layout and temp_path is not None:
                try:
                    with Image.open(temp_path) as img:
                        img_rgb = img.convert("RGB")
                        layout = self._enrich_table_regions(layout, img_rgb)
                except Exception as exc:  # noqa: BLE001
                    table_enrich_warning = f"Tabellenstruktur-Erkennung fehlgeschlagen: {exc}"
            page_infos = self._extract_page_infos(parse_result, layout=layout)
            text = self._extract_markdown(parse_result)
            layout_visualizations = (
                self._extract_layout_visualizations(parse_result)
                if selected_enable_layout
                else None
            )
            page_texts = self._build_page_texts_from_layout(layout)
            warnings: list[str] = []
            if table_enrich_warning:
                warnings.append(table_enrich_warning)
            if not text:
                text = self._build_text_from_layout(layout)
                if text:
                    warnings.append(
                        "Leere Markdown-Hülle entfernt; Text aus Layout-Regionen rekonstruiert."
                    )
            if not text:
                raise OllamaError("Expert-Backend hat keinen OCR-Text zurückgegeben")
            error_message = self._extract_error_message(parse_result)
            if error_message:
                warnings.append(f"Expert-Backend Hinweis: {error_message}")
            if token_limit is not None:
                warnings.append("token_limit wird vom Expert-Backend nicht verwendet.")
            if gif_max_frames is not None:
                warnings.append("gif_max_frames ist nur für GIF-Verarbeitung relevant.")
            if layout:
                region_count = sum(len(page.get("regions", [])) for page in layout)
                warnings.append(
                    f"Expert-Layout: {region_count} Regionen auf {len(layout)} Seite(n) erkannt."
                )
            elif selected_enable_layout:
                warnings.append(
                    "Expert-Layout war aktiviert, aber GLM-OCR hat keine Layout-Regionen geliefert."
                )
            if layout_visualizations:
                warnings.append(
                    f"Expert-Layout-Visualisierung: {len(layout_visualizations)} Ansicht(en) verfügbar."
                )
            elif selected_enable_layout:
                warnings.append(
                    "Interne GLM-OCR-Layout-Visualisierung bleibt wegen einer Upstream-Inkompatibilität deaktiviert."
                )
            warnings.append(
                f"Expert-Backend wurde mit enable_layout={str(selected_enable_layout).lower()} ausgeführt."
            )

            latency_ms = int((time.perf_counter() - start) * 1000)
            return OCRResult(
                text=text,
                structured=None,
                model=selected_model,
                mode=mode,
                schema_name=None,
                latency_ms=latency_ms,
                warnings=warnings,
                layout=layout,
                layout_visualizations=layout_visualizations,
                page_infos=page_infos,
                page_texts=page_texts or [text],
                markdown=text if text else None,
            )
        except OllamaError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"Expert-Backend Anfrage fehlgeschlagen: {exc}") from exc
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
