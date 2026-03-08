from __future__ import annotations

import base64
import copy
import importlib
import mimetypes
import time
from importlib.util import find_spec
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from app.services.ocr_pipeline import (
    PLAIN_TASK_OCR_TEXT,
    OCRPipeline,
    OCRResult,
    normalize_ocr_text_output,
)
from app.services.ollama_client import OllamaError

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
    ) -> None:
        self.direct_pipeline = direct_pipeline
        self.default_model = default_model
        self.mode = mode
        self.ocr_api_host = ocr_api_host
        self.ocr_api_port = ocr_api_port
        self.timeout_s = timeout_s
        self.enable_layout = enable_layout
        self._parser_cache: dict[tuple[str, bool], Any] = {}

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

    def _build_parser(self, *, model: str, enable_layout: bool) -> Any:
        parser_class = self._load_glmocr_class()

        try:
            parser = parser_class(
                mode=self.mode,
                model=model,
                ocr_api_host=self.ocr_api_host,
                ocr_api_port=self.ocr_api_port,
                timeout=max(1, int(self.timeout_s)),
                enable_layout=enable_layout,
            )
            self._enable_layout_score_preservation(parser)
            return parser
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"Expert-Backend konnte nicht initialisiert werden: {exc}") from exc

    def _get_parser(self, *, model: str, enable_layout: bool) -> Any:
        cache_key = (model, enable_layout)
        parser = self._parser_cache.get(cache_key)
        if parser is None:
            parser = self._build_parser(model=model, enable_layout=enable_layout)
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
                round(float(value), 4) if isinstance(value, (int, float)) else value for value in bbox
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
                region_scores.append(
                    (
                        cls._region_signature(raw_region),
                        cls._coerce_confidence(raw_region.get("score") or raw_region.get("confidence")),
                    )
                )
            page_scores.append(region_scores)
        return page_scores

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
            if not isinstance(formatted_page, dict):
                continue
            formatted_regions = formatted_page.get("regions")
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
                if formatted_region.get("score") is not None or formatted_region.get("confidence") is not None:
                    continue

                confidence = scores_by_signature.get(cls._region_signature(formatted_region))
                if confidence is None and region_index < len(positional_scores):
                    confidence = positional_scores[region_index]
                if confidence is not None:
                    formatted_region["score"] = confidence

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

        confidence = GLMOCRExpertPipeline._coerce_confidence(
            region.get("confidence") or region.get("score")
        )
        if confidence is not None:
            normalized_region["confidence"] = confidence

        if not normalized_region:
            return None
        return normalized_region

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
    def _infer_page_size_from_regions(regions: list[dict[str, object]]) -> tuple[float, float] | None:
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
            page_number = raw_page_dict.get("page_number") or raw_page_dict.get("page") or page_index
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
    ) -> OCRResult:
        selected_task = (task or PLAIN_TASK_OCR_TEXT).strip()
        selected_model = (model or "").strip() or self.default_model
        selected_enable_layout = (
            self.enable_layout if expert_enable_layout is None else expert_enable_layout
        )

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
            )

        suffix = _CONTENT_TYPE_SUFFIX_MAP.get(content_type or "", ".bin")
        temp_path: Path | None = None
        start = time.perf_counter()
        try:
            with NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_path = Path(temp_file.name)

            parser = self._get_parser(model=selected_model, enable_layout=selected_enable_layout)
            parse_result = parser.parse(
                str(temp_path),
                save_results=False,
                save_layout_visualization=selected_enable_layout,
            )

            layout = self._extract_layout(parse_result)
            page_infos = self._extract_page_infos(parse_result, layout=layout)
            text = self._extract_markdown(parse_result)
            layout_visualizations = (
                self._extract_layout_visualizations(parse_result)
                if selected_enable_layout
                else None
            )
            page_texts = self._build_page_texts_from_layout(layout)
            warnings: list[str] = []
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
