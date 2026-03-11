from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import cast
from uuid import uuid4

import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, Response

from app.config import get_settings
from app.schemas import SCHEMA_REGISTRY
from app.services.analyze_operation_store import AnalyzeOperationStore
from app.services.backend_router import OCRBackendRouter
from app.services.ollama_client import OllamaClient, OllamaError

router = APIRouter(prefix="/api")
compat_router = APIRouter()
API_VERSION = "2026-03-09-preview"
STRING_INDEX_TYPE = "textElements"
AZURE_API_VERSION = "2022-08-31"
AZURE_MODEL_ID = "prebuilt-read"
SUPPORTED_STRING_INDEX_TYPES = {"textElements", "unicodeCodePoint", "utf16CodeUnit"}
ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/tif",
    "image/tiff",
    "image/x-tiff",
    "application/pdf",
}
_WORD_RE = re.compile(r"\S+")


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    normalized = content_type.split(";", 1)[0].strip().lower()
    return normalized or None


def _sniff_content_type(payload: bytes) -> str | None:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if payload.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return "image/tiff"
    if payload.startswith(b"%PDF-"):
        return "application/pdf"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return None


def _resolve_effective_content_type(content_type: str | None, payload: bytes) -> str:
    normalized = _normalize_content_type(content_type)
    if normalized in ALLOWED_MIME_TYPES:
        return cast(str, normalized)
    if normalized in {None, "application/octet-stream"}:
        sniffed = _sniff_content_type(payload)
        if sniffed is not None:
            return sniffed
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "application/octet-stream konnte keinem unterstützten Bild- oder PDF-Typ "
                "zugeordnet werden."
            ),
        )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Nicht unterstützter Datei-Inhaltstyp: {normalized}",
    )


def _query_param(request: Request, name: str) -> str | None:
    value = request.query_params.get(name)
    if value is None or value == "":
        return None
    return value


def _resolve_text_param(form_value: object, query_value: str | None, default: str | None) -> str | None:
    if isinstance(form_value, str) and form_value != "":
        return form_value
    if query_value is not None:
        return query_value
    return default


def _resolve_int_param(form_value: object, query_value: str | None, field_name: str) -> int | None:
    if isinstance(form_value, int) and not isinstance(form_value, bool):
        return form_value
    if query_value is None:
        return None
    try:
        return int(query_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ungültiger Integer-Parameter: {field_name}",
        ) from exc


def _resolve_bool_param(
    form_value: object, query_value: str | None, field_name: str
) -> bool | None:
    if isinstance(form_value, bool):
        return form_value
    if query_value is None:
        return None
    normalized = query_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Ungültiger Boolean-Parameter: {field_name}",
    )


def _page_number(value: object, fallback: int) -> int:
    return value if isinstance(value, int) and value > 0 else fallback


def _coerce_confidence(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        confidence = float(value)
    elif isinstance(value, str):
        if not value.strip():
            return None
        try:
            confidence = float(value)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(confidence):
        return None
    if confidence < 0.0 or confidence > 1.0:
        return None
    return confidence


def _default_confidence(value: object) -> float:
    confidence = _coerce_confidence(value)
    return confidence if confidence is not None else 0.0


def _bbox_to_polygon(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(point, (int, float)) for point in value):
        return None
    x1, y1, x2, y2 = [float(point) for point in value]
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _coerce_polygon(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None

    if value and all(isinstance(point, (int, float)) for point in value):
        if len(value) < 8 or len(value) % 2 != 0:
            return None
        try:
            return [float(point) for point in value]
        except (TypeError, ValueError):
            return None

    polygon: list[float] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        if not all(isinstance(coordinate, (int, float)) for coordinate in point):
            return None
        polygon.extend((float(point[0]), float(point[1])))

    if len(polygon) < 8 or len(polygon) % 2 != 0:
        return None
    return polygon


def _bbox_to_rect(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(point, (int, float)) for point in value):
        return None
    x1, y1, x2, y2 = [float(point) for point in value]
    return x1, y1, x2, y2


def _rect_to_polygon(*, x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _empty_polygon() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _slice_line_rect(
    region_rect: tuple[float, float, float, float] | None,
    *,
    line_index: int,
    line_count: int,
) -> tuple[float, float, float, float] | None:
    if region_rect is None:
        return None
    x1, y1, x2, y2 = region_rect
    if line_count <= 1:
        return x1, y1, x2, y2
    total_height = max(y2 - y1, 0.0)
    line_y1 = y1 + (total_height * line_index / line_count)
    line_y2 = y1 + (total_height * (line_index + 1) / line_count)
    return x1, line_y1, x2, line_y2


def _split_page_paragraphs(page_text: str) -> list[str]:
    normalized = page_text.strip()
    if not normalized:
        return []

    paragraph_blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    if len(paragraph_blocks) > 1:
        return paragraph_blocks

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if 1 < len(lines) <= 12:
        return lines
    return [normalized]


def _page_entry_base(page_info: object, page_index: int) -> dict[str, object]:
    info = page_info if isinstance(page_info, dict) else {}
    page_entry: dict[str, object] = {
        "pageNumber": _page_number(info.get("page_number"), page_index),
        "angle": float(info["angle"]) if isinstance(info.get("angle"), (int, float)) else 0.0,
        "width": info["width"] if isinstance(info.get("width"), (int, float)) else 0,
        "height": info["height"] if isinstance(info.get("height"), (int, float)) else 0,
        "unit": info["unit"].strip()
        if isinstance(info.get("unit"), str) and info["unit"].strip()
        else "pixel",
        "words": info["words"] if isinstance(info.get("words"), list) else [],
        "lines": info["lines"] if isinstance(info.get("lines"), list) else [],
        "spans": info["spans"] if isinstance(info.get("spans"), list) else [],
        "kind": info["kind"].strip()
        if isinstance(info.get("kind"), str) and info["kind"].strip()
        else "document",
    }
    return page_entry


def _page_content_from_layout(page: object) -> str:
    if not isinstance(page, dict):
        return ""
    regions = page.get("regions")
    if not isinstance(regions, list):
        return ""
    page_content = [
        str(region.get("content") or "").strip()
        for region in regions
        if isinstance(region, dict) and str(region.get("content") or "").strip()
    ]
    return "\n".join(page_content).strip()


def _string_unit_length(text: str, string_index_type: str) -> int:
    if string_index_type == "utf16CodeUnit":
        return len(text.encode("utf-16-le")) // 2
    return len(text)


def _make_span(*, offset: int, text: str, string_index_type: str) -> dict[str, int]:
    return {
        "offset": offset,
        "length": _string_unit_length(text, string_index_type),
    }


def _locate_span_in_page_content(
    *,
    page_content: str,
    fragment: str,
    page_offset: int,
    string_index_type: str,
    search_cursor: int,
) -> tuple[dict[str, int], int]:
    if not fragment:
        return {"offset": page_offset, "length": 0}, search_cursor

    position = page_content.find(fragment, search_cursor)
    if position < 0:
        position = page_content.find(fragment)
    if position < 0:
        position = min(search_cursor, len(page_content))

    return (
        _make_span(
            offset=page_offset + _string_unit_length(page_content[:position], string_index_type),
            text=fragment,
            string_index_type=string_index_type,
        ),
        position + len(fragment),
    )


def _build_line_and_word_entries(
    *,
    page_content: str,
    page_offset: int,
    page_layout: object,
    string_index_type: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lines: list[dict[str, object]] = []
    words: list[dict[str, object]] = []
    search_cursor = 0

    regions = page_layout.get("regions") if isinstance(page_layout, dict) else None
    if isinstance(regions, list) and regions:
        for region in regions:
            if not isinstance(region, dict):
                continue
            region_content = str(region.get("content") or "").strip()
            if not region_content:
                continue
            region_rect = _bbox_to_rect(region.get("bbox_2d"))
            region_polygon = _coerce_polygon(region.get("polygon"))
            raw_confidence = region.get("confidence")
            if raw_confidence is None:
                raw_confidence = region.get("score")
            region_confidence = _default_confidence(raw_confidence)
            segments = [line.strip() for line in region_content.splitlines() if line.strip()] or [
                region_content
            ]
            for line_index, segment in enumerate(segments):
                line_rect = _slice_line_rect(region_rect, line_index=line_index, line_count=len(segments))
                if region_polygon is not None and len(segments) == 1:
                    polygon = region_polygon
                else:
                    polygon = (
                        _empty_polygon()
                        if line_rect is None
                        else _rect_to_polygon(
                            x1=line_rect[0],
                            y1=line_rect[1],
                            x2=line_rect[2],
                            y2=line_rect[3],
                        )
                    )
                line_span, search_cursor = _locate_span_in_page_content(
                    page_content=page_content,
                    fragment=segment,
                    page_offset=page_offset,
                    string_index_type=string_index_type,
                    search_cursor=search_cursor,
                )
                line_entry: dict[str, object] = {
                    "content": segment,
                    "spans": [line_span],
                    "confidence": region_confidence,
                    "polygon": polygon,
                }
                lines.append(line_entry)
                for word_match in _WORD_RE.finditer(segment):
                    word_content = word_match.group(0)
                    word_entry: dict[str, object] = {
                        "content": word_content,
                        "span": _make_span(
                            offset=line_span["offset"]
                            + _string_unit_length(
                                segment[: word_match.start()], string_index_type
                            ),
                            text=word_content,
                            string_index_type=string_index_type,
                        ),
                        "confidence": region_confidence,
                        "polygon": _empty_polygon(),
                    }
                    if line_rect is not None:
                        x1, y1, x2, y2 = line_rect
                        line_width = max(x2 - x1, 0.0)
                        segment_length = max(len(segment), 1)
                        word_x1 = x1 + (line_width * word_match.start() / segment_length)
                        word_x2 = x1 + (line_width * word_match.end() / segment_length)
                        word_entry["polygon"] = _rect_to_polygon(
                            x1=word_x1,
                            y1=y1,
                            x2=word_x2,
                            y2=y2,
                        )
                    words.append(word_entry)
        return lines, words

    for segment in [line.strip() for line in page_content.splitlines() if line.strip()]:
        line_span, search_cursor = _locate_span_in_page_content(
            page_content=page_content,
            fragment=segment,
            page_offset=page_offset,
            string_index_type=string_index_type,
            search_cursor=search_cursor,
        )
        lines.append(
            {
                "content": segment,
                "spans": [line_span],
                "confidence": 0.0,
                "polygon": _empty_polygon(),
            }
        )
        for word_match in _WORD_RE.finditer(segment):
            word_content = word_match.group(0)
            words.append(
                {
                    "content": word_content,
                    "span": _make_span(
                        offset=line_span["offset"]
                        + _string_unit_length(segment[: word_match.start()], string_index_type),
                        text=word_content,
                        string_index_type=string_index_type,
                    ),
                    "confidence": 0.0,
                    "polygon": _empty_polygon(),
                }
            )
    return lines, words


def _build_paragraph_entries_for_page(
    *,
    page_number: int,
    page_content: str,
    page_offset: int,
    page_layout: object,
    string_index_type: str,
) -> list[dict[str, object]]:
    paragraphs: list[dict[str, object]] = []
    search_cursor = 0

    regions = page_layout.get("regions") if isinstance(page_layout, dict) else None
    if isinstance(regions, list) and regions:
        for region in regions:
            if not isinstance(region, dict):
                continue
            region_content = str(region.get("content") or "").strip()
            if not region_content:
                continue
            paragraph_span, search_cursor = _locate_span_in_page_content(
                page_content=page_content,
                fragment=region_content,
                page_offset=page_offset,
                string_index_type=string_index_type,
                search_cursor=search_cursor,
            )
            paragraph: dict[str, object] = {
                "content": region_content,
                "spans": [paragraph_span],
                "boundingRegions": [],
            }
            polygon = _coerce_polygon(region.get("polygon")) or _bbox_to_polygon(region.get("bbox_2d"))
            if polygon is not None:
                paragraph["boundingRegions"] = [
                    {
                        "pageNumber": page_number,
                        "polygon": polygon,
                    }
                ]
            paragraphs.append(paragraph)
        return paragraphs

    for paragraph_text in _split_page_paragraphs(page_content):
        paragraph_span, search_cursor = _locate_span_in_page_content(
            page_content=page_content,
            fragment=paragraph_text,
            page_offset=page_offset,
            string_index_type=string_index_type,
            search_cursor=search_cursor,
        )
        paragraphs.append(
            {
                "content": paragraph_text,
                "spans": [paragraph_span],
                "boundingRegions": [],
            }
        )
    return paragraphs


def _build_document_projection(
    *,
    page_infos: list[dict[str, object]] | None,
    page_texts: list[str] | None,
    layout: list[dict[str, object]] | None,
    content: str,
    string_index_type: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    page_count = max(len(page_infos or []), len(page_texts or []), len(layout or []))
    if page_count == 0:
        if not content.strip():
            return [], []
        return (
            [
                {
                    **_page_entry_base({}, 1),
                    "content": content.strip(),
                    "spans": [_make_span(offset=0, text=content.strip(), string_index_type=string_index_type)],
                    "lines": [],
                    "words": [],
                }
            ],
            [{"content": content.strip(), "spans": [_make_span(offset=0, text=content.strip(), string_index_type=string_index_type)]}],
        )

    page_contexts: list[dict[str, object]] = []
    for page_index in range(1, page_count + 1):
        page_info = (page_infos or [])[page_index - 1] if page_index - 1 < len(page_infos or []) else {}
        page_layout = (layout or [])[page_index - 1] if page_index - 1 < len(layout or []) else None
        raw_page_text = (
            (page_texts or [])[page_index - 1]
            if page_index - 1 < len(page_texts or [])
            else _page_content_from_layout(page_layout)
        )
        page_text = str(raw_page_text or "").strip()
        page_number = _page_number(
            page_info.get("page_number") if isinstance(page_info, dict) else None,
            page_index,
        )
        page_contexts.append(
            {
                "page_index": page_index,
                "page_number": page_number,
                "page_info": page_info,
                "page_layout": page_layout,
                "page_content": page_text,
            }
        )

    offsets: list[int] = []
    current_offset = 0
    has_previous_content = False
    for page_context in page_contexts:
        page_content = str(page_context["page_content"])
        if page_content and has_previous_content:
            current_offset += _string_unit_length("\n\n", string_index_type)
        offsets.append(current_offset)
        if page_content:
            current_offset += _string_unit_length(page_content, string_index_type)
            has_previous_content = True

    pages: list[dict[str, object]] = []
    paragraphs: list[dict[str, object]] = []
    for page_context, page_offset in zip(page_contexts, offsets, strict=False):
        page_index = cast(int, page_context["page_index"])
        page_number = cast(int, page_context["page_number"])
        page_info = page_context["page_info"]
        page_layout = page_context["page_layout"]
        page_content = cast(str, page_context["page_content"])

        page_entry = _page_entry_base(page_info, page_index)
        page_entry["content"] = page_content
        if page_content:
            page_entry["spans"] = [
                _make_span(
                    offset=page_offset,
                    text=page_content,
                    string_index_type=string_index_type,
                )
            ]
            lines, words = _build_line_and_word_entries(
                page_content=page_content,
                page_offset=page_offset,
                page_layout=page_layout,
                string_index_type=string_index_type,
            )
            page_entry["lines"] = lines
            page_entry["words"] = words
            paragraphs.extend(
                _build_paragraph_entries_for_page(
                    page_number=page_number,
                    page_content=page_content,
                    page_offset=page_offset,
                    page_layout=page_layout,
                    string_index_type=string_index_type,
                )
            )
        else:
            page_entry["lines"] = []
            page_entry["words"] = []
            page_entry["spans"] = []
        pages.append(page_entry)

    return pages, paragraphs


def _build_analyze_result(
    *,
    content: str,
    model_id: str,
    layout: list[dict[str, object]] | None,
    page_infos: list[dict[str, object]] | None,
    page_texts: list[str] | None,
    api_version: str = API_VERSION,
    string_index_type: str = STRING_INDEX_TYPE,
) -> dict[str, object]:
    pages, paragraphs = _build_document_projection(
        page_infos=page_infos,
        page_texts=page_texts,
        layout=layout,
        content=content,
        string_index_type=string_index_type,
    )
    return {
        "apiVersion": api_version,
        "modelId": model_id,
        "stringIndexType": string_index_type,
        "content": content,
        "pages": pages,
        "paragraphs": paragraphs,
        "styles": [],
        "languages": [],
    }


def get_ocr_backend_router(request: Request) -> OCRBackendRouter:
    return cast(OCRBackendRouter, request.app.state.ocr_backend_router)


def get_ollama_client(request: Request) -> OllamaClient:
    return cast(OllamaClient, request.app.state.ollama_client)


def get_analyze_operation_store(request: Request) -> AnalyzeOperationStore:
    return cast(AnalyzeOperationStore, request.app.state.analyze_operation_store)


def _service_status_payload() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "prebuilt-read",
        "apiStatus": "Healthy",
        "apiStatusMessage": "Service is running.",
    }


def _usage_logs_payload() -> dict[str, object]:
    return {
        "apiType": "prebuilt-read",
        "serviceName": "ocr-demo",
        "type": "UsageLogs",
        "meters": [],
    }


def _validate_azure_model_id(model_id: str) -> str:
    if model_id != AZURE_MODEL_ID:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unbekanntes Modell: {model_id}",
        )
    return model_id


def _validate_azure_api_version(api_version: str) -> str:
    if api_version != AZURE_API_VERSION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Nicht unterstützte api-version: {api_version}",
        )
    return api_version


def _normalize_string_index_type(string_index_type: str | None) -> str:
    if string_index_type is None:
        return "textElements"
    if string_index_type not in SUPPORTED_STRING_INDEX_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Nicht unterstützter stringIndexType: {string_index_type}",
        )
    return string_index_type


def _parse_pages_spec(pages: str | None) -> set[int] | None:
    if pages is None or not pages.strip():
        return None

    selected_pages: set[int] = set()
    for part in pages.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            try:
                start_page = int(start_text)
                end_page = int(end_text)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ungültiger Seitenbereich: {chunk}",
                ) from exc
            if start_page < 1 or end_page < start_page:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ungültiger Seitenbereich: {chunk}",
                )
            selected_pages.update(range(start_page, end_page + 1))
            continue
        try:
            page_number = int(chunk)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ungültige Seitenauswahl: {chunk}",
            ) from exc
        if page_number < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ungültige Seitenauswahl: {chunk}",
            )
        selected_pages.add(page_number)
    return selected_pages or None


def _filter_result_pages(
    *,
    selected_pages: set[int] | None,
    layout: list[dict[str, object]] | None,
    page_infos: list[dict[str, object]] | None,
    page_texts: list[str] | None,
) -> tuple[list[dict[str, object]] | None, list[dict[str, object]] | None, list[str] | None]:
    if not selected_pages:
        return layout, page_infos, page_texts

    filtered_layout = None
    if layout is not None:
        filtered_layout = [
            page
            for index, page in enumerate(layout, start=1)
            if _page_number(page.get("page_number") if isinstance(page, dict) else None, index)
            in selected_pages
        ]

    filtered_page_infos = None
    if page_infos is not None:
        filtered_page_infos = [
            page_info
            for index, page_info in enumerate(page_infos, start=1)
            if _page_number(page_info.get("page_number") if isinstance(page_info, dict) else None, index)
            in selected_pages
        ]

    filtered_page_texts = None
    if page_texts is not None:
        filtered_page_texts = [
            page_text
            for index, page_text in enumerate(page_texts, start=1)
            if index in selected_pages
        ]

    return filtered_layout, filtered_page_infos, filtered_page_texts


def _content_from_page_texts_or_layout(
    page_texts: list[str] | None,
    layout: list[dict[str, object]] | None,
    fallback_text: str,
) -> str:
    if page_texts:
        normalized_pages = [page_text.strip() for page_text in page_texts if page_text.strip()]
        if normalized_pages:
            return "\n\n".join(normalized_pages)
    if layout:
        page_contents = [_page_content_from_layout(page) for page in layout]
        normalized_pages = [page_content for page_content in page_contents if page_content]
        if normalized_pages:
            return "\n\n".join(normalized_pages)
    return fallback_text


async def _resolve_compat_request_input(request: Request) -> tuple[bytes, str]:
    settings = get_settings()
    normalized_content_type = _normalize_content_type(request.headers.get("content-type"))

    if normalized_content_type == "application/json":
        raw_body = await request.body()
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ungültiger JSON-Body für Analyze-Anfrage.",
            ) from exc
        url_source = payload.get("urlSource")
        if not isinstance(url_source, str) or not url_source.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="application/json erfordert das Feld 'urlSource'.",
            )
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
            try:
                response = await client.get(url_source.strip())
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"urlSource konnte nicht geladen werden: {exc}",
                ) from exc
        image_bytes = response.content
        content_type = _resolve_effective_content_type(response.headers.get("content-type"), image_bytes)
    else:
        image_bytes = await request.body()
        content_type = _resolve_effective_content_type(request.headers.get("content-type"), image_bytes)

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Die hochgeladene Datei ist leer.",
        )
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Datei überschreitet {settings.max_upload_bytes} Bytes.",
        )
    return image_bytes, content_type


async def _run_plain_ocr(
    *,
    pipeline: OCRBackendRouter,
    image_bytes: bytes,
    content_type: str,
    backend: str | None = None,
    expert_enable_layout: bool | None = None,
) -> tuple[object, str]:
    try:
        return await pipeline.run(
            backend=backend,
            image_bytes=image_bytes,
            content_type=content_type,
            mode="plain",
            schema_name=None,
            model=None,
            task="ocr_text",
            custom_prompt=None,
            token_limit=None,
            gif_max_frames=None,
            expert_enable_layout=expert_enable_layout,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


async def _execute_compat_analyze_operation(
    *,
    request: Request,
    store: AnalyzeOperationStore,
    operation_id: str,
    started_at: datetime,
    model_id: str,
    string_index_type: str,
    selected_pages: set[int] | None,
    pipeline: OCRBackendRouter,
    image_bytes: bytes,
    content_type: str,
    backend: str | None,
    expert_enable_layout: bool | None,
) -> None:
    try:
        await store.mark_running(operation_id, started_at=datetime.now(timezone.utc))
        result, _ = await _run_plain_ocr(
            pipeline=pipeline,
            image_bytes=image_bytes,
            content_type=content_type,
            backend=backend,
            expert_enable_layout=expert_enable_layout,
        )
        completed_at = datetime.now(timezone.utc)
        payload = _build_compat_response_payload(
            started_at=started_at,
            completed_at=completed_at,
            model_id=model_id,
            string_index_type=string_index_type,
            selected_pages=selected_pages,
            result=result,
        )
        await store.mark_succeeded(operation_id, payload=payload, completed_at=completed_at)
    except HTTPException as exc:
        failed_at = datetime.now(timezone.utc)
        await store.mark_failed(
            operation_id,
            code="AnalyzeFailed",
            message=str(exc.detail),
            failed_at=failed_at,
        )
    except Exception as exc:  # noqa: BLE001
        failed_at = datetime.now(timezone.utc)
        logger = cast(logging.Logger, request.app.state.logger)
        logger.exception("Unerwarteter OCR-Fehler in compat_analyze")
        await store.mark_failed(
            operation_id,
            code="InternalServerError",
            message=str(exc),
            failed_at=failed_at,
        )


def _build_compat_response_payload(
    *,
    started_at: datetime,
    completed_at: datetime,
    model_id: str,
    string_index_type: str,
    selected_pages: set[int] | None,
    result: object,
) -> dict[str, object]:
    layout = getattr(result, "layout", None)
    page_infos = getattr(result, "page_infos", None)
    page_texts = getattr(result, "page_texts", None)
    filtered_layout, filtered_page_infos, filtered_page_texts = _filter_result_pages(
        selected_pages=selected_pages,
        layout=layout,
        page_infos=page_infos,
        page_texts=page_texts,
    )
    content = _content_from_page_texts_or_layout(filtered_page_texts, filtered_layout, getattr(result, "text", ""))
    return {
        "status": "succeeded",
        "createdDateTime": _isoformat_utc(started_at),
        "lastUpdatedDateTime": _isoformat_utc(completed_at),
        "analyzeResult": _build_analyze_result(
            content=content,
            model_id=model_id,
            layout=filtered_layout,
            page_infos=filtered_page_infos,
            page_texts=filtered_page_texts,
            api_version=AZURE_API_VERSION,
            string_index_type=string_index_type,
        ),
    }


def _operation_location(request: Request, *, model_id: str, result_id: str, api_version: str) -> str:
    base_location = str(
        request.url_for(
            "compat_get_analyze_result",
            modelId=model_id,
            rId=result_id,
        )
    )
    separator = "&" if "?" in base_location else "?"
    return f"{base_location}{separator}api-version={api_version}"


def _compat_headers(*, request_id: str | None = None, retry_after: str | None = None) -> dict[str, str]:
    headers = {"apim-request-id": request_id or str(uuid4())}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return headers


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ollama_base_url": settings.ollama_base_url,
        "default_model": settings.ollama_model,
        "default_backend": settings.ocr_backend,
        "default_token_limit": settings.default_token_limit,
    }


@router.get("/models")
async def models(client: OllamaClient = Depends(get_ollama_client)) -> dict:
    try:
        model_names = await client.list_models()
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return {"models": model_names}


@router.get("/schemas")
async def schemas() -> dict:
    return {"schemas": SCHEMA_REGISTRY}


@router.post("/ocr")
async def ocr(
    request: Request,
    file: UploadFile | None = File(None),
    mode: str | None = Form(None),
    schema_name: str | None = Form(None),
    model: str | None = Form(None),
    task: str | None = Form(None),
    custom_prompt: str | None = Form(None),
    token_limit: int | None = Form(None),
    gif_max_frames: int | None = Form(None),
    expert_enable_layout: bool | None = Form(None),
    backend: str | None = Form(None),
    pipeline: OCRBackendRouter = Depends(get_ocr_backend_router),
) -> dict:
    settings = get_settings()
    started_at = datetime.now(timezone.utc)

    mode = cast(str, _resolve_text_param(mode, _query_param(request, "mode"), "plain"))
    schema_name = _resolve_text_param(schema_name, _query_param(request, "schema_name"), None)
    model = _resolve_text_param(model, _query_param(request, "model"), None)
    task = _resolve_text_param(task, _query_param(request, "task"), None)
    custom_prompt = _resolve_text_param(custom_prompt, _query_param(request, "custom_prompt"), None)
    token_limit = _resolve_int_param(token_limit, _query_param(request, "token_limit"), "token_limit")
    gif_max_frames = _resolve_int_param(
        gif_max_frames, _query_param(request, "gif_max_frames"), "gif_max_frames"
    )
    expert_enable_layout = _resolve_bool_param(
        expert_enable_layout,
        _query_param(request, "expert_enable_layout"),
        "expert_enable_layout",
    )
    backend = _resolve_text_param(backend, _query_param(request, "backend"), None)

    if file is not None:
        image_bytes = await file.read()
        content_type = _resolve_effective_content_type(file.content_type, image_bytes)
    else:
        image_bytes = await request.body()
        content_type = _resolve_effective_content_type(request.headers.get("content-type"), image_bytes)

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Die hochgeladene Datei ist leer.",
        )
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Datei überschreitet {settings.max_upload_bytes} Bytes.",
        )

    try:
        result, selected_backend = await pipeline.run(
            backend=backend,
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
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger = cast(logging.Logger, request.app.state.logger)
        logger.exception("Unerwarteter OCR-Fehler")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unerwarteter OCR-Fehler: {exc}",
        ) from exc

    completed_at = datetime.now(timezone.utc)
    content = result.text
    analyze_result = _build_analyze_result(
        content=content,
        model_id=result.model,
        layout=result.layout,
        page_infos=getattr(result, "page_infos", None),
        page_texts=getattr(result, "page_texts", None),
    )

    return {
        "status": "succeeded",
        "createdDateTime": _isoformat_utc(started_at),
        "lastUpdatedDateTime": _isoformat_utc(completed_at),
        "analyzeResult": analyze_result,
        "text": result.text,
        "markdown": result.markdown,
        "structured": result.structured,
        "layout": result.layout,
        "layout_visualizations": result.layout_visualizations,
        "model": result.model,
        "mode": result.mode,
        "schema_name": result.schema_name,
        "backend": selected_backend,
        "latency_ms": result.latency_ms,
        "warnings": result.warnings,
    }


@compat_router.get("/ready")
@compat_router.get("/ContainerReadiness")
@compat_router.get("/ContainerLiveness")
async def compat_service_ready() -> JSONResponse:
    return JSONResponse(_service_status_payload(), headers=_compat_headers())


@compat_router.post("/authentication/renew")
async def compat_authentication_renew(token: str | None = None) -> JSONResponse:
    return JSONResponse({"status": "ok", "token": token}, headers=_compat_headers())


@compat_router.get("/records/usage-logs")
@compat_router.get("/records/usage-logs/{month}/{year}")
async def compat_usage_logs(month: str | None = None, year: str | None = None) -> JSONResponse:
    payload = _usage_logs_payload()
    if month is not None and year is not None:
        payload["month"] = month
        payload["year"] = year
    return JSONResponse(payload, headers=_compat_headers())


@compat_router.post("/formrecognizer/documentModels/{modelId}:syncAnalyze")
async def compat_sync_analyze(
    request: Request,
    modelId: str,
    api_version: str = Query(..., alias="api-version"),
    pages: str | None = Query(None),
    locale: str | None = Query(None),
    string_index_type: str | None = Query(None, alias="stringIndexType"),
    backend: str | None = Query(None),
    expert_enable_layout: bool | None = Query(None),
    pipeline: OCRBackendRouter = Depends(get_ocr_backend_router),
) -> JSONResponse:
    del locale
    model_id = _validate_azure_model_id(modelId)
    _validate_azure_api_version(api_version)
    normalized_string_index_type = _normalize_string_index_type(string_index_type)
    selected_pages = _parse_pages_spec(pages)

    started_at = datetime.now(timezone.utc)
    image_bytes, content_type = await _resolve_compat_request_input(request)
    result, _ = await _run_plain_ocr(
        pipeline=pipeline,
        image_bytes=image_bytes,
        content_type=content_type,
        backend=backend,
        expert_enable_layout=expert_enable_layout,
    )
    completed_at = datetime.now(timezone.utc)
    return JSONResponse(
        _build_compat_response_payload(
            started_at=started_at,
            completed_at=completed_at,
            model_id=model_id,
            string_index_type=normalized_string_index_type,
            selected_pages=selected_pages,
            result=result,
        ),
        headers=_compat_headers(),
    )


@compat_router.post("/formrecognizer/documentModels/{modelId}:analyze")
async def compat_analyze(
    request: Request,
    modelId: str,
    api_version: str = Query(..., alias="api-version"),
    pages: str | None = Query(None),
    locale: str | None = Query(None),
    string_index_type: str | None = Query(None, alias="stringIndexType"),
    backend: str | None = Query(None),
    expert_enable_layout: bool | None = Query(None),
    pipeline: OCRBackendRouter = Depends(get_ocr_backend_router),
    store: AnalyzeOperationStore = Depends(get_analyze_operation_store),
) -> Response:
    del locale
    model_id = _validate_azure_model_id(modelId)
    _validate_azure_api_version(api_version)
    normalized_string_index_type = _normalize_string_index_type(string_index_type)
    selected_pages = _parse_pages_spec(pages)

    started_at = datetime.now(timezone.utc)
    image_bytes, content_type = await _resolve_compat_request_input(request)
    operation = await store.create(model_id=model_id, created_at=started_at)
    asyncio.create_task(
        _execute_compat_analyze_operation(
            request=request,
            store=store,
            operation_id=operation.id,
            started_at=started_at,
            model_id=model_id,
            string_index_type=normalized_string_index_type,
            selected_pages=selected_pages,
            pipeline=pipeline,
            image_bytes=image_bytes,
            content_type=content_type,
            backend=backend,
            expert_enable_layout=expert_enable_layout,
        )
    )

    return Response(
        status_code=status.HTTP_202_ACCEPTED,
        headers={
            "Operation-Location": _operation_location(
                request,
                model_id=model_id,
                result_id=operation.id,
                api_version=api_version,
            ),
            **_compat_headers(request_id=operation.request_id, retry_after="1"),
        },
    )


@compat_router.get(
    "/formrecognizer/documentModels/{modelId}/analyzeResults/{rId}",
    name="compat_get_analyze_result",
)
async def compat_get_analyze_result(
    modelId: str,
    rId: str,
    api_version: str = Query(..., alias="api-version"),
    store: AnalyzeOperationStore = Depends(get_analyze_operation_store),
) -> JSONResponse:
    model_id = _validate_azure_model_id(modelId)
    _validate_azure_api_version(api_version)

    operation = await store.get(rId)
    if operation is None or operation.model_id != model_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analyze-Ergebnis nicht gefunden: {rId}",
        )

    if operation.status == "succeeded" and operation.payload is not None:
        return JSONResponse(
            operation.payload,
            headers=_compat_headers(request_id=operation.request_id),
        )

    response_payload: dict[str, object] = {
        "status": operation.status,
        "createdDateTime": _isoformat_utc(operation.created_at),
        "lastUpdatedDateTime": _isoformat_utc(operation.updated_at),
    }
    if operation.error is not None:
        response_payload["error"] = operation.error
    retry_after = None if operation.status in {"succeeded", "failed"} else "1"
    return JSONResponse(
        response_payload,
        headers=_compat_headers(request_id=operation.request_id, retry_after=retry_after),
    )


# Backward-compatible alias paths for external clients.
router.add_api_route("/health/", health, methods=["GET"], include_in_schema=False)
router.add_api_route("/models/", models, methods=["GET"], include_in_schema=False)
router.add_api_route("/schemas/", schemas, methods=["GET"], include_in_schema=False)
router.add_api_route("/ocr/", ocr, methods=["POST"], include_in_schema=False)

router.add_api_route("/v1/health", health, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/models", models, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/schemas", schemas, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/ocr", ocr, methods=["POST"], include_in_schema=False)
router.add_api_route("/v1/health/", health, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/models/", models, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/schemas/", schemas, methods=["GET"], include_in_schema=False)
router.add_api_route("/v1/ocr/", ocr, methods=["POST"], include_in_schema=False)

compat_router.add_api_route("/ready/", compat_service_ready, methods=["GET"], include_in_schema=False)
compat_router.add_api_route(
    "/ContainerReadiness/",
    compat_service_ready,
    methods=["GET"],
    include_in_schema=False,
)
compat_router.add_api_route(
    "/ContainerLiveness/",
    compat_service_ready,
    methods=["GET"],
    include_in_schema=False,
)
