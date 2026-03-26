"""Document analysis pipeline using HFLayoutDetector + Ollama.

Replaces the GLM-OCR expert pipeline with a standalone implementation:
layout detection via any HuggingFace model, full-page OCR as ground truth,
and per-region OCR with fuzzy matching back to the ground truth text.
"""

from __future__ import annotations

import io
import logging
import re
import time
from html import unescape
from typing import Any, cast

from rapidfuzz import fuzz

from PIL import Image

from app.services.layout_detector import HFLayoutDetector, LayoutDetectorConfig
from app.services.ocr_pipeline import (
    PLAIN_TASK_OCR_TEXT,
    OCRPipeline,
    OCRResult,
    normalize_ocr_text_output,
)
from app.services.ollama_client import OllamaClient, OllamaError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label classification
# ---------------------------------------------------------------------------

_IMAGE_LABELS = {
    "figure", "figure_caption", "image", "picture", "illustration",
    "chart", "seal", "signature", "stamp", "barcode", "qr_code", "Picture",
}
_TABLE_LABELS = {"table", "table_title", "table_caption", "table_footnote", "Table"}
_FORMULA_LABELS = {
    "formula", "formula_caption", "formula_number",
    "equation_footnote", "isolate_formula", "Formula",
}


def _classify_label(label: str) -> str:
    """Map a layout label to a task type: text, table, formula, or skip."""
    normalized = label.strip().lower()
    if not normalized:
        return "text"
    if label in _IMAGE_LABELS or normalized in _IMAGE_LABELS:
        return "skip"
    if label in _TABLE_LABELS or normalized in _TABLE_LABELS or "table" in normalized:
        return "table"
    if (
        label in _FORMULA_LABELS
        or normalized in _FORMULA_LABELS
        or "formula" in normalized
        or "equation" in normalized
    ):
        return "formula"
    return "text"


def _build_label_task_mapping(id2label: dict[int, str] | None) -> dict[str, list[str]]:
    if not id2label:
        return {}
    mapping: dict[str, list[str]] = {}
    for label in id2label.values():
        task = _classify_label(label)
        mapping.setdefault(task, []).append(label)
    return mapping


# ---------------------------------------------------------------------------
# Table HTML parsing
# ---------------------------------------------------------------------------


def _parse_table_html(text: str) -> list[list[str]]:
    """Parse HTML ``<tr><td>`` or markdown tables into rows × columns."""
    if "<tr" in text.lower():
        rows: list[list[str]] = []
        for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", text, re.IGNORECASE | re.DOTALL):
            cells: list[str] = []
            for cell_match in re.finditer(
                r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>",
                tr_match.group(1),
                re.IGNORECASE | re.DOTALL,
            ):
                val = re.sub(r"<[^>]+>", "", cell_match.group(1))
                cells.append(unescape(val).strip())
            if cells:
                rows.append(cells)
        if rows:
            return rows

    # Markdown fallback
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


def _strip_table_markup(text: str) -> str:
    """Strip HTML/markdown table markup to plain text."""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = unescape(cleaned)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line)


def _build_table_cells(
    parsed_rows: list[list[str]],
    region_bbox: list[float],
) -> list[dict[str, object]]:
    """Build cell dicts with proportionally computed bboxes."""
    if not parsed_rows:
        return []
    row_count = len(parsed_rows)
    col_count = max(len(row) for row in parsed_rows)
    if col_count == 0:
        return []

    rx1, ry1, rx2, ry2 = region_bbox
    row_h = (ry2 - ry1) / row_count
    col_w = (rx2 - rx1) / col_count

    cells: list[dict[str, object]] = []
    for r_idx, row in enumerate(parsed_rows):
        for c_idx, content in enumerate(row):
            cx1 = rx1 + c_idx * col_w
            cy1 = ry1 + r_idx * row_h
            cx2 = rx1 + (c_idx + 1) * col_w
            cy2 = ry1 + (r_idx + 1) * row_h
            cells.append({
                "row": r_idx,
                "column": c_idx,
                "content": content,
                "bbox_2d": [cx1, cy1, cx2, cy2],
                "polygon": [cx1, cy1, cx2, cy1, cx2, cy2, cx1, cy2],
                "is_header": r_idx == 0,
            })
    return cells


# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------


def _sort_reading_order(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort regions top-to-bottom, left-to-right with row banding."""
    if not regions:
        return regions
    # Band tolerance: regions within ~3% of page height are on the same line.
    tolerance = 30  # in 0-1000 normalized coords
    sorted_by_y = sorted(regions, key=lambda r: r.get("bbox_2d", [0, 0, 0, 0])[1])
    bands: list[list[dict[str, Any]]] = []
    current_band: list[dict[str, Any]] = []
    band_y = 0.0
    for region in sorted_by_y:
        bbox = region.get("bbox_2d", [0, 0, 0, 0])
        y_center = (bbox[1] + bbox[3]) / 2
        if not current_band or abs(y_center - band_y) <= tolerance:
            current_band.append(region)
            band_y = sum((r.get("bbox_2d", [0, 0, 0, 0])[1] + r.get("bbox_2d", [0, 0, 0, 0])[3]) / 2 for r in current_band) / len(current_band)
        else:
            bands.append(sorted(current_band, key=lambda r: r.get("bbox_2d", [0, 0, 0, 0])[0]))
            current_band = [region]
            band_y = y_center
    if current_band:
        bands.append(sorted(current_band, key=lambda r: r.get("bbox_2d", [0, 0, 0, 0])[0]))
    result: list[dict[str, Any]] = []
    for band in bands:
        result.extend(band)
    return result


# ---------------------------------------------------------------------------
# Ground-truth matching
# ---------------------------------------------------------------------------


def _match_to_ground_truth(
    candidate: str,
    full_text: str,
    threshold: float = 60.0,
    max_window: int = 5,
) -> str:
    """Find the best matching passage in full_text for candidate.

    Builds sliding windows of 1..max_window consecutive lines from full_text,
    scores each with token_set_ratio, and returns the best match if it exceeds
    threshold. Falls back to candidate if nothing scores high enough.
    """
    if not candidate or not full_text:
        return candidate

    lines = [l for l in full_text.splitlines() if l.strip()]
    if not lines:
        return candidate

    best_score = 0.0
    best_text = candidate

    for win_size in range(1, min(max_window + 1, len(lines) + 1)):
        for start in range(len(lines) - win_size + 1):
            window = " ".join(lines[start : start + win_size])
            score = fuzz.token_set_ratio(candidate, window)
            if score > best_score:
                best_score = score
                best_text = window

    return best_text if best_score >= threshold else candidate


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DocumentPipeline:
    """Layout-aware OCR pipeline using HFLayoutDetector + Ollama."""

    def __init__(
        self,
        *,
        direct_pipeline: OCRPipeline,
        ollama_client: OllamaClient,
        default_model: str,
        enable_layout: bool,
        layout_model: str,
        timeout_s: float,
    ) -> None:
        self.direct_pipeline = direct_pipeline
        self.ollama_client = ollama_client
        self.default_model = default_model
        self.enable_layout = enable_layout
        self.layout_model = layout_model
        self.timeout_s = timeout_s
        self._detector_cache: dict[str, HFLayoutDetector] = {}

    # ------------------------------------------------------------------
    # Detector management
    # ------------------------------------------------------------------

    def _get_detector(self, model: str, threshold: float | None = None) -> HFLayoutDetector:
        detector = self._detector_cache.get(model)
        if detector is None:
            config = LayoutDetectorConfig(
                model_dir=model,
                threshold=threshold or 0.2,
                layout_nms=True,
                layout_unclip_ratio=[1.0, 1.0],
                layout_merge_bboxes_mode="large",
                batch_size=8,
            )
            detector = HFLayoutDetector(config)
            detector.start()
            # Build task mapping from the model's labels.
            if detector.id2label:
                detector.label_task_mapping = _build_label_task_mapping(detector.id2label)
            self._detector_cache[model] = detector
        if threshold is not None:
            detector.threshold = threshold
        return detector

    # ------------------------------------------------------------------
    # Per-region OCR
    # ------------------------------------------------------------------

    def _crop_region(self, image: Image.Image, bbox: list[float]) -> bytes | None:
        """Crop a region from the image; return PNG bytes or None if too small."""
        img_w, img_h = image.size
        x1 = int(bbox[0] / 1000 * img_w)
        y1 = int(bbox[1] / 1000 * img_h)
        x2 = int(bbox[2] / 1000 * img_w)
        y2 = int(bbox[3] / 1000 * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        crop = image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    async def _ocr_region(
        self,
        image: Image.Image,
        region: dict[str, Any],
        *,
        model: str,
    ) -> str:
        """Crop a region and OCR it with the plain prompt."""
        crop_bytes = self._crop_region(image, region.get("bbox_2d", [0, 0, 0, 0]))
        if not crop_bytes:
            return ""
        raw = await self.ollama_client.run_ocr(
            image_bytes=crop_bytes,
            prompt=self.direct_pipeline.plain_prompt_template,
            model=model,
        )
        return normalize_ocr_text_output(raw)

    # ------------------------------------------------------------------
    # Page processing
    # ------------------------------------------------------------------

    async def _process_page(
        self,
        image: Image.Image,
        image_bytes: bytes,
        *,
        model: str,
        detector: HFLayoutDetector,
        page_number: int,
    ) -> tuple[dict[str, object], str, list[str]]:
        """Full-page OCR + layout detection, return (page_layout, page_text, warnings)."""
        warnings: list[str] = []

        # 1. Full-page OCR — one call, no cropping, no clipping.
        try:
            page_text = await self.ollama_client.run_ocr(
                image_bytes=image_bytes,
                prompt=self.direct_pipeline.plain_prompt_template,
                model=model,
            )
            page_text = normalize_ocr_text_output(page_text)
        except OllamaError as exc:
            warnings.append(f"Seiten-OCR fehlgeschlagen: {exc}")
            page_text = ""

        # 2. Layout detection — for structure only.
        detection_results = detector.process([image])
        raw_regions = detection_results[0] if detection_results else []
        regions = _sort_reading_order(raw_regions)

        # 3. Build layout regions — crop OCR each, fuzzy-match against ground truth.
        layout_regions: list[dict[str, object]] = []
        for idx, region in enumerate(regions):
            task_type = region.get("task_type", "text")
            label = region.get("label", "")

            layout_region: dict[str, object] = {
                "index": idx,
                "label": label,
                "bbox_2d": region.get("bbox_2d"),
                "polygon": region.get("polygon"),
                "confidence": region.get("score"),
            }

            if task_type == "skip":
                layout_region["content"] = ""
            else:
                try:
                    candidate = await self._ocr_region(image, region, model=model)
                except OllamaError as exc:
                    warnings.append(f"Region {idx} ({label}) OCR fehlgeschlagen: {exc}")
                    candidate = ""
                layout_region["content"] = (
                    _match_to_ground_truth(candidate, page_text) if candidate else ""
                )

            layout_regions.append(layout_region)

        page_layout: dict[str, object] = {
            "page_number": page_number,
            "regions": layout_regions,
        }
        return page_layout, page_text, warnings

    # ------------------------------------------------------------------
    # Public API (OCRService protocol)
    # ------------------------------------------------------------------

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
        selected_model = (model or "").strip() or self.default_model
        selected_task = (task or PLAIN_TASK_OCR_TEXT).strip()
        selected_enable_layout = (
            self.enable_layout if expert_enable_layout is None else expert_enable_layout
        )
        selected_layout_model = (expert_layout_model or "").strip() or self.layout_model

        # Fallback to direct pipeline for unsupported modes
        if mode != "plain":
            return await self._fallback(
                reason="Document-Pipeline unterstützt derzeit nur mode=plain; direkte Pipeline wurde verwendet.",
                image_bytes=image_bytes, content_type=content_type, mode=mode,
                schema_name=schema_name, model=model, task=task,
                custom_prompt=custom_prompt, token_limit=token_limit,
                gif_max_frames=gif_max_frames,
            )

        if selected_task != PLAIN_TASK_OCR_TEXT or (custom_prompt and custom_prompt.strip()):
            return await self._fallback(
                reason="Document-Pipeline unterstützt derzeit nur ocr_text ohne custom_prompt; direkte Pipeline wurde verwendet.",
                image_bytes=image_bytes, content_type=content_type, mode=mode,
                schema_name=schema_name, model=model, task=task,
                custom_prompt=custom_prompt, token_limit=token_limit,
                gif_max_frames=gif_max_frames,
            )

        if content_type == "image/gif":
            return await self._fallback(
                reason="GIF-Verarbeitung bleibt in der direkten Pipeline.",
                image_bytes=image_bytes, content_type=content_type, mode=mode,
                schema_name=schema_name, model=model, task=task,
                custom_prompt=custom_prompt, token_limit=token_limit,
                gif_max_frames=gif_max_frames,
            )

        if not selected_enable_layout:
            return await self._fallback(
                reason="Layout deaktiviert; direkte Pipeline wurde verwendet.",
                image_bytes=image_bytes, content_type=content_type, mode=mode,
                schema_name=schema_name, model=model, task=task,
                custom_prompt=custom_prompt, token_limit=token_limit,
                gif_max_frames=gif_max_frames,
            )

        start = time.perf_counter()
        warnings: list[str] = []

        # Prepare pages (PDF or single image)
        pages: list[tuple[Image.Image, bytes]] = []
        if content_type == "application/pdf":
            rendered_pages, pdf_warnings = self.direct_pipeline._render_pdf_pages(image_bytes)
            warnings.extend(pdf_warnings)
            for page_bytes in rendered_pages:
                pages.append((Image.open(io.BytesIO(page_bytes)).convert("RGB"), page_bytes))
        else:
            pages.append((Image.open(io.BytesIO(image_bytes)).convert("RGB"), image_bytes))

        # Get layout detector
        detector = self._get_detector(selected_layout_model, expert_layout_threshold)

        # Process each page
        layout: list[dict[str, object]] = []
        all_page_texts: list[str] = []
        page_infos: list[dict[str, object]] = []
        for page_idx, (page_image, page_bytes) in enumerate(pages):
            page_number = page_idx + 1
            page_layout, page_text, page_warnings = await self._process_page(
                page_image,
                page_bytes,
                model=selected_model,
                detector=detector,
                page_number=page_number,
            )
            layout.append(page_layout)
            all_page_texts.append(page_text)
            warnings.extend(
                f"Seite {page_number}: {w}" if len(pages) > 1 else w
                for w in page_warnings
            )
            page_infos.append({
                "page_number": page_number,
                "angle": 0.0,
                "width": page_image.width,
                "height": page_image.height,
                "unit": "pixel",
                "kind": "document",
                "words": [],
                "lines": [],
                "spans": [],
            })

        # Assemble result
        if len(all_page_texts) <= 1:
            text = all_page_texts[0] if all_page_texts else ""
        else:
            text = "\n\n".join(
                f"--- Seite {i + 1} ---\n{t}" for i, t in enumerate(all_page_texts)
            )

        region_count = sum(len(cast(list, p.get("regions", []))) for p in layout)
        warnings.append(
            f"Document-Layout: {region_count} Regionen auf {len(layout)} Seite(n) erkannt."
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
            layout_visualizations=None,
            page_infos=page_infos,
            page_texts=all_page_texts,
            markdown=text,
        )

    async def _fallback(
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
        )
        result.warnings.append(reason)
        return result
