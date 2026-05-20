"""Document analysis pipeline using HFLayoutDetector + Ollama.

Replaces the GLM-OCR expert pipeline with a standalone implementation:
layout detection via any HuggingFace model, full-page OCR as source text,
per-region OCR with fuzzy matching back to the source text, and optional
Table Transformer for precise cell bounding boxes.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from html import unescape
from typing import Any, cast

from PIL import Image
from rapidfuzz import fuzz

from app.services.deskew import detect_cardinal_rotation, deskew_image
from app.services.layout_detector import HFLayoutDetector, LayoutDetectorConfig
from app.services.ocr_pipeline import (
    PLAIN_TASK_OCR_TEXT,
    OCRPipeline,
    OCRResult,
    encode_page_images,
    normalize_ocr_text_output,
)
from app.services.inference import InferenceError
from app.services.inference.registry import VisionClientRegistry
from app.services.table_structure_recognizer import TableStructureRecognizer
from app.services.word_detector import WordDetector, create_word_detector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label classification
# ---------------------------------------------------------------------------

_IMAGE_LABELS = {
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
_TABLE_LABELS = {"table", "table_title", "table_caption", "table_footnote", "Table"}
_FORMULA_LABELS = {
    "formula",
    "formula_caption",
    "formula_number",
    "equation_footnote",
    "isolate_formula",
    "Formula",
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
            cells.append(
                {
                    "row": r_idx,
                    "column": c_idx,
                    "content": content,
                    "bbox_2d": [cx1, cy1, cx2, cy2],
                    "polygon": [cx1, cy1, cx2, cy1, cx2, cy2, cx1, cy2],
                    "is_header": r_idx == 0,
                }
            )
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
            band_y = sum(
                (r.get("bbox_2d", [0, 0, 0, 0])[1] + r.get("bbox_2d", [0, 0, 0, 0])[3]) / 2
                for r in current_band
            ) / len(current_band)
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


def _match_to_source_text(
    candidate: str,
    source_text: str,
    threshold: float = 60.0,
) -> str:
    """Find the best matching passage in source_text for candidate.

    Anchors on where the first candidate line best matches in source_text, then
    takes as many lines as the candidate has. Falls back to candidate if the
    anchor score is below threshold.
    """
    if not candidate or not source_text:
        return candidate

    lines = [ln for ln in source_text.splitlines() if ln.strip()]
    if not lines:
        return candidate

    candidate_lines_list = [ln for ln in candidate.splitlines() if ln.strip()]
    if not candidate_lines_list:
        return candidate
    first_line = candidate_lines_list[0]

    # Find the line in source_text where the first candidate line matches best.
    best_start_score = 0.0
    best_start = 0
    for i, line in enumerate(lines):
        score = fuzz.ratio(first_line, line)
        if score > best_start_score:
            best_start_score = score
            best_start = i

    if best_start_score < threshold:
        return candidate

    # Sequential alignment: advance through source text one candidate line at a time.
    # For each candidate line, try joining 1..max_join consecutive source text lines
    # to handle crop OCR (one row per line) vs full-page OCR (one value per line).
    max_join = 6
    gt_pos = best_start
    for cand_line in candidate_lines_list[1:]:
        best_line_score = 0.0
        best_line_pos = gt_pos
        lookahead = min(len(lines), gt_pos + max_join * 2)
        for i in range(gt_pos, lookahead):
            for join in range(1, min(max_join + 1, len(lines) - i + 1)):
                window = " ".join(lines[i : i + join])
                score = fuzz.ratio(cand_line, window)
                if score > best_line_score:
                    best_line_score = score
                    best_line_pos = i + join - 1
        if best_line_score >= threshold:
            gt_pos = best_line_pos

    return "\n".join(lines[best_start : gt_pos + 1])


# ---------------------------------------------------------------------------
# Table Transformer cell remapping
# ---------------------------------------------------------------------------


def _remap_cells_to_page_coords(
    cells: list[dict[str, Any]],
    region_bbox: list[float],
    crop_w: int,
    crop_h: int,
) -> list[dict[str, Any]]:
    """Convert pixel-coord cell bboxes (relative to crop) to 0-1000 page coords."""
    rx1, ry1, rx2, ry2 = region_bbox
    rw = rx2 - rx1
    rh = ry2 - ry1
    remapped: list[dict[str, Any]] = []
    for cell in cells:
        cx1, cy1, cx2, cy2 = cell["bbox_2d"]
        px1 = rx1 + (cx1 / crop_w) * rw
        py1 = ry1 + (cy1 / crop_h) * rh
        px2 = rx1 + (cx2 / crop_w) * rw
        py2 = ry1 + (cy2 / crop_h) * rh
        remapped.append(
            {
                **cell,
                "bbox_2d": [px1, py1, px2, py2],
                "polygon": [px1, py1, px2, py1, px2, py2, px1, py2],
            }
        )
    return remapped


# ---------------------------------------------------------------------------
# Word-polygon content assignment
# ---------------------------------------------------------------------------


_WORD_CONTENT_SIMILARITY_THRESHOLD = 70.0


def _assign_word_content(
    word_polys: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign text content to detected word polygons.

    Detector text (DocTR / PaddleOCR per-word recognition) is the floor:
    every polygon starts with whatever the detector read. Per-region OCR
    tokens are then matched to polygons by text similarity (rapidfuzz),
    not by position. A polygon adopts a layout token only when the match
    score crosses ``_WORD_CONTENT_SIMILARITY_THRESHOLD`` — this lets
    Ollama's reading correct or polish detector text where the two
    agree, while cells whose per-region OCR was missed entirely (common
    for tables where Ollama on the cropped image collapses values) keep
    DocTR's reading instead of being overridden by a neighbouring
    token. Each layout token is consumed at most once.
    """
    result: list[dict[str, Any]] = [dict(wp) for wp in word_polys]

    # Group all polygon indices by the smallest containing region (by bbox).
    region_poly_indices: dict[int, list[int]] = {}
    for poly_idx, wp in enumerate(word_polys):
        poly = wp.get("polygon") or []
        if not poly or len(poly) < 8:
            continue

        x_c = poly[0::2]
        y_c = poly[1::2]
        px1, px2 = min(x_c), max(x_c)
        py1, py2 = min(y_c), max(y_c)
        word_area = max(1.0, (px2 - px1) * (py2 - py1))

        best: int | None = None
        best_area = float("inf")

        for r_idx, region in enumerate(regions):
            bbox = region.get("bbox_2d") or []
            if len(bbox) != 4:
                continue
            rx1, ry1, rx2, ry2 = bbox

            ix1 = max(px1, rx1)
            iy1 = max(py1, ry1)
            ix2 = min(px2, rx2)
            iy2 = min(py2, ry2)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                ioa = intersection / word_area

                # If at least 10% of the word is inside this region, consider it.
                if ioa > 0.1:
                    area = (rx2 - rx1) * (ry2 - ry1)
                    if area < best_area:
                        best = r_idx
                        best_area = area

        if best is not None:
            region_poly_indices.setdefault(best, []).append(poly_idx)

    for r_idx, indices in region_poly_indices.items():
        region_content = str(regions[r_idx].get("content") or "")
        flat_tokens = [
            t for line in region_content.splitlines() if line.strip() for t in line.split()
        ]
        if not flat_tokens:
            continue

        used_tokens: set[int] = set()
        for poly_idx in indices:
            if poly_idx >= len(result):
                continue
            detector_text = str(result[poly_idx].get("content") or "").strip()

            if detector_text:
                best_token_idx: int | None = None
                best_score = 0.0
                detector_lc = detector_text.lower()
                for token_idx, token in enumerate(flat_tokens):
                    if token_idx in used_tokens:
                        continue
                    score = fuzz.ratio(detector_lc, token.lower())
                    if score > best_score:
                        best_score = score
                        best_token_idx = token_idx
                if best_token_idx is not None and best_score >= _WORD_CONTENT_SIMILARITY_THRESHOLD:
                    result[poly_idx]["content"] = flat_tokens[best_token_idx]
                    used_tokens.add(best_token_idx)
                # else: keep detector text — region OCR didn't catch this word.
            else:
                # Detector couldn't recognize this polygon — fall back to the
                # first unused layout token (positional, since we have no
                # detector text to match against).
                for token_idx, token in enumerate(flat_tokens):
                    if token_idx not in used_tokens:
                        result[poly_idx]["content"] = token
                        used_tokens.add(token_idx)
                        break

    return [r for r in result if r and r.get("polygon")]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DocumentPipeline:
    """Layout-aware OCR pipeline using HFLayoutDetector + Ollama."""

    def __init__(
        self,
        *,
        direct_pipeline: OCRPipeline,
        vision_registry: VisionClientRegistry,
        default_model: str,
        enable_layout: bool,
        layout_model: str,
        timeout_s: float,
        enable_table_transformer: bool = False,
        enable_per_region_ocr: bool = True,
        enable_text_anchor: bool = True,
        text_anchor_threshold: float = 60.0,
        word_detector: WordDetector | None = None,
        layout_max_dim: int = 1800,
    ) -> None:
        self.direct_pipeline = direct_pipeline
        self.vision_registry = vision_registry
        self._run_client = None
        self.default_model = default_model
        self.enable_layout = enable_layout
        self.layout_model = layout_model
        self.timeout_s = timeout_s
        self.enable_table_transformer = enable_table_transformer
        self.enable_per_region_ocr = enable_per_region_ocr
        self.enable_text_anchor = enable_text_anchor
        self.text_anchor_threshold = text_anchor_threshold
        self.word_detector: WordDetector | None = word_detector
        self.layout_max_dim = max(256, int(layout_max_dim))
        # Inherited from direct_pipeline so both pipelines share one setting.
        self.deskew_enabled = bool(getattr(direct_pipeline, "deskew_enabled", False))
        self.deskew_min_angle_deg = float(
            getattr(direct_pipeline, "deskew_min_angle_deg", 0.5)
        )
        self._detector_cache: dict[str, HFLayoutDetector] = {}
        self._table_recognizer: TableStructureRecognizer | None = None
        self._word_detector_cache: dict[str, WordDetector | None] = {}

    @property
    def _vision_client(self):
        if self._run_client is not None:
            return self._run_client
        return self.vision_registry.get(self.vision_registry.default_provider)

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
    # Table Transformer
    # ------------------------------------------------------------------

    def _get_table_recognizer(self) -> TableStructureRecognizer:
        if self._table_recognizer is None:
            recognizer = TableStructureRecognizer()
            recognizer.start()
            self._table_recognizer = recognizer
        return self._table_recognizer

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

    def _crop_region_image(self, image: Image.Image, bbox: list[float]) -> Image.Image | None:
        """Crop a region from the image; return PIL Image or None if too small."""
        img_w, img_h = image.size
        x1 = int(bbox[0] / 1000 * img_w)
        y1 = int(bbox[1] / 1000 * img_h)
        x2 = int(bbox[2] / 1000 * img_w)
        y2 = int(bbox[3] / 1000 * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        return image.crop((x1, y1, x2, y2))

    async def _ocr_region(
        self,
        image: Image.Image,
        region: dict[str, Any],
        *,
        model: str,
        precomputed_bytes: bytes | None = None,
    ) -> str:
        """Crop a region and OCR it with the plain prompt.

        Pass ``precomputed_bytes`` to supply an already-corrected crop (e.g.
        after cardinal rotation); otherwise the crop is taken from ``image``.
        """
        crop_bytes = precomputed_bytes or self._crop_region(
            image, region.get("bbox_2d", [0, 0, 0, 0])
        )
        if not crop_bytes:
            return ""
        raw = await self._vision_client.run_vision_chat(
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
        use_table_transformer: bool = False,
        per_region_ocr: bool = True,
        text_anchor: bool = True,
        text_anchor_threshold: float = 60.0,
        assemble_from_regions: bool = False,
    ) -> tuple[dict[str, object], str, list[str]]:
        """Full-page OCR + layout detection, return (page_layout, page_text, warnings)."""
        warnings: list[str] = []

        # 1. Full-page OCR — one call, no cropping, no clipping.
        try:
            page_text = await self._vision_client.run_vision_chat(
                image_bytes=image_bytes,
                prompt=self.direct_pipeline.plain_prompt_template,
                model=model,
            )
            page_text = normalize_ocr_text_output(page_text)
        except InferenceError as exc:
            warnings.append(f"Seiten-OCR fehlgeschlagen: {exc}")
            page_text = ""

        # 2. Layout detection — for structure only.
        # Layout-Modelle (PP-DocLayout, Deformable DETR, RT-DETR) sind auf
        # ~800–1024 px Input trainiert. Großformatige Scans/Fotos hier nochmal
        # auf layout_max_dim runterskalieren, sonst zerfasern die Regionen.
        # Die Bboxes werden danach wieder auf die Originalauflösung skaliert,
        # damit die Per-Region-OCR auf dem hochaufgelösten Bild crops macht.
        layout_image = image
        if max(image.size) > self.layout_max_dim:
            ratio = self.layout_max_dim / max(image.size)
            new_size = (
                max(1, int(image.size[0] * ratio)),
                max(1, int(image.size[1] * ratio)),
            )
            layout_image = image.resize(new_size, Image.Resampling.LANCZOS)

        detection_results = detector.process([layout_image])
        raw_regions = detection_results[0] if detection_results else []

        regions = _sort_reading_order(raw_regions)

        # 3. Build layout regions — crop OCR each, fuzzy-match against source text.
        _CARDINAL_CONF_THRESHOLD = 0.70
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
                "region_angle": 0,
            }

            if task_type == "skip":
                layout_region["content"] = ""
            elif not per_region_ocr:
                layout_region["content"] = ""
            else:
                # Per-region cardinal orientation detection (handles multi-doc scans).
                precomputed_crop: bytes | None = None
                if self.deskew_enabled:
                    crop_img = self._crop_region_image(
                        image, region.get("bbox_2d", [0, 0, 0, 0])
                    )
                    if crop_img is not None:
                        try:
                            rot, conf = await asyncio.to_thread(
                                detect_cardinal_rotation, crop_img
                            )
                        except Exception as exc:  # noqa: BLE001
                            warnings.append(
                                f"Region {idx} Orientierungserkennung fehlgeschlagen: {exc}"
                            )
                            rot, conf = 0, 0.0
                        if rot != 0 and conf >= _CARDINAL_CONF_THRESHOLD:
                            _TRANSPOSE = {
                                90: Image.Transpose.ROTATE_90,
                                180: Image.Transpose.ROTATE_180,
                                270: Image.Transpose.ROTATE_270,
                            }
                            corrected = crop_img.transpose(_TRANSPOSE[rot])
                            buf = io.BytesIO()
                            corrected.save(buf, format="PNG")
                            precomputed_crop = buf.getvalue()
                            layout_region["region_angle"] = rot
                            warnings.append(
                                f"Region {idx} ({label}): {rot}° CCW Rotation erkannt"
                            )

                try:
                    candidate = await self._ocr_region(
                        image, region, model=model, precomputed_bytes=precomputed_crop
                    )
                except InferenceError as exc:
                    warnings.append(f"Region {idx} ({label}) OCR fehlgeschlagen: {exc}")
                    candidate = ""
                if not candidate:
                    layout_region["content"] = ""
                elif text_anchor:
                    layout_region["content"] = _match_to_source_text(
                        candidate, page_text, threshold=text_anchor_threshold
                    )
                else:
                    layout_region["content"] = candidate

                # Table Transformer: detect precise cell bboxes for table regions.
                if task_type == "table" and use_table_transformer:
                    crop_img = self._crop_region_image(image, region.get("bbox_2d", [0, 0, 0, 0]))
                    if crop_img is not None:
                        try:
                            raw_cells = self._get_table_recognizer().recognize(crop_img)
                            if raw_cells:
                                layout_region["cells"] = _remap_cells_to_page_coords(
                                    raw_cells,
                                    region.get("bbox_2d", [0, 0, 0, 0]),
                                    crop_img.width,
                                    crop_img.height,
                                )
                        except Exception as exc:  # noqa: BLE001
                            warnings.append(f"Region {idx} Table Transformer fehlgeschlagen: {exc}")

            layout_regions.append(layout_region)

        page_layout: dict[str, object] = {
            "page_number": page_number,
            "regions": layout_regions,
        }

        if assemble_from_regions:
            region_texts = [str(r.get("content", "")) for r in layout_regions if r.get("content")]
            if region_texts:
                page_text = "\n\n".join(region_texts)

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
        expert_table_transformer: bool | None = None,
        expert_per_region_ocr: bool | None = None,
        expert_text_anchor: bool | None = None,
        expert_text_anchor_threshold: float | None = None,
        expert_word_detector: str | None = None,
        expert_assemble_from_regions: bool | None = None,
        inference_provider: str | None = None,
    ) -> OCRResult:
        resolved = self.vision_registry.resolve(
            inference_provider=inference_provider,
            model=model,
        )
        self._run_client = resolved.client
        try:
            return await self._run_with_client(
                selected_model=resolved.model_id,
                inference_provider=resolved.provider_id,
                image_bytes=image_bytes,
                content_type=content_type,
                mode=mode,
                schema_name=schema_name,
                task=task,
                custom_prompt=custom_prompt,
                token_limit=token_limit,
                gif_max_frames=gif_max_frames,
                expert_enable_layout=expert_enable_layout,
                expert_layout_model=expert_layout_model,
                expert_layout_threshold=expert_layout_threshold,
                expert_table_transformer=expert_table_transformer,
                expert_per_region_ocr=expert_per_region_ocr,
                expert_text_anchor=expert_text_anchor,
                expert_text_anchor_threshold=expert_text_anchor_threshold,
                expert_word_detector=expert_word_detector,
                expert_assemble_from_regions=expert_assemble_from_regions,
            )
        finally:
            self._run_client = None

    async def _run_with_client(
        self,
        *,
        selected_model: str,
        inference_provider: str,
        image_bytes: bytes,
        content_type: str | None,
        mode: str,
        schema_name: str | None,
        task: str | None,
        custom_prompt: str | None,
        token_limit: int | None,
        gif_max_frames: int | None,
        expert_enable_layout: bool | None,
        expert_layout_model: str | None,
        expert_layout_threshold: float | None,
        expert_table_transformer: bool | None,
        expert_per_region_ocr: bool | None,
        expert_text_anchor: bool | None,
        expert_text_anchor_threshold: float | None,
        expert_word_detector: str | None,
        expert_assemble_from_regions: bool | None,
    ) -> OCRResult:
        selected_task = (task or PLAIN_TASK_OCR_TEXT).strip()
        selected_enable_layout = (
            self.enable_layout if expert_enable_layout is None else expert_enable_layout
        )
        selected_layout_model = (expert_layout_model or "").strip() or self.layout_model
        selected_table_transformer = (
            self.enable_table_transformer
            if expert_table_transformer is None
            else expert_table_transformer
        )
        selected_per_region_ocr = (
            self.enable_per_region_ocr if expert_per_region_ocr is None else expert_per_region_ocr
        )
        selected_text_anchor = (
            self.enable_text_anchor if expert_text_anchor is None else expert_text_anchor
        )
        selected_text_anchor_threshold = (
            self.text_anchor_threshold
            if expert_text_anchor_threshold is None
            else expert_text_anchor_threshold
        )
        selected_assemble_from_regions = bool(expert_assemble_from_regions)
        selected_word_detector: WordDetector | None = self.word_detector
        word_detector_warning: str | None = None
        if expert_word_detector is not None and expert_word_detector != "":
            key = expert_word_detector.strip().lower()
            if key not in self._word_detector_cache:
                try:
                    self._word_detector_cache[key] = await asyncio.to_thread(
                        create_word_detector, expert_word_detector
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Wort-Detektor-Initialisierung fehlgeschlagen: %s", exc)
                    word_detector_warning = f"Wort-Detektor nicht verfügbar: {exc}"
                    self._word_detector_cache[key] = None
            selected_word_detector = self._word_detector_cache[key]

        # Fallback to direct pipeline for unsupported modes
        if mode != "plain":
            return await self._fallback(
                reason="Document-Pipeline unterstützt derzeit nur mode=plain; direkte Pipeline wurde verwendet.",
                inference_provider=inference_provider,
                image_bytes=image_bytes,
                content_type=content_type,
                mode=mode,
                schema_name=schema_name,
                model=selected_model,
                task=task,
                custom_prompt=custom_prompt,
                token_limit=token_limit,
                gif_max_frames=gif_max_frames,
            )

        if selected_task != PLAIN_TASK_OCR_TEXT or (custom_prompt and custom_prompt.strip()):
            return await self._fallback(
                reason="Document-Pipeline unterstützt derzeit nur ocr_text ohne custom_prompt; direkte Pipeline wurde verwendet.",
                inference_provider=inference_provider,
                image_bytes=image_bytes,
                content_type=content_type,
                mode=mode,
                schema_name=schema_name,
                model=selected_model,
                task=task,
                custom_prompt=custom_prompt,
                token_limit=token_limit,
                gif_max_frames=gif_max_frames,
            )

        if content_type == "image/gif":
            return await self._fallback(
                reason="GIF-Verarbeitung bleibt in der direkten Pipeline.",
                inference_provider=inference_provider,
                image_bytes=image_bytes,
                content_type=content_type,
                mode=mode,
                schema_name=schema_name,
                model=selected_model,
                task=task,
                custom_prompt=custom_prompt,
                token_limit=token_limit,
                gif_max_frames=gif_max_frames,
            )

        if not selected_enable_layout:
            return await self._fallback(
                reason="Layout deaktiviert; direkte Pipeline wurde verwendet.",
                inference_provider=inference_provider,
                image_bytes=image_bytes,
                content_type=content_type,
                mode=mode,
                schema_name=schema_name,
                model=selected_model,
                task=task,
                custom_prompt=custom_prompt,
                token_limit=token_limit,
                gif_max_frames=gif_max_frames,
            )

        start = time.perf_counter()
        warnings: list[str] = []
        if word_detector_warning:
            warnings.append(word_detector_warning)

        # Prepare pages (PDF or single image)
        pages: list[tuple[Image.Image, bytes]] = []
        raw_page_images: list[bytes] | None = None
        if content_type == "application/pdf":
            rendered_pages, pdf_warnings = self.direct_pipeline._render_pdf_pages(image_bytes)
            warnings.extend(pdf_warnings)
            raw_page_images = list(rendered_pages)
            for page_bytes in rendered_pages:
                pages.append((Image.open(io.BytesIO(page_bytes)).convert("RGB"), page_bytes))
        else:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            pages.append((pil_img, png_bytes))
            if content_type in {"image/tif", "image/tiff", "image/x-tiff"}:
                raw_page_images = [png_bytes]

        # Get layout detector
        detector = self._get_detector(selected_layout_model, expert_layout_threshold)

        # Process each page
        layout: list[dict[str, object]] = []
        all_page_texts: list[str] = []
        page_infos: list[dict[str, object]] = []
        for page_idx, (page_image, page_bytes) in enumerate(pages):
            page_number = page_idx + 1

            # Page-level deskew: correct cardinal rotation and fine skew before
            # layout detection so regions are detected on a straight image.
            page_angle = 0.0
            if self.deskew_enabled:
                try:
                    page_image, page_angle = await asyncio.to_thread(
                        deskew_image,
                        page_image,
                        min_angle_deg=self.deskew_min_angle_deg,
                    )
                    if page_angle != 0.0:
                        buf = io.BytesIO()
                        page_image.save(buf, format="PNG")
                        page_bytes = buf.getvalue()
                        deskew_msg = f"Deskew: {page_angle:.1f}° CCW Korrektur angewendet"
                        warnings.append(
                            f"Seite {page_number}: {deskew_msg}" if len(pages) > 1 else deskew_msg
                        )
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Deskew fehlgeschlagen: {exc}")

            page_layout, page_text, page_warnings = await self._process_page(
                page_image,
                page_bytes,
                model=selected_model,
                detector=detector,
                page_number=page_number,
                use_table_transformer=selected_table_transformer,
                per_region_ocr=selected_per_region_ocr,
                assemble_from_regions=selected_assemble_from_regions,
                text_anchor=selected_text_anchor,
                text_anchor_threshold=selected_text_anchor_threshold,
            )
            if selected_word_detector is not None:
                try:
                    word_polys = await asyncio.to_thread(selected_word_detector.detect, page_image)
                    # Detector-provided text (DocTR / PaddleOCR per-word
                    # recognition) is preserved as a fallback. Per-region
                    # OCR tokens are mapped onto the polygons in reading
                    # order inside _assign_word_content and override the
                    # detector text where they cover.
                    page_layout["word_polys"] = _assign_word_content(
                        cast(list[dict[str, Any]], word_polys),
                        cast(list[dict[str, Any]], page_layout.get("regions", [])),
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Wort-Erkennung fehlgeschlagen: {exc}")
            layout.append(page_layout)
            all_page_texts.append(page_text)
            warnings.extend(
                f"Seite {page_number}: {w}" if len(pages) > 1 else w for w in page_warnings
            )
            page_infos.append(
                {
                    "page_number": page_number,
                    "angle": page_angle,
                    "width": page_image.width,
                    "height": page_image.height,
                    "unit": "pixel",
                    "kind": "document",
                    "words": [],
                    "lines": [],
                    "spans": [],
                }
            )

        # Assemble result
        if len(all_page_texts) <= 1:
            text = all_page_texts[0] if all_page_texts else ""
        else:
            text = "\n\n".join(f"--- Seite {i + 1} ---\n{t}" for i, t in enumerate(all_page_texts))

        region_count = sum(len(cast(list, p.get("regions", []))) for p in layout)
        warnings.append(
            f"Document layout: {region_count} regions detected on {len(layout)} page(s)."
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
            inference_provider=inference_provider,
            layout=layout,
            layout_visualizations=None,
            page_infos=page_infos,
            page_texts=all_page_texts,
            markdown=text,
            page_images=(
                await asyncio.to_thread(encode_page_images, raw_page_images)
                if raw_page_images
                else None
            ),
        )

    async def _fallback(
        self,
        *,
        reason: str,
        inference_provider: str,
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
            inference_provider=inference_provider,
        )
        result.warnings.append(reason)
        return result
