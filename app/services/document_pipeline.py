"""Document analysis pipeline using HFLayoutDetector + Ollama.

Replaces the GLM-OCR expert pipeline with a standalone implementation:
layout detection via any HuggingFace model, full-page OCR as source text,
per-region OCR with fuzzy matching back to the source text, and optional
Table Transformer for precise cell bounding boxes.
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
    encode_page_images,
    normalize_ocr_text_output,
)
from app.services.ollama_client import OllamaClient, OllamaError
from app.services.table_structure_recognizer import TableStructureRecognizer
from app.services.word_detector import WordDetector, create_word_detector

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

    lines = [l for l in source_text.splitlines() if l.strip()]
    if not lines:
        return candidate

    candidate_lines_list = [l for l in candidate.splitlines() if l.strip()]
    if not candidate_lines_list:
        return candidate
    first_line = candidate_lines_list[0]
    last_line = candidate_lines_list[-1]

    # Find the line in source_text where the first candidate line matches best.
    best_start_score = 0.0
    best_start = 0
    for i, line in enumerate(lines):
        score = fuzz.ratio(first_line, line)
        if score > best_start_score:
            best_start_score = score
            best_start = i

    if best_start_score < threshold:
        return ""

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
        remapped.append({
            **cell,
            "bbox_2d": [px1, py1, px2, py2],
            "polygon": [px1, py1, px2, py1, px2, py2, px1, py2],
        })
    return remapped


# ---------------------------------------------------------------------------
# Word-polygon content assignment
# ---------------------------------------------------------------------------


def _poly_centroid(polygon: list[float]) -> tuple[float, float]:
    n = max(len(polygon) // 2, 1)
    return sum(polygon[0::2]) / n, sum(polygon[1::2]) / n


def _assign_word_content(
    word_polys: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign text content to word polygons.

    If a word polygon already has content from the detector (e.g. DocTR/PaddleOCR
    recognition), keep it as-is.  For polygons without content, fall back to
    assigning tokens from the containing layout region's OCR text in reading order.
    """
    result: list[dict[str, Any]] = [dict(wp) for wp in word_polys]

    # If all polygons already carry detector-recognised content, nothing to do.
    needs_assignment = [i for i, wp in enumerate(word_polys) if not wp.get("content")]
    if not needs_assignment:
        return result

    # Group unassigned polygon indices by the region whose bbox most tightly contains them.
    region_poly_indices: dict[int, list[int]] = {}
    for poly_idx in needs_assignment:
        wp = word_polys[poly_idx]
        poly = wp["polygon"]
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
                
                # If at least 10% of the word is inside this region, consider it a candidate.
                if ioa > 0.1:
                    area = (rx2 - rx1) * (ry2 - ry1)
                    if area < best_area:
                        best = r_idx
                        best_area = area
                        
        if best is not None:
            region_poly_indices.setdefault(best, []).append(poly_idx)

    for r_idx, indices in region_poly_indices.items():
        region_content = str(regions[r_idx].get("content") or "")
        lines = [line.strip() for line in region_content.splitlines() if line.strip()]
        
        # Sort polygons by reading order using row-banding
        band_tolerance = 15  # in 0-1000 normalised coords
        centroids = {i: _poly_centroid(word_polys[i]["polygon"]) for i in indices}
        by_y = sorted(indices, key=lambda i: centroids[i][1])
        bands: list[list[int]] = []
        current: list[int] = []
        band_y = 0.0
        for i in by_y:
            cy = centroids[i][1]
            if not current or abs(cy - band_y) <= band_tolerance:
                current.append(i)
                band_y = sum(centroids[j][1] for j in current) / len(current)
            else:
                bands.append(sorted(current, key=lambda j: centroids[j][0]))
                current = [i]
                band_y = cy
        if current:
            bands.append(sorted(current, key=lambda j: centroids[j][0]))
        
        # indices_sorted is a flat list of polygon indices in this region.
        # Check if we should perform line-level proportional splitting.
        flat_tokens = [t for line in lines for t in line.split()]
        total_tokens = len(flat_tokens)
        polys_in_region = sum(len(band) for band in bands)
        
        if polys_in_region > 0 and total_tokens > polys_in_region * 1.5:
            # Reconstruct result polygons by splitting the bounding boxes
            # We pair each band (row of polygons) with a chunk of text.
            # Map each token chunk to its individual physical polygon
            valid_bands = []
            flat_poly_geometries = []
            for band_indices in bands:
                band_pgs = []
                for p_idx in band_indices:
                    poly = word_polys[p_idx]["polygon"]
                    if len(poly) >= 8:
                        x_c = poly[0::2]
                        x1, x2 = min(x_c), max(x_c)
                        poly_width = max(1.0, x2 - x1)
                        
                        # Filter out purely artefactual/noise lines
                        if poly_width >= 5:
                            pg = {
                                "width": poly_width,
                                "poly": poly,
                                "confidence": word_polys[p_idx].get("confidence", 1.0)
                            }
                            band_pgs.append(pg)
                            flat_poly_geometries.append(pg)
                    result[p_idx]["polygon"] = []  # ALWAYS clear original from result array
                if band_pgs:
                    valid_bands.append(band_pgs)
            
            def _distribute_tokens(tokens_list, pgs_list):
                if not tokens_list or not pgs_list:
                    return

                # Proportional font width approximation to prevent wide/narrow letter box drift
                def _char_width(c: str) -> float:
                    cl = c.lower()
                    if cl in "il1.,'!;:|": return 0.25
                    if cl in "tfj()[]{}": return 0.45
                    if cl in "mw": return 1.45
                    if cl in "abcdegknopqrsuyz": return 0.9
                    if cl in "xhv": return 0.95
                    return 1.0
                def _tok_width(tok: str) -> float:
                    return sum(_char_width(c) for c in tok)
                space_width = 1.6
                left_margin = 0.5
                right_margin = 2.0

                t_width = sum(pg["width"] for pg in pgs_list)
                t_chars = sum(_tok_width(t) for t in tokens_list) + len(tokens_list) * space_width + left_margin + right_margin

                t_idx = 0
                for i, pg in enumerate(pgs_list):
                    if i == len(pgs_list) - 1:
                        poly_tokens = tokens_list[t_idx:]
                    else:
                        target_chars = t_chars * (pg["width"] / max(1.0, t_width))
                        current_chars = 0
                        poly_tokens = []
                        while t_idx < len(tokens_list):
                            tok = tokens_list[t_idx]
                            tok_w = _tok_width(tok) + (space_width if current_chars > 0 else 0)
                            if current_chars > 0 and current_chars + tok_w - target_chars > target_chars - current_chars:
                                break
                            poly_tokens.append(tok)
                            current_chars += tok_w
                            t_idx += 1

                    if not poly_tokens:
                        continue

                    px1, py1, px2, py2, px3, py3, px4, py4 = pg["poly"]
                    dx_top, dy_top = px2 - px1, py2 - py1
                    dx_bot, dy_bot = px3 - px4, py3 - py4

                    b_total_len = sum(_tok_width(t) for t in poly_tokens) + max(0, len(poly_tokens) - 1) * space_width + left_margin + right_margin
                    b_current_len = left_margin
                    for tok_i, token in enumerate(poly_tokens):
                        s_r = b_current_len / max(1.0, b_total_len)
                        b_current_len += _tok_width(token)
                        e_r = b_current_len / max(1.0, b_total_len)
                        if tok_i < len(poly_tokens) - 1:
                            b_current_len += space_width
                        sub_poly = [
                            px1 + dx_top * s_r, py1 + dy_top * s_r,
                            px1 + dx_top * e_r, py1 + dy_top * e_r,
                            px4 + dx_bot * e_r, py4 + dy_bot * e_r,
                            px4 + dx_bot * s_r, py4 + dy_bot * s_r,
                        ]
                        result.append({"polygon": sub_poly, "content": token, "confidence": pg["confidence"]})

            # If Ollama provided exactly as many lines as we have visual layout bands,
            # we can perfectly match line tokens to line polygons, preventing cascading overflow!
            if len(lines) > 0 and len(valid_bands) == len(lines):
                for b_i, band_pgs in enumerate(valid_bands):
                    band_tokens = lines[b_i].split()
                    _distribute_tokens(band_tokens, band_pgs)
            else:
                # Fall back to distributing all tokens mathematically across all raw geometries
                _distribute_tokens(flat_tokens, flat_poly_geometries)
        else:
            # 1-to-1 fallback mapping
            indices_sorted = [i for band in bands for i in band]
            for token_idx, poly_idx in enumerate(indices_sorted):
                if poly_idx < len(result):
                    result[poly_idx]["content"] = flat_tokens[token_idx] if token_idx < len(flat_tokens) else ""

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
        ollama_client: OllamaClient,
        default_model: str,
        enable_layout: bool,
        layout_model: str,
        timeout_s: float,
        enable_table_transformer: bool = False,
        word_detector: WordDetector | None = None,
    ) -> None:
        self.direct_pipeline = direct_pipeline
        self.ollama_client = ollama_client
        self.default_model = default_model
        self.enable_layout = enable_layout
        self.layout_model = layout_model
        self.timeout_s = timeout_s
        self.enable_table_transformer = enable_table_transformer
        self.word_detector: WordDetector | None = word_detector
        self._detector_cache: dict[str, HFLayoutDetector] = {}
        self._table_recognizer: TableStructureRecognizer | None = None
        self._word_detector_cache: dict[str, WordDetector | None] = {}

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
            self._table_recognizer = TableStructureRecognizer()
            self._table_recognizer.start()
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
        use_table_transformer: bool = False,
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

        # 3. Build layout regions — crop OCR each, fuzzy-match against source text.
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
                    _match_to_source_text(candidate, page_text) if candidate else ""
                )

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
        expert_word_detector: str | None = None,
    ) -> OCRResult:
        selected_model = (model or "").strip() or self.default_model
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
        selected_word_detector: WordDetector | None = self.word_detector
        word_detector_warning: str | None = None
        if expert_word_detector is not None and expert_word_detector != "":
            key = expert_word_detector.strip().lower()
            if key not in self._word_detector_cache:
                try:
                    self._word_detector_cache[key] = create_word_detector(expert_word_detector)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Wort-Detektor-Initialisierung fehlgeschlagen: %s", exc)
                    word_detector_warning = f"Wort-Detektor nicht verfügbar: {exc}"
                    self._word_detector_cache[key] = None
            selected_word_detector = self._word_detector_cache[key]

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
                use_table_transformer=selected_table_transformer,
            )
            if selected_word_detector is not None:
                try:
                    word_polys = selected_word_detector.detect(page_image)
                    # Strip detector-provided text so word content always
                    # comes from the primary OCR's region text.
                    for wp in word_polys:
                        wp.pop("content", None)
                    page_layout["word_polys"] = _assign_word_content(
                        word_polys, cast(list, page_layout.get("regions", []))
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Wort-Erkennung fehlgeschlagen: {exc}")
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
            page_images=encode_page_images(raw_page_images) if raw_page_images else None,
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
