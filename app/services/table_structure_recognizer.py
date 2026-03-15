"""Table structure recognition using Microsoft Table Transformer.

Takes a cropped table image and detects rows, columns, headers and spanning
cells.  Individual cell bounding boxes are computed from row × column
intersections.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"

# Table Transformer id2label:
# 0: table, 1: table column, 2: table row,
# 3: table column header, 4: table projected row header,
# 5: table spanning cell
_LABEL_ROW = "table row"
_LABEL_COLUMN = "table column"
_LABEL_COLUMN_HEADER = "table column header"
_LABEL_PROJECTED_ROW_HEADER = "table projected row header"
_LABEL_SPANNING_CELL = "table spanning cell"


class TableStructureRecognizer:
    """Detect table rows, columns, headers and cells within a table crop."""

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._requested_device = device
        self._model: Any = None
        self._image_processor: Any = None
        self._device: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        logger.info("Loading table structure model from %s …", self.model_name)
        self._image_processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            use_fast=False,
            size={"shortest_edge": 800, "longest_edge": 800},
        )
        self._model = AutoModelForObjectDetection.from_pretrained(self.model_name)
        self._model.eval()

        if self._requested_device is not None:
            self._device = self._requested_device
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        self._model = self._model.to(self._device)
        logger.info("Table structure model loaded on device: %s", self._device)

    def stop(self) -> None:
        if self._model is not None:
            if self._device and self._device.startswith("cuda"):
                torch.cuda.empty_cache()
            self._model = None
        self._image_processor = None
        self._device = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, table_crop: Image.Image) -> list[dict[str, Any]]:
        """Return a list of cell dicts for the given table crop.

        Each cell dict has keys: ``row``, ``column``, ``bbox_2d`` (pixel
        coords relative to the crop), ``polygon``, ``is_header``.
        """
        if self._model is None:
            raise RuntimeError("Table structure recognizer not started.")

        elements = self._detect_elements(table_crop)
        rows = sorted(elements.get(_LABEL_ROW, []), key=lambda b: b[1])
        columns = sorted(elements.get(_LABEL_COLUMN, []), key=lambda b: b[0])
        header_boxes = elements.get(_LABEL_COLUMN_HEADER, [])
        projected_header_boxes = elements.get(_LABEL_PROJECTED_ROW_HEADER, [])
        spanning_boxes = elements.get(_LABEL_SPANNING_CELL, [])

        if not rows or not columns:
            return []

        cells = self._compute_cells(
            rows=rows,
            columns=columns,
            header_boxes=header_boxes,
            projected_header_boxes=projected_header_boxes,
            spanning_boxes=spanning_boxes,
        )
        return cells

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _detect_elements(self, image: Image.Image) -> dict[str, list[list[float]]]:
        """Run the model and return detections grouped by label name."""
        inputs = self._image_processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target = torch.tensor([image.size[::-1]], device=self._device)
        results = self._image_processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target
        )[0]

        id2label: dict[int, str] = self._model.config.id2label
        elements: dict[str, list[list[float]]] = {}
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"], strict=False
        ):
            label = id2label.get(label_id.item(), "")
            score_val = float(score.item())
            logger.debug(
                "Detection: label=%r score=%.3f box=%s",
                label,
                score_val,
                box.tolist(),
            )
            if not label or label == "table":
                continue
            box_list = box.tolist()
            elements.setdefault(label, []).append(box_list)
        logger.info(
            "Table structure elements: %s",
            {k: len(v) for k, v in elements.items()},
        )
        return elements

    # ------------------------------------------------------------------
    # Cell computation
    # ------------------------------------------------------------------

    @staticmethod
    def _box_intersection(box_a: list[float], box_b: list[float]) -> list[float] | None:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        if x1 >= x2 or y1 >= y2:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _center_in_box(cx: float, cy: float, box: list[float]) -> bool:
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    @staticmethod
    def _box_overlap_ratio(inner: list[float], outer: list[float]) -> float:
        ix1 = max(inner[0], outer[0])
        iy1 = max(inner[1], outer[1])
        ix2 = min(inner[2], outer[2])
        iy2 = min(inner[3], outer[3])
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        inner_area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-6)
        return inter_area / inner_area

    @classmethod
    def _compute_cells(
        cls,
        *,
        rows: list[list[float]],
        columns: list[list[float]],
        header_boxes: list[list[float]],
        projected_header_boxes: list[list[float]],
        spanning_boxes: list[list[float]],
    ) -> list[dict[str, Any]]:
        all_header_boxes = header_boxes + projected_header_boxes

        # Build basic grid cells from row × column intersections.
        grid: list[dict[str, Any]] = []
        for row_idx, row_box in enumerate(rows):
            for col_idx, col_box in enumerate(columns):
                cell_box = cls._box_intersection(row_box, col_box)
                if cell_box is None:
                    continue
                cx = (cell_box[0] + cell_box[2]) / 2
                cy = (cell_box[1] + cell_box[3]) / 2
                is_header = any(cls._center_in_box(cx, cy, hbox) for hbox in all_header_boxes)
                grid.append(
                    {
                        "row": row_idx,
                        "column": col_idx,
                        "bbox_2d": cell_box,
                        "polygon": _bbox_to_polygon(cell_box),
                        "is_header": is_header,
                    }
                )

        # Mark cells covered by spanning cell detections.
        for span_box in spanning_boxes:
            covered_rows: set[int] = set()
            covered_cols: set[int] = set()
            for cell in grid:
                if cls._box_overlap_ratio(cell["bbox_2d"], span_box) >= 0.5:
                    covered_rows.add(cell["row"])
                    covered_cols.add(cell["column"])
            if len(covered_rows) > 1 or len(covered_cols) > 1:
                min_row = min(covered_rows) if covered_rows else 0
                min_col = min(covered_cols) if covered_cols else 0
                for cell in grid:
                    if cell["row"] in covered_rows and cell["column"] in covered_cols:
                        cell["row_span"] = len(covered_rows)
                        cell["col_span"] = len(covered_cols)
                        cell["span_root"] = cell["row"] == min_row and cell["column"] == min_col

        return grid


def _bbox_to_polygon(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2, y1, x2, y2, x1, y2]
