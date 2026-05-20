"""Generic HuggingFace layout detector.

Uses ``AutoModelForObjectDetection`` / ``AutoImageProcessor`` so any
HuggingFace object-detection checkpoint can be loaded (Docling Heron,
Deformable DETR, RT-DETR variants, PP-DocLayout safetensors, …).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def resolve_layout_device(requested: str | None) -> str:
    """Map ``auto`` / ``cpu`` / ``cuda`` / ``cuda:N`` to a torch device string."""
    raw = (requested or "auto").strip().lower()
    if raw in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cpu":
        return "cpu"
    if raw in {"cuda", "gpu", "rocm", "amd"}:
        if not torch.cuda.is_available():
            logger.warning("Layout device %r requested but CUDA is unavailable; using CPU", raw)
            return "cpu"
        return "cuda"
    if raw.startswith("cuda:"):
        if not torch.cuda.is_available():
            logger.warning(
                "Layout device %r requested but CUDA is unavailable; using CPU", raw
            )
            return "cpu"
        return raw
    logger.warning("Unknown layout device %r; falling back to auto", requested)
    return resolve_layout_device("auto")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass
class LayoutDetectorConfig:
    """Settings for :class:`HFLayoutDetector`."""

    model_dir: str
    threshold: float = 0.2
    threshold_by_class: dict[str, float] | None = None
    layout_nms: bool = True
    layout_unclip_ratio: list[float] | None = None
    layout_merge_bboxes_mode: str = "large"
    batch_size: int = 8
    label_task_mapping: dict[str, list[str]] | None = None
    id2label: dict[int, str] | None = None
    device: str = "auto"
    cuda_visible_devices: str | None = None
    # Extra fields for forward-compat; ignored by the detector.
    extra: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Inlined post-processing helpers (originally from glmocr)
# ------------------------------------------------------------------


def _iou(box1: Any, box2: Any) -> float:
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    x1_i = max(x1, x1_p)
    y1_i = max(y1, y1_p)
    x2_i = min(x2, x2_p)
    y2_i = min(y2, y2_p)
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    return float(inter_area) / float(box1_area + box2_area - inter_area)


def _is_contained(box1: Any, box2: Any) -> bool:
    _, _, x1, y1, x2, y2 = box1
    _, _, x1_p, y1_p, x2_p, y2_p = box2
    box1_area = (x2 - x1) * (y2 - y1)
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersect_area = inter_width * inter_height
    ratio = intersect_area / box1_area if box1_area > 0 else 0
    return ratio >= 0.8


def _nms(boxes: Any, iou_same: float = 0.6, iou_diff: float = 0.95) -> list[int]:
    scores = boxes[:, 1]
    indices: list[int] = np.argsort(scores)[::-1].tolist()
    selected: list[int] = []
    while indices:
        current = indices[0]
        selected.append(current)
        indices = indices[1:]
        current_box = boxes[current]
        current_class = current_box[0]
        current_coords = current_box[2:]
        filtered: list[int] = []
        for i in indices:
            box = boxes[i]
            threshold = iou_same if current_class == box[0] else iou_diff
            if _iou(current_coords, box[2:]) < threshold:
                filtered.append(i)
        indices = filtered
    return selected


def _check_containment(
    boxes: Any,
    preserve_indices: set[int] | None = None,
    category_index: int | None = None,
    mode: str | None = None,
) -> tuple[Any, Any]:
    n = len(boxes)
    contains_other = np.zeros(n, dtype=int)
    contained_by_other = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if preserve_indices is not None and boxes[i][0] in preserve_indices:
                continue
            if category_index is not None and mode is not None:
                if mode == "large" and boxes[j][0] == category_index:
                    if _is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
                if mode == "small" and boxes[i][0] == category_index:
                    if _is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
            else:
                if _is_contained(boxes[i], boxes[j]):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
    return contains_other, contained_by_other


def _unclip_boxes(boxes: Any, unclip_ratio: Any) -> Any:
    if unclip_ratio is None:
        return boxes
    if isinstance(unclip_ratio, dict):
        expanded = []
        for box in boxes:
            class_id = box[0]
            if class_id in unclip_ratio:
                wr, hr = unclip_ratio[class_id]
                x1, y1, x2, y2 = box[2], box[3], box[4], box[5]
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w / 2, y1 + h / 2
                row = [
                    class_id,
                    box[1],
                    cx - w * wr / 2,
                    cy - h * hr / 2,
                    cx + w * wr / 2,
                    cy + h * hr / 2,
                ]
                if len(box) > 6:
                    row.extend(box[6:])
                expanded.append(row)
            else:
                expanded.append(box)
        return np.array(expanded)
    widths = boxes[:, 4] - boxes[:, 2]
    heights = boxes[:, 5] - boxes[:, 3]
    cx = boxes[:, 2] + widths / 2
    cy = boxes[:, 3] + heights / 2
    nw = widths * unclip_ratio[0]
    nh = heights * unclip_ratio[1]
    result = np.column_stack(
        (boxes[:, 0], boxes[:, 1], cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2)
    )
    if boxes.shape[1] > 6:
        result = np.column_stack((result, boxes[:, 6:]))
    return result


def _apply_layout_postprocess(
    raw_results: list[dict[str, Any]],
    id2label: dict[int, str],
    img_sizes: list[tuple[int, int]],
    layout_nms: bool = True,
    layout_unclip_ratio: Any = None,
    layout_merge_bboxes_mode: Any = None,
) -> list[list[dict[str, Any]]]:
    all_labels = list(id2label.values())
    paddle_results: list[list[dict[str, Any]]] = []

    for img_idx, result in enumerate(raw_results):
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        boxes = result["boxes"].cpu().numpy()
        order_seq = result["order_seq"].cpu().numpy()
        polygon_points = result.get("polygon_points", [])
        img_size = img_sizes[img_idx]

        rows = []
        for i in range(len(scores)):
            rows.append([int(labels[i]), float(scores[i]), *boxes[i].tolist(), int(order_seq[i])])
        if not rows:
            paddle_results.append([])
            continue
        arr = np.array(rows)

        # NMS
        if layout_nms:
            sel = _nms(arr[:, :6], iou_same=0.6, iou_diff=0.98)
            arr = arr[sel]

        # Filter oversized image detections
        if len(arr) > 1:
            area_thres = 0.82 if img_size[0] > img_size[1] else 0.93
            image_idx = all_labels.index("image") if "image" in all_labels else None
            img_area = img_size[0] * img_size[1]
            keep = []
            for box in arr:
                if box[0] == image_idx:
                    bx1 = max(0, box[2])
                    by1 = max(0, box[3])
                    bx2 = min(img_size[0], box[4])
                    by2 = min(img_size[1], box[5])
                    if (bx2 - bx1) * (by2 - by1) <= area_thres * img_area:
                        keep.append(box)
                else:
                    keep.append(box)
            if keep:
                arr = np.array(keep)

        # Containment merging
        if layout_merge_bboxes_mode:
            preserve_labels = ["image", "seal", "chart"]
            preserve_ids: set[int] = set()
            for lbl in preserve_labels:
                if lbl in all_labels:
                    preserve_ids.add(all_labels.index(lbl))
            if isinstance(layout_merge_bboxes_mode, str):
                if layout_merge_bboxes_mode != "union":
                    _co, cbo = _check_containment(arr[:, :6], preserve_ids)
                    if layout_merge_bboxes_mode == "large":
                        arr = arr[cbo == 0]
                    elif layout_merge_bboxes_mode == "small":
                        arr = arr[(_co == 0) | (cbo == 1)]
            elif isinstance(layout_merge_bboxes_mode, dict):
                mask = np.ones(len(arr), dtype=bool)
                for cat_idx, lm in layout_merge_bboxes_mode.items():
                    if lm != "union":
                        _co, cbo = _check_containment(arr[:, :6], preserve_ids, cat_idx, mode=lm)
                        if lm == "large":
                            mask &= cbo == 0
                        elif lm == "small":
                            mask &= (_co == 0) | (cbo == 1)
                arr = arr[mask]

        if len(arr) == 0:
            paddle_results.append([])
            continue

        # Sort by order
        arr = arr[np.argsort(arr[:, 6])]

        # Unclip
        if layout_unclip_ratio:
            ratio = layout_unclip_ratio
            if isinstance(ratio, (int, float)):
                ratio = (ratio, ratio)
            arr = _unclip_boxes(arr, ratio)

        # Convert to output format
        img_w, img_h = img_size
        page: list[dict[str, Any]] = []
        for box_data in arr:
            cls_id = int(box_data[0])
            score = float(box_data[1])
            x1 = max(0.0, min(float(box_data[2]), img_w))
            y1 = max(0.0, min(float(box_data[3]), img_h))
            x2 = max(0.0, min(float(box_data[4]), img_w))
            y2 = max(0.0, min(float(box_data[5]), img_h))
            if x1 >= x2 or y1 >= y2:
                continue
            label_name = id2label.get(cls_id, f"class_{cls_id}")
            poly = None
            if polygon_points:
                for orig_idx in range(len(boxes)):
                    if np.allclose(boxes[orig_idx], box_data[2:6], atol=1.0):
                        if orig_idx < len(polygon_points) and polygon_points[orig_idx] is not None:
                            poly = polygon_points[orig_idx].astype(np.float32)
                        break
            if poly is None:
                poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            else:
                poly[:, 0] = np.clip(poly[:, 0], 0, img_w)
                poly[:, 1] = np.clip(poly[:, 1], 0, img_h)
            page.append(
                {
                    "cls_id": cls_id,
                    "label": label_name,
                    "score": score,
                    "coordinate": [int(x1), int(y1), int(x2), int(y2)],
                    "order": int(box_data[6]) if box_data[6] > 0 else None,
                    "polygon_points": poly,
                }
            )
        paddle_results.append(page)

    return paddle_results


# ------------------------------------------------------------------
# Detector
# ------------------------------------------------------------------


class HFLayoutDetector:
    """Layout detector that loads any HuggingFace object-detection model."""

    def __init__(self, config: LayoutDetectorConfig) -> None:
        self.model_dir = config.model_dir
        self.layout_device = config.device
        self.cuda_visible_devices = config.cuda_visible_devices

        self.threshold = config.threshold
        self.threshold_by_class = config.threshold_by_class
        self.layout_nms = config.layout_nms
        self.layout_unclip_ratio = config.layout_unclip_ratio
        self.layout_merge_bboxes_mode = config.layout_merge_bboxes_mode
        self.batch_size = config.batch_size

        self.label_task_mapping = config.label_task_mapping
        self.id2label: dict[int, str] | None = getattr(config, "id2label", None)

        self._model: Any = None
        self._image_processor: Any = None
        self._device: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load model and processor via ``Auto*`` classes."""
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        logger.info("Loading HF layout model from %s …", self.model_dir)

        self._image_processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self._model = AutoModelForObjectDetection.from_pretrained(self.model_dir)
        self._model.eval()

        if self.cuda_visible_devices is not None and self.layout_device in {
            "auto",
            "",
        }:
            legacy = (
                f"cuda:{self.cuda_visible_devices}"
                if torch.cuda.is_available()
                else "cpu"
            )
            self._device = legacy
        else:
            self._device = resolve_layout_device(self.layout_device)
        self._model = self._model.to(self._device)

        if self.id2label is None:
            self.id2label = self._model.config.id2label
        logger.info("HF layout model loaded on device: %s", self._device)

    def stop(self) -> None:
        if self._model is not None:
            if self._device and self._device.startswith("cuda"):
                torch.cuda.empty_cache()
            self._model = None
        self._image_processor = None
        self._device = None

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _empty_detection_result(self) -> dict[str, Any]:
        return {
            "scores": torch.tensor([], device=self._device),
            "labels": torch.tensor([], dtype=torch.long, device=self._device),
            "boxes": torch.tensor([], device=self._device).reshape(0, 4),
            "order_seq": torch.tensor([], dtype=torch.long, device=self._device),
        }

    def _run_single(self, image: Image.Image, pre_threshold: float) -> dict[str, Any]:
        inputs = self._image_processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        target = torch.tensor([image.size[::-1]], device=self._device)
        results = self._image_processor.post_process_object_detection(
            outputs, threshold=pre_threshold, target_sizes=target
        )
        return cast(dict[str, Any], results[0])

    def _post_process_chunk(
        self,
        chunk_pil: list[Image.Image],
        outputs: Any,
        target_sizes: Any,
        pre_threshold: float,
    ) -> list[dict[str, Any]]:
        try:
            return cast(
                list[dict[str, Any]],
                self._image_processor.post_process_object_detection(
                    outputs, threshold=pre_threshold, target_sizes=target_sizes
                ),
            )
        except Exception as exc:
            logger.warning("Batch post_process failed, retrying per-image: %s", exc)

        results: list[dict[str, Any]] = []
        for img in chunk_pil:
            try:
                results.append(self._run_single(img, pre_threshold))
            except Exception as exc2:
                logger.warning("Single-image post_process failed: %s", exc2)
                results.append(self._empty_detection_result())
        return results

    def _apply_per_class_threshold(self, raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.threshold_by_class or not self.id2label:
            return raw_results

        label2id = {name: int(cls_id) for cls_id, name in self.id2label.items()}
        class_thresholds: dict[int, float] = {}
        for key, value in self.threshold_by_class.items():
            if isinstance(key, str):
                if key in label2id:
                    class_thresholds[label2id[key]] = float(value)
            else:
                class_thresholds[int(key)] = float(value)

        fallback = self.threshold
        filtered: list[dict[str, Any]] = []
        for result in raw_results:
            scores = result["scores"]
            labels = result["labels"]
            thresholds = torch.full_like(scores, fallback)
            for class_id, thresh in class_thresholds.items():
                thresholds[labels == class_id] = thresh
            keep = scores >= thresholds
            new_result: dict[str, Any] = {
                "scores": scores[keep],
                "labels": labels[keep],
                "boxes": result["boxes"][keep],
            }
            if "order_seq" in result:
                new_result["order_seq"] = result["order_seq"][keep]
            if "polygon_points" in result:
                keep_list = keep.tolist()
                new_result["polygon_points"] = [
                    p for p, k in zip(result["polygon_points"], keep_list, strict=False) if k
                ]
            filtered.append(new_result)
        return filtered

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        images: list[Image.Image],
        save_visualization: bool = False,
        visualization_output_dir: str | None = None,
        global_start_idx: int = 0,
    ) -> list[list[dict[str, Any]]]:
        if self._model is None:
            raise RuntimeError("Layout detector not started. Call start() first.")

        num_images = len(images)
        image_batch: list[tuple[Any, int, int]] = []
        for image in images:
            w, h = image.size
            image_batch.append((np.array(image.convert("RGB")), w, h))

        pil_images = [Image.fromarray(ib[0]) for ib in image_batch]
        all_paddle_results: list[list[dict[str, Any]]] = []

        for chunk_start in range(0, num_images, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, num_images)
            chunk_pil = pil_images[chunk_start:chunk_end]

            inputs = self._image_processor(images=chunk_pil, return_tensors="pt")
            inputs = {
                k: v.to(self._device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
            }

            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor([img.size[::-1] for img in chunk_pil], device=self._device)

            if self.threshold_by_class:
                pre_threshold = min(self.threshold, min(self.threshold_by_class.values()))
            else:
                pre_threshold = self.threshold

            raw_results = self._post_process_chunk(chunk_pil, outputs, target_sizes, pre_threshold)

            if self.threshold_by_class:
                raw_results = self._apply_per_class_threshold(raw_results)

            for result in raw_results:
                if "polygon_points" not in result:
                    bxs = result["boxes"]
                    pp: list[Any] = []
                    for box in bxs:
                        x1, y1, x2, y2 = box.tolist()
                        pp.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
                    result["polygon_points"] = pp
                if "order_seq" not in result:
                    nd = len(result["scores"])
                    result["order_seq"] = torch.arange(nd, dtype=torch.long, device=self._device)

            img_sizes = [img.size for img in chunk_pil]
            paddle_results = _apply_layout_postprocess(
                raw_results=raw_results,
                id2label=self.id2label or {},
                img_sizes=img_sizes,
                layout_nms=self.layout_nms,
                layout_unclip_ratio=self.layout_unclip_ratio,
                layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            )
            all_paddle_results.extend(paddle_results)

            if self._device and self._device.startswith("cuda") and chunk_end < num_images:
                del inputs, outputs, raw_results
                torch.cuda.empty_cache()

        # Normalize to 0-1000 coordinates
        all_results: list[list[dict[str, Any]]] = []
        for img_idx, page_results in enumerate(all_paddle_results):
            img_w = image_batch[img_idx][1]
            img_h = image_batch[img_idx][2]
            results: list[dict[str, Any]] = []
            valid_index = 0
            for item in page_results:
                label = item["label"]
                score = item["score"]
                box = item["coordinate"]
                task_type = None
                if self.label_task_mapping:
                    for task_item, labels in self.label_task_mapping.items():
                        if isinstance(labels, list) and label in labels:
                            task_type = task_item
                            break
                if task_type is None:
                    task_type = "skip"
                if task_type == "abandon":
                    continue

                x1, y1, x2, y2 = box
                x1_norm = int(float(x1) / img_w * 1000)
                y1_norm = int(float(y1) / img_h * 1000)
                x2_norm = int(float(x2) / img_w * 1000)
                y2_norm = int(float(y2) / img_h * 1000)

                poly_array = item["polygon_points"]
                polygon = [
                    [
                        int(float(pt[0]) / img_w * 1000),
                        int(float(pt[1]) / img_h * 1000),
                    ]
                    for pt in poly_array
                ]

                results.append(
                    {
                        "index": valid_index,
                        "label": label,
                        "score": float(score),
                        "bbox_2d": [x1_norm, y1_norm, x2_norm, y2_norm],
                        "polygon": polygon,
                        "task_type": task_type,
                    }
                )
                valid_index += 1
            all_results.append(results)

        return all_results
