"""Generic HuggingFace layout detector for GLM-OCR.

Drop-in replacement for GLM-OCR's built-in PPDocLayoutDetector. Uses
``AutoModelForObjectDetection`` / ``AutoImageProcessor`` so any HuggingFace
object-detection checkpoint can be loaded (Docling Heron, DocLayout-YOLO,
RT-DETR variants, …).

The class implements ``glmocr.layout.base.BaseLayoutDetector`` and produces
the same output format that GLM-OCR's pipeline expects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from glmocr.layout.base import BaseLayoutDetector
from glmocr.utils.layout_postprocess_utils import apply_layout_postprocess
from PIL import Image

if TYPE_CHECKING:
    from glmocr.config import LayoutConfig

logger = logging.getLogger(__name__)


class HFLayoutDetector(BaseLayoutDetector):
    """Layout detector that loads any HuggingFace object-detection model.

    Produces the same region dicts as ``PPDocLayoutDetector`` so it can be
    used as a transparent replacement inside the GLM-OCR pipeline.
    """

    def __init__(self, config: LayoutConfig) -> None:
        super().__init__(config)

        self.model_dir = config.model_dir
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

        if torch.cuda.is_available():
            self._device = (
                f"cuda:{self.cuda_visible_devices}"
                if self.cuda_visible_devices is not None
                else "cuda"
            )
        else:
            self._device = "cpu"
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
        return results[0]

    def _post_process_chunk(
        self,
        chunk_pil: list[Image.Image],
        outputs: Any,
        target_sizes: Any,
        pre_threshold: float,
    ) -> list[dict[str, Any]]:
        try:
            return self._image_processor.post_process_object_detection(
                outputs, threshold=pre_threshold, target_sizes=target_sizes
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
    # Main entry point (same signature as PPDocLayoutDetector.process)
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

            # Synthesize polygon_points and order_seq from boxes when the
            # model doesn't produce them (most AutoModel checkpoints don't).
            for result in raw_results:
                if "polygon_points" not in result:
                    boxes = result["boxes"]
                    polygon_points: list[Any] = []
                    for box in boxes:
                        x1, y1, x2, y2 = box.tolist()
                        polygon_points.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
                    result["polygon_points"] = polygon_points
                if "order_seq" not in result:
                    num_detections = len(result["scores"])
                    result["order_seq"] = torch.arange(
                        num_detections, dtype=torch.long, device=self._device
                    )

            img_sizes = [img.size for img in chunk_pil]
            paddle_results = apply_layout_postprocess(
                raw_results=raw_results,
                id2label=self.id2label,
                img_sizes=img_sizes,
                layout_nms=self.layout_nms,
                layout_unclip_ratio=self.layout_unclip_ratio,
                layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            )
            all_paddle_results.extend(paddle_results)

            if self._device and self._device.startswith("cuda") and chunk_end < num_images:
                del inputs, outputs, raw_results
                torch.cuda.empty_cache()

        # Normalize to 0-1000 coordinates (same as PPDocLayoutDetector)
        all_results: list[list[dict[str, Any]]] = []
        for img_idx, paddle_results in enumerate(all_paddle_results):
            img_w = image_batch[img_idx][1]
            img_h = image_batch[img_idx][2]
            results: list[dict[str, Any]] = []
            valid_index = 0
            for item in paddle_results:
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
