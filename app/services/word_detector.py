"""Word-level bounding box detectors.

Two optional backends — neither is installed by default:

- ``paddleocr``:  PaddleOCR full OCR (``pip install paddleocr>=3.0``).
  Returns oriented quadrilateral polygons with recognised text in original image space.

- ``doctr``:  DocTR OCR predictor (``pip install 'python-doctr[torch]'``).
  Returns rotated or straight bounding boxes with recognised text.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class WordPoly(TypedDict, total=False):
    polygon: list[float]  # flat [x0,y0,x1,y1,x2,y2,x3,y3] in 0-1000 coords
    confidence: float
    content: str  # recognised text from the detector (may be absent)


class WordDetector(Protocol):
    """Detect word/line-level bounding polygons in a page image."""

    def detect(self, image: "Image.Image") -> list[WordPoly]:
        """Return word polygons with coordinates normalised to 0-1000."""
        ...


# ---------------------------------------------------------------------------
# PaddleOCR / paddlex oneDNN fix
# ---------------------------------------------------------------------------


def _disable_paddle_mkldnn() -> None:
    """Prevent PaddlePaddle 3.x PIR+oneDNN crash on CPU.

    Root cause (traced through paddlex source):
    1. ``paddlex.inference.utils.misc.is_mkldnn_available()`` checks
       ``hasattr(paddle.inference.Config, "set_mkldnn_cache_capacity")`` — always
       True on x86 — so ``get_default_run_mode()`` returns ``"mkldnn"``.
    2. ``PaddleInfer._create()`` sees ``run_mode == "mkldnn"`` and calls
       ``config.enable_mkldnn()``.  This arms the oneDNN path in the C++ predictor.
    3. During ``predictor.run()``, the PIR executor tries to convert
       ``pir::ArrayAttribute<pir::DoubleAttribute>`` for an oneDNN instruction —
       a known PaddlePaddle 3.x bug → ``NotImplementedError``.

    Fix (two complementary layers):
    A. Patch ``paddlex.is_mkldnn_available`` → ``False`` so the default run_mode
       becomes ``"paddle"``, which makes paddlex call ``config.disable_mkldnn()``
       explicitly.
    B. Redirect ``paddle.inference.Config.enable_mkldnn`` to call
       ``disable_mkldnn()`` instead, as a safety net for any code path that
       hard-codes ``run_mode="mkldnn"`` (e.g. the LaTeX_OCR_rec special case).
    """
    # --- Layer A: patch paddlex availability check ---
    try:
        import paddlex.inference.utils.misc as _pxm  # type: ignore[import-untyped]
        import paddlex.inference.utils.pp_option as _pxo  # type: ignore[import-untyped]

        _pxm.is_mkldnn_available = lambda: False
        # pp_option imports is_mkldnn_available with `from .misc import …`,
        # so we must also replace the name in pp_option's own globals:
        _pxo.is_mkldnn_available = lambda: False
        logger.debug("paddlex.is_mkldnn_available → False gesetzt.")
    except Exception as e:  # noqa: BLE001
        logger.debug("paddlex-Patch nicht anwendbar: %s", e)

    # --- Layer B: redirect Config.enable_mkldnn → disable_mkldnn ---
    try:
        import paddle.inference as _pi  # type: ignore[import-untyped]

        if not getattr(_pi.Config, "_mkldnn_redirected", False):
            # Redirect enable_mkldnn to actually call disable_mkldnn.
            _pi.Config.enable_mkldnn = lambda self, *a, **kw: self.disable_mkldnn()
            _pi.Config.enable_mkldnn_bfloat16 = lambda self, *a, **kw: None
            _pi.Config.set_mkldnn_cache_capacity = lambda self, *a, **kw: None
            _pi.Config._mkldnn_redirected = True  # type: ignore[attr-defined]
            logger.debug("paddle.inference.Config.enable_mkldnn → disable_mkldnn umgeleitet.")
    except Exception as e:  # noqa: BLE001
        logger.debug("paddle.inference.Config-Patch nicht anwendbar: %s", e)


# ---------------------------------------------------------------------------
# PaddleOCR backend
# ---------------------------------------------------------------------------


class PaddleOCRWordDetector:
    """Word detector using PaddleOCR det-only mode.

    Requires: ``pip install paddleocr>=3.0``
    """

    def __init__(self) -> None:
        import os

        # Set C++ gflags before paddle initialises its shared library.
        for _flag in (
            "FLAGS_use_mkldnn",
            "FLAGS_new_executor_use_mkldnn",
            "FLAGS_enable_pir_in_executor",
        ):
            os.environ[_flag] = "0"  # force-override, not setdefault

        _disable_paddle_mkldnn()

        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "paddleocr ist nicht installiert. "
                "Installieren Sie es mit: pip install paddleocr>=3.0"
            ) from exc
        self._ocr: Any = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        logger.info("PaddleOCRWordDetector initialisiert.")

    def detect(self, image: "Image.Image") -> list[WordPoly]:
        import numpy as np  # type: ignore[import-untyped]

        img = image.convert("RGB")
        w, h = img.size
        img_array = np.array(img)
        results: list[WordPoly] = []

        try:
            raw = self._ocr.predict(img_array)
            pred = raw if isinstance(raw, list) else list(raw)
            if not pred:
                return results
            page = pred[0]

            # Paddlex >= 3.0 result objects
            page_dict = None
            if isinstance(page, dict):
                page_dict = page
            else:
                if hasattr(page, "keys") and hasattr(page, "get"):
                    page_dict = dict(page)
                elif hasattr(page, "__dict__") and ("dt_polys" in page.__dict__ or "det_polys" in page.__dict__):
                    page_dict = page.__dict__
                elif hasattr(page, "json"):
                    if callable(page.json):
                        page_dict = page.json()
                    elif isinstance(page.json, dict):
                        page_dict = page.json
                elif hasattr(page, "res") and isinstance(page.res, dict):
                    page_dict = page.res
                elif isinstance(page, (list, tuple)):
                    page_dict = None  # paddlev2 output

            if page_dict is not None:
                polys = page_dict.get("dt_polys") or page_dict.get("det_polys") or []
                scores = page_dict.get("rec_scores") or page_dict.get("det_scores") or []
                for idx, poly in enumerate(polys):
                    pts = np.array(poly).reshape(-1, 2)
                    flat = _pts_to_flat(pts, w, h)
                    if flat:
                        entry: WordPoly = {
                            "polygon": flat,
                            "confidence": float(scores[idx]) if idx < len(scores) else 1.0,
                        }
                        results.append(entry)
            elif isinstance(page, list):
                # PaddleOCR v2.x: [[polygon_4x2, (text, score)], ...]
                for item in page:
                    if isinstance(item, (list, tuple)) and item:
                        pts = np.array(item[0]).reshape(-1, 2)
                        flat = _pts_to_flat(pts, w, h)
                        if flat:
                            entry = {"polygon": flat, "confidence": 1.0}
                            if len(item) >= 2 and isinstance(item[1], (list, tuple)) and item[1]:
                                entry["content"] = str(item[1][0])
                            results.append(entry)
        except Exception:
            logger.exception("PaddleOCRWordDetector.detect fehlgeschlagen")

        return results


# ---------------------------------------------------------------------------
# DocTR backend
# ---------------------------------------------------------------------------


class DocTRWordDetector:
    """Word detector using python-doctr OCR predictor (detection + recognition).

    Requires: ``pip install 'python-doctr[torch]'``
    """

    def __init__(self) -> None:
        try:
            from doctr.models import ocr_predictor  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "python-doctr ist nicht installiert. "
                "Installieren Sie es mit: pip install 'python-doctr[torch]'"
            ) from exc
        # assume_straight_pages=False → rotated quadrilateral output
        self._predictor: Any = ocr_predictor(
            pretrained=True, assume_straight_pages=False
        )
        logger.info("DocTRWordDetector initialisiert.")

    def detect(self, image: "Image.Image") -> list[WordPoly]:
        import numpy as np  # type: ignore[import-untyped]

        img_array = np.array(image.convert("RGB"))
        results: list[WordPoly] = []

        try:
            # ocr_predictor returns a Document with pages → blocks → lines → words
            doc = self._predictor([img_array])
            for page in doc.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            geo = np.array(word.geometry)
                            if geo.ndim == 2 and geo.shape[1] == 2 and geo.shape[0] >= 4:
                                # N×2 rotated polygon in 0-1 relative coords
                                flat = [
                                    v
                                    for pt in geo[:4]
                                    for v in [float(pt[0]) * 1000, float(pt[1]) * 1000]
                                ]
                            elif geo.ndim == 2 and geo.shape == (2, 2):
                                # Straight bbox: [[x_min, y_min], [x_max, y_max]]
                                x0, y0 = geo[0]
                                x1, y1 = geo[1]
                                flat = [
                                    x0 * 1000, y0 * 1000,
                                    x1 * 1000, y0 * 1000,
                                    x1 * 1000, y1 * 1000,
                                    x0 * 1000, y1 * 1000,
                                ]
                            elif geo.ndim == 1 and geo.shape == (4,):
                                x0, y0, x1, y1 = geo
                                flat = [
                                    x0 * 1000, y0 * 1000,
                                    x1 * 1000, y0 * 1000,
                                    x1 * 1000, y1 * 1000,
                                    x0 * 1000, y1 * 1000,
                                ]
                            else:
                                continue
                            if len(flat) >= 8:
                                entry: WordPoly = {
                                    "polygon": flat,
                                    "confidence": float(word.confidence) if word.confidence is not None else 1.0,
                                }
                                if word.value:
                                    entry["content"] = word.value
                                results.append(entry)
        except Exception:
            logger.exception("DocTRWordDetector.detect fehlgeschlagen")

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _pts_to_flat(pts: "Any", img_w: int, img_h: int) -> list[float]:
    """Convert Nx2 pixel-coord array to flat 0-1000 normalised polygon."""
    flat: list[float] = []
    for pt in pts:
        flat.append(float(pt[0]) / img_w * 1000)
        flat.append(float(pt[1]) / img_h * 1000)
    return flat if len(flat) >= 8 else []



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_VALID = {"none", "paddleocr", "doctr"}


def create_word_detector(name: str) -> WordDetector | None:
    """Return a WordDetector instance or None for 'none'."""
    normalized = name.strip().lower()
    if normalized not in _VALID:
        logger.warning(
            "Unbekannter OCR_WORD_DETECTOR-Wert %r – verwende 'none'.", name
        )
        return None
    if normalized == "none":
        return None
    if normalized == "paddleocr":
        return PaddleOCRWordDetector()
    return DocTRWordDetector()
