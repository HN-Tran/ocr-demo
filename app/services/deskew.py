"""Detect and correct document skew and cardinal misorientation.

Uses OpenCV (already in pyproject.toml) — no new dependencies required.
"""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_gray(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.uint8)


def _proj_variance(gray: np.ndarray) -> float:
    """Variance of row-wise dark-pixel sums. High when text lines are horizontal."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(np.var(binary.sum(axis=1).astype(np.float64)))


def _gradient_asymmetry(gray: np.ndarray) -> float:
    """Return a score that is positive when the image is correctly oriented (0°).

    In upright Latin text the transition from whitespace INTO text (going
    downward) is sharper than the transition OUT of text, because cap-height
    starts cleanly while descenders trail off.  The score compares the mean
    squared energy of rising vs falling gradients of the horizontal projection.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = binary.sum(axis=1).astype(np.float64)
    grad = np.diff(proj)
    pos = grad[grad > 0]
    neg = grad[grad < 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(np.mean(pos ** 2) - np.mean(neg ** 2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PIL_TRANSPOSE = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def detect_cardinal_rotation(img: Image.Image) -> tuple[int, float]:
    """Detect how many degrees CCW to rotate `img` to make text upright.

    Returns ``(rotation_to_apply, confidence)`` where:
    - ``rotation_to_apply`` ∈ {0, 90, 180, 270} — counter-clockwise degrees
    - ``confidence`` ∈ [0, 1]; values below ~0.70 indicate an uncertain result
    """
    gray = _to_gray(img)

    # Evaluate all 4 candidates via numpy rotation (avoids PIL overhead).
    # np.rot90(k=1) = 90° CCW, k=2 = 180°, k=3 = 270° CCW.
    candidates = [np.rot90(gray, k=k) for k in range(4)]
    variances = [_proj_variance(c) for c in candidates]

    # Step 1: which axis-aligned pair has text running horizontally?
    score_0_180 = max(variances[0], variances[2])  # upright or upside-down
    score_90_270 = max(variances[1], variances[3])  # rotated 90° either way

    if score_0_180 >= score_90_270:
        # Text is roughly horizontal → distinguish 0° from 180° via gradient asymmetry.
        sym_0 = _gradient_asymmetry(candidates[0])
        sym_180 = _gradient_asymmetry(candidates[2])
        if sym_0 >= sym_180:
            winner_k = 0
            margin = abs(sym_0 - sym_180)
        else:
            winner_k = 2
            margin = abs(sym_180 - sym_0)
        total = abs(sym_0) + abs(sym_180) + 1e-6
        confidence = float(np.clip(margin / total, 0.0, 1.0))
    else:
        # Text is vertical → distinguish 90° from 270° by higher variance.
        if variances[1] >= variances[3]:
            winner_k = 1
        else:
            winner_k = 3
        # 90° vs 270° mismatch is almost always obvious (large variance gap).
        confidence_raw = (score_90_270 - score_0_180) / (score_90_270 + score_0_180 + 1e-6)
        confidence = float(np.clip(confidence_raw * 2.0, 0.0, 1.0))

    return winner_k * 90, confidence


def detect_fine_skew(img: Image.Image) -> float:
    """Detect fine skew angle of text from horizontal (degrees, CCW positive).

    Returns 0.0 when fewer than 5 text-line blobs are detected (not confident).
    To correct, apply: ``image.rotate(-result, expand=True, ...)``.
    """
    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to connect words into line-shaped blobs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles: list[float] = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        _, (w, h), angle = cv2.minAreaRect(contour)
        if w < h:
            angle = 90.0 + angle  # align to long axis
        if -45.0 < angle < 45.0:
            angles.append(angle)

    if len(angles) < 5:
        return 0.0
    return float(np.median(angles))


def deskew_image(
    img: Image.Image,
    *,
    min_angle_deg: float = 0.5,
    cardinal_confidence_threshold: float = 0.70,
) -> tuple[Image.Image, float]:
    """Detect and correct cardinal misorientation and fine skew.

    Returns ``(corrected_image, net_ccw_correction)`` where
    ``net_ccw_correction`` is the total CCW degrees applied (stored in
    ``page_info["angle"]``; positive = original was tilted CW by that amount).
    """
    net_ccw = 0.0

    # Step 1: Correct cardinal rotation (lossless via PIL transpose).
    rotation, confidence = detect_cardinal_rotation(img)
    if rotation != 0 and confidence >= cardinal_confidence_threshold:
        img = img.transpose(_PIL_TRANSPOSE[rotation])
        net_ccw += rotation

    # Step 2: Fine skew — angle is CCW-positive (negative = CW tilt).
    skew = detect_fine_skew(img)
    fine_correction = -skew  # CCW rotation needed to straighten
    if abs(fine_correction) >= min_angle_deg:
        img = img.rotate(
            fine_correction,
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        )
        net_ccw += fine_correction

    return img, net_ccw
