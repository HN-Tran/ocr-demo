"""Detect and correct document skew and cardinal misorientation.

Uses OpenCV (already in pyproject.toml) — no new dependencies required.
"""

from __future__ import annotations

import cv2
import numpy as np
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


def _boundary_asymmetry(gray: np.ndarray) -> float:
    """Return a score > 0 when the image is correctly oriented (0°).

    Measures whether text band ENTRIES (top edge) are sharper than EXITS
    (bottom edge) using only cross-zero transitions in the projection profile.
    Within-band fluctuations are excluded to avoid the signal being swamped by
    internal variation.

    For upright Latin text the entry (cap-height) is at full density from the
    first row, while the exit tapers through descenders — giving a sharper entry
    transition. Score = mean(entry²) − mean(exit²).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = binary.sum(axis=1).astype(np.float64)

    threshold = proj.max() * 0.05  # ignore sub-threshold noise
    if threshold == 0:
        return 0.0

    entry_sq: list[float] = []
    exit_sq: list[float] = []
    in_band = False
    prev_val = 0.0

    for _i, val in enumerate(proj):
        if not in_band and val > threshold:
            entry_sq.append(val**2)
            in_band = True
        elif in_band and val <= threshold:
            exit_sq.append(prev_val**2)
            in_band = False
        prev_val = val

    # Close any open band at the image edge.
    if in_band and prev_val > threshold:
        exit_sq.append(prev_val**2)

    if not entry_sq or not exit_sq:
        return 0.0
    return float(np.mean(entry_sq) - np.mean(exit_sq))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PIL_TRANSPOSE = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}

# Snapping tolerance: if detected angle is within this many degrees of a
# cardinal multiple, use the lossless PIL transpose instead of bicubic rotate.
_CARDINAL_SNAP_DEG = 3.0

# Minimum centre-of-mass offset from 0.5 (as a fraction of 0..1) needed to
# trigger a 180° flip in the small-angle branch.  confidence = |vcenter−0.5|×2,
# so 0.15 corresponds to vcenter ≥ 0.575 (text clearly biased to one half).
_VCENTER_MIN_CONFIDENCE = 0.15


def detect_page_angle(img: Image.Image, *, max_scan_dim: int = 600) -> float:
    """Detect the CCW rotation angle (degrees) needed to make text horizontal.

    Scans -90° to +90° in two passes (coarse 5°, fine 1°) and returns the
    angle that maximises horizontal projection variance.  Returns 0.0 when
    the image is already straight.

    Does NOT resolve 0° vs 180° (identical variance) — ``deskew_image``
    applies a separate orientation check for that.
    """
    # Downsample for the scan so the ~36 rotations stay fast.
    scan_img = img
    if max(img.size) > max_scan_dim:
        ratio = max_scan_dim / max(img.size)
        scan_img = img.resize(
            (max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
            Image.Resampling.LANCZOS,
        )

    gray = _to_gray(scan_img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    base = Image.fromarray(binary)

    def _var(angle: float) -> float:
        arr = np.array(
            base.rotate(angle, expand=True, fillcolor=0, resample=Image.Resampling.NEAREST)
        )
        return float(np.var(arr.sum(axis=1).astype(np.float64)))

    # Coarse pass: every 5° from -90° to +90° inclusive.
    best_angle = 0.0
    best_var = _var(0.0)
    for coarse_deg in range(-90, 91, 5):
        if coarse_deg == 0:
            continue
        v = _var(float(coarse_deg))
        if v > best_var:
            best_var = v
            best_angle = float(coarse_deg)

    # Fine pass: ±4° around the coarse winner (always, even when winner is 0°
    # so that small CW tilts are detected even when no coarse angle beats 0°).
    for da in range(-4, 5):
        if da == 0:
            continue
        fine_angle = best_angle + float(da)
        if -90.0 <= fine_angle <= 90.0:
            v = _var(fine_angle)
            if v > best_var:
                best_var = v
                best_angle = fine_angle

    return best_angle


def detect_cardinal_rotation(img: Image.Image) -> tuple[int, float]:
    """Fast cardinal-only detector used for per-region orientation correction.

    Returns ``(rotation_to_apply, confidence)`` where:
    - ``rotation_to_apply`` ∈ {0, 90, 180, 270} — counter-clockwise degrees
    - ``confidence`` ∈ [0, 1]; values below ~0.70 indicate an uncertain result

    For full-page deskew (including intermediate angles) use ``deskew_image``.
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
        # Text is roughly horizontal → distinguish 0° from 180° via boundary asymmetry.
        sym_0 = _boundary_asymmetry(candidates[0])
        sym_180 = _boundary_asymmetry(candidates[2])
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
        confidence_raw = (score_90_270 - score_0_180) / (score_90_270 + score_0_180 + 1e-6)
        confidence = float(np.clip(confidence_raw * 2.0, 0.0, 1.0))

    return winner_k * 90, confidence


def deskew_image(
    img: Image.Image,
    *,
    min_angle_deg: float = 0.5,
    cardinal_confidence_threshold: float = 0.70,
) -> tuple[Image.Image, float]:
    """Detect and correct any skew angle plus orientation (0° vs 180°).

    Works for any rotation angle, not only cardinal multiples.  Cardinal
    snapping (within ``_CARDINAL_SNAP_DEG``) uses a lossless PIL transpose;
    all other angles use bicubic interpolation.

    Returns ``(corrected_image, net_ccw_correction)`` where
    ``net_ccw_correction`` is the total CCW degrees applied (stored in
    ``page_info["angle"]``).
    """
    net_ccw = 0.0

    # Step 1: Apply the dominant text angle first (any value in -90°..+90°).
    # This ensures the orientation check in Step 2 always runs on a roughly
    # horizontal image where boundary_asymmetry is reliable.  A tilted image
    # can have blurred projection band edges that fool the 0°/180° heuristic.
    angle = detect_page_angle(img)

    if abs(angle) >= min_angle_deg:
        # Snap angles close to a cardinal multiple to use lossless transpose.
        snapped = round(angle / 90) * 90  # nearest multiple of 90
        if abs(angle - snapped) <= _CARDINAL_SNAP_DEG and snapped != 0:
            img = img.transpose(_PIL_TRANSPOSE[int(snapped) % 360])
            net_ccw += snapped
            residual = angle - snapped
        else:
            residual = angle

        if abs(residual) >= min_angle_deg:
            img = img.rotate(
                residual,
                expand=True,
                resample=Image.Resampling.BICUBIC,
                fillcolor=(255, 255, 255),
            )
            net_ccw += residual

    # Step 2: Orientation check on the now-horizontal image.
    # 0° vs 180° have identical projection variance, so detect_page_angle
    # cannot distinguish them.  Two complementary signals are used:
    #
    # • For small applied corrections (|net_ccw| < 10°): bicubic resampling
    #   can distort band entry/exit values enough to flip the BA sign on small
    #   images.  Instead, compare the relative vertical position of text in the
    #   image: upright documents have text closer to the top (smaller top
    #   margin), so centre-of-mass < 0.5 means keep; > 0.5 means flip.
    #
    # • For larger corrections the padding from expand=True is substantial, so
    #   centre-of-mass becomes unreliable; fall back to boundary asymmetry.
    gray = _to_gray(img)
    if abs(net_ccw) < 10.0:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_dark = binary.sum(axis=1).astype(np.float64)
        total_dark = row_dark.sum()
        if total_dark > 0:
            rows_idx = np.arange(len(row_dark), dtype=np.float64)
            vcenter = float((rows_idx * row_dark).sum() / total_dark) / len(row_dark)
            inverted = vcenter > 0.5
            confidence = abs(vcenter - 0.5) * 2.0  # 0..1
        else:
            inverted = False
            confidence = 0.0
    else:
        ba = _boundary_asymmetry(gray)
        ba_flip = _boundary_asymmetry(np.rot90(gray, k=2))
        inverted = ba_flip > ba
        total_ba = abs(ba) + abs(ba_flip) + 1e-6
        confidence = float(np.clip((ba_flip - ba) / total_ba, 0.0, 1.0)) if inverted else 0.0

    threshold = _VCENTER_MIN_CONFIDENCE if abs(net_ccw) < 10.0 else cardinal_confidence_threshold
    if inverted and confidence >= threshold:
        img = img.transpose(Image.Transpose.ROTATE_180)
        net_ccw += 180

    return img, net_ccw
