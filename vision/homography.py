import json
import os
from typing import Tuple

import cv2
import numpy as np

# Base directory for this module
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CALIB_FILE = os.path.join(_BASE_DIR, "homography.json")

# Default calibration points – replace with your measured values.
# Image (pixel) coordinates of the 4 board corners, in order:
# top-left, top-right, bottom-right, bottom-left.
_DEFAULT_PTS_SRC = np.array(
    [[141, 131], [480, 159], [493, 630], [64, 601]],
    dtype=np.float32,
)

# Target laser/DAC space (0-4095 in both axes).
_DEFAULT_PTS_DST = np.array(
    [[0, 0], [4095, 0], [4095, 4095], [0, 4095]],
    dtype=np.float32,
)


def _compute_homography(
    pts_src: np.ndarray = None, pts_dst: np.ndarray = None
) -> np.ndarray:
    """Compute homography from image space to laser space."""
    if pts_src is None:
        pts_src = _DEFAULT_PTS_SRC
    if pts_dst is None:
        pts_dst = _DEFAULT_PTS_DST

    h, status = cv2.findHomography(pts_src, pts_dst)
    if h is None:
        raise RuntimeError("Failed to compute homography matrix.")
    return h


def _save_homography(h: np.ndarray, path: str = _CALIB_FILE) -> None:
    """Persist the homography matrix as JSON."""
    data = {"h": h.tolist()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _load_or_create_homography() -> np.ndarray:
    """Load homography from disk or recompute using defaults."""
    if os.path.exists(_CALIB_FILE):
        try:
            with open(_CALIB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            h = np.asarray(data["h"], dtype=np.float64)
            if h.shape == (3, 3):
                return h
        except Exception as exc:  # noqa: BLE001
            print(f"[homography] Failed to load saved homography, recomputing: {exc}")

    h = _compute_homography()
    try:
        _save_homography(h)
    except Exception as exc:  # noqa: BLE001
        print(f"[homography] Warning: could not save homography to disk: {exc}")
    return h


# Single shared homography for this process
_H = _load_or_create_homography()


def reload_homography() -> None:
    """Reload the homography matrix from disk."""
    global _H
    _H = _load_or_create_homography()
    print("[homography] Matrix reloaded from disk.")


def pixel_to_laser(
    pixel_x: float,
    pixel_y: float,
    clamp: bool = True,
    max_val: int = 4095,
) -> Tuple[int, int]:
    """Map image pixel coordinates to laser/DAC coordinates.

    Returns a pair of integers (x, y) in 0..max_val.

    Raises ValueError if the transform produces NaNs.
    """
    point = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, _H)
    x, y = transformed[0, 0]

    if np.isnan(x) or np.isnan(y):
        raise ValueError(f"Homography produced NaN for input ({pixel_x}, {pixel_y})")

    if clamp:
        x = max(0.0, min(float(max_val), float(x)))
        y = max(0.0, min(float(max_val), float(y)))

    return int(round(x)), int(round(y))


def get_laser_coords(pixel_x: float, pixel_y: float) -> Tuple[int, int]:
    """Backwards-compatible wrapper for existing code."""
    return pixel_to_laser(pixel_x, pixel_y)
