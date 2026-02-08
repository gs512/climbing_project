# climbing_project/vision/diff_route.py

from typing import List, Tuple

import cv2
import numpy as np

from .homography import pixel_to_laser  # note the leading dot: relative import

# TUNABLE
DIFF_THRESH = 30
MIN_BLOB_AREA = 120
KERNEL_SIZE = 7

# HSV color ranges (OpenCV H: 0-179)
COLOR_RANGES = {
    "red": [(0, 10), (160, 179)],
    "yellow": [(18, 35)],
    "green": [(40, 85)],
    "blue": [(95, 135)],
}
MIN_SAT = 90
MIN_VAL = 90

_KERNEL = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)


def _classify_hue(mean_h: float) -> str:
    for name, ranges in COLOR_RANGES.items():
        for lo, hi in ranges:
            if lo <= mean_h <= hi:
                return name
    return "unknown"


def detect_circles_by_diff_with_colors(
    clean_path: str,
    route_path: str,
) -> List[Tuple[int, int, str]]:
    """
    Given a clean (no-route) screenshot and a route screenshot,
    return a list of (cx, cy, color_name) in **clean-image pixel coordinates**.
    If shapes differ we resize the route image to match the clean image size.
    """
    clean = cv2.imread(clean_path)
    route = cv2.imread(route_path)
    if clean is None or route is None:
        raise FileNotFoundError("Could not load clean or route image")

    # If shapes differ, resize route to match clean.
    if clean.shape != route.shape:
        h, w = clean.shape[:2]
        print(
            f"[diff_route] WARNING: shape mismatch clean={clean.shape}, "
            f"route={route.shape}. Resizing route to ({w}x{h})."
        )
        route = cv2.resize(route, (w, h), interpolation=cv2.INTER_LINEAR)

            
    diff = cv2.absdiff(route, clean)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    # Close thin rings -> solid blobs
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(route, cv2.COLOR_BGR2HSV)

    results: List[Tuple[int, int, str]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

        h, s, v = cv2.split(hsv)
        h = h[contour_mask == 255]
        s = s[contour_mask == 255]
        v = v[contour_mask == 255]

        valid = (s >= MIN_SAT) & (v >= MIN_VAL)
        if np.count_nonzero(valid) == 0:
            color = "unknown"
        else:
            mean_h = float(np.mean(h[valid]))
            color = _classify_hue(mean_h)

        results.append((cx, cy, color))
    return results


def detect_and_map_to_laser_with_colors(
    clean_path: str,
    route_path: str,
) -> List[Tuple[int, int, str]]:
    """
    Returns list of (x_dac, y_dac, color_name) in laser/DAC coordinates.
    """
    centers = detect_circles_by_diff_with_colors(clean_path, route_path)
    mapped: List[Tuple[int, int, str]] = []
    for cx, cy, color in centers:
        lx, ly = pixel_to_laser(cx, cy)
        mapped.append((lx, ly, color))
    return mapped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diff-based Boulderbot detector with colors")
    parser.add_argument("clean")
    parser.add_argument("route")
    args = parser.parse_args()

    pts = detect_circles_by_diff_with_colors(args.clean, args.route)
    print(pts)