import json
import os
import time
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import serial  # type: ignore
from homography import pixel_to_laser

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_BASE_DIR, "vision_config.json")

# Default configuration – can be overridden by vision_config.json
_DEFAULT_CONFIG = {
    "color_ranges": {
        "green": {"lower": [40, 100, 100], "upper": [80, 255, 255]},
        "blue": {"lower": [100, 150, 0], "upper": [140, 255, 255]},
        "yellow": {"lower": [20, 100, 100], "upper": [30, 255, 255]},
        # Red often wraps in HSV; you can add a second range if needed
        "red": {"lower": [0, 150, 50], "upper": [10, 255, 255]},
    },
    "area_range": [150.0, 5000.0],  # min, max contour area
    "circularity_min": 0.7,
    "kernel_size": 5,
    "serial_port": "/dev/ttyACM0",
    "baudrate": 115200,
    "dwell_seconds": 1.0,
}


def _load_config() -> Dict:
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Merge shallowly over defaults
            merged = {**_DEFAULT_CONFIG, **cfg}
            if "color_ranges" in cfg:
                merged["color_ranges"] = cfg["color_ranges"]
            return merged
        except Exception as exc:  # noqa: BLE001
            print(
                f"[render_upload] Failed to load vision_config.json, using defaults: {exc}"
            )
    return _DEFAULT_CONFIG


_CONFIG = _load_config()

_COLOR_RANGES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
for name, bounds in _CONFIG["color_ranges"].items():
    lower = np.array(bounds["lower"], dtype=np.uint8)
    upper = np.array(bounds["upper"], dtype=np.uint8)
    _COLOR_RANGES[name] = (lower, upper)

_AREA_RANGE: Tuple[float, float] = tuple(_CONFIG["area_range"])  # type: ignore[arg-type]
_CIRCULARITY_MIN: float = float(_CONFIG["circularity_min"])
_KERNEL = np.ones((_CONFIG["kernel_size"], _CONFIG["kernel_size"]), np.uint8)


def detect_holds(
    img_bgr: np.ndarray,
    color_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    area_range: Sequence[float] = None,
    circularity_min: float = None,
) -> List[Tuple[int, int, str]]:
    """Detect approximately circular colored holds in an image.

    Returns a list of (cx, cy, color_name) in pixel coordinates.
    """
    if color_ranges is None:
        color_ranges = _COLOR_RANGES
    if area_range is None:
        area_range = _AREA_RANGE
    if circularity_min is None:
        circularity_min = _CIRCULARITY_MIN

    min_area, max_area = float(area_range[0]), float(area_range[1])

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    holds: List[Tuple[int, int, str]] = []

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0.0:
                continue

            circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
            if circularity < circularity_min:
                continue

            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])

            holds.append((cx, cy, color_name))

    return holds


def extract_and_trace(
    image_path: str,
    serial_port: str = None,
    baudrate: int = None,
    dwell_seconds: float = None,
) -> None:
    """End-to-end helper: load image, detect holds, map to laser space, and trace.

    Intended for CLI/debug use. For Flask, use a thin wrapper that calls this.
    """
    if serial_port is None:
        serial_port = _CONFIG["serial_port"]
    if baudrate is None:
        baudrate = int(_CONFIG["baudrate"])
    if dwell_seconds is None:
        dwell_seconds = float(_CONFIG["dwell_seconds"])

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    holds = detect_holds(img)
    if not holds:
        print("[render_upload] No holds detected.")
        return

    print(
        f"[render_upload] Detected {len(holds)} holds, starting trace on {serial_port}."
    )

    with serial.Serial(serial_port, baudrate, timeout=0.1) as ser:
        for px, py, color_name in holds:
            lx, ly = pixel_to_laser(px, py)
            msg = f"{lx},{ly}\n"
            ser.write(msg.encode("ascii"))
            print(f"  -> {color_name}: pixel=({px},{py}) laser=({lx},{ly})")
            time.sleep(dwell_seconds)


def trace_boulderbot_route(clean_path: str, route_path: str) -> None:
    pts = detect_and_map_to_laser_with_colors(clean_path, route_path)
    if not pts:
        print("[boulderbot] No circles detected.")
        return

    with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1) as ser:
        print(f"[boulderbot] Tracing {len(pts)} circles")
        for x_dac, y_dac, color_name in pts:
            # c = COLOR_CODE.get(color_name, "U")
            msg = f"{x_dac},{y_dac}\n"
            ser.write(msg.encode("ascii"))
            print(f"  -> {color_name}: ({x_dac},{y_dac})")
            time.sleep(DWELL_SECONDS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect holds in an image and trace them with the laser."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--port", help="Serial port for ESP32 (default from config)")
    parser.add_argument("--baud", type=int, help="Baudrate (default from config)")
    parser.add_argument(
        "--dwell", type=float, help="Dwell seconds per hold (default from config)"
    )
    args = parser.parse_args()

    extract_and_trace(
        args.image, serial_port=args.port, baudrate=args.baud, dwell_seconds=args.dwell
    )
