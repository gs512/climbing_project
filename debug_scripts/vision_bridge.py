import os
import time

import cv2
import serial  # type: ignore
from vision.homography import pixel_to_laser
from render_upload import detect_holds

# Serial configuration – keep this in sync with render_upload/config
SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE = 115200

# Camera configuration
CAM_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
DEBUG_FRAME_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vision_debug.jpg"
)


def main() -> None:
    # 1. Setup Serial to ESP32-S3
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    except Exception as exc:  # noqa: BLE001
        print(f"[vision_bridge] Failed to open serial port {SERIAL_PORT}: {exc}")
        return

    # 2. Setup camera with MJPEG at 1080p
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[vision_bridge] ERROR: Could not open camera.")
        ser.close()
        return

    print("[vision_bridge] Vision Bridge Active. Press Ctrl+C to stop.")

    last_debug_save = 0.0
    debug_interval = 2.0  # seconds

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[vision_bridge] WARN: Failed to grab frame, retrying...")
                time.sleep(0.1)
                continue

            holds = detect_holds(frame)

            for px, py, color_name in holds:
                try:
                    lx, ly = pixel_to_laser(px, py)
                except ValueError as exc:
                    print(f"[vision_bridge] Homography error for ({px},{py}): {exc}")
                    continue

                msg = f"{lx},{ly}\n"
                ser.write(msg.encode("ascii"))
                print(
                    f"[vision_bridge] {color_name}: pixel=({px},{py}) laser=({lx},{ly})"
                )
                time.sleep(0.05)  # small dwell to avoid flooding

            # Periodically save a frame for offline inspection
            now = time.time()
            if now - last_debug_save >= debug_interval:
                last_debug_save = now
                try:
                    cv2.imwrite(DEBUG_FRAME_PATH, frame)
                except Exception as exc:  # noqa: BLE001
                    print(f"[vision_bridge] Failed to save debug frame: {exc}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[vision_bridge] Stopping...")
    finally:
        cap.release()
        ser.close()


if __name__ == "__main__":
    main()
