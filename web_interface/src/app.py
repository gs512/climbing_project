# climbing_project/web_interface/src/app.py
import os
import random
import sqlite3
import string
import sys
import time
import json
from typing import Optional, List, Tuple

import numpy as np
import cv2
import serial  # type: ignore
from flask import Flask, redirect, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename

# ---- Path setup ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # web_interface/src
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- Vision imports ----
from vision.diff_route import (  # type: ignore  # noqa: E402
    detect_and_map_to_laser_with_colors,
    detect_circles_by_diff_with_colors,
)
from vision.homography import pixel_to_laser, reload_homography  # type: ignore  # noqa: E402

# ---- Flask app ----
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
DB_PATH = os.path.join(BASE_DIR, "routes.db")

# Serial settings
SERIAL_PORT = os.environ.get("LASER_SERIAL_PORT", "/dev/ttyACM0")
BAUDRATE = 115200
_SERIAL_HANDLE: Optional[serial.Serial] = None

# Homography & reference image paths
CLEAN_IMAGE_PATH = os.path.join(BASE_DIR, "static", "uploads", "clean.png")
HOMOGRAPHY_JSON_PATH = os.path.join(PROJECT_ROOT, "vision", "homography.json")

# Role mapping & sizes (diameter in DAC units 0..4095)
ROLE_BY_COLOR = {
    "red": "start",
    "green": "finish",
    "blue": "mid",
    "yellow": "mid",
    "unknown": "mid",
}
SHAPE_BY_ROLE = {"start": "X", "finish": "S", "mid": "C"}
DIAM_BY_ROLE = {"start": 260, "mid": 200, "finish": 300}


# ---- Utilities ----
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_serial() -> Optional[serial.Serial]:
    """Return a cached serial handle or open one."""
    global _SERIAL_HANDLE
    if _SERIAL_HANDLE and getattr(_SERIAL_HANDLE, "is_open", False):
        return _SERIAL_HANDLE
    try:
        _SERIAL_HANDLE = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        print(f"[app] Opened serial on {SERIAL_PORT}")
    except Exception as exc:  # noqa: BLE001
        print(f"[app] ERROR opening serial port {SERIAL_PORT}: {exc}")
        _SERIAL_HANDLE = None
    return _SERIAL_HANDLE


def send_shape_cmds(pts: List[Tuple[int, int, str]], dwell_gap: float = 0.02):
    """
    pts: list of (x_dac, y_dac, color_name). Send encoded shape commands over serial.
    Protocol: ASCII CSV: X,Y,TYPE,SZ\n
      TYPE: 'C' circle, 'X' cross, 'S' square
      SZ: diameter in DAC units
    """
    ser = get_serial()
    if ser is None:
        print("[app] No serial available; skipping sending shape cmds.")
        return

    enriched = []
    for x, y, color in pts:
        role = ROLE_BY_COLOR.get(color, "mid")
        shape = SHAPE_BY_ROLE.get(role, "C")
        size = DIAM_BY_ROLE.get(role, DIAM_BY_ROLE["mid"])
        enriched.append((x, y, shape, int(size), role, color))

    # Ordering: start first, mids bottom->top, finish last
    def ordering_key(item):
        x, y, shape, size, role, color = item
        if role == "start":
            order = 0
        elif role == "finish":
            order = 3
        else:
            order = 1
        return (order, -y if order == 1 else 0)

    enriched.sort(key=ordering_key)

    for x, y, shape, size, role, color in enriched:
        line = f"{int(x)},{int(y)},{shape},{int(size)}\n"
        try:
            ser.write(line.encode("ascii"))
            time.sleep(dwell_gap)
            print(f"[app] SENT {line.strip()} role={role} color={color}")
        except Exception as exc:  # noqa: BLE001
            print(f"[app] ERROR sending serial command: {exc}")


def send_shape_cmds_from_roles(
    points: List[dict],
    clean_size: Tuple[int, int],
    route_size: Tuple[int, int],
    dwell_gap: float = 0.02,
):
    """
    points: list of dicts with keys:
      - x, y: pixel coords in ROUTE image space
      - role: 'start' | 'mid' | 'finish'
    clean_size: (width, height) of clean.png
    route_size: (width, height) of this route image

    We scale route coords → clean coords, then map with pixel_to_laser.
    """
    ser = get_serial()
    if ser is None:
        print("[app] No serial available; skipping custom trace.")
        return

    wc, hc = clean_size
    wr, hr = route_size
    sx = wc / float(wr)
    sy = hc / float(hr)

    enriched = []
    for p in points:
        px_route = float(p["x"])
        py_route = float(p["y"])
        role = str(p.get("role", "mid"))
        if role not in ("start", "mid", "finish"):
            role = "mid"

        # scale route → clean
        px_clean = px_route * sx
        py_clean = py_route * sy

        lx, ly = pixel_to_laser(px_clean, py_clean)
        shape = SHAPE_BY_ROLE.get(role, "C")
        size = DIAM_BY_ROLE.get(role, DIAM_BY_ROLE["mid"])
        enriched.append((lx, ly, shape, int(size), role))

    # ordering same as above
    def ordering_key(item):
        x, y, shape, size, role = item
        if role == "start":
            order = 0
        elif role == "finish":
            order = 3
        else:
            order = 1
        return (order, -y if order == 1 else 0)

    enriched.sort(key=ordering_key)

    for x, y, shape, size, role in enriched:
        line = f"{int(x)},{int(y)},{shape},{int(size)}\n"
        try:
            ser.write(line.encode("ascii"))
            time.sleep(dwell_gap)
            print(f"[app] SENT {line.strip()} role={role}")
        except Exception as exc:  # noqa: BLE001
            print(f"[app] ERROR sending serial command: {exc}")


def _save_homography_matrix(h: np.ndarray) -> None:
    data = {"h": h.tolist()}
    os.makedirs(os.path.dirname(HOMOGRAPHY_JSON_PATH), exist_ok=True)
    with open(HOMOGRAPHY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[calib] Saved homography to {HOMOGRAPHY_JSON_PATH!r}")
    try:
        reload_homography()
    except Exception as exc:
        print(f"[app] Warning: failed to reload homography: {exc}")


def _load_homography_matrix() -> np.ndarray:
    if not os.path.exists(HOMOGRAPHY_JSON_PATH):
        raise FileNotFoundError(f"Homography file not found: {HOMOGRAPHY_JSON_PATH}")
    with open(HOMOGRAPHY_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    h = np.asarray(data["h"], dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError("Homography matrix must be 3x3")
    return h


def _compute_homography_preview(pts: List[List[float]]):
    """
    Helper: given 4 image points TL,TR,BR,BL, compute homography and return:
      h, mapped_dac (list of 4 [x,y]), preview_pixels (corners+center inverse-mapped).
    Does NOT save h.
    """
    pts_src = np.array(pts, dtype=np.float32)
    if pts_src.shape != (4, 2):
        raise ValueError("Expected 4x2 points")

    pts_dst = np.array(
        [[0.0, 0.0], [4095.0, 0.0], [4095.0, 4095.0], [0.0, 4095.0]],
        dtype=np.float32,
    )

    h, status = cv2.findHomography(pts_src, pts_dst)
    if h is None or h.shape != (3, 3):
        raise RuntimeError("cv2.findHomography failed")

    # Map image points to DAC
    pts_in = pts_src.reshape((4, 1, 2)).astype(np.float32)
    mapped_dac = cv2.perspectiveTransform(pts_in, h).reshape(-1, 2).tolist()

    # Inverse: DAC corners + center for pixel preview
    h_inv = np.linalg.inv(h)
    preview_dac = np.array(
        [[0.0, 0.0], [4095.0, 0.0], [4095.0, 4095.0], [0.0, 4095.0], [2048.0, 2048.0]],
        dtype=np.float32,
    )
    preview_in = preview_dac.reshape((preview_dac.shape[0], 1, 2))
    preview_pixels = cv2.perspectiveTransform(preview_in, h_inv).reshape(-1, 2).tolist()

    return h, mapped_dac, preview_pixels


# ---- Web UI routes (index/upload/delete/trace) ----
@app.route("/")
def index():
    with get_db_connection() as conn:
        routes = conn.execute("SELECT * FROM routes ORDER BY created_at DESC").fetchall()
    return render_template("index.html", routes=routes, request=request)


@app.route("/upload", methods=["POST"])
def upload_route():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        return redirect(url_for("index"))
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(save_path):
        prefix = "".join(random.choice(string.ascii_uppercase) for _ in range(4))
        filename = f"{prefix}_{filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # infer image size for convenience (not critical)
    w = h = None
    try:
        img = cv2.imread(save_path)
        if img is not None:
            h, w = img.shape[:2]
    except Exception:
        pass

    route_name = request.form.get("name", "") or "".join(
        random.choice(string.ascii_uppercase) for _ in range(6)
    )
    difficulty = request.form.get("difficulty", "V0")
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO routes (filename, name, difficulty, img_width, img_height) "
            "VALUES (?, ?, ?, ?, ?)",
            (filename, route_name, difficulty, w, h),
        )
        conn.commit()
    return redirect(url_for("index"))


@app.route("/trace/<int:route_id>")
def trace_route(route_id):
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM routes WHERE id = ?", (route_id,)).fetchone()
    if row is None:
        return redirect(url_for("index"))
    route_img = os.path.join(app.config["UPLOAD_FOLDER"], row["filename"])
    try:
        pts = detect_and_map_to_laser_with_colors(CLEAN_IMAGE_PATH, route_img)
    except Exception as exc:  # noqa: BLE001
        print(f"[app] ERROR detecting route (instant trace): {exc}")
        return redirect(url_for("index"))
    if not pts:
        print("[app] No points detected for tracing.")
        return redirect(url_for("index"))
    send_shape_cmds(pts)
    return redirect(url_for("index"))


@app.route("/delete/<int:route_id>", methods=["POST"])
def delete_route(route_id):
    with get_db_connection() as conn:
        r = conn.execute("SELECT * FROM routes WHERE id = ?", (route_id,)).fetchone()
        if r:
            conn.execute("DELETE FROM routes WHERE id = ?", (route_id,))
            conn.commit()
            fp = os.path.join(app.config["UPLOAD_FOLDER"], r["filename"])
            try:
                os.remove(fp)
            except OSError:
                pass
    return redirect(url_for("index"))


# ---- Detection preview + editable trace ----
@app.route("/route/<int:route_id>/preview", methods=["GET"])
def preview_route(route_id: int):
    """
    Show detection result, allow user to edit/add points before sending to ESP.

    If the route already has saved points in the DB (points_json), we use those
    as the initial points and SKIP detection. Detection is only used as a
    fallback for new routes.
    """
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM routes WHERE id = ?", (route_id,)).fetchone()
    if row is None:
        return redirect(url_for("index"))

    image_rel = row["filename"]
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_rel)
    saved_points_raw = row["points_json"] if "points_json" in row.keys() else None

    # Load images for size / scaling
    clean = cv2.imread(CLEAN_IMAGE_PATH)
    route_img = cv2.imread(image_path)
    if clean is None or route_img is None:
        print("[app] ERROR: could not load clean or route image for preview.")
        return redirect(url_for("index"))

    hc, wc = clean.shape[:2]
    hr, wr = route_img.shape[:2]
    sx = wr / float(wc)
    sy = hr / float(hc)

    points = []

    # 1) Try to use saved points (already in route coords)
    if saved_points_raw:
        try:
            stored = json.loads(saved_points_raw)
            for i, p in enumerate(stored):
                try:
                    x = int(round(float(p["x"])))
                    y = int(round(float(p["y"])))
                except Exception:
                    continue
                role = str(p.get("role", "mid"))
                color = str(p.get("color", "manual"))
                points.append(
                    {"id": i, "x": x, "y": y, "color": color, "role": role}
                )
            print(f"[app] Using {len(points)} saved points for route {route_id}")
        except Exception as exc:
            print(f"[app] ERROR parsing saved points_json: {exc}; falling back to detection")

    # 2) If we still have no points, run detection once
    if not points:
        try:
            detections = detect_circles_by_diff_with_colors(CLEAN_IMAGE_PATH, image_path)
        except Exception as exc:
            print(f"[app] ERROR detecting route for preview: {exc}")
            detections = []
        for i, (cx_clean, cy_clean, color) in enumerate(detections):
            rx = int(round(cx_clean * sx))
            ry = int(round(cy_clean * sy))
            role = ROLE_BY_COLOR.get(color, "mid")
            points.append(
                {"id": i, "x": rx, "y": ry, "color": color, "role": role}
            )
        print(f"[app] Detection produced {len(points)} points for route {route_id}")

    points_json = json.dumps(points)
    image_url = url_for("static", filename=f"uploads/{image_rel}")
    return render_template(
        "preview.html",
        route=row,
        image_url=image_url,
        points_json=points_json,
    )
    

@app.route("/route/<int:route_id>/trace_custom", methods=["POST"])
def trace_custom_route(route_id: int):
    """
    Receive edited points from preview and send to ESP.
    Also persists the points (route-space coords) in the DB so that next time
    we open the preview we skip detection and load these points instead.
    """
    payload = request.get_json(silent=True)
    if not payload or "points" not in payload:
        return jsonify({"ok": False, "error": "Missing 'points'"}), 400

    pts_raw = payload["points"]
    if not isinstance(pts_raw, list) or len(pts_raw) == 0:
        return jsonify({"ok": False, "error": "No points provided"}), 400

    filtered_for_send = []
    persisted_points = []
    for p in pts_raw:
        try:
            x = float(p["x"])
            y = float(p["y"])
            role = str(p.get("role", "mid"))
        except Exception:
            continue
        filtered_for_send.append({"x": x, "y": y, "role": role})
        persisted_points.append(
            {
                "x": x,
                "y": y,
                "role": role,
                "color": p.get("color", "manual"),
            }
        )

    if not filtered_for_send:
        return jsonify({"ok": False, "error": "No valid points"}), 400

    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM routes WHERE id = ?", (route_id,)).fetchone()
        if row is None:
            return jsonify({"ok": False, "error": "Route not found"}), 404

        # Persist points_json so next preview re-uses them
        conn.execute(
            "UPDATE routes SET points_json = ? WHERE id = ?",
            (json.dumps(persisted_points), route_id),
        )
        conn.commit()

    image_rel = row["filename"]
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_rel)

    clean = cv2.imread(CLEAN_IMAGE_PATH)
    route_img = cv2.imread(image_path)
    if clean is None or route_img is None:
        return jsonify({"ok": False, "error": "Could not load clean or route image"}), 500

    hc, wc = clean.shape[:2]
    hr, wr = route_img.shape[:2]

    try:
        # this is the function we already wrote earlier:
        #   - scales route coords -> clean
        #   - maps clean -> DAC via pixel_to_laser
        #   - applies role->shape/size
        send_shape_cmds_from_roles(filtered_for_send, (wc, hc), (wr, hr))
    except Exception as exc:
        print(f"[app] ERROR sending custom trace: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({"ok": True})

@app.route("/route/<int:route_id>/update_meta", methods=["POST"])
def update_route_meta(route_id: int):
    """Update route name and difficulty from preview page."""
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"ok": False, "error": "Missing JSON body"}), 400

    name = payload.get("name", "").strip()
    difficulty = payload.get("difficulty", "").strip()

    if not name:
        return jsonify({"ok": False, "error": "Name cannot be empty"}), 400
    if not difficulty:
        difficulty = "V0"

    with get_db_connection() as conn:
        cur = conn.execute(
            "UPDATE routes SET name = ?, difficulty = ? WHERE id = ?",
            (name, difficulty, route_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            return jsonify({"ok": False, "error": "Route not found"}), 404

    return jsonify({"ok": True})


# ---- Calibration endpoints ----
@app.route("/calibrate", methods=["GET"])
def calibrate_form():
    if not os.path.exists(CLEAN_IMAGE_PATH):
        return f"<p>Reference image not found. Place clean.png at: {CLEAN_IMAGE_PATH}</p>", 404
    clean_url = url_for("static", filename="uploads/clean.png")
    return render_template("calibrate.html", clean_image_url=clean_url)


@app.route("/calibrate/submit", methods=["POST"])
def calibrate_submit():
    payload = request.get_json(silent=True)
    if not payload or "points" not in payload:
        return jsonify({"ok": False, "error": "Missing JSON 'points' field."}), 400

    pts = payload["points"]
    if not isinstance(pts, list) or len(pts) != 4:
        return jsonify({"ok": False, "error": "Expected 4 points (TL, TR, BR, BL)."}), 400

    try:
        h, mapped_dac, preview_pixels = _compute_homography_preview(pts)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Preview compute failed: {exc}"}), 500

    return jsonify({"ok": True, "mapped_dac": mapped_dac, "preview_pixels": preview_pixels})


@app.route("/calibrate/save", methods=["POST"])
def calibrate_save():
    payload = request.get_json(silent=True)
    if not payload or "points" not in payload:
        return jsonify({"ok": False, "error": "Missing JSON 'points' field."}), 400

    pts = payload["points"]
    if not isinstance(pts, list) or len(pts) != 4:
        return jsonify({"ok": False, "error": "Expected 4 points (TL, TR, BR, BL)."}), 400

    try:
        h, mapped_dac, preview_pixels = _compute_homography_preview(pts)
        _save_homography_matrix(h)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Save failed: {exc}"}), 500

    return jsonify({"ok": True, "mapped_dac": mapped_dac, "preview_pixels": preview_pixels})


@app.route("/calibrate/test_draw", methods=["POST"])
def calibrate_test_draw():
    payload = request.get_json(silent=True)
    if not payload or "mapped_dac" not in payload:
        return jsonify({"ok": False, "error": "Missing mapped_dac list"}), 400
    try:
        pts = payload["mapped_dac"]
        pts_parsed = []
        for pair in pts:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            x = float(pair[0])
            y = float(pair[1])
            pts_parsed.append((int(round(x)), int(round(y))))
        ser = get_serial()
        if ser is None:
            return jsonify({"ok": False, "error": "No serial available"}), 500
        for x, y in pts_parsed:
            line = f"{int(x)},{int(y)},C,180\n"
            ser.write(line.encode("ascii"))
            time.sleep(0.05)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/calibrate/draw_grid", methods=["POST"])
def calibrate_draw_grid():
    try:
        h = _load_homography_matrix()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Cannot load homography: {exc}"}), 500

    n_rows = 4
    n_cols = 4
    dac_points = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = (j + 0.5) / n_cols * 4095.0
            y = (i + 0.5) / n_rows * 4095.0
            dac_points.append([x, y])

    ser = get_serial()
    if ser is None:
        return jsonify({"ok": False, "error": "No serial available"}), 500
    try:
        for x, y in dac_points:
            line = f"{int(round(x))},{int(round(y))},C,150\n"
            ser.write(line.encode("ascii"))
            time.sleep(0.04)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Serial send failed: {exc}"}), 500

    try:
        h_inv = np.linalg.inv(h)
        dac_arr = np.array(dac_points, dtype=np.float32).reshape(-1, 1, 2)
        pix = cv2.perspectiveTransform(dac_arr, h_inv).reshape(-1, 2).tolist()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Inverse transform failed: {exc}"}), 500

    return jsonify({"ok": True, "grid_pixels": pix, "dac_points": dac_points})


@app.route("/calibrate/manual", methods=["POST"])
def calibrate_manual():
    """Manually move the laser to a specific X,Y DAC coordinate."""
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"ok": False, "error": "Missing JSON body"}), 400

    try:
        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))
        # Clamp to 0-4095
        x = max(0, min(4095, x))
        y = max(0, min(4095, y))

        ser = get_serial()
        if ser is None:
            return jsonify({"ok": False, "error": "No serial available"}), 500
        
        # Send 'P' for Point (fallback in ESP firmware)
        # Format: X,Y,P,1
        line = f"{x},{y},P,1\n"
        ser.write(line.encode("ascii"))
        return jsonify({"ok": True, "x": x, "y": y})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def _ensure_schema():
    """Create / migrate routes table (adds points_json,img_width,img_height if missing)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                name TEXT NOT NULL,
                difficulty TEXT DEFAULT 'V0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # add new columns if they don't exist
        cols = {row[1] for row in c.execute("PRAGMA table_info(routes)")}
        if "points_json" not in cols:
            c.execute("ALTER TABLE routes ADD COLUMN points_json TEXT")
        if "img_width" not in cols:
            c.execute("ALTER TABLE routes ADD COLUMN img_width INTEGER")
        if "img_height" not in cols:
            c.execute("ALTER TABLE routes ADD COLUMN img_height INTEGER")
        c.commit()


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    _ensure_schema()
    app.run(host="0.0.0.0", port=8000, debug=True)