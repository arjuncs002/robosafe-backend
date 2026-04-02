#!/usr/bin/env python3
"""
ROBOSAFE - Jetson Nano Main (FINAL)
Camera: Essecloud via RJ45 at 192.168.1.88
"""

import cv2
import time
import math
import serial
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
BACKEND_URL    = "https://robosafe-backend.onrender.com"
RTSP_URL       = "rtsp://admin:@192.168.1.88:554/ch0_0.264"  # RJ45 direct
MODEL_PATH     = "/home/project/robosafe/yolov8n.pt"
SERIAL_PORT    = "/dev/ttyTHS1"
SERIAL_BAUD    = 115200
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

# Auto-drive thresholds
STOP_AREA_RATIO         = 0.30
LOST_FRAMES_BEFORE_SCAN = 20
DEAD_ZONE_PX            = 40

# Odometry
BASE_SPEED_MPS    = 0.15
SPIN_360_DURATION = 3.5

# HTTP rate limit
HTTP_CMD_RATE_LIMIT = 0.3

# ============================================================
# SERIAL
# ============================================================
ser = None

def init_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        print(f"[Serial] Connected: {SERIAL_PORT}")
    except Exception as e:
        print(f"[Serial] WARN: {e} — HTTP fallback only")
        ser = None

def send_serial(cmd: str):
    if ser and ser.is_open:
        try:
            ser.write((cmd.strip() + "\n").encode())
        except Exception:
            pass

# ============================================================
# HTTP COMMAND
# ============================================================
_http_lock     = threading.Lock()
_last_http_cmd = ""
_last_http_ts  = 0.0

def send_http_command(cmd: str):
    global _last_http_cmd, _last_http_ts
    now = time.time()
    with _http_lock:
        if cmd == _last_http_cmd and (now - _last_http_ts) < HTTP_CMD_RATE_LIMIT:
            return
        _last_http_cmd = cmd
        _last_http_ts  = now
    try:
        requests.post(f"{BACKEND_URL}/api/auto_command",
                      json={"command": cmd}, timeout=1.5)
    except Exception:
        pass

def send_command(cmd: str):
    send_serial(cmd)
    send_http_command(cmd)

# ============================================================
# BACKEND HELPERS
# ============================================================
_executor = ThreadPoolExecutor(max_workers=4)

def post_state(human_count: int, detections: list):
    def _do():
        try:
            requests.post(f"{BACKEND_URL}/api/state",
                          json={"human_count": human_count,
                                "detections": detections}, timeout=2)
        except Exception:
            pass
    _executor.submit(_do)

def post_map_flag(flag_type: str, x: float, y: float, label: str = ""):
    def _do():
        try:
            requests.post(f"{BACKEND_URL}/api/map/flag",
                          json={"flag_type": flag_type, "x": x, "y": y,
                                "label": label, "ts": time.time()}, timeout=2)
        except Exception:
            pass
    _executor.submit(_do)

def post_rover_position(x: float, y: float, heading: float):
    def _do():
        try:
            requests.post(f"{BACKEND_URL}/api/map/position",
                          json={"x": x, "y": y, "heading": heading,
                                "ts": time.time()}, timeout=1)
        except Exception:
            pass
    _executor.submit(_do)

def get_mode() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/api/drive_mode/public", timeout=2)
        if r.status_code == 200:
            return r.json().get("mode", "MANUAL")
    except Exception:
        pass
    return "MANUAL"

def post_alert():
    try:
        requests.post(f"{BACKEND_URL}/api/alert",
                      json={"type": "siren", "ts": time.time()}, timeout=1)
    except Exception:
        pass

def get_life_confirm_pending():
    try:
        r = requests.get(f"{BACKEND_URL}/api/life_confirm/pending", timeout=1)
        if r.status_code == 200:
            d = r.json()
            return d.get("confirmed", False), d.get("result", "")
    except Exception:
        pass
    return False, ""

def clear_life_confirm():
    try:
        requests.post(f"{BACKEND_URL}/api/life_confirm/clear", timeout=1)
    except Exception:
        pass

# ============================================================
# ODOMETRY
# ============================================================
class Odometry:
    def __init__(self):
        self.x        = 0.0
        self.y        = 0.0
        self.heading  = 0.0
        self._last_ts = time.time()

    def update(self, cmd: str):
        now  = time.time()
        dt   = now - self._last_ts
        self._last_ts = now
        dist = BASE_SPEED_MPS * dt
        if cmd == "FORWARD":
            self.x += dist * math.sin(self.heading)
            self.y += dist * math.cos(self.heading)
        elif cmd == "BACKWARD":
            self.x -= dist * math.sin(self.heading)
            self.y -= dist * math.cos(self.heading)
        elif cmd in ("LEFT", "RIGHT"):
            sign = -1 if cmd == "LEFT" else 1
            self.heading += sign * math.radians(60) * dt

    def position(self):
        return self.x, self.y, self.heading

# ============================================================
# STEERING
# ============================================================
def get_steering_command(bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox
    area       = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h
    if area / frame_area >= STOP_AREA_RATIO:
        return "STOP", True
    cx    = (x1 + x2) // 2
    error = cx - (frame_w // 2)
    if abs(error) < DEAD_ZONE_PX:
        return "FORWARD", False
    return ("LEFT", False) if error < 0 else ("RIGHT", False)

# ============================================================
# AUTO-DRIVE STATE MACHINE
# ============================================================
class AutoDriver:
    SCAN     = "SCAN"
    APPROACH = "APPROACH"
    ARRIVED  = "ARRIVED"
    WAITING  = "WAITING"
    SPIN360  = "SPIN360"

    def __init__(self, odo: Odometry):
        self.state       = self.SCAN
        self.odo         = odo
        self.lost_frames = 0
        self.scan_dir    = "LEFT"
        self.alert_sent  = False
        self.spin_start  = 0.0

    def reset(self):
        self.state       = self.SCAN
        self.lost_frames = 0
        self.alert_sent  = False

    def step(self, detections: list, frame_w: int, frame_h: int) -> str:
        x, y, heading = self.odo.position()

        if self.state == self.SCAN:
            if detections:
                print("[AutoDrive] Human found → APPROACH")
                self.state       = self.APPROACH
                self.lost_frames = 0
                post_map_flag("detected", x, y, "Human detected")
            return self.scan_dir

        elif self.state == self.APPROACH:
            if not detections:
                self.lost_frames += 1
                if self.lost_frames > LOST_FRAMES_BEFORE_SCAN:
                    print("[AutoDrive] Lost human → SCAN")
                    self.state = self.SCAN
                    return self.scan_dir
                return "FORWARD"
            self.lost_frames = 0
            best = max(detections,
                       key=lambda d: (d["bbox"][2]-d["bbox"][0]) *
                                     (d["bbox"][3]-d["bbox"][1]))
            cmd, stop = get_steering_command(best["bbox"], frame_w, frame_h)
            if stop:
                print("[AutoDrive] Arrived → ARRIVED")
                post_map_flag("arrived", x, y, "Rover arrived")
                self.state      = self.ARRIVED
                self.alert_sent = False
            return cmd

        elif self.state == self.ARRIVED:
            if not self.alert_sent:
                self.alert_sent = True
                _executor.submit(post_alert)
                print("[AutoDrive] Siren alert sent → WAITING")
                self.state = self.WAITING
            return "STOP"

        elif self.state == self.WAITING:
            confirmed, result = get_life_confirm_pending()
            if confirmed:
                flag_type = "alive" if result == "alive" else "not_alive"
                post_map_flag(flag_type, x, y, f"Life: {result}")
                clear_life_confirm()
                print(f"[AutoDrive] Vitals: {result} → SPIN360")
                self.state      = self.SPIN360
                self.spin_start = time.time()
            return "STOP"

        elif self.state == self.SPIN360:
            if time.time() - self.spin_start >= SPIN_360_DURATION:
                print("[AutoDrive] Spin done → SCAN")
                self.state = self.SCAN
                return "STOP"
            return "LEFT"

        return "STOP"

# ============================================================
# FRAME GRABBER THREAD
# Runs in background — grabs frames continuously so buffer
# never fills up. Main thread reads latest frame only.
# ============================================================
class FrameGrabber:
    def __init__(self, url: str):
        self.url     = url
        self._frame  = None
        self._lock   = threading.Lock()
        self.running = True
        self._t      = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Camera] Opened {w}x{h} ✓")
        else:
            print(f"[Camera] FAILED to open!")
        return cap

    def _loop(self):
        cap        = self._open()
        fail_count = 0
        while self.running:
            if not cap or not cap.isOpened():
                time.sleep(2.0)
                cap = self._open()
                continue
            ret, frame = cap.read()
            if ret and frame is not None:
                fail_count = 0
                # Resize immediately — faster YOLO + less memory
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                with self._lock:
                    self._frame = frame
            else:
                fail_count += 1
                if fail_count > 10:
                    print("[Camera] Reconnecting...")
                    cap.release()
                    time.sleep(2.0)
                    cap        = self._open()
                    fail_count = 0
                else:
                    time.sleep(0.03)

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False

# ============================================================
# MJPEG STREAM SERVER
# Runs on port 5000 — website fetches from here
# ============================================================
from http.server import BaseHTTPRequestHandler, HTTPServer
import struct

latest_jpeg = None
jpeg_lock   = threading.Lock()

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress access logs

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with jpeg_lock:
                        jpg = latest_jpeg
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)   # ~30fps
            except Exception:
                pass
        else:
            self.send_response(404)
            self.end_headers()

def start_stream_server():
    server = HTTPServer(("0.0.0.0", 5000), MJPEGHandler)
    print("[Stream] MJPEG server running on port 5000")
    server.serve_forever()

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print("  ROBOSAFE Jetson Main — FINAL")
    print("  Camera: RJ45 @ 192.168.1.88")
    print("=" * 55)

    init_serial()

    print(f"[YOLO] Loading: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[YOLO] Ready.")

    # Start MJPEG stream server in background
    stream_thread = threading.Thread(target=start_stream_server, daemon=True)
    stream_thread.start()

    # Start background frame grabber
    grabber = FrameGrabber(RTSP_URL)

    # Wait up to 20s for first frame
    print("[Camera] Waiting for first frame...")
    for i in range(20):
        if grabber.get_frame() is not None:
            print(f"[Camera] First frame received after {i+1}s ✓")
            break
        time.sleep(1.0)
        print(f"[Camera] Still waiting... {i+1}s")
    else:
        print("[Camera] WARNING: No frames — check RJ45 connection")

    odo    = Odometry()
    driver = AutoDriver(odo)

    current_mode    = "MANUAL"
    last_mode_check = time.time()
    last_pos_post   = time.time()

    print("[Main] MANUAL mode — waiting for commands.")
    print("=" * 55)

    global latest_jpeg

    while True:
        frame = grabber.get_frame()

        if frame is None:
            time.sleep(0.05)
            continue

        # Encode frame as JPEG for stream server
        ret, jpg = cv2.imencode(".jpg", frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ret:
            with jpeg_lock:
                latest_jpeg = jpg.tobytes()

        # YOLO inference
        try:
            results    = model.predict(frame, verbose=False, conf=0.35)[0]
            detections = []
            if results.boxes is not None:
                for box in results.boxes:
                    cls  = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if cls == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        detections.append({
                            "cls":        cls,
                            "confidence": conf,
                            "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                        })
        except Exception as e:
            print(f"[YOLO] Error: {e}")
            detections = []

        # Post to backend
        post_state(len(detections), detections)

        if len(detections) > 0:
            print(f"[YOLO] {len(detections)} human(s) detected ✓")

        # Mode poll every 1s
        now = time.time()
        if now - last_mode_check > 1.0:
            last_mode_check = now
            m = get_mode()
            if m != current_mode:
                current_mode = m
                print(f"[Mode] → {current_mode}")
                if current_mode == "AUTO":
                    driver.reset()
                elif current_mode == "MANUAL":
                    send_command("STOP")

        # Drive logic
        if current_mode == "AUTO":
            cmd = driver.step(detections, FRAME_WIDTH, FRAME_HEIGHT)
            odo.update(cmd)
            send_command(cmd)
        else:
            odo.update("STOP")

        # Post rover position every 0.5s
        if now - last_pos_post > 0.5:
            last_pos_post = now
            px, py, ph    = odo.position()
            post_rover_position(px, py, ph)

    grabber.stop()
    _executor.shutdown(wait=False)
    if ser:
        ser.close()


if __name__ == "__main__":
    main()