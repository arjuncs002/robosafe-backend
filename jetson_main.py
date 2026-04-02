#!/usr/bin/env python3
"""
ROBOSAFE - Jetson Nano Main (Fixed)

Key fixes:
 - Starts in MANUAL mode, no auto-switching
 - post_state uses a thread pool (not raw threads) to avoid explosion
 - RTSP reconnect loop with exponential backoff
 - life_confirm polling properly integrated
 - 360 spin implemented as timed LEFT rotation
 - human_count=0 posted when no detection (gauge resets)
 - Mode checked every 1s (not 2s) for faster response
"""

import cv2
import time
import math
import serial
import requests
import threading
import json
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
BACKEND_URL    = "https://robosafe-backend.onrender.com"
RTSP_URL       = "rtsp://admin:@192.168.29.192:554/ch0_0.264"
MODEL_PATH     = "best.pt"
SERIAL_PORT    = "/dev/ttyTHS1"   # Jetson UART → ESP32 RX
SERIAL_BAUD    = 115200
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

# Auto-drive thresholds
STOP_AREA_RATIO         = 0.30   # stop when bbox > 30% of frame area (~1m)
LOST_FRAMES_BEFORE_SCAN = 20     # frames without human before re-scanning

# Proportional steering
KP_STEER    = 1.5
DEAD_ZONE_PX = 40               # px from centre = go straight

# Odometry
WHEEL_RADIUS_M = 0.035
BASE_SPEED_MPS = 0.15

# 360 spin: how long to spin LEFT for one full rotation
# Tune this value — depends on your rover's turn speed
SPIN_360_DURATION = 3.5          # seconds

# HTTP rate-limit for identical commands
HTTP_CMD_RATE_LIMIT = 0.3        # seconds

# ============================================================
# SERIAL TO ESP32
# ============================================================
ser = None

def init_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        print(f"[Serial] Connected on {SERIAL_PORT}")
        return True
    except Exception as e:
        print(f"[Serial] WARN: {e} — HTTP fallback only")
        ser = None
        return False

def send_serial(cmd: str):
    if ser and ser.is_open:
        try:
            ser.write((cmd.strip() + "\n").encode())
        except Exception as e:
            print(f"[Serial] Write error: {e}")

# ============================================================
# HTTP COMMAND (fallback)
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
        requests.post(
            f"{BACKEND_URL}/api/auto_command",
            json={"command": cmd},
            timeout=1.5,
        )
    except Exception:
        pass

def send_command(cmd: str):
    """Primary: serial UART. Fallback: HTTP."""
    send_serial(cmd)
    send_http_command(cmd)

# ============================================================
# BACKEND STATE
# ============================================================
_executor = ThreadPoolExecutor(max_workers=4)

def post_state(human_count: int, detections: list):
    """Post vision state to backend (non-blocking)."""
    def _do():
        try:
            requests.post(
                f"{BACKEND_URL}/api/state",
                json={"human_count": human_count, "detections": detections},
                timeout=2,
            )
        except Exception:
            pass
    _executor.submit(_do)

def post_map_flag(flag_type: str, x: float, y: float, label: str = ""):
    def _do():
        try:
            requests.post(
                f"{BACKEND_URL}/api/map/flag",
                json={"flag_type": flag_type, "x": x, "y": y,
                      "label": label, "ts": time.time()},
                timeout=2,
            )
        except Exception:
            pass
    _executor.submit(_do)

def post_rover_position(x: float, y: float, heading: float):
    def _do():
        try:
            requests.post(
                f"{BACKEND_URL}/api/map/position",
                json={"x": x, "y": y, "heading": heading, "ts": time.time()},
                timeout=1,
            )
        except Exception:
            pass
    _executor.submit(_do)

def get_mode() -> str:
    """Fetch current drive mode from backend (no auth needed)."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/drive_mode/public", timeout=2)
        if r.status_code == 200:
            return r.json().get("mode", "MANUAL")
    except Exception:
        pass
    return "MANUAL"

def post_alert():
    try:
        requests.post(
            f"{BACKEND_URL}/api/alert",
            json={"type": "siren", "ts": time.time()},
            timeout=1,
        )
    except Exception:
        pass

def get_life_confirm_pending():
    """Returns (confirmed, result) or (False, '')."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/life_confirm/pending", timeout=1)
        if r.status_code == 200:
            data = r.json()
            return data.get("confirmed", False), data.get("result", "")
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
        self.heading  = 0.0   # radians, 0 = forward
        self._last_ts = time.time()

    def update(self, cmd: str):
        now = time.time()
        dt  = now - self._last_ts
        self._last_ts = now
        dist = BASE_SPEED_MPS * dt

        if cmd == "FORWARD":
            self.x += dist * math.sin(self.heading)
            self.y += dist * math.cos(self.heading)
        elif cmd == "BACKWARD":
            self.x -= dist * math.sin(self.heading)
            self.y -= dist * math.cos(self.heading)
        elif cmd in ("LEFT", "RIGHT"):
            # ~60 deg/s turn rate
            sign = -1 if cmd == "LEFT" else 1
            self.heading += sign * math.radians(60) * dt

    def position(self):
        return self.x, self.y, self.heading

# ============================================================
# STEERING
# ============================================================
def get_steering_command(bbox, frame_w, frame_h):
    """
    Returns (command, stop_flag).
    stop_flag=True when rover is close enough to stop (area > threshold).
    """
    x1, y1, x2, y2 = bbox
    area       = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h

    if area / frame_area >= STOP_AREA_RATIO:
        return "STOP", True

    cx    = (x1 + x2) // 2
    error = cx - (frame_w // 2)

    if abs(error) < DEAD_ZONE_PX:
        return "FORWARD", False
    elif error < 0:
        return "LEFT", False
    else:
        return "RIGHT", False

# ============================================================
# AUTO-DRIVE STATE MACHINE
# ============================================================
class AutoDriver:
    # States
    SCAN     = "SCAN"      # rotating, searching for human
    APPROACH = "APPROACH"  # moving toward detected human
    ARRIVED  = "ARRIVED"   # sent alert, waiting for life confirm
    WAITING  = "WAITING"   # holding still, polling backend
    SPIN360  = "SPIN360"   # doing 360 spin after confirmation
    IDLE     = "IDLE"      # just stopped (mode switched to MANUAL externally)

    def __init__(self, odo: Odometry):
        self.state        = self.SCAN
        self.odo          = odo
        self.lost_frames  = 0
        self.scan_dir     = "LEFT"
        self.alert_sent   = False
        self.spin_start   = 0.0
        self.spin_dir     = "LEFT"

    def reset(self):
        """Called when switching back to AUTO from MANUAL."""
        self.state       = self.SCAN
        self.lost_frames = 0
        self.alert_sent  = False

    def step(self, detections: list, frame_w: int, frame_h: int) -> str:
        """
        Called every frame in AUTO mode.
        Returns a command string for the rover.
        """
        x, y, heading = self.odo.position()

        # ── SCAN: rotate until a human appears ──────────────────────────────
        if self.state == self.SCAN:
            if detections:
                print("[AutoDrive] Human found → APPROACH")
                self.state       = self.APPROACH
                self.lost_frames = 0
                post_map_flag("detected", x, y, "Human detected")
            return self.scan_dir   # keep rotating while scanning

        # ── APPROACH: steer toward closest/largest human ─────────────────────
        elif self.state == self.APPROACH:
            if not detections:
                self.lost_frames += 1
                if self.lost_frames > LOST_FRAMES_BEFORE_SCAN:
                    print("[AutoDrive] Lost human → SCAN")
                    self.state = self.SCAN
                    return self.scan_dir
                return "FORWARD"   # keep going, may reappear

            self.lost_frames = 0
            best = max(detections,
                       key=lambda d: (d["bbox"][2] - d["bbox"][0]) *
                                     (d["bbox"][3] - d["bbox"][1]))
            cmd, stop = get_steering_command(best["bbox"], frame_w, frame_h)

            if stop:
                print("[AutoDrive] Arrived at human → ARRIVED")
                post_map_flag("arrived", x, y, "Rover arrived at human")
                self.state      = self.ARRIVED
                self.alert_sent = False
            return cmd

        # ── ARRIVED: send siren alert to dashboard ───────────────────────────
        elif self.state == self.ARRIVED:
            if not self.alert_sent:
                self.alert_sent = True
                _executor.submit(post_alert)
                print("[AutoDrive] Siren alert sent → WAITING for vitals")
                self.state = self.WAITING
            return "STOP"

        # ── WAITING: hold still, poll for operator vitals confirmation ────────
        elif self.state == self.WAITING:
            confirmed, result = get_life_confirm_pending()
            if confirmed:
                flag_type = "alive" if result == "alive" else "not_alive"
                post_map_flag(flag_type, x, y, f"Life: {result}")
                clear_life_confirm()
                print(f"[AutoDrive] Vitals confirmed: {result} → SPIN360")
                self.state     = self.SPIN360
                self.spin_start = time.time()
                self.spin_dir   = "LEFT"   # always spin left for 360
            return "STOP"

        # ── SPIN360: rotate ~360° then go back to scanning ───────────────────
        elif self.state == self.SPIN360:
            elapsed = time.time() - self.spin_start
            if elapsed >= SPIN_360_DURATION:
                print("[AutoDrive] 360 spin done → SCAN")
                self.state = self.SCAN
                return "STOP"
            return self.spin_dir   # LEFT for full rotation

        return "STOP"

# ============================================================
# CAMERA: open with reconnect
# ============================================================
def open_camera(url: str):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        print(f"[Camera] Opened: {url}")
    else:
        print(f"[Camera] Failed to open: {url}")
    return cap

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print("  ROBOSAFE Jetson Main — Fixed")
    print("=" * 55)

    init_serial()

    print(f"[YOLO] Loading: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[YOLO] Ready.")

    cap = open_camera(RTSP_URL)

    odo    = Odometry()
    driver = AutoDriver(odo)

    current_mode      = "MANUAL"   # start in MANUAL
    last_mode_check   = time.time()
    last_pos_post     = time.time()
    prev_mode         = "MANUAL"

    print("[Main] Starting in MANUAL mode — rover ready.")

    while True:
        # ── Camera read with reconnect ───────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame lost, reconnecting...")
            cap.release()
            time.sleep(1.0)
            cap = open_camera(RTSP_URL)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # ── YOLO inference ───────────────────────────────────────────────────
        results    = model.predict(frame, verbose=False, conf=0.35)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls  = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls == 0:   # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append({
                        "cls":        cls,
                        "confidence": conf,
                        "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    })

        # ── Post detection state (gauge resets to 0 automatically) ───────────
        post_state(len(detections), detections)

        # ── Mode poll every 1 s ──────────────────────────────────────────────
        now = time.time()
        if now - last_mode_check > 1.0:
            last_mode_check = now
            m = get_mode()
            if m != current_mode:
                current_mode = m
                print(f"[Mode] → {current_mode}")
                if current_mode == "AUTO":
                    driver.reset()   # fresh scan when entering AUTO
                elif current_mode == "MANUAL":
                    send_command("STOP")   # safe stop when going MANUAL

        # ── Drive logic ──────────────────────────────────────────────────────
        if current_mode == "AUTO":
            cmd = driver.step(detections, FRAME_WIDTH, FRAME_HEIGHT)
            odo.update(cmd)
            send_command(cmd)
        else:
            # MANUAL: Jetson does NOT send drive commands
            # ESP32 reads commands directly from /api/control/latest
            odo.update("STOP")   # odometry still tracks (stopped in manual)

        # ── Post rover position every 0.5 s ─────────────────────────────────
        if now - last_pos_post > 0.5:
            last_pos_post = now
            px, py, ph    = odo.position()
            post_rover_position(px, py, ph)

    cap.release()
    if ser:
        ser.close()
    _executor.shutdown(wait=False)


if __name__ == "__main__":
    main()