#!/usr/bin/env python3
"""
ROBOSAFE - Jetson Nano Main - FINAL v2
=======================================
AUTO MODE:
  - YOLO detects human
  - Sends FORWARD/LEFT/RIGHT/STOP directly to ESP32 via UART
  - Keeps human centered in frame
  - Stops at ~1-1.5m (bbox area threshold)
  - Posts alert to backend → popup on dashboard
  - Waits for operator vitals confirm
  - Does 360 spin → scans again
  - After 2x 360 with no human → posts manual mode request

MANUAL MODE:
  - Jetson sends NO drive commands
  - Only posts detection state + rover position to backend
  - ESP32 gets commands from dashboard via WiFi

RUNS ON BOOT via systemd service.
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
# CONFIG — Edit these to match your setup
# ============================================================
BACKEND_URL   = "https://robosafe-backend.onrender.com"
RTSP_URL      = "rtsp://admin:@192.168.29.192:554/ch0_0.264"
MODEL_PATH    = "/home/jetson/robosafe/best.pt"   # absolute path
SERIAL_PORT   = "/dev/ttyTHS1"                    # Jetson UART TX → ESP32 GPIO32
SERIAL_BAUD   = 115200
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

# Auto-drive
STOP_AREA_RATIO         = 0.28   # stop when human bbox > 28% frame area (~1-1.5m)
DEAD_ZONE_PX            = 50     # pixels from centre = go straight
LOST_FRAMES_BEFORE_SCAN = 25     # frames without human → back to scan

# Odometry
BASE_SPEED_MPS = 0.15
TURN_DEG_PER_S = 60.0

# 360 spin duration — tune based on your rover's turn speed
SPIN_360_DURATION = 3.5   # seconds

# Max 360 spins before asking operator to switch to manual
MAX_EMPTY_SPINS = 2

# ============================================================
# SERIAL
# ============================================================
ser = None

def init_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        time.sleep(0.5)
        print(f"[Serial] Connected: {SERIAL_PORT}")
        return True
    except Exception as e:
        print(f"[Serial] WARN: {e}")
        ser = None
        return False

def send_serial(cmd: str):
    """Send command to ESP32 via UART."""
    global ser
    if ser and ser.is_open:
        try:
            ser.write((cmd.strip() + "\n").encode())
        except Exception as e:
            print(f"[Serial] Write error: {e}")
            ser = None

def send_command(cmd: str):
    send_serial(cmd)

# ============================================================
# BACKEND CALLS (non-blocking thread pool)
# ============================================================
_executor = ThreadPoolExecutor(max_workers=4)

def post_state(human_count: int, detections: list):
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

def post_alert():
    def _do():
        try:
            requests.post(
                f"{BACKEND_URL}/api/alert",
                json={"type": "siren", "ts": time.time()},
                timeout=2,
            )
        except Exception:
            pass
    _executor.submit(_do)

def post_manual_request():
    """Ask dashboard to switch to manual — no humans found after 2 spins."""
    def _do():
        try:
            requests.post(
                f"{BACKEND_URL}/api/manual_request",
                json={"reason": "No human found after 2 scans", "ts": time.time()},
                timeout=2,
            )
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

def get_life_confirm():
    try:
        r = requests.get(f"{BACKEND_URL}/api/life_confirm/pending", timeout=2)
        if r.status_code == 200:
            data = r.json()
            return data.get("confirmed", False), data.get("result", "")
    except Exception:
        pass
    return False, ""

def clear_life_confirm():
    try:
        requests.post(f"{BACKEND_URL}/api/life_confirm/clear", timeout=2)
    except Exception:
        pass

# ============================================================
# ODOMETRY
# ============================================================
class Odometry:
    def __init__(self):
        self.x       = 0.0
        self.y       = 0.0
        self.heading = 0.0
        self._last   = time.time()

    def update(self, cmd: str):
        now  = time.time()
        dt   = now - self._last
        self._last = now
        dist = BASE_SPEED_MPS * dt

        if cmd == "FORWARD":
            self.x += dist * math.sin(self.heading)
            self.y += dist * math.cos(self.heading)
        elif cmd == "BACKWARD":
            self.x -= dist * math.sin(self.heading)
            self.y -= dist * math.cos(self.heading)
        elif cmd == "LEFT":
            self.heading -= math.radians(TURN_DEG_PER_S) * dt
        elif cmd == "RIGHT":
            self.heading += math.radians(TURN_DEG_PER_S) * dt

    def pos(self):
        return self.x, self.y, self.heading

# ============================================================
# STEERING
# ============================================================
def get_steering(bbox, frame_w, frame_h):
    """Returns (cmd, arrived)."""
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
# AUTO DRIVE STATE MACHINE
# ============================================================
class AutoDriver:
    SCAN     = "SCAN"
    APPROACH = "APPROACH"
    ARRIVED  = "ARRIVED"
    WAITING  = "WAITING"
    SPIN360  = "SPIN360"

    def __init__(self, odo: Odometry):
        self.odo         = odo
        self.state       = self.SCAN
        self.lost_frames = 0
        self.alert_sent  = False
        self.spin_start  = 0.0
        self.empty_spins = 0    # count spins with no human found

    def reset(self):
        self.state       = self.SCAN
        self.lost_frames = 0
        self.alert_sent  = False
        self.empty_spins = 0

    def step(self, detections: list, frame_w: int, frame_h: int) -> str:
        x, y, heading = self.odo.pos()

        # ── SCAN ────────────────────────────────────────────────────────────
        if self.state == self.SCAN:
            if detections:
                print("[Auto] Human found → APPROACH")
                self.state       = self.APPROACH
                self.lost_frames = 0
                self.empty_spins = 0
                post_map_flag("detected", x, y, "Human detected")
            return "LEFT"   # keep spinning

        # ── APPROACH ────────────────────────────────────────────────────────
        elif self.state == self.APPROACH:
            if not detections:
                self.lost_frames += 1
                if self.lost_frames > LOST_FRAMES_BEFORE_SCAN:
                    print("[Auto] Lost human → SCAN")
                    self.state = self.SCAN
                    return "LEFT"
                return "FORWARD"

            self.lost_frames = 0
            # Target largest detection
            best = max(detections,
                       key=lambda d: (d["bbox"][2]-d["bbox"][0]) *
                                     (d["bbox"][3]-d["bbox"][1]))
            cmd, arrived = get_steering(best["bbox"], frame_w, frame_h)

            if arrived:
                print("[Auto] Arrived at human → ARRIVED")
                post_map_flag("arrived", x, y, "Rover arrived")
                self.state      = self.ARRIVED
                self.alert_sent = False

            return cmd

        # ── ARRIVED ─────────────────────────────────────────────────────────
        elif self.state == self.ARRIVED:
            if not self.alert_sent:
                self.alert_sent = True
                post_alert()
                print("[Auto] Alert sent → WAITING")
                self.state = self.WAITING
            return "STOP"

        # ── WAITING ─────────────────────────────────────────────────────────
        elif self.state == self.WAITING:
            confirmed, result = get_life_confirm()
            if confirmed:
                flag_type = "alive" if result == "alive" else "not_alive"
                post_map_flag(flag_type, x, y, f"Life: {result}")
                clear_life_confirm()
                print(f"[Auto] Vitals: {result} → SPIN360")
                self.state      = self.SPIN360
                self.spin_start = time.time()
            return "STOP"

        # ── SPIN360 ─────────────────────────────────────────────────────────
        elif self.state == self.SPIN360:
            elapsed = time.time() - self.spin_start
            if elapsed >= SPIN_360_DURATION:
                self.empty_spins += 1
                print(f"[Auto] Spin done ({self.empty_spins}/{MAX_EMPTY_SPINS}) → SCAN")

                if self.empty_spins >= MAX_EMPTY_SPINS:
                    print("[Auto] No human found after 2 spins → requesting MANUAL")
                    post_manual_request()
                    self.empty_spins = 0

                self.state = self.SCAN
                return "STOP"
            return "LEFT"

        return "STOP"

# ============================================================
# CAMERA
# ============================================================
def open_camera(url: str):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        print(f"[Camera] Opened: {url}")
    else:
        print(f"[Camera] Failed: {url}")
    return cap

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 55)
    print("  ROBOSAFE Jetson Main - FINAL v2")
    print("=" * 55)

    init_serial()

    print(f"[YOLO] Loading: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[YOLO] Ready.")

    cap = open_camera(RTSP_URL)

    odo    = Odometry()
    driver = AutoDriver(odo)

    current_mode    = "MANUAL"
    last_mode_check = time.time()
    last_pos_post   = time.time()

    print("[Main] MANUAL mode — waiting for commands.")

    while True:
        # ── Camera frame ─────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame lost, reconnecting...")
            cap.release()
            time.sleep(1.0)
            cap = open_camera(RTSP_URL)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # ── YOLO ─────────────────────────────────────────────────────────
        results    = model.predict(frame, verbose=False, conf=0.35)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls  = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append({
                        "cls": cls, "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    })

        # Always post state to backend (gauge + history)
        post_state(len(detections), detections)

        # ── Mode poll every 1s ───────────────────────────────────────────
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

        # ── Drive logic ──────────────────────────────────────────────────
        if current_mode == "AUTO":
            cmd = driver.step(detections, FRAME_WIDTH, FRAME_HEIGHT)
            odo.update(cmd)
            send_command(cmd)
        else:
            odo.update("STOP")

        # ── Post position every 0.5s ─────────────────────────────────────
        if now - last_pos_post > 0.5:
            last_pos_post = now
            px, py, ph = odo.pos()
            post_rover_position(px, py, ph)

    cap.release()
    if ser:
        ser.close()
    _executor.shutdown(wait=False)


if __name__ == "__main__":
    main()