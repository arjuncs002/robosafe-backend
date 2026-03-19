#!/usr/bin/env python3.8
"""
ROBOSAFE - Jetson Nano Main
Phase 2: YOLO Detection + Auto Drive + Serial to ESP32 + Backend Sync
"""

import cv2
import time
import math
import serial
import requests
import threading
import json
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
BACKEND_URL     = "https://robosafe-backend.onrender.com"
RTSP_URL        = "rtsp://admin:@192.168.29.192:554/ch0_0.264"
MODEL_PATH      = "best.pt"
SERIAL_PORT     = "/dev/ttyTHS1"   # Jetson UART TX->ESP32 RX
SERIAL_BAUD     = 115200
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
FRAME_CENTER_X  = FRAME_WIDTH // 2

# Auto drive thresholds
STOP_AREA_RATIO      = 0.40   # stop when bbox area > 40% of frame
LOST_FRAMES_BEFORE_SCAN = 20  # frames without human before scanning

# Proportional steering gains
KP_STEER        = 1.5        # how aggressive the turn is
DEAD_ZONE_PX    = 40         # pixels from center = go straight

# Odometry (dead reckoning)
WHEEL_RADIUS_M  = 0.035      # 3.5 cm
WHEEL_CIRC_M    = 2 * math.pi * WHEEL_RADIUS_M
BASE_SPEED_MPS  = 0.15       # approx rover speed at full PWM

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
        print(f"[Serial] WARN: {e} — will use HTTP fallback only")
        ser = None
        return False

def send_serial(cmd: str):
    """Send command string over UART to ESP32."""
    if ser and ser.is_open:
        try:
            ser.write((cmd.strip() + "\n").encode())
        except Exception as e:
            print(f"[Serial] Write error: {e}")

# ============================================================
# HTTP FALLBACK TO BACKEND
# ============================================================
_http_lock = threading.Lock()
_last_http_cmd = ""
_last_http_ts   = 0

def send_http_command(cmd: str):
    """Post command to backend (ESP32 will poll it)."""
    global _last_http_cmd, _last_http_ts
    now = time.time()
    with _http_lock:
        if cmd == _last_http_cmd and (now - _last_http_ts) < 0.3:
            return   # rate-limit identical commands
        _last_http_cmd = cmd
        _last_http_ts  = now

    try:
        requests.post(
            f"{BACKEND_URL}/api/auto_command",
            json={"command": cmd},
            timeout=1.5
        )
    except Exception as e:
        pass  # non-fatal

def send_command(cmd: str):
    """Primary: serial. Fallback: HTTP."""
    send_serial(cmd)
    send_http_command(cmd)

# ============================================================
# BACKEND STATE SYNC
# ============================================================
def post_state(human_count: int, detections: list):
    try:
        requests.post(
            f"{BACKEND_URL}/api/state",
            json={"human_count": human_count, "detections": detections},
            timeout=2
        )
    except:
        pass

def post_map_flag(flag_type: str, x: float, y: float, label: str = ""):
    """Send a map flag to the backend."""
    try:
        requests.post(
            f"{BACKEND_URL}/api/map/flag",
            json={
                "flag_type": flag_type,
                "x": x,
                "y": y,
                "label": label,
                "ts": time.time()
            },
            timeout=2
        )
    except:
        pass

def post_rover_position(x: float, y: float, heading: float):
    try:
        requests.post(
            f"{BACKEND_URL}/api/map/position",
            json={"x": x, "y": y, "heading": heading, "ts": time.time()},
            timeout=1
        )
    except:
        pass

def get_mode():
    """Fetch current mode (AUTO / MANUAL) from backend."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/drive_mode", timeout=2)
        if r.status_code == 200:
            return r.json().get("mode", "MANUAL")
    except:
        pass
    return "MANUAL"

# ============================================================
# ODOMETRY
# ============================================================
class Odometry:
    def __init__(self):
        self.x       = 0.0   # metres from start
        self.y       = 0.0
        self.heading = 0.0   # radians, 0 = forward (North)
        self._last_ts = time.time()
        self._current_cmd = "STOP"

    def update(self, cmd: str):
        now  = time.time()
        dt   = now - self._last_ts
        self._last_ts = now
        self._current_cmd = cmd

        dist = BASE_SPEED_MPS * dt

        if cmd == "FORWARD":
            self.x += dist * math.sin(self.heading)
            self.y += dist * math.cos(self.heading)
        elif cmd == "BACKWARD":
            self.x -= dist * math.sin(self.heading)
            self.y -= dist * math.cos(self.heading)
        elif cmd == "LEFT":
            self.heading -= math.radians(60) * dt   # ~60 deg/s turn
        elif cmd == "RIGHT":
            self.heading += math.radians(60) * dt

    def position(self):
        return self.x, self.y, self.heading

# ============================================================
# STEERING LOGIC
# ============================================================
def get_steering_command(bbox, frame_w, frame_h):
    """
    Returns (command, stop_flag)
    command  : FORWARD / LEFT / RIGHT
    stop_flag: True if rover is close enough to stop
    """
    x1, y1, x2, y2 = bbox
    box_w  = x2 - x1
    box_h  = y2 - y1
    area   = box_w * box_h
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
    SCAN      = "SCAN"      # rotating, searching
    APPROACH  = "APPROACH"  # moving toward human
    ARRIVED   = "ARRIVED"   # at human, siren playing
    WAITING   = "WAITING"   # waiting for life confirmation
    RESUME    = "RESUME"    # spin before next scan

    def __init__(self, odo: Odometry):
        self.state       = self.SCAN
        self.odo         = odo
        self.lost_frames = 0
        self.scan_dir    = "LEFT"        # direction of scan rotation
        self.arrived_ts  = 0
        self.siren_sent  = False
        self.flag_sent   = False
        self.resume_start = 0
        self.resume_duration = 2.5       # seconds of spin after confirm

    def step(self, detections: list, frame_w: int, frame_h: int):
        """
        Called every frame.
        detections: list of {"bbox": [x1,y1,x2,y2], "confidence": float}
        Returns command string.
        """
        x, y, heading = self.odo.position()

        # ---- SCAN state: rotate until human found ----
        if self.state == self.SCAN:
            if detections:
                self.state     = self.APPROACH
                self.lost_frames = 0
                self.flag_sent   = False
                # Flag initial detection on map
                post_map_flag("detected", x, y, "Human detected")
                print("[AutoDrive] Human found → APPROACH")
                return self.scan_dir   # one last frame before switching

            return self.scan_dir   # keep rotating

        # ---- APPROACH state: steer toward human ----
        elif self.state == self.APPROACH:
            if not detections:
                self.lost_frames += 1
                if self.lost_frames > LOST_FRAMES_BEFORE_SCAN:
                    print("[AutoDrive] Lost human → SCAN")
                    self.state = self.SCAN
                    return self.scan_dir
                return "FORWARD"   # keep going, might reappear

            self.lost_frames = 0
            # Use largest bbox
            best = max(detections, key=lambda d: (
                (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1])
            ))
            cmd, stop = get_steering_command(best["bbox"], frame_w, frame_h)

            if stop:
                # Post arrived flag
                post_map_flag("arrived", x, y, "Rover arrived at human")
                self.state      = self.ARRIVED
                self.arrived_ts = time.time()
                self.siren_sent = False
                print("[AutoDrive] Arrived at human → ARRIVED")
                return "STOP"

            return cmd

        # ---- ARRIVED: play siren, notify backend ----
        elif self.state == self.ARRIVED:
            if not self.siren_sent:
                self.siren_sent = True
                # Tell backend to play siren on website
                try:
                    requests.post(
                        f"{BACKEND_URL}/api/alert",
                        json={"type": "siren", "ts": time.time()},
                        timeout=1
                    )
                except:
                    pass
                print("[AutoDrive] Siren sent → WAITING for confirmation")
                self.state = self.WAITING

            return "STOP"

        # ---- WAITING: hold until operator confirms life ----
        elif self.state == self.WAITING:
            # Backend will change state via /api/life_confirm endpoint
            # jetson polls that
            try:
                r = requests.get(f"{BACKEND_URL}/api/life_confirm/pending", timeout=1)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("confirmed"):
                        result  = data.get("result", "unknown")   # alive / not_alive
                        flag_t  = "alive" if result == "alive" else "not_alive"
                        post_map_flag(flag_t, x, y, f"Life: {result}")
                        # Clear the pending confirm
                        try:
                            requests.post(
                                f"{BACKEND_URL}/api/life_confirm/clear",
                                timeout=1
                            )
                        except:
                            pass
                        print(f"[AutoDrive] Life confirmed: {result} → RESUME")
                        self.state        = self.RESUME
                        self.resume_start = time.time()
                        # Alternate scan direction
                        self.scan_dir = "RIGHT" if self.scan_dir == "LEFT" else "LEFT"
            except:
                pass

            return "STOP"

        # ---- RESUME: spin briefly before scanning again ----
        elif self.state == self.RESUME:
            if time.time() - self.resume_start >= self.resume_duration:
                self.state = self.SCAN
                print("[AutoDrive] Spin done → SCAN")
                return "STOP"
            return self.scan_dir

        return "STOP"

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 50)
    print("ROBOSAFE Jetson Main - Phase 2")
    print("=" * 50)

    init_serial()

    print(f"[YOLO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[YOLO] Model loaded.")

    print(f"[Camera] Opening {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[Camera] FAILED to open. Check RTSP URL.")
        return

    print("[Camera] Open ✓")

    odo    = Odometry()
    driver = AutoDriver(odo)

    frame_skip   = 0
    last_pos_post = time.time()
    last_mode_check = time.time()
    current_mode = "MANUAL"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame read failed, retrying...")
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # ---- YOLO inference ----
        results   = model.predict(frame, verbose=False, conf=0.35)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls  = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls == 0:   # person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append({
                        "cls": cls,
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })

        # ---- Post state to backend (non-blocking thread) ----
        threading.Thread(
            target=post_state,
            args=(len(detections), detections),
            daemon=True
        ).start()

        # ---- Mode check every 2s ----
        now = time.time()
        if now - last_mode_check > 2.0:
            last_mode_check = now
            m = get_mode()
            if m != current_mode:
                current_mode = m
                print(f"[Mode] Switched to {current_mode}")

        # ---- Send drive command ----
        if current_mode == "AUTO":
            cmd = driver.step(detections, FRAME_WIDTH, FRAME_HEIGHT)
            odo.update(cmd)
            send_command(cmd)
        else:
            # MANUAL mode: Jetson still posts vision but doesn't drive
            odo.update("STOP")

        # ---- Post rover position every 0.5s ----
        if now - last_pos_post > 0.5:
            last_pos_post = now
            px, py, ph = odo.position()
            threading.Thread(
                target=post_rover_position,
                args=(px, py, ph),
                daemon=True
            ).start()

    cap.release()
    if ser:
        ser.close()


if __name__ == "__main__":
    main()