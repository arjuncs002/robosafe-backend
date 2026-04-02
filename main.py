#!/usr/bin/env python3
"""
ROBOSAFE Backend v2 - Fixed
Key fixes:
 - Boot state: MANUAL mode, mmWave OFF, motors enabled
 - /api/life_confirm endpoints added for Jetson <-> dashboard flow
 - Python 3.8 compatible type hints (Optional instead of str | None)
 - /api/state POST now properly updates human_count to 0 when no detection
 - Token refresh window extended to 10 minutes
 - Drive mode persisted properly
 - Alert state cleaned up
"""

import time
import asyncio
import secrets
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2   = None
    YOLO  = None
    np    = None

from db import (
    init_db, SessionLocal,
    Detection, ControlCommand, MMWaveState,
    MapFlag, RoverPosition, DriveMode, AutoCommand, LifeConfirm
)

app = FastAPI(title="ROBOSAFE Backend v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# ─── ENSURE INITIAL DB ROWS ───────────────────────────────────────────────────
def _ensure_defaults():
    try:
        db = SessionLocal()
        # Drive mode defaults to MANUAL
        if db.query(DriveMode).count() == 0:
            db.add(DriveMode(mode="MANUAL"))
            db.commit()
        # Life confirm defaults to cleared/not confirmed
        if db.query(LifeConfirm).count() == 0:
            db.add(LifeConfirm(confirmed=False, result="", cleared=True))
            db.commit()
        db.close()
    except Exception as e:
        print(f"[Init] DB defaults error: {e}")

_ensure_defaults()

# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────
TOKEN_TTL_SEC = 600          # 10 minutes
TOKENS        = {}
DASH_PASSWORD = {"value": os.getenv("ADMIN_PASSWORD", "GROUP5")}

def issue_token():
    token = secrets.token_urlsafe(32)
    TOKENS[token] = time.time()
    return token

def verify_token(request: Request, token_query: Optional[str] = None):
    token = None
    auth  = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    if token is None and token_query:
        token = token_query.strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    last = TOKENS.get(token)
    if not last:
        raise HTTPException(status_code=401, detail="Invalid token")
    now = time.time()
    if (now - last) > TOKEN_TTL_SEC:
        TOKENS.pop(token, None)
        raise HTTPException(status_code=401, detail="Token expired")
    TOKENS[token] = now   # sliding window
    return token

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA (optional local mode)
# ─────────────────────────────────────────────────────────────────────────────
USE_LOCAL_CAMERA = os.getenv("USE_LOCAL_CAMERA", "false").lower() == "true"

model = None
cap   = None

if USE_LOCAL_CAMERA and CV2_AVAILABLE:
    MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
    try:
        model = YOLO(MODEL_PATH)
        cap   = cv2.VideoCapture(0)
        print("[Camera] Local camera opened")
    except Exception as e:
        print(f"[Camera] Failed to open local camera: {e}")
        model = None
        cap   = None

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY STATE  (all initialised to safe/off defaults)
# ─────────────────────────────────────────────────────────────────────────────

# Vision state — updated by Jetson POST /api/state
STATE = {
    "ts":          time.time(),
    "human_count": 0,
    "detections":  [],
}

# Rover command state — updated by dashboard POST /api/control
ROVER_STATE = {
    "last_command":    "STOP",
    "last_command_ts": time.time(),
}

# mmWave — OFF by default on boot
MMWAVE_STATE = {
    "status":               "SENSOR DISABLED",
    "last_presence":        0,
    "energy_delta":         0,
    "respiration_detected": False,
    "distance":             0.0,
    "energy_min":           0,
    "energy_max":           0,
    "last_update":          0.0,
    "enabled":              False,   # ← OFF on boot
}

# Drive mode — MANUAL on boot
DRIVE_MODE_STATE = {"mode": "MANUAL"}

# Map
MAP_STATE = {
    "rover_x":       0.0,
    "rover_y":       0.0,
    "rover_heading": 0.0,
    "rover_ts":      time.time(),
    "flags":         [],
    "track":         [],
}

# Alert (rover arrived at human)
ALERT_STATE = {
    "type":   "",
    "ts":     0.0,
    "active": False,
}

# Auto command from Jetson
AUTO_CMD_STATE = {
    "command": "STOP",
    "ts":      time.time(),
}

# Life-confirm handshake (in-memory mirror)
LIFE_CONFIRM_STATE = {
    "confirmed": False,
    "result":    "",     # 'alive' | 'not_alive'
    "cleared":   True,
}

MIN_LOG_GAP_SEC   = 3.0
_last_save_ts     = 0.0
_last_saved_count = -1

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_detection(count: int):
    try:
        db = SessionLocal()
        db.add(Detection(ts=time.time(), count=count, source="jetson"))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] save_detection error: {e}")

def detect_and_draw(frame):
    if model is None or not CV2_AVAILABLE:
        return frame, 0, []
    results = model.predict(frame, verbose=False)[0]
    count   = 0
    dets    = []
    if results.boxes is not None:
        for box in results.boxes:
            cls  = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls == 0 and conf >= 0.35:
                count += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                dets.append({"cls": cls, "confidence": conf,
                             "bbox": [int(x1), int(y1), int(x2), int(y2)]})
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"Human {conf:.2f}", (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame, count, dets

def video_generator():
    global _last_save_ts, _last_saved_count

    # --- offline / no camera ---
    if not USE_LOCAL_CAMERA or cap is None or not CV2_AVAILABLE:
        if CV2_AVAILABLE and np is not None:
            blank = np.ones((480, 640, 3), dtype=np.uint8) * 30
            cv2.putText(blank, "Camera Offline", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            ret, jpeg = cv2.imencode(".jpg", blank,
                                     [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            frame_bytes = jpeg.tobytes() if ret else b""
        else:
            frame_bytes = b""
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + frame_bytes + b"\r\n")
            time.sleep(0.1)
        return

    # --- local camera loop ---
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame = cv2.resize(frame, (960, 540))
        frame, count, dets = detect_and_draw(frame)

        STATE["ts"]          = time.time()
        STATE["human_count"] = count
        STATE["detections"]  = dets

        if count > 0:
            now = time.time()
            if (now - _last_save_ts) >= MIN_LOG_GAP_SEC or count != _last_saved_count:
                save_detection(count)
                _last_save_ts     = now
                _last_saved_count = count

        ret, jpeg = cv2.imencode(".jpg", frame,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpeg.tobytes() + b"\r\n")

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"ok": True, "service": "robosafe-backend-v2"}


# ── AUTH ──────────────────────────────────────────────────────────────────────

@app.post("/api/login")
async def login(payload: Optional[dict] = None):
    if payload is None:
        payload = {}
    pw = str(payload.get("password", "")).strip()
    if pw != DASH_PASSWORD["value"]:
        raise HTTPException(status_code=401, detail="Wrong password")
    return {"token": issue_token()}


@app.post("/api/password")
async def change_password(request: Request, payload: Optional[dict] = None):
    verify_token(request)
    if payload is None:
        payload = {}
    cur  = str(payload.get("current_password", "")).strip()
    new  = str(payload.get("new_password",     "")).strip()
    conf = str(payload.get("confirm_password", "")).strip()
    if cur != DASH_PASSWORD["value"]:
        raise HTTPException(status_code=400, detail="Current password wrong")
    if not new or not conf:
        raise HTTPException(status_code=400, detail="New password required")
    if new != conf:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if len(new) < 4:
        raise HTTPException(status_code=400, detail="Password too short")
    DASH_PASSWORD["value"] = new
    return {"ok": True}


# ── STATE ─────────────────────────────────────────────────────────────────────

@app.get("/api/state")
def get_state(request: Request, overlays: int = 1):
    verify_token(request)
    now = time.time()
    # Auto-reset mmWave if no update for 15 s
    if MMWAVE_STATE["enabled"] and MMWAVE_STATE["last_update"] > 0 and \
            (now - MMWAVE_STATE["last_update"]) > 15:
        MMWAVE_STATE.update({
            "status":               "NO PRESENCE DETECTED",
            "last_presence":        0,
            "energy_delta":         0,
            "respiration_detected": False,
            "distance":             0.0,
        })
    return JSONResponse({
        "ts":          STATE["ts"],
        "human_count": STATE["human_count"],
        "detections":  STATE["detections"] if overlays == 1 else [],
        "mmwave": {
            "status":               MMWAVE_STATE["status"],
            "last_presence":        MMWAVE_STATE["last_presence"],
            "energy_delta":         MMWAVE_STATE["energy_delta"],
            "respiration_detected": MMWAVE_STATE["respiration_detected"],
            "distance":             MMWAVE_STATE["distance"],
            "energy_min":           MMWAVE_STATE["energy_min"],
            "energy_max":           MMWAVE_STATE["energy_max"],
            "last_update":          MMWAVE_STATE["last_update"],
            "enabled":              MMWAVE_STATE["enabled"],
        },
        "drive_mode": DRIVE_MODE_STATE["mode"],
        "alert":      ALERT_STATE,
    })


@app.post("/api/state")
async def update_state(payload: Optional[dict] = None):
    """
    Called by Jetson every frame.
    No auth required — internal service call.
    Updates human_count to 0 when no detections (gauge resets correctly).
    """
    if payload is None:
        payload = {}
    global _last_save_ts, _last_saved_count
    STATE["ts"]          = time.time()
    STATE["human_count"] = int(payload.get("human_count", 0))
    STATE["detections"]  = payload.get("detections",  [])

    count = STATE["human_count"]
    if count > 0:
        now = time.time()
        if (now - _last_save_ts) >= MIN_LOG_GAP_SEC or count != _last_saved_count:
            save_detection(count)
            _last_save_ts     = now
            _last_saved_count = count

    return {"ok": True}


# ── mmWAVE ────────────────────────────────────────────────────────────────────

@app.post("/api/mmwave")
async def update_mmwave(payload: Optional[dict] = None):
    """ESP32 posts sensor data here every sync interval."""
    if payload is None:
        payload = {}
    if MMWAVE_STATE["enabled"]:
        MMWAVE_STATE.update({
            "status":               payload.get("status",               "NO PRESENCE DETECTED"),
            "last_presence":        payload.get("last_presence",        0),
            "energy_delta":         payload.get("energy_delta",         0),
            "respiration_detected": payload.get("respiration_detected", False),
            "distance":             float(payload.get("distance",       0)),
            "energy_min":           payload.get("energy_min",           0),
            "energy_max":           payload.get("energy_max",           0),
            "last_update":          time.time(),
        })
        asyncio.create_task(_save_mmwave(
            MMWAVE_STATE["status"],
            MMWAVE_STATE["last_presence"],
            MMWAVE_STATE["energy_delta"],
            MMWAVE_STATE["respiration_detected"],
        ))
    # Always return current enabled state so ESP32 can sync its local flag
    return {"ok": True, "enabled": MMWAVE_STATE["enabled"]}


async def _save_mmwave(status, presence, energy_delta, respiration):
    try:
        db = SessionLocal()
        db.add(MMWaveState(ts=time.time(), status=status, presence=presence,
                           energy_delta=energy_delta,
                           respiration_detected=respiration))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] _save_mmwave error: {e}")


@app.post("/api/mmwave/toggle")
async def toggle_mmwave(request: Request, payload: Optional[dict] = None):
    verify_token(request)
    if payload is None:
        payload = {}
    enabled = bool(payload.get("enabled", False))
    MMWAVE_STATE["enabled"] = enabled
    if not enabled:
        MMWAVE_STATE.update({
            "status":               "SENSOR DISABLED",
            "last_presence":        0,
            "energy_delta":         0,
            "distance":             0.0,
            "energy_min":           0,
            "energy_max":           0,
            "respiration_detected": False,
        })
        # Stop rover when mmWave is disabled (safe default)
        ROVER_STATE["last_command"]    = "STOP"
        ROVER_STATE["last_command_ts"] = time.time()
    else:
        MMWAVE_STATE["status"]      = "NO PRESENCE DETECTED"
        MMWAVE_STATE["last_update"] = 0.0
    print(f"[mmWave] {'ENABLED' if enabled else 'DISABLED'}")
    return {"ok": True, "enabled": enabled}


@app.get("/api/mmwave")
def get_mmwave(request: Request):
    verify_token(request)
    return JSONResponse(MMWAVE_STATE)


# ── ROVER CONTROL ─────────────────────────────────────────────────────────────

@app.post("/api/control")
async def control_rover(request: Request, payload: Optional[dict] = None):
    """Dashboard sends manual drive commands here."""
    verify_token(request)
    if payload is None:
        payload = {}
    command = str(payload.get("command", "STOP")).strip().upper()
    valid   = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}
    if command not in valid:
        raise HTTPException(status_code=400, detail="Invalid command")
    # mmWave active blocks movement (safety)
    if MMWAVE_STATE["enabled"] and command != "STOP":
        return JSONResponse({"ok": False, "reason": "mmWave active — motors disabled"})
    ROVER_STATE["last_command"]    = command
    ROVER_STATE["last_command_ts"] = time.time()
    asyncio.create_task(_save_command(command))
    return {"ok": True, "command": command, "ts": ROVER_STATE["last_command_ts"]}


async def _save_command(command: str):
    try:
        db = SessionLocal()
        db.add(ControlCommand(ts=time.time(), command=command))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] _save_command error: {e}")


@app.get("/api/control/latest")
async def get_latest_command():
    """
    ESP32 polls this every 150 ms.
    In AUTO mode  → return the latest Jetson auto command.
    In MANUAL mode → return the latest dashboard command.
    """
    if DRIVE_MODE_STATE["mode"] == "AUTO":
        return {
            "command": AUTO_CMD_STATE["command"],
            "ts":      AUTO_CMD_STATE["ts"],
            "source":  "auto",
        }
    return {
        "command": ROVER_STATE["last_command"],
        "ts":      ROVER_STATE["last_command_ts"],
        "source":  "manual",
    }


# ── AUTO COMMAND (Jetson → backend → ESP32) ───────────────────────────────────

@app.post("/api/auto_command")
async def set_auto_command(payload: Optional[dict] = None):
    """Jetson posts its drive decision here (HTTP fallback alongside serial)."""
    if payload is None:
        payload = {}
    cmd = str(payload.get("command", "STOP")).strip().upper()
    valid = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}
    if cmd not in valid:
        cmd = "STOP"
    AUTO_CMD_STATE["command"] = cmd
    AUTO_CMD_STATE["ts"]      = time.time()
    return {"ok": True}


# ── DRIVE MODE ────────────────────────────────────────────────────────────────

@app.get("/api/drive_mode")
def get_drive_mode(request: Request):
    verify_token(request)
    return {"mode": DRIVE_MODE_STATE["mode"]}


@app.get("/api/drive_mode/public")
def get_drive_mode_public():
    """Jetson polls this without a token."""
    return {"mode": DRIVE_MODE_STATE["mode"]}


@app.post("/api/drive_mode")
async def set_drive_mode(request: Request, payload: Optional[dict] = None):
    verify_token(request)
    if payload is None:
        payload = {}
    mode = str(payload.get("mode", "MANUAL")).upper()
    if mode not in ("AUTO", "MANUAL"):
        raise HTTPException(status_code=400, detail="mode must be AUTO or MANUAL")
    DRIVE_MODE_STATE["mode"] = mode
    if mode == "MANUAL":
        # Immediately stop rover when switching to manual
        ROVER_STATE["last_command"]    = "STOP"
        ROVER_STATE["last_command_ts"] = time.time()
        AUTO_CMD_STATE["command"]      = "STOP"
    try:
        db  = SessionLocal()
        row = db.query(DriveMode).first()
        if row:
            row.mode = mode
        else:
            db.add(DriveMode(mode=mode))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] set_drive_mode error: {e}")
    print(f"[Mode] Switched to {mode}")
    return {"ok": True, "mode": mode}


# ── ALERT (rover arrived at human) ───────────────────────────────────────────

@app.post("/api/alert")
async def post_alert(payload: Optional[dict] = None):
    """Jetson calls this when rover arrives near a human."""
    if payload is None:
        payload = {}
    ALERT_STATE["type"]   = payload.get("type", "siren")
    ALERT_STATE["ts"]     = time.time()
    ALERT_STATE["active"] = True
    print("[Alert] Arrival alert triggered")
    return {"ok": True}


@app.post("/api/alert/clear")
async def clear_alert(request: Request):
    """Dashboard operator dismisses popup."""
    verify_token(request)
    ALERT_STATE["active"] = False
    return {"ok": True}


# ── LIFE CONFIRM ──────────────────────────────────────────────────────────────

@app.post("/api/life_confirm")
async def submit_life_confirm(request: Request, payload: Optional[dict] = None):
    """
    Dashboard operator submits vitals result after arrival.
    payload: { "result": "alive" | "not_alive" }
    """
    verify_token(request)
    if payload is None:
        payload = {}
    result = str(payload.get("result", "")).strip().lower()
    if result not in ("alive", "not_alive"):
        raise HTTPException(status_code=400, detail="result must be 'alive' or 'not_alive'")

    LIFE_CONFIRM_STATE["confirmed"] = True
    LIFE_CONFIRM_STATE["result"]    = result
    LIFE_CONFIRM_STATE["cleared"]   = False

    # Persist to DB
    try:
        db  = SessionLocal()
        row = db.query(LifeConfirm).first()
        if row:
            row.confirmed = True
            row.result    = result
            row.cleared   = False
        else:
            db.add(LifeConfirm(confirmed=True, result=result, cleared=False))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] life_confirm error: {e}")

    print(f"[LifeConfirm] Result submitted: {result}")
    return {"ok": True, "result": result}


@app.get("/api/life_confirm/pending")
async def get_life_confirm_pending():
    """
    Jetson polls this to check if operator has responded.
    Returns confirmed=True only once — after that it waits for /clear.
    """
    return {
        "confirmed": LIFE_CONFIRM_STATE["confirmed"],
        "result":    LIFE_CONFIRM_STATE["result"],
    }


@app.post("/api/life_confirm/clear")
async def clear_life_confirm():
    """Jetson calls this after it has processed the confirmation."""
    LIFE_CONFIRM_STATE["confirmed"] = False
    LIFE_CONFIRM_STATE["result"]    = ""
    LIFE_CONFIRM_STATE["cleared"]   = True

    try:
        db  = SessionLocal()
        row = db.query(LifeConfirm).first()
        if row:
            row.confirmed = False
            row.result    = ""
            row.cleared   = True
            db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] clear_life_confirm error: {e}")

    return {"ok": True}


# ── MAP ───────────────────────────────────────────────────────────────────────

@app.post("/api/map/flag")
async def add_map_flag(payload: Optional[dict] = None):
    if payload is None:
        payload = {}
    flag = {
        "id":        int(time.time() * 1000),
        "flag_type": payload.get("flag_type", "detected"),
        "x":         float(payload.get("x",     0)),
        "y":         float(payload.get("y",     0)),
        "label":     payload.get("label",        ""),
        "ts":        payload.get("ts",  time.time()),
    }
    MAP_STATE["flags"].append(flag)
    asyncio.create_task(_save_flag(flag))
    return {"ok": True}


async def _save_flag(flag: dict):
    try:
        db = SessionLocal()
        db.add(MapFlag(ts=flag["ts"], flag_type=flag["flag_type"],
                       x=flag["x"], y=flag["y"], label=flag["label"]))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] _save_flag error: {e}")


@app.get("/api/map/flags")
def get_map_flags(request: Request):
    verify_token(request)
    return MAP_STATE["flags"]


@app.delete("/api/map/flags")
def clear_map_flags(request: Request):
    verify_token(request)
    MAP_STATE["flags"].clear()
    MAP_STATE["track"].clear()
    try:
        db = SessionLocal()
        db.query(MapFlag).delete()
        db.query(RoverPosition).delete()
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] clear_map_flags error: {e}")
    return {"ok": True}


@app.post("/api/map/position")
async def update_rover_position(payload: Optional[dict] = None):
    """Jetson posts rover position every 0.5 s for map tracking."""
    if payload is None:
        payload = {}
    MAP_STATE["rover_x"]       = float(payload.get("x",       0))
    MAP_STATE["rover_y"]       = float(payload.get("y",       0))
    MAP_STATE["rover_heading"] = float(payload.get("heading", 0))
    MAP_STATE["rover_ts"]      = time.time()

    point = {
        "x":  MAP_STATE["rover_x"],
        "y":  MAP_STATE["rover_y"],
        "ts": MAP_STATE["rover_ts"],
    }
    MAP_STATE["track"].append(point)
    # Keep last 2000 points (~17 min at 2 pts/s)
    if len(MAP_STATE["track"]) > 2000:
        MAP_STATE["track"] = MAP_STATE["track"][-2000:]

    asyncio.create_task(_save_position(
        MAP_STATE["rover_x"],
        MAP_STATE["rover_y"],
        MAP_STATE["rover_heading"],
    ))
    return {"ok": True}


async def _save_position(x, y, heading):
    try:
        db = SessionLocal()
        db.add(RoverPosition(ts=time.time(), x=x, y=y, heading=heading))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] _save_position error: {e}")


@app.get("/api/map/state")
def get_map_state(request: Request):
    verify_token(request)
    return {
        "rover_x":       MAP_STATE["rover_x"],
        "rover_y":       MAP_STATE["rover_y"],
        "rover_heading": MAP_STATE["rover_heading"],
        "rover_ts":      MAP_STATE["rover_ts"],
        "flags":         MAP_STATE["flags"],
        "track":         MAP_STATE["track"][-500:],  # last 500 pts to frontend
    }


# ── VIDEO ─────────────────────────────────────────────────────────────────────

@app.get("/video")
def video(request: Request, token: Optional[str] = None):
    verify_token(request, token_query=token)
    return StreamingResponse(
        video_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── HISTORY ───────────────────────────────────────────────────────────────────

@app.get("/api/history")
def get_history(request: Request, limit: int = 100):
    verify_token(request)
    try:
        db   = SessionLocal()
        rows = db.query(Detection).order_by(Detection.ts.desc()).limit(limit).all()
        db.close()
        return [{"ts": r.ts, "count": r.count, "source": r.source} for r in rows]
    except Exception as e:
        print(f"[DB] get_history error: {e}")
        return []


@app.delete("/api/history")
def delete_history(request: Request):
    verify_token(request)
    try:
        db = SessionLocal()
        db.query(Detection).delete()
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] delete_history error: {e}")
    return {"ok": True}


# ── WEBSOCKET ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json({
                "ts":          STATE["ts"],
                "human_count": STATE["human_count"],
                "drive_mode":  DRIVE_MODE_STATE["mode"],
                "alert":       ALERT_STATE,
            })
            await asyncio.sleep(0.25)
    except Exception:
        pass