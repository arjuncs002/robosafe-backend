#!/usr/bin/env python3
"""
ROBOSAFE Backend - FINAL
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
    cv2  = None
    YOLO = None
    np   = None

from db import (
    init_db, SessionLocal,
    Detection, ControlCommand, MMWaveState,
    MapFlag, RoverPosition, DriveMode, AutoCommand, LifeConfirm
)

app = FastAPI(title="ROBOSAFE Backend FINAL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
def _ensure_defaults():
    try:
        db = SessionLocal()
        if db.query(DriveMode).count() == 0:
            db.add(DriveMode(mode="MANUAL"))
            db.commit()
        if db.query(LifeConfirm).count() == 0:
            db.add(LifeConfirm(confirmed=False, result="", cleared=True))
            db.commit()
        db.close()
    except Exception as e:
        print(f"[Init] DB error: {e}")

_ensure_defaults()

# ── AUTH ──────────────────────────────────────────────────────────────────────
TOKEN_TTL_SEC = 600
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
    TOKENS[token] = now
    return token

# ── CAMERA ────────────────────────────────────────────────────────────────────
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
        print(f"[Camera] Failed: {e}")

# ── IN-MEMORY STATE ───────────────────────────────────────────────────────────
STATE = {"ts": time.time(), "human_count": 0, "detections": []}

ROVER_STATE = {"last_command": "STOP", "last_command_ts": time.time()}

MMWAVE_STATE = {
    "status": "SENSOR DISABLED", "last_presence": 0,
    "energy_delta": 0, "respiration_detected": False,
    "distance": 0.0, "energy_min": 0, "energy_max": 0,
    "last_update": 0.0, "enabled": False,
}

DRIVE_MODE_STATE = {"mode": "MANUAL"}

MAP_STATE = {
    "rover_x": 0.0, "rover_y": 0.0, "rover_heading": 0.0,
    "rover_ts": time.time(), "flags": [], "track": [],
}

ALERT_STATE = {"type": "", "ts": 0.0, "active": False}

AUTO_CMD_STATE = {"command": "STOP", "ts": time.time()}

LIFE_CONFIRM_STATE = {"confirmed": False, "result": "", "cleared": True}

# New: manual mode request from Jetson
MANUAL_REQUEST_STATE = {"active": False, "reason": "", "ts": 0.0}

MIN_LOG_GAP_SEC   = 3.0
_last_save_ts     = 0.0
_last_saved_count = -1

# ── HELPERS ───────────────────────────────────────────────────────────────────
def save_detection(count: int):
    try:
        db = SessionLocal()
        db.add(Detection(ts=time.time(), count=count, source="jetson"))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] save_detection: {e}")

def video_generator():
    if not USE_LOCAL_CAMERA or cap is None or not CV2_AVAILABLE:
        if CV2_AVAILABLE and np is not None:
            blank = np.ones((480, 640, 3), dtype=np.uint8) * 30
            cv2.putText(blank, "Camera Offline", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            ret, jpeg = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            frame_bytes = jpeg.tobytes() if ret else b""
        else:
            frame_bytes = b""
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.1)
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame = cv2.resize(frame, (960, 540))
        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"ok": True, "service": "robosafe-backend-final"}

# AUTH
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
    if not new or new != conf:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if len(new) < 4:
        raise HTTPException(status_code=400, detail="Too short")
    DASH_PASSWORD["value"] = new
    return {"ok": True}

# STATE
@app.get("/api/state")
def get_state(request: Request, overlays: int = 1):
    verify_token(request)
    now = time.time()
    if MMWAVE_STATE["enabled"] and MMWAVE_STATE["last_update"] > 0 and \
            (now - MMWAVE_STATE["last_update"]) > 15:
        MMWAVE_STATE.update({
            "status": "NO PRESENCE DETECTED", "last_presence": 0,
            "energy_delta": 0, "respiration_detected": False, "distance": 0.0,
        })
    return JSONResponse({
        "ts":          STATE["ts"],
        "human_count": STATE["human_count"],
        "detections":  STATE["detections"] if overlays == 1 else [],
        "mmwave":      MMWAVE_STATE,
        "drive_mode":  DRIVE_MODE_STATE["mode"],
        "alert":       ALERT_STATE,
        "manual_request": MANUAL_REQUEST_STATE,
    })

@app.post("/api/state")
async def update_state(payload: Optional[dict] = None):
    if payload is None:
        payload = {}
    global _last_save_ts, _last_saved_count
    STATE["ts"]          = time.time()
    STATE["human_count"] = int(payload.get("human_count", 0))
    STATE["detections"]  = payload.get("detections", [])
    count = STATE["human_count"]
    if count > 0:
        now = time.time()
        if (now - _last_save_ts) >= MIN_LOG_GAP_SEC or count != _last_saved_count:
            save_detection(count)
            _last_save_ts     = now
            _last_saved_count = count
    return {"ok": True}

# MMWAVE
@app.post("/api/mmwave")
async def update_mmwave(payload: Optional[dict] = None):
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
    return {"ok": True, "enabled": MMWAVE_STATE["enabled"]}

@app.post("/api/mmwave/toggle")
async def toggle_mmwave(request: Request, payload: Optional[dict] = None):
    verify_token(request)
    if payload is None:
        payload = {}
    enabled = bool(payload.get("enabled", False))
    MMWAVE_STATE["enabled"] = enabled
    if not enabled:
        MMWAVE_STATE.update({
            "status": "SENSOR DISABLED", "last_presence": 0,
            "energy_delta": 0, "distance": 0.0,
            "energy_min": 0, "energy_max": 0, "respiration_detected": False,
        })
        ROVER_STATE["last_command"]    = "STOP"
        ROVER_STATE["last_command_ts"] = time.time()
    else:
        MMWAVE_STATE["status"]      = "NO PRESENCE DETECTED"
        MMWAVE_STATE["last_update"] = 0.0
    return {"ok": True, "enabled": enabled}

@app.get("/api/mmwave")
def get_mmwave(request: Request):
    verify_token(request)
    return JSONResponse(MMWAVE_STATE)

# ROVER CONTROL
@app.post("/api/control")
async def control_rover(request: Request, payload: Optional[dict] = None):
    verify_token(request)
    if payload is None:
        payload = {}
    command = str(payload.get("command", "STOP")).strip().upper()
    if command not in {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}:
        raise HTTPException(status_code=400, detail="Invalid command")
    if MMWAVE_STATE["enabled"] and command != "STOP":
        return JSONResponse({"ok": False, "reason": "mmWave active"})
    ROVER_STATE["last_command"]    = command
    ROVER_STATE["last_command_ts"] = time.time()
    asyncio.create_task(_save_command(command))
    return {"ok": True, "command": command}

async def _save_command(command: str):
    try:
        db = SessionLocal()
        db.add(ControlCommand(ts=time.time(), command=command))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] _save_command: {e}")

@app.get("/api/control/latest")
async def get_latest_command():
    """ESP32 polls this in MANUAL mode."""
    return {
        "command": ROVER_STATE["last_command"],
        "ts":      ROVER_STATE["last_command_ts"],
        "source":  "manual",
    }

# DRIVE MODE
@app.get("/api/drive_mode/public")
def get_drive_mode_public():
    return {"mode": DRIVE_MODE_STATE["mode"]}

@app.get("/api/drive_mode")
def get_drive_mode(request: Request):
    verify_token(request)
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
        ROVER_STATE["last_command"]    = "STOP"
        ROVER_STATE["last_command_ts"] = time.time()
        AUTO_CMD_STATE["command"]      = "STOP"
    # Clear manual request when operator manually switches
    MANUAL_REQUEST_STATE["active"] = False
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
        print(f"[DB] set_drive_mode: {e}")
    print(f"[Mode] → {mode}")
    return {"ok": True, "mode": mode}

# ALERT
@app.post("/api/alert")
async def post_alert_endpoint(payload: Optional[dict] = None):
    if payload is None:
        payload = {}
    ALERT_STATE["type"]   = payload.get("type", "siren")
    ALERT_STATE["ts"]     = time.time()
    ALERT_STATE["active"] = True
    print("[Alert] Triggered")
    return {"ok": True}

@app.post("/api/alert/clear")
async def clear_alert(request: Request):
    verify_token(request)
    ALERT_STATE["active"] = False
    return {"ok": True}

# MANUAL REQUEST (Jetson → dashboard popup to switch to manual)
@app.post("/api/manual_request")
async def post_manual_request(payload: Optional[dict] = None):
    if payload is None:
        payload = {}
    MANUAL_REQUEST_STATE["active"] = True
    MANUAL_REQUEST_STATE["reason"] = payload.get("reason", "No human found")
    MANUAL_REQUEST_STATE["ts"]     = time.time()
    print("[ManualRequest] Jetson requests manual mode")
    return {"ok": True}

@app.post("/api/manual_request/clear")
async def clear_manual_request(request: Request):
    verify_token(request)
    MANUAL_REQUEST_STATE["active"] = False
    return {"ok": True}

# LIFE CONFIRM
@app.post("/api/life_confirm")
async def submit_life_confirm(request: Request, payload: Optional[dict] = None):
    verify_token(request)
    if payload is None:
        payload = {}
    result = str(payload.get("result", "")).strip().lower()
    if result not in ("alive", "not_alive"):
        raise HTTPException(status_code=400, detail="result must be alive or not_alive")
    LIFE_CONFIRM_STATE["confirmed"] = True
    LIFE_CONFIRM_STATE["result"]    = result
    LIFE_CONFIRM_STATE["cleared"]   = False
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
        print(f"[DB] life_confirm: {e}")
    print(f"[LifeConfirm] {result}")
    return {"ok": True, "result": result}

@app.get("/api/life_confirm/pending")
async def get_life_confirm_pending():
    return {
        "confirmed": LIFE_CONFIRM_STATE["confirmed"],
        "result":    LIFE_CONFIRM_STATE["result"],
    }

@app.post("/api/life_confirm/clear")
async def clear_life_confirm():
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
        print(f"[DB] clear_life_confirm: {e}")
    return {"ok": True}

# MAP
@app.post("/api/map/flag")
async def add_map_flag(payload: Optional[dict] = None):
    if payload is None:
        payload = {}
    flag = {
        "id":        int(time.time() * 1000),
        "flag_type": payload.get("flag_type", "detected"),
        "x":         float(payload.get("x",   0)),
        "y":         float(payload.get("y",   0)),
        "label":     payload.get("label",      ""),
        "ts":        payload.get("ts", time.time()),
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
        print(f"[DB] _save_flag: {e}")

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
        print(f"[DB] clear_map_flags: {e}")
    return {"ok": True}

@app.post("/api/map/position")
async def update_rover_position(payload: Optional[dict] = None):
    if payload is None:
        payload = {}
    MAP_STATE["rover_x"]       = float(payload.get("x",       0))
    MAP_STATE["rover_y"]       = float(payload.get("y",       0))
    MAP_STATE["rover_heading"] = float(payload.get("heading", 0))
    MAP_STATE["rover_ts"]      = time.time()
    point = {"x": MAP_STATE["rover_x"], "y": MAP_STATE["rover_y"], "ts": MAP_STATE["rover_ts"]}
    MAP_STATE["track"].append(point)
    if len(MAP_STATE["track"]) > 2000:
        MAP_STATE["track"] = MAP_STATE["track"][-2000:]
    asyncio.create_task(_save_position(
        MAP_STATE["rover_x"], MAP_STATE["rover_y"], MAP_STATE["rover_heading"]))
    return {"ok": True}

async def _save_position(x, y, heading):
    try:
        db = SessionLocal()
        db.add(RoverPosition(ts=time.time(), x=x, y=y, heading=heading))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[DB] _save_position: {e}")

@app.get("/api/map/state")
def get_map_state(request: Request):
    verify_token(request)
    return {
        "rover_x":       MAP_STATE["rover_x"],
        "rover_y":       MAP_STATE["rover_y"],
        "rover_heading": MAP_STATE["rover_heading"],
        "rover_ts":      MAP_STATE["rover_ts"],
        "flags":         MAP_STATE["flags"],
        "track":         MAP_STATE["track"][-500:],
    }

# VIDEO
@app.get("/video")
def video(request: Request, token: Optional[str] = None):
    verify_token(request, token_query=token)
    return StreamingResponse(video_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# HISTORY
@app.get("/api/history")
def get_history(request: Request, limit: int = 100):
    verify_token(request)
    try:
        db   = SessionLocal()
        rows = db.query(Detection).order_by(Detection.ts.desc()).limit(limit).all()
        db.close()
        return [{"ts": r.ts, "count": r.count, "source": r.source} for r in rows]
    except Exception as e:
        print(f"[DB] get_history: {e}")
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
        print(f"[DB] delete_history: {e}")
    return {"ok": True}

# WEBSOCKET
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json({
                "ts":             STATE["ts"],
                "human_count":    STATE["human_count"],
                "drive_mode":     DRIVE_MODE_STATE["mode"],
                "alert":          ALERT_STATE,
                "manual_request": MANUAL_REQUEST_STATE,
            })
            await asyncio.sleep(0.25)
    except Exception:
        pass