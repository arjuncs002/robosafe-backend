import time
import asyncio
import secrets
import os
import numpy as np

from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

try:
    import cv2
    from ultralytics import YOLO
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    YOLO = None

from db import (
    init_db, SessionLocal,
    Detection, ControlCommand, MMWaveState,
    MapFlag, RoverPosition, DriveMode, LifeConfirmation, AutoCommand
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

# ── ensure a DriveMode row exists ──────────────────────────────────────────────
def _ensure_drive_mode():
    try:
        db = SessionLocal()
        if db.query(DriveMode).count() == 0:
            db.add(DriveMode(mode="MANUAL"))
            db.commit()
        db.close()
    except:
        pass

_ensure_drive_mode()

# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────
TOKEN_TTL_SEC = 300
TOKENS = {}
DASH_PASSWORD = {"value": os.getenv("ADMIN_PASSWORD", "GROUP5")}


def issue_token():
    token = secrets.token_urlsafe(32)
    TOKENS[token] = time.time()
    return token


def verify_token(request: Request, token_query: str | None = None):
    token = None
    auth = request.headers.get("authorization") or ""
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


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA (unchanged from Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
USE_LOCAL_CAMERA = os.getenv("USE_LOCAL_CAMERA", "false").lower() == "true"

if USE_LOCAL_CAMERA and CV2_AVAILABLE:
    MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
    try:
        model = YOLO(MODEL_PATH)
        cap = cv2.VideoCapture(0)
    except:
        model = None
        cap = None
else:
    model = None
    cap = None

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY STATE
# ─────────────────────────────────────────────────────────────────────────────
STATE = {
    "ts": time.time(),
    "human_count": 0,
    "detections": [],
}

ROVER_STATE = {
    "last_command": "STOP",
    "last_command_ts": time.time(),
}

MMWAVE_STATE = {
    "status": "NO PRESENCE DETECTED",
    "last_presence": 0,
    "energy_delta": 0,
    "respiration_detected": False,
    "distance": 0,
    "energy_min": 0,
    "energy_max": 0,
    "last_update": 0,
    "enabled": True,
}

# Phase 2 in-memory state
DRIVE_MODE_STATE = {"mode": "MANUAL"}   # MANUAL | AUTO

MAP_STATE = {
    "rover_x":       0.0,
    "rover_y":       0.0,
    "rover_heading": 0.0,
    "rover_ts":      time.time(),
    "flags":         [],          # list of flag dicts
    "track":         [],          # list of {x,y,ts} for path drawing
}

ALERT_STATE = {
    "type":    "",
    "ts":      0,
    "active":  False,
}

LIFE_CONFIRM_STATE = {
    "confirmed": False,
    "result":    "",
    "ts":        0,
}

AUTO_CMD_STATE = {
    "command": "STOP",
    "ts":      time.time(),
}

MIN_LOG_GAP_SEC   = 3.0
_last_save_ts     = 0.0
_last_saved_count = -1

MAX_TRACK_POINTS = 2000   # keep last N track points in memory


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
        print(f"DB Error: {e}")


def detect_and_draw(frame):
    if model is None or not CV2_AVAILABLE:
        return frame, 0, []
    results = model.predict(frame, verbose=False)[0]
    count = 0
    dets  = []
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
    if not USE_LOCAL_CAMERA or cap is None or not CV2_AVAILABLE:
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 30
        if CV2_AVAILABLE:
            cv2.putText(blank, "Camera Offline", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            ret, jpeg = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = jpeg.tobytes()
        else:
            frame_bytes = b''
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + frame_bytes + b"\r\n")
            time.sleep(0.1)

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
            allow_save = (now - _last_save_ts) >= MIN_LOG_GAP_SEC or count != _last_saved_count
            if allow_save:
                save_detection(count)
                _last_save_ts     = now
                _last_saved_count = count
        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpeg.tobytes() + b"\r\n")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 ENDPOINTS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"ok": True, "service": "robosafe-backend-v2",
            "camera_enabled": USE_LOCAL_CAMERA and CV2_AVAILABLE}


@app.post("/api/login")
async def login(payload: dict | None = None):
    if payload is None:
        payload = {}
    pw = str(payload.get("password", "")).strip()
    if pw != DASH_PASSWORD["value"]:
        raise HTTPException(status_code=401, detail="Wrong password")
    return {"token": issue_token()}


@app.post("/api/password")
async def change_password(request: Request, payload: dict | None = None):
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


@app.get("/api/state")
def get_state(request: Request, overlays: int = 1):
    verify_token(request)
    now = time.time()
    if MMWAVE_STATE["last_update"] > 0 and (now - MMWAVE_STATE["last_update"]) > 15:
        MMWAVE_STATE.update({
            "status": "NO PRESENCE DETECTED",
            "last_presence": 0, "energy_delta": 0,
            "respiration_detected": False, "distance": 0,
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
async def update_state(request: Request, payload: dict | None = None):
    if payload is None:
        payload = {}
    STATE["ts"]          = time.time()
    STATE["human_count"] = payload.get("human_count", 0)
    STATE["detections"]  = payload.get("detections",  [])
    if STATE["human_count"] > 0:
        save_detection(STATE["human_count"])
    return {"ok": True}


@app.post("/api/mmwave")
async def update_mmwave(payload: dict | None = None):
    if payload is None:
        payload = {}
    enabled = MMWAVE_STATE["enabled"]
    if enabled:
        MMWAVE_STATE.update({
            "status":               payload.get("status",               "NO PRESENCE DETECTED"),
            "last_presence":        payload.get("last_presence",        0),
            "energy_delta":         payload.get("energy_delta",         0),
            "respiration_detected": payload.get("respiration_detected", False),
            "distance":             payload.get("distance",             0),
            "energy_min":           payload.get("energy_min",           0),
            "energy_max":           payload.get("energy_max",           0),
            "last_update":          time.time(),
        })
        asyncio.create_task(_save_mmwave(
            MMWAVE_STATE["status"], MMWAVE_STATE["last_presence"],
            MMWAVE_STATE["energy_delta"], MMWAVE_STATE["respiration_detected"]
        ))
    return {"ok": True, "enabled": enabled}


async def _save_mmwave(status, presence, energy_delta, respiration):
    try:
        db = SessionLocal()
        db.add(MMWaveState(ts=time.time(), status=status, presence=presence,
                           energy_delta=energy_delta,
                           respiration_detected=respiration))
        db.commit()
        db.close()
    except Exception as e:
        print(f"DB Error: {e}")


@app.post("/api/mmwave/toggle")
async def toggle_mmwave(request: Request, payload: dict | None = None):
    verify_token(request)
    if payload is None:
        payload = {}
    enabled = payload.get("enabled", True)
    MMWAVE_STATE["enabled"] = enabled
    if not enabled:
        MMWAVE_STATE.update({
            "status": "SENSOR DISABLED", "last_presence": 0,
            "energy_delta": 0, "distance": 0,
            "energy_min": 0, "energy_max": 0, "respiration_detected": False,
        })
        ROVER_STATE["last_command"]    = "STOP"
        ROVER_STATE["last_command_ts"] = time.time()
    else:
        MMWAVE_STATE["status"]      = "NO PRESENCE DETECTED"
        MMWAVE_STATE["last_update"] = 0
    return {"ok": True, "enabled": enabled}


@app.get("/api/mmwave")
def get_mmwave(request: Request):
    verify_token(request)
    return JSONResponse(MMWAVE_STATE)


@app.get("/api/history")
def get_history(request: Request, limit: int = 100):
    verify_token(request)
    try:
        db   = SessionLocal()
        rows = db.query(Detection).order_by(Detection.ts.desc()).limit(limit).all()
        db.close()
        return [{"ts": r.ts, "count": r.count, "source": r.source} for r in rows]
    except Exception as e:
        print(f"DB Error: {e}")
        return []


@app.delete("/api/history")
def delete_history(request: Request):
    verify_token(request)
    try:
        db = SessionLocal()
        db.query(Detection).delete()
        db.commit()
        db.close()
    except:
        pass
    return {"ok": True}


@app.get("/video")
def video(request: Request, token: str | None = None):
    verify_token(request, token_query=token)
    return StreamingResponse(video_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/control")
async def control_rover(request: Request, payload: dict | None = None):
    verify_token(request)
    if payload is None:
        payload = {}
    command = str(payload.get("command", "STOP")).strip().upper()
    valid   = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]
    if command not in valid:
        raise HTTPException(status_code=400, detail="Invalid command")
    if MMWAVE_STATE["enabled"] and command != "STOP":
        return JSONResponse({"ok": False, "reason": "mmWave active - motors disabled"})
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
        print(f"DB Error: {e}")


@app.get("/api/control/latest")
async def get_latest_command():
    # In AUTO mode, Jetson drives via serial; HTTP control is manual override
    if DRIVE_MODE_STATE["mode"] == "AUTO":
        # Return auto command so ESP32 can execute via HTTP fallback
        return {
            "command": AUTO_CMD_STATE["command"],
            "ts":      AUTO_CMD_STATE["ts"],
            "source":  "auto"
        }
    return {
        "command": ROVER_STATE["last_command"],
        "ts":      ROVER_STATE["last_command_ts"],
        "source":  "manual"
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(STATE)
            await asyncio.sleep(0.25)
    except:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Drive Mode ────────────────────────────────────────────────────────────────
@app.get("/api/drive_mode/public")
def get_drive_mode_public():
    """No-auth endpoint for ESP32 to check drive mode."""
    return {"mode": DRIVE_MODE_STATE["mode"]}


@app.get("/api/drive_mode")
def get_drive_mode(request: Request):
    verify_token(request)
    return {"mode": DRIVE_MODE_STATE["mode"]}


@app.post("/api/drive_mode")
async def set_drive_mode(request: Request, payload: dict | None = None):
    verify_token(request)
    if payload is None:
        payload = {}
    mode = str(payload.get("mode", "MANUAL")).upper()
    if mode not in ("AUTO", "MANUAL"):
        raise HTTPException(status_code=400, detail="mode must be AUTO or MANUAL")
    DRIVE_MODE_STATE["mode"] = mode
    # If switching to MANUAL, stop rover
    if mode == "MANUAL":
        ROVER_STATE["last_command"]    = "STOP"
        ROVER_STATE["last_command_ts"] = time.time()
    # Persist
    try:
        db  = SessionLocal()
        row = db.query(DriveMode).first()
        if row:
            row.mode = mode
        else:
            db.add(DriveMode(mode=mode))
        db.commit()
        db.close()
    except:
        pass
    print(f"[DriveMode] → {mode}")
    return {"ok": True, "mode": mode}


# ── Auto Command (Jetson → Backend → ESP32 fallback) ─────────────────────────
@app.post("/api/auto_command")
async def set_auto_command(payload: dict | None = None):
    """Called by Jetson when serial is unavailable. ESP32 polls /api/control/latest."""
    if payload is None:
        payload = {}
    cmd = str(payload.get("command", "STOP")).strip().upper()
    AUTO_CMD_STATE["command"] = cmd
    AUTO_CMD_STATE["ts"]      = time.time()
    return {"ok": True}


# ── Map Flags ─────────────────────────────────────────────────────────────────
@app.post("/api/map/flag")
async def add_map_flag(payload: dict | None = None):
    """Called by Jetson to drop a flag on the map."""
    if payload is None:
        payload = {}
    flag = {
        "id":        int(time.time() * 1000),
        "flag_type": payload.get("flag_type", "detected"),
        "x":         float(payload.get("x", 0)),
        "y":         float(payload.get("y", 0)),
        "label":     payload.get("label", ""),
        "ts":        payload.get("ts", time.time()),
    }
    MAP_STATE["flags"].append(flag)
    # Persist
    asyncio.create_task(_save_flag(flag))
    print(f"[Map] Flag: {flag['flag_type']} @ ({flag['x']:.2f}, {flag['y']:.2f})")
    return {"ok": True}


async def _save_flag(flag: dict):
    try:
        db = SessionLocal()
        db.add(MapFlag(ts=flag["ts"], flag_type=flag["flag_type"],
                       x=flag["x"], y=flag["y"], label=flag["label"]))
        db.commit()
        db.close()
    except Exception as e:
        print(f"DB Error: {e}")


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
    except:
        pass
    return {"ok": True}


# ── Rover Position ────────────────────────────────────────────────────────────
@app.post("/api/map/position")
async def update_rover_position(payload: dict | None = None):
    """Called by Jetson every ~0.5s with dead-reckoning position."""
    if payload is None:
        payload = {}
    MAP_STATE["rover_x"]       = float(payload.get("x",       0))
    MAP_STATE["rover_y"]       = float(payload.get("y",       0))
    MAP_STATE["rover_heading"] = float(payload.get("heading", 0))
    MAP_STATE["rover_ts"]      = time.time()

    point = {"x": MAP_STATE["rover_x"], "y": MAP_STATE["rover_y"],
             "ts": MAP_STATE["rover_ts"]}
    MAP_STATE["track"].append(point)
    if len(MAP_STATE["track"]) > 2000:
        MAP_STATE["track"] = MAP_STATE["track"][-2000:]

    asyncio.create_task(_save_position(
        MAP_STATE["rover_x"], MAP_STATE["rover_y"], MAP_STATE["rover_heading"]
    ))
    return {"ok": True}


async def _save_position(x, y, heading):
    try:
        db = SessionLocal()
        db.add(RoverPosition(ts=time.time(), x=x, y=y, heading=heading))
        db.commit()
        db.close()
    except Exception as e:
        print(f"DB Error: {e}")


@app.get("/api/map/state")
def get_map_state(request: Request):
    verify_token(request)
    return {
        "rover_x":       MAP_STATE["rover_x"],
        "rover_y":       MAP_STATE["rover_y"],
        "rover_heading": MAP_STATE["rover_heading"],
        "rover_ts":      MAP_STATE["rover_ts"],
        "flags":         MAP_STATE["flags"],
        "track":         MAP_STATE["track"][-500:],  # last 500 for the client
    }


# ── Alert ─────────────────────────────────────────────────────────────────────
@app.post("/api/alert")
async def post_alert(payload: dict | None = None):
    """Jetson sends this when rover arrives at human."""
    if payload is None:
        payload = {}
    ALERT_STATE["type"]   = payload.get("type", "siren")
    ALERT_STATE["ts"]     = time.time()
    ALERT_STATE["active"] = True
    return {"ok": True}


@app.post("/api/alert/clear")
async def clear_alert(request: Request):
    verify_token(request)
    ALERT_STATE["active"] = False
    return {"ok": True}


# ── Life Confirmation ─────────────────────────────────────────────────────────
@app.get("/api/life_confirm/pending")
async def life_confirm_pending():
    """Jetson polls this to see if operator has confirmed life status."""
    return {
        "confirmed": LIFE_CONFIRM_STATE["confirmed"],
        "result":    LIFE_CONFIRM_STATE["result"],
        "ts":        LIFE_CONFIRM_STATE["ts"],
    }


@app.post("/api/life_confirm")
async def submit_life_confirm(request: Request, payload: dict | None = None):
    """Website calls this when operator clicks Alive / Not Alive."""
    verify_token(request)
    if payload is None:
        payload = {}
    result = str(payload.get("result", "")).lower()   # alive | not_alive
    if result not in ("alive", "not_alive"):
        raise HTTPException(status_code=400, detail="result must be alive or not_alive")
    LIFE_CONFIRM_STATE["confirmed"] = True
    LIFE_CONFIRM_STATE["result"]    = result
    LIFE_CONFIRM_STATE["ts"]        = time.time()
    # Also update the latest flag colour on the map
    if MAP_STATE["flags"]:
        # Update last arrived flag
        for flag in reversed(MAP_STATE["flags"]):
            if flag["flag_type"] == "arrived":
                flag["flag_type"] = result   # "alive" or "not_alive"
                flag["label"]     = f"Confirmed: {result}"
                break
    # Persist
    try:
        db = SessionLocal()
        db.add(LifeConfirmation(ts=time.time(), confirmed=True, result=result))
        db.commit()
        db.close()
    except:
        pass
    return {"ok": True, "result": result}


@app.post("/api/life_confirm/clear")
async def clear_life_confirm():
    """Jetson calls this after reading the confirmation."""
    LIFE_CONFIRM_STATE["confirmed"] = False
    LIFE_CONFIRM_STATE["result"]    = ""
    return {"ok": True}