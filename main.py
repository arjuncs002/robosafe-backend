import time
import asyncio
import secrets
import cv2
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

from db import init_db, SessionLocal, Detection, ControlCommand, MMWaveState

app = FastAPI(title="ROBOSAFE Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

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
    "status": "NO CONFIRMATION",
    "last_presence": 0,
    "energy_delta": 0,
    "respiration_detected": False,
    "last_update": 0,
}

MIN_LOG_GAP_SEC = 3.0
_last_save_ts = 0.0
_last_saved_count = -1

def save_detection(count: int):
    try:
        db = SessionLocal()
        db.add(Detection(ts=time.time(), count=count, source="webcam"))
        db.commit()
        db.close()
    except Exception as e:
        print(f"DB Error: {e}")

def detect_and_draw(frame):
    if model is None or not CV2_AVAILABLE:
        return frame, 0, []
    
    results = model.predict(frame, verbose=False)[0]
    count = 0
    dets = []

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls == 0 and conf >= 0.35:
                count += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                dets.append(
                    {"cls": cls, "confidence": conf, "bbox": [int(x1), int(y1), int(x2), int(y2)]}
                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"Human {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

    return frame, count, dets

def video_generator():
    global _last_save_ts, _last_saved_count

    if not USE_LOCAL_CAMERA or cap is None or not CV2_AVAILABLE:
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 30
        if CV2_AVAILABLE:
            cv2.putText(blank, "Camera Offline", (180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            ret, jpeg = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        else:
            from io import BytesIO
            try:
                from PIL import Image
                img = Image.fromarray(blank.astype('uint8'), 'RGB')
                buf = BytesIO()
                img.save(buf, format='JPEG')
                jpeg_bytes = buf.getvalue()
            except:
                jpeg_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xfc\x28\xa2\x8a\x00\xff\xd9'
                ret = True
                
            frame_bytes = jpeg_bytes
        
        while True:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            time.sleep(0.1)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (960, 540))
        frame, count, dets = detect_and_draw(frame)

        STATE["ts"] = time.time()
        STATE["human_count"] = count
        STATE["detections"] = dets

        if count > 0:
            now = time.time()
            allow_save = False
            if (now - _last_save_ts) >= MIN_LOG_GAP_SEC:
                allow_save = True
            if count != _last_saved_count:
                allow_save = True

            if allow_save:
                save_detection(count)
                _last_save_ts = now
                _last_saved_count = count

        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )

@app.get("/")
def root():
    return {
        "ok": True, 
        "service": "robosafe-backend",
        "camera_enabled": USE_LOCAL_CAMERA and CV2_AVAILABLE
    }

@app.post("/api/login")
async def login(payload: dict | None = None):
    if payload is None:
        payload = {}

    pw = str(payload.get("password", "")).strip()
    if pw != DASH_PASSWORD["value"]:
        raise HTTPException(status_code=401, detail="Wrong password")

    token = issue_token()
    return {"token": token}

@app.post("/api/password")
async def change_password(request: Request, payload: dict | None = None):
    verify_token(request)

    if payload is None:
        payload = {}

    cur = str(payload.get("current_password", "")).strip()
    new = str(payload.get("new_password", "")).strip()
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
    
    # Auto-reset mmWave to NO CONFIRMATION if no update in 15 seconds
    now = time.time()
    if MMWAVE_STATE["last_update"] > 0 and (now - MMWAVE_STATE["last_update"]) > 15:
        MMWAVE_STATE["status"] = "NO CONFIRMATION"
        MMWAVE_STATE["last_presence"] = 0
        MMWAVE_STATE["energy_delta"] = 0
        MMWAVE_STATE["respiration_detected"] = False

    response_data = {
        "ts": STATE["ts"],
        "human_count": STATE["human_count"],
        "detections": STATE["detections"] if overlays == 1 else [],
        "mmwave": MMWAVE_STATE,
    }

    return JSONResponse(response_data)

@app.post("/api/state")
async def update_state(request: Request, payload: dict | None = None):
    if payload is None:
        payload = {}
    
    STATE["ts"] = time.time()
    STATE["human_count"] = payload.get("human_count", 0)
    STATE["detections"] = payload.get("detections", [])
    
    count = STATE["human_count"]
    if count > 0:
        save_detection(count)
    
    return {"ok": True}

@app.post("/api/mmwave")
async def update_mmwave(payload: dict | None = None):
    if payload is None:
        payload = {}
    
    status = payload.get("status", "NO CONFIRMATION")
    presence = payload.get("last_presence", 0)
    energy_delta = payload.get("energy_delta", 0)
    respiration = payload.get("respiration_detected", False)
    
    MMWAVE_STATE["status"] = status
    MMWAVE_STATE["last_presence"] = presence
    MMWAVE_STATE["energy_delta"] = energy_delta
    MMWAVE_STATE["respiration_detected"] = respiration
    MMWAVE_STATE["last_update"] = time.time()
    
    try:
        db = SessionLocal()
        db.add(MMWaveState(
            ts=time.time(),
            status=status,
            presence=presence,
            energy_delta=energy_delta,
            respiration_detected=respiration
        ))
        db.commit()
        db.close()
    except Exception as e:
        print(f"DB Error: {e}")
    
    print(f"[mmWave] Status: {status}, Presence: {presence}, Delta: {energy_delta}, Respiration: {respiration}")
    
    return {"ok": True}

@app.get("/api/mmwave")
def get_mmwave(request: Request):
    verify_token(request)
    return JSONResponse(MMWAVE_STATE)

@app.get("/api/history")
def get_history(request: Request, limit: int = 100):
    verify_token(request)

    try:
        db = SessionLocal()
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
    except Exception as e:
        print(f"DB Error: {e}")
    
    return {"ok": True}

@app.get("/video")
def video(request: Request, token: str | None = None):
    verify_token(request, token_query=token)
    return StreamingResponse(
        video_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.post("/api/control")
async def control_rover(request: Request, payload: dict | None = None):
    verify_token(request)

    if payload is None:
        payload = {}

    command = str(payload.get("command", "STOP")).strip().upper()

    valid_commands = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]
    if command not in valid_commands:
        raise HTTPException(status_code=400, detail="Invalid command")

    ROVER_STATE["last_command"] = command
    ROVER_STATE["last_command_ts"] = time.time()

    try:
        db = SessionLocal()
        db.add(ControlCommand(ts=time.time(), command=command))
        db.commit()
        db.close()
    except Exception as e:
        print(f"DB Error: {e}")

    print(f"[ROVER CONTROL] Command received: {command}")

    return {"ok": True, "command": command, "ts": ROVER_STATE["last_command_ts"]}

@app.get("/api/control/latest")
async def get_latest_command():
    return {
        "command": ROVER_STATE["last_command"],
        "ts": ROVER_STATE["last_command_ts"]
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