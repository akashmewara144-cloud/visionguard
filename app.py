import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import time
import base64
import threading
import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

PORT = int(os.environ.get("PORT", 5000))

CONFIDENCE = 0.45
INFERENCE_INTERVAL = 0.35
MODEL_SIZE = 320
MAX_DETECTIONS = 10
CAMERA_TIMEOUT = 5

LOITERING_SECONDS = 10
ABNORMAL_SPEED = 450
CROWD_THRESHOLD = 4
INCIDENT_COOLDOWN = 8
TRACK_TIMEOUT = 3

ENTRY_END = 0.25
RESTRICTED_START = 0.65

app = Flask(__name__, static_folder=BASE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
CORS(app)

os.makedirs(EVIDENCE_DIR, exist_ok=True)

model = None
latest_jpeg = None
last_frame_time = 0
last_inference_time = 0

frame_lock = threading.Lock()
state_lock = threading.Lock()
processing_lock = threading.Lock()

state = {
    "people": 0,
    "objects": 0,
    "fps": 0.0,
    "inferenceTime": 0.0,
    "entered": 0,
    "exited": 0,
    "risk": 5,
    "riskLevel": "LOW",
    "riskReasons": [],
    "camera": False,
    "yolo": False
}

tracks = {}
cooldowns = {}
incidents = []
incident_id = 1
next_track_id = 1

try:
    model = YOLO(MODEL_PATH)
    model.fuse()
    state["yolo"] = True
    print("YOLOv8n loaded successfully")
except Exception as e:
    print("YOLO loading failed:", e)
    model = None


def get_regions(frame):
    h, w = frame.shape[:2]
    return (
        int(w * ENTRY_END),
        int(w * RESTRICTED_START),
        w,
        h
    )


def get_person_zone(x1, x2, frame):
    _, _, w, _ = get_regions(frame)
    center_x = (x1 + x2) / 2
    ratio = center_x / max(w, 1)

    if ratio < ENTRY_END:
        return "ENTRY"

    if ratio >= RESTRICTED_START:
        return "RESTRICTED"

    return "MONITORING"


def create_incident(
    event_type,
    track_id,
    confidence,
    frame,
    score,
    severity,
    reason
):
    global incident_id

    key = f"{event_type}_{track_id}"
    now = time.time()

    if now - cooldowns.get(key, 0) < INCIDENT_COOLDOWN:
        return False

    cooldowns[key] = now

    stamp = datetime.now()

    safe_type = "".join(
        c if c.isalnum() else "_"
        for c in event_type
    )

    filename = (
        f"incident_"
        f"{stamp.strftime('%Y%m%d_%H%M%S_%f')}_"
        f"{safe_type}_"
        f"{track_id}.jpg"
    )

    path = os.path.join(EVIDENCE_DIR, filename)

    evidence = None

    try:
        small = cv2.resize(
            frame,
            (320, 240),
            interpolation=cv2.INTER_AREA
        )

        if cv2.imwrite(
            path,
            small,
            [cv2.IMWRITE_JPEG_QUALITY, 65]
        ):
            evidence = f"/evidence/{filename}"

        del small

    except Exception as e:
        print("Evidence error:", e)

    incident = {
        "id": incident_id,
        "type": event_type,
        "trackingId": int(track_id),
        "confidence": round(float(confidence) * 100, 1),
        "risk": score,
        "severity": severity,
        "reason": reason,
        "camera": "Browser Webcam",
        "timestamp": stamp.strftime("%Y-%m-%d %H:%M:%S"),
        "evidence": evidence
    }

    with state_lock:
        incidents.insert(0, incident)

        if len(incidents) > 30:
            del incidents[30:]

    print(
        f"INCIDENT | {event_type} | "
        f"ID {track_id} | {severity}"
    )

    incident_id += 1

    return True


def find_person_track(cx, cy):
    global next_track_id

    best_id = None
    best_distance = 999999

    for track_id, track in tracks.items():
        distance = (
            (cx - track["previous_x"]) ** 2 +
            (cy - track["previous_y"]) ** 2
        ) ** 0.5

        if distance < 80 and distance < best_distance:
            best_distance = distance
            best_id = track_id

    if best_id is None:
        best_id = next_track_id
        next_track_id += 1

    return best_id


def process_person(
    track_id,
    confidence,
    x1,
    y1,
    x2,
    y2,
    frame
):
    now = time.time()

    zone = get_person_zone(
        x1,
        x2,
        frame
    )

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    if track_id not in tracks:
        tracks[track_id] = {
            "zone": zone,
            "previous_zone": zone,
            "previous_x": center_x,
            "previous_y": center_y,
            "previous_time": now,
            "zone_start": now if zone == "RESTRICTED" else None,
            "loitering": False,
            "abnormal": False,
            "last_seen": now
        }

        if zone == "RESTRICTED":
            create_incident(
                "Restricted Zone Intrusion",
                track_id,
                confidence,
                frame,
                70,
                "HIGH",
                "Person detected inside restricted zone"
            )

        return

    track = tracks[track_id]

    previous_zone = track["zone"]
    previous_x = track["previous_x"]
    previous_y = track["previous_y"]
    previous_time = track["previous_time"]

    dt = max(now - previous_time, 0.05)

    distance = (
        (center_x - previous_x) ** 2 +
        (center_y - previous_y) ** 2
    ) ** 0.5

    speed = distance / dt

    abnormal = speed >= ABNORMAL_SPEED

    if abnormal and not track["abnormal"]:
        create_incident(
            "Abnormal Movement",
            track_id,
            confidence,
            frame,
            75,
            "HIGH",
            "Unusually rapid movement detected"
        )

    track["abnormal"] = abnormal

    if previous_zone == "ENTRY" and zone == "MONITORING":
        with state_lock:
            state["entered"] += 1

        create_incident(
            "Entry Detected",
            track_id,
            confidence,
            frame,
            30,
            "MEDIUM",
            "Person moved from entry area into monitoring area"
        )

    if previous_zone == "MONITORING" and zone == "ENTRY":
        with state_lock:
            state["exited"] += 1

        create_incident(
            "Exit Detected",
            track_id,
            confidence,
            frame,
            20,
            "LOW",
            "Person returned to entry area"
        )

    if zone == "RESTRICTED" and previous_zone != "RESTRICTED":
        track["zone_start"] = now
        track["loitering"] = False

        create_incident(
            "Restricted Zone Intrusion",
            track_id,
            confidence,
            frame,
            70,
            "HIGH",
            "Person entered restricted zone"
        )

    if zone != "RESTRICTED":
        track["zone_start"] = None
        track["loitering"] = False

    if zone == "RESTRICTED":
        if track["zone_start"] is None:
            track["zone_start"] = now

        duration = now - track["zone_start"]

        if (
            duration >= LOITERING_SECONDS
            and not track["loitering"]
        ):
            track["loitering"] = True

            create_incident(
                "Loitering",
                track_id,
                confidence,
                frame,
                90,
                "CRITICAL",
                f"Person remained in restricted zone for {duration:.1f} seconds"
            )

    track["previous_zone"] = previous_zone
    track["zone"] = zone
    track["previous_x"] = center_x
    track["previous_y"] = center_y
    track["previous_time"] = now
    track["last_seen"] = now


def cleanup_tracks():
    now = time.time()

    expired = [
        track_id
        for track_id, track in tracks.items()
        if now - track["last_seen"] > TRACK_TIMEOUT
    ]

    for track_id in expired:
        tracks.pop(track_id, None)


def calculate_risk(people_count):
    restricted = any(
        t.get("zone") == "RESTRICTED"
        for t in tracks.values()
    )

    loitering = any(
        t.get("loitering")
        for t in tracks.values()
    )

    abnormal = any(
        t.get("abnormal")
        for t in tracks.values()
    )

    score = 5
    reasons = []

    if restricted:
        score = max(score, 70)
        reasons.append("Restricted zone intrusion")

    if loitering:
        score = max(score, 90)
        reasons.append("Loitering detected")

    if abnormal:
        score = max(score, 75)
        reasons.append("Abnormal movement detected")

    if people_count >= CROWD_THRESHOLD:
        score = max(score, 65)
        reasons.append("Crowd detected")

    if score >= 81:
        level = "CRITICAL"
    elif score >= 61:
        level = "HIGH"
    elif score >= 31:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level, reasons


def draw_detection(
    frame,
    x1,
    y1,
    x2,
    y2,
    label,
    confidence,
    track_id,
    zone,
    loitering,
    abnormal
):
    if label == "person":
        if abnormal:
            color = (0, 0, 255)
        elif loitering:
            color = (0, 140, 255)
        elif zone == "RESTRICTED":
            color = (0, 140, 255)
        elif zone == "ENTRY":
            color = (0, 220, 255)
        else:
            color = (80, 220, 100)
    else:
        color = (100, 180, 255)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    text = f"{label} {confidence:.0%}"

    if track_id >= 0:
        text += f" ID:{track_id}"

    if label == "person":
        text += f" [{zone}]"

    if loitering:
        text += " LOITERING"

    if abnormal:
        text += " ABNORMAL"

    cv2.putText(
        frame,
        text,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1
    )


def draw_overlay(frame):
    h, w = frame.shape[:2]

    entry_x, restricted_x, _, _ = get_regions(frame)

    cv2.line(
        frame,
        (entry_x, 0),
        (entry_x, h),
        (0, 220, 255),
        1
    )

    cv2.line(
        frame,
        (restricted_x, 0),
        (restricted_x, h),
        (0, 140, 255),
        1
    )

    with state_lock:
        risk = state["risk"]
        risk_level = state["riskLevel"]
        people = state["people"]
        objects = state["objects"]
        fps = state["fps"]

    if risk_level == "LOW":
        color = (50, 200, 80)
    elif risk_level == "MEDIUM":
        color = (0, 220, 255)
    elif risk_level == "HIGH":
        color = (0, 140, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (w - 190, 5),
        (w - 5, 35),
        color,
        -1
    )

    cv2.putText(
        frame,
        f"RISK {risk} - {risk_level}",
        (w - 180, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    text = (
        f"People:{people} "
        f"Objects:{objects} "
        f"FPS:{fps:.1f}"
    )

    cv2.putText(
        frame,
        text,
        (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )


def process_frame(frame):
    global latest_jpeg
    global last_frame_time
    global last_inference_time

    with processing_lock:
        now = time.time()

        last_frame_time = now

        if (
            now - last_inference_time
            < INFERENCE_INTERVAL
        ):
            return

        last_inference_time = now

        height, width = frame.shape[:2]

        target_width = 640

        if width > target_width:
            scale = target_width / width
            frame = cv2.resize(
                frame,
                (
                    target_width,
                    int(height * scale)
                ),
                interpolation=cv2.INTER_AREA
            )

        display_frame = frame.copy()

        people = 0
        objects = 0

        start = time.time()

        if model is not None:
            try:
                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE,
                    imgsz=MODEL_SIZE,
                    max_det=MAX_DETECTIONS,
                    device="cpu",
                    verbose=False
                )

                if results:
                    boxes = results[0].boxes

                    if boxes is not None:
                        for i in range(len(boxes)):
                            confidence = float(
                                boxes.conf[i]
                            )

                            cls = int(
                                boxes.cls[i]
                            )

                            x1, y1, x2, y2 = map(
                                int,
                                boxes.xyxy[i].tolist()
                            )

                            objects += 1

                            if isinstance(model.names, dict):
                                label = model.names.get(
                                    cls,
                                    str(cls)
                                )
                            else:
                                label = model.names[cls]

                            track_id = -1
                            zone = "MONITORING"
                            loitering = False
                            abnormal = False

                            if label == "person":
                                people += 1

                                center_x = int(
                                    (x1 + x2) / 2
                                )

                                center_y = int(
                                    (y1 + y2) / 2
                                )

                                track_id = find_person_track(
                                    center_x,
                                    center_y
                                )

                                process_person(
                                    track_id,
                                    confidence,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    display_frame
                                )

                                track = tracks.get(
                                    track_id,
                                    {}
                                )

                                zone = track.get(
                                    "zone",
                                    "MONITORING"
                                )

                                loitering = track.get(
                                    "loitering",
                                    False
                                )

                                abnormal = track.get(
                                    "abnormal",
                                    False
                                )

                            draw_detection(
                                display_frame,
                                x1,
                                y1,
                                x2,
                                y2,
                                label,
                                confidence,
                                track_id,
                                zone,
                                loitering,
                                abnormal
                            )

            except Exception as e:
                print("YOLO error:", e)

        cleanup_tracks()

        score, level, reasons = calculate_risk(
            people
        )

        inference = (
            time.time() - start
        ) * 1000

        previous_fps = state["fps"]

        instant_fps = (
            1000 / inference
            if inference > 0
            else previous_fps
        )

        fps = (
            previous_fps * 0.7
            +
            instant_fps * 0.3
        )

        with state_lock:
            state["people"] = people
            state["objects"] = objects
            state["fps"] = fps
            state["inferenceTime"] = inference
            state["risk"] = score
            state["riskLevel"] = level
            state["riskReasons"] = reasons
            state["camera"] = True
            state["yolo"] = model is not None

        draw_overlay(display_frame)

        ok, encoded = cv2.imencode(
            ".jpg",
            display_frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                60
            ]
        )

        if ok:
            jpeg = encoded.tobytes()

            with frame_lock:
                latest_jpeg = jpeg

            del jpeg

        del display_frame


def generate_video():
    while True:
        with frame_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

        time.sleep(0.05)


@app.route("/")
def home():
    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/frame", methods=["POST"])
def receive_frame():
    try:
        data = request.get_json(
            silent=True
        ) or {}

        image_data = data.get("image")

        if not image_data:
            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400

        if "," in image_data:
            image_data = image_data.split(
                ",",
                1
            )[1]

        raw = base64.b64decode(
            image_data,
            validate=False
        )

        if len(raw) > 1500000:
            return jsonify({
                "success": False,
                "error": "Frame too large"
            }), 413

        array = np.frombuffer(
            raw,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )

        del raw
        del array

        if frame is None:
            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400

        process_frame(frame)

        del frame

        with state_lock:
            result = dict(state)

        return jsonify({
            "success": True,
            "state": result
        })

    except Exception as e:
        print("Frame error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/health")
def health():
    with state_lock:
        yolo_ok = state["yolo"]

    camera_ok = (
        time.time() - last_frame_time
        <= CAMERA_TIMEOUT
    )

    return jsonify({
        "success": True,
        "service": "VisionGuard",
        "camera": camera_ok,
        "yolo": yolo_ok,
        "status": "running"
    })


@app.route("/api/status")
def status():
    camera_ok = (
        time.time() - last_frame_time
        <= CAMERA_TIMEOUT
    )

    with state_lock:
        state["camera"] = camera_ok
        result = dict(state)

    result.update({
        "tracker": "Lightweight",
        "model": "YOLOv8n",
        "cameraName": "Browser Webcam",
        "features": {
            "entryZone": True,
            "restrictedZone": True,
            "loitering": True,
            "abnormalMovement": True,
            "crowdDetection": True,
            "entryExit": True,
            "incidentEvidence": True
        }
    })

    return jsonify(result)


@app.route("/api/incidents")
def get_incidents():
    with state_lock:
        data = list(incidents)

    return jsonify({
        "success": True,
        "count": len(data),
        "incidents": data
    })


@app.route("/api/incidents", methods=["POST"])
def add_incident():
    global incident_id

    data = request.get_json(
        silent=True
    ) or {}

    incident = {
        "id": incident_id,
        "type": data.get(
            "type",
            "Manual Incident"
        ),
        "trackingId": data.get(
            "trackingId",
            -1
        ),
        "confidence": data.get(
            "confidence",
            0
        ),
        "risk": data.get(
            "risk",
            50
        ),
        "severity": data.get(
            "severity",
            "MEDIUM"
        ),
        "reason": data.get(
            "reason",
            "Manual incident"
        ),
        "camera": "Browser Webcam",
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "evidence": None
    }

    with state_lock:
        incidents.insert(
            0,
            incident
        )

        if len(incidents) > 30:
            del incidents[30:]

    incident_id += 1

    return jsonify({
        "success": True,
        "incident": incident
    })


@app.route(
    "/api/incidents/<int:incident_number>",
    methods=["DELETE"]
)
def delete_incident(incident_number):
    global incidents

    with state_lock:
        before = len(incidents)

        incidents = [
            x for x in incidents
            if x["id"] != incident_number
        ]

        deleted = len(incidents) < before

    return jsonify({
        "success": deleted
    })


@app.route("/evidence/<path:filename>")
def evidence(filename):
    return send_from_directory(
        EVIDENCE_DIR,
        filename
    )


@app.route("/api/config")
def config():
    return jsonify({
        "confidence": CONFIDENCE,
        "loiteringSeconds": LOITERING_SECONDS,
        "abnormalSpeed": ABNORMAL_SPEED,
        "crowdThreshold": CROWD_THRESHOLD,
        "entryZoneEnd": ENTRY_END,
        "restrictedZoneStart": RESTRICTED_START,
        "camera": "Browser Webcam",
        "model": "YOLOv8n",
        "inferenceSize": MODEL_SIZE,
        "inferenceInterval": INFERENCE_INTERVAL
    })


@app.route("/api/features")
def features():
    return jsonify({
        "entryZone": {
            "enabled": True,
            "description": "Left-side access and entry area"
        },
        "monitoringZone": {
            "enabled": True,
            "description": "Central normal monitoring area"
        },
        "restrictedZone": {
            "enabled": True,
            "description": "Right-side restricted security area"
        },
        "loitering": {
            "enabled": True,
            "thresholdSeconds": LOITERING_SECONDS
        },
        "abnormalMovement": {
            "enabled": True,
            "speedThreshold": ABNORMAL_SPEED
        },
        "crowdDetection": {
            "enabled": True,
            "threshold": CROWD_THRESHOLD
        },
        "entryExit": {
            "enabled": True,
            "description": "Zone-to-zone movement tracking"
        },
        "incidentEvidence": {
            "enabled": True
        }
    })


@app.route("/api/reset")
def reset_statistics():
    with state_lock:
        state["entered"] = 0
        state["exited"] = 0
        state["risk"] = 5
        state["riskLevel"] = "LOW"
        state["riskReasons"] = []

    tracks.clear()
    cooldowns.clear()

    return jsonify({
        "success": True,
        "message": "Detection statistics reset"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )