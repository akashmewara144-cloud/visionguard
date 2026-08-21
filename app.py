import os
import cv2
import time
import threading
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

CAMERA_INDEX = 0
PORT = 5000
CONFIDENCE = 0.45
LOITERING_SECONDS = 10
ABNORMAL_SPEED = 450
CROWD_THRESHOLD = 4
INCIDENT_COOLDOWN = 8
TRACK_TIMEOUT = 3

ENTRY_END = 0.25
RESTRICTED_START = 0.65

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

model = None
camera = None
camera_running = False
latest_jpeg = None

frame_lock = threading.Lock()
state_lock = threading.Lock()

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

try:
    model = YOLO(MODEL_PATH)
    state["yolo"] = True
    print("YOLOv8 loaded successfully")
except Exception as e:
    print("YOLO loading failed:", e)

def open_camera():
    global camera, camera_running

    if camera is not None:
        try:
            camera.release()
        except:
            pass

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        try:
            camera.release()
        except:
            pass
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        camera = None
        camera_running = False
        state["camera"] = False
        return False

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    try:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass

    for _ in range(15):
        ok, frame = camera.read()
        if ok and frame is not None and frame.size:
            camera_running = True
            state["camera"] = True
            print("Laptop webcam connected:", frame.shape)
            return True
        time.sleep(0.1)

    try:
        camera.release()
    except:
        pass

    camera = None
    camera_running = False
    state["camera"] = False
    return False

def get_regions(frame):
    h, w = frame.shape[:2]
    entry_x = int(w * ENTRY_END)
    restricted_x = int(w * RESTRICTED_START)
    return entry_x, restricted_x, w, h

def get_person_zone(x1, y1, x2, y2, frame):
    _, _, w, _ = get_regions(frame)
    center_x = (x1 + x2) / 2
    ratio = center_x / w

    if ratio < ENTRY_END:
        return "ENTRY"
    if ratio >= RESTRICTED_START:
        return "RESTRICTED"
    return "MONITORING"

def create_incident(event_type, track_id, confidence, frame, score, severity, reason):
    global incident_id

    key = f"{event_type}_{track_id}"
    now = time.time()

    if now - cooldowns.get(key, 0) < INCIDENT_COOLDOWN:
        return False

    cooldowns[key] = now

    stamp = datetime.now()
    safe_type = "".join(c if c.isalnum() else "_" for c in event_type)
    filename = f"incident_{stamp.strftime('%Y%m%d_%H%M%S_%f')}_{safe_type}_{track_id}.jpg"
    path = os.path.join(EVIDENCE_DIR, filename)

    evidence = None

    try:
        if cv2.imwrite(path, frame):
            evidence = f"/evidence/{filename}"
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
        "camera": "Laptop Webcam",
        "timestamp": stamp.strftime("%Y-%m-%d %H:%M:%S"),
        "evidence": evidence
    }

    with state_lock:
        incidents.insert(0, incident)
        if len(incidents) > 100:
            incidents.pop()

    print(f"INCIDENT | {event_type} | ID {track_id} | {severity}")
    incident_id += 1
    return True

def process_person(track_id, confidence, x1, y1, x2, y2, frame):
    now = time.time()
    zone = get_person_zone(x1, y1, x2, y2, frame)

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

    dt = max(now - previous_time, 0.01)
    distance = ((center_x - previous_x) ** 2 + (center_y - previous_y) ** 2) ** 0.5
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
            "Person returned from monitoring area to entry area"
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

        if duration >= LOITERING_SECONDS and not track["loitering"]:
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
        for track_id, track in list(tracks.items())
        if now - track["last_seen"] > TRACK_TIMEOUT
    ]

    for track_id in expired:
        tracks.pop(track_id, None)

def calculate_risk(people_count):
    restricted = any(t.get("zone") == "RESTRICTED" for t in tracks.values())
    loitering = any(t.get("loitering") for t in tracks.values())
    abnormal = any(t.get("abnormal") for t in tracks.values())

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

def draw_overlay(frame):
    h, w = frame.shape[:2]
    entry_x, restricted_x, _, _ = get_regions(frame)

    cv2.line(frame, (entry_x, 0), (entry_x, h), (0, 220, 255), 2)
    cv2.line(frame, (restricted_x, 0), (restricted_x, h), (0, 140, 255), 2)

    cv2.rectangle(frame, (restricted_x, 0), (w, h), (0, 140, 255), 2)

    cv2.putText(frame, "ENTRY / ACCESS", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(frame, "MONITORING AREA", (entry_x + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, "RESTRICTED ZONE", (restricted_x + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)

    if state["riskLevel"] == "LOW":
        color = (50, 200, 80)
    elif state["riskLevel"] == "MEDIUM":
        color = (0, 220, 255)
    elif state["riskLevel"] == "HIGH":
        color = (0, 140, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame, (w - 270, 10), (w - 10, 50), color, -1)
    cv2.putText(frame, f"RISK {state['risk']} - {state['riskLevel']}", (w - 255, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    status_text = (
        f"People: {state['people']} | "
        f"Objects: {state['objects']} | "
        f"Entered: {state['entered']} | "
        f"Exited: {state['exited']} | "
        f"FPS: {state['fps']:.1f}"
    )

    cv2.rectangle(frame, (10, h - 48), (w - 10, h - 10), (15, 23, 42), -1)
    cv2.putText(frame, status_text, (20, h - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_detection(frame, x1, y1, x2, y2, label, confidence, track_id, zone, loitering, abnormal):
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

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {confidence:.0%}"

    if track_id >= 0:
        text += f" ID:{track_id}"

    if label == "person":
        text += f" [{zone}]"

    if loitering:
        text += " LOITERING"

    if abnormal:
        text += " ABNORMAL"

    y_text = max(20, y1 - 8)

    cv2.putText(
        frame,
        text,
        (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2
    )

def detection_loop():
    global camera, camera_running, latest_jpeg

    if not open_camera():
        return

    frame_counter = 0
    fps_start = time.time()

    while True:
        if camera is None or not camera.isOpened():
            camera_running = False
            state["camera"] = False

            if not open_camera():
                time.sleep(1)
                continue

        ok, frame = camera.read()

        if not ok or frame is None or not frame.size:
            camera_running = False
            state["camera"] = False

            try:
                camera.release()
            except:
                pass

            camera = None
            time.sleep(0.2)
            continue

        camera_running = True
        state["camera"] = True

        start = time.time()
        people = 0
        objects = 0

        if model is not None:
            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=CONFIDENCE,
                    verbose=False,
                    imgsz=640,
                    device="cpu"
                )

                if results:
                    boxes = results[0].boxes

                    if boxes is not None:
                        for i in range(len(boxes)):
                            confidence = float(boxes.conf[i].item())

                            if confidence < CONFIDENCE:
                                continue

                            cls = int(boxes.cls[i].item())
                            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())

                            objects += 1

                            track_id = -1

                            if boxes.id is not None:
                                try:
                                    track_id = int(boxes.id[i].item())
                                except:
                                    track_id = -1

                            if isinstance(model.names, dict):
                                label = model.names.get(cls, str(cls))
                            else:
                                label = model.names[cls]

                            zone = "MONITORING"
                            loitering = False
                            abnormal = False

                            if label == "person":
                                people += 1

                                if track_id >= 0:
                                    process_person(
                                        track_id,
                                        confidence,
                                        x1,
                                        y1,
                                        x2,
                                        y2,
                                        frame
                                    )

                                    track = tracks.get(track_id, {})
                                    zone = track.get("zone", "MONITORING")
                                    loitering = track.get("loitering", False)
                                    abnormal = track.get("abnormal", False)

                            draw_detection(
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
                            )

            except Exception as e:
                print("YOLO error:", e)

        cleanup_tracks()

        score, level, reasons = calculate_risk(people)

        elapsed = time.time() - fps_start
        frame_counter += 1

        current_fps = state["fps"]

        if elapsed >= 1:
            current_fps = frame_counter / elapsed
            frame_counter = 0
            fps_start = time.time()

        inference = (time.time() - start) * 1000

        with state_lock:
            state["people"] = people
            state["objects"] = objects
            state["fps"] = current_fps
            state["inferenceTime"] = inference
            state["risk"] = score
            state["riskLevel"] = level
            state["riskReasons"] = reasons
            state["camera"] = camera_running
            state["yolo"] = model is not None

        draw_overlay(frame)

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if ok:
            with frame_lock:
                latest_jpeg = encoded.tobytes()

def generate_video():
    while True:
        with frame_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/health")
def health():
    with state_lock:
        camera_ok = state["camera"]
        yolo_ok = state["yolo"]

    return jsonify({
        "success": True,
        "service": "VisionGuard",
        "camera": camera_ok,
        "yolo": yolo_ok,
        "status": "running"
    })

@app.route("/api/status")
def status():
    with state_lock:
        result = dict(state)

    result.update({
        "tracker": "ByteTrack",
        "model": "YOLOv8n",
        "cameraName": "Laptop Webcam",
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

    data = request.get_json(silent=True) or {}

    incident = {
        "id": incident_id,
        "type": data.get("type", "Manual Incident"),
        "trackingId": data.get("trackingId", -1),
        "confidence": data.get("confidence", 0),
        "risk": data.get("risk", 50),
        "severity": data.get("severity", "MEDIUM"),
        "reason": data.get("reason", "Manual incident"),
        "camera": "Laptop Webcam",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evidence": None
    }

    with state_lock:
        incidents.insert(0, incident)

    incident_id += 1

    return jsonify({
        "success": True,
        "incident": incident
    })

@app.route("/api/incidents/<int:incident_number>", methods=["DELETE"])
def delete_incident(incident_number):
    global incidents

    with state_lock:
        before = len(incidents)
        incidents = [
            x for x in incidents
            if x["id"] != incident_number
        ]

    return jsonify({
        "success": len(incidents) < before
    })

@app.route("/evidence/<path:filename>")
def evidence(filename):
    return send_from_directory(EVIDENCE_DIR, filename)

@app.route("/api/config")
def config():
    return jsonify({
        "confidence": CONFIDENCE,
        "loiteringSeconds": LOITERING_SECONDS,
        "abnormalSpeed": ABNORMAL_SPEED,
        "crowdThreshold": CROWD_THRESHOLD,
        "entryZoneEnd": ENTRY_END,
        "restrictedZoneStart": RESTRICTED_START,
        "camera": "Laptop Webcam",
        "model": "YOLOv8n"
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

def start_detection():
    thread = threading.Thread(
        target=detection_loop,
        daemon=True
    )
    thread.start()

if __name__ == "__main__":
    print("VisionGuard AI Surveillance")
    print("YOLOv8:", "READY" if model else "UNAVAILABLE")
    print("Camera: Laptop Webcam")
    print("Entry Zone: LEFT")
    print("Monitoring Zone: CENTER")
    print("Restricted Zone: RIGHT")
    print("Loitering: ENABLED")
    print("Abnormal Movement: ENABLED")
    print("Crowd Detection: ENABLED")
    print("Dashboard: http://localhost:5000")

    start_detection()

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )