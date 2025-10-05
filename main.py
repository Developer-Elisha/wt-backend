# backend/main.py
import cv2
import torch
import base64
import asyncio
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ---------------- Tracker ----------------
class TinyTracker:
    def __init__(self, iou_th=0.3):
        self.iou_th = iou_th
        self.next_id = 1
        self.tracks = {}

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        aa = (ax2 - ax1) * (ay2 - ay1)
        bb = (bx2 - bx1) * (by2 - by1)
        return inter / (aa + bb - inter + 1e-6)

    def update(self, boxes):
        results = []
        new_tracks = {}
        for b in boxes:
            assigned = False
            for tid, info in self.tracks.items():
                if self._iou(info['box'], b) > self.iou_th:
                    new_tracks[tid] = {'box': b}
                    results.append((tid, b))
                    assigned = True
                    break
            if not assigned:
                tid = self.next_id
                self.next_id += 1
                new_tracks[tid] = {'box': b}
                results.append((tid, b))
        self.tracks = new_tracks
        return results

# ---------------- FastAPI App ----------------
app = FastAPI(title="Watch Tower - Accident Detection")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRAME_SKIP = 2
MIN_OVERLAP_AREA = 500

# Load YOLO model
model = YOLO("yolov8n.pt")
if torch.cuda.is_available():
    model.to("cuda")

# Classes to detect: person, car, bike, bus, truck, hands
TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # modify based on YOLO class mapping

# ---------------- Video Upload WebSocket ----------------
@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_bytes()
        tfile_path = "uploaded_video.mp4"
        with open(tfile_path, "wb") as f:
            f.write(data)

        cap = cv2.VideoCapture(tfile_path)
        tracker = TinyTracker()
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        accidents = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SKIP == 0:
                results = model(frame, conf=0.25, verbose=False)[0]
                boxes = []
                for b, c in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                    if int(c) in TARGET_CLASSES:
                        boxes.append(tuple(map(int, b)))
                tracks = tracker.update(boxes)
            else:
                tracks = tracker.update([])

            accident_ids = set()
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    id1, b1 = tracks[i]
                    id2, b2 = tracks[j]
                    x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                    y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                    if x_overlap * y_overlap > MIN_OVERLAP_AREA:
                        accident_ids.update([id1, id2])
                        if len(accidents) == 0 or frame_idx / fps - accidents[-1] > 1:
                            accidents.append(frame_idx / fps)

            # draw boxes
            for tid, box in tracks:
                x1, y1, x2, y2 = box
                color = (0, 0, 255) if tid in accident_ids else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _, buffer = cv2.imencode(".jpg", frame)
            b64_frame = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_json({"frame": b64_frame, "accidents": accidents})

            frame_idx += 1
            await asyncio.sleep(0.01)

        cap.release()
        await websocket.close()

    except WebSocketDisconnect:
        cap.release()
        print("Video WebSocket disconnected")

# ---------------- Live Camera WebSocket ----------------
@app.websocket("/ws/live")
async def live_camera_ws(websocket: WebSocket):
    await websocket.accept()
    tracker = TinyTracker()
    accidents = []
    frame_idx = 0
    fps = 30

    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            if frame_idx % FRAME_SKIP == 0:
                results = model(frame, conf=0.25, verbose=False)[0]
                boxes = []
                for b, c in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                    if int(c) in TARGET_CLASSES:
                        boxes.append(tuple(map(int, b)))
                tracks = tracker.update(boxes)
            else:
                tracks = tracker.update([])

            accident_ids = set()
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    id1, b1 = tracks[i]
                    id2, b2 = tracks[j]
                    x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                    y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                    if x_overlap * y_overlap > MIN_OVERLAP_AREA:
                        accident_ids.update([id1, id2])
                        if len(accidents) == 0 or frame_idx / fps - accidents[-1] > 1:
                            accidents.append(frame_idx / fps)

            # draw boxes
            for tid, box in tracks:
                x1, y1, x2, y2 = box
                color = (0, 0, 255) if tid in accident_ids else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _, buffer = cv2.imencode(".jpg", frame)
            b64_frame = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_json({"frame": b64_frame, "accidents": accidents})

            frame_idx += 1
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("Live camera disconnected")
