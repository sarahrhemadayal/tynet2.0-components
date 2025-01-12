from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import json
import asyncio
import uvicorn
from typing import Dict, Any
import base64

app = FastAPI(title="Live YOLO Detection API")

# Load YOLO model and configurations globally
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load COCO classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define object categories
biodegradable_objects = ["apple", "banana", "orange", "broccoli", "carrot"]
recyclable_objects = ["bottle", "can", "plastic", "paper", "cardboard"]
hazardous_objects = ["battery", "lighter", "fire_extinguisher", "toxic_chemical"]

class ObjectDetector:
    def __init__(self):
        self.object_tracker: Dict[str, Any] = {}
        self.cap = None

    async def process_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame")

        height, width, _ = frame.shape

        # Convert frame to blob for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        detections = yolo_net.forward(output_layers)

        # Process detections
        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    x = center_x - w // 2
                    y = center_y - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Clear previous tracker data
        self.object_tracker.clear()

        # Process detected objects
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                # Classify object
                if label in biodegradable_objects:
                    category = "Biodegradable"
                    color = (0, 255, 0)
                elif label in recyclable_objects:
                    category = "Recyclable"
                    color = (255, 0, 0)
                elif label in hazardous_objects:
                    category = "Hazardous"
                    color = (0, 0, 255)
                else:
                    category = "Other"
                    color = (0, 255, 255)

                # Track object if not "Other"
                if category != "Other":
                    object_key = f"{label}_{x}_{y}_{w}_{h}"
                    self.object_tracker[object_key] = {
                        "label": label,
                        "category": category,
                        "confidence": float(confidence),
                        "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                    }

                # Draw bounding box and label on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} ({category}) {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert frame to base64 for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare detection summary
        count_summary = {}
        for detection in self.object_tracker.values():
            key = (detection["label"], detection["category"])
            count_summary[key] = count_summary.get(key, 0) + 1

        return {
            "frame": frame_base64,
            "detections": list(self.object_tracker.values()),
            "summary": [
                {
                    "item": item,
                    "category": category,
                    "count": count
                }
                for (item, category), count in count_summary.items()
            ]
        }

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    detector = ObjectDetector()
    
    try:
        while True:
            try:
                # Process frame and send results
                results = await detector.process_frame()
                await websocket.send_json(results)
                
                # Small delay to control frame rate
                await asyncio.sleep(0.03)  # Approximately 30 FPS
                
            except RuntimeError as e:
                await websocket.send_json({"error": str(e)})
                break
                
    except Exception as e:
        await websocket.send_json({"error": f"Unexpected error: {str(e)}"})
    
    finally:
        detector.cleanup()

@app.get("/")
async def root():
    return {"message": "Live YOLO Detection API. Connect to /ws/detect for live detection stream."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)