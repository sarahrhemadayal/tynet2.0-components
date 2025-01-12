import cv2
import numpy as np
import json

# Load YOLO model
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define object categories
biodegradable_objects = ["apple", "banana", "orange", "broccoli", "carrot"]
recyclable_objects = ["bottle", "can", "plastic", "paper", "book", "cardboard"]
hazardous_objects = ["battery", "lighter", "fire_extinguisher", "toxic_chemical"]

# Dictionary to track object detections
object_tracker = {}

# Set up camera feed
cap = cv2.VideoCapture(0)  # Use 0 for webcam or path to video file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, channels = frame.shape

    # Convert frame to blob for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward(output_layers)

    # Process the detections
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Rectangle coordinates
                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw the bounding boxes and classify objects
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Get the label
            confidence = confidences[i]

            # Classify object based on predefined categories
            if label in biodegradable_objects:
                category = "Biodegradable"
                color = (0, 255, 0)  # Green for biodegradable
            elif label in recyclable_objects:
                category = "Recyclable"
                color = (255, 0, 0)  # Blue for recyclable
            elif label in hazardous_objects:
                category = "Hazardous"
                color = (0, 0, 255)  # Red for hazardous
            else:
                category = "Other"
                color = (0, 255, 255)  # Yellow for irrelevant objects

            # Track object
            object_key = f"{label}_{x}_{y}_{w}_{h}"
            if object_key not in object_tracker:
                object_tracker[object_key] = {"count": 1, "label": label, "category": category}
            else:
                object_tracker[object_key]["count"] += 1

            # Output JSON for objects detected for more than 5 frames
            if object_tracker[object_key]["count"] > 5:
                output_data = {
                    "item": label,
                    "category": category
                }
                print(json.dumps(output_data, indent=4))  # Print the JSON

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({category}) {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with detections
    cv2.imshow("YOLOv3 Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
