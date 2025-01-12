import torch
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 3),  # Biodegradable or Non-biodegradable
    torch.nn.Softmax(dim=1)
)
model.load_state_dict(torch.load("ai/trash_classifier_best.pth", map_location=device))
model = model.to(device)
model.eval()

# Labels for the classes
labels = {0: "Biodegradable", 1: "Recyclable", 2: "Hazardous"}


# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image):
    """Classify an image (from file or camera)."""
    input_image = transform(image).unsqueeze(0).to(device)

    # Predict using the model
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted_class = torch.max(outputs, 1)
        return labels[predicted_class.item()]

def live_capture():
    """Use the live camera to capture and classify images."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'q' to quit or 'c' to capture and classify an image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break

        # Display the live feed
        cv2.imshow("Live Capture - Press 'c' to classify, 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Preprocess and classify the captured frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = Image.fromarray(image)  # Convert to PIL format
            prediction = classify_image(image)

            # Display the result
            print(f"Prediction: {prediction}")
            cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Result", frame)

    cap.release()
    cv2.destroyAllWindows()

def file_input():
    """Classify an image file."""
    file_path = input("Enter the file path of the image: ").strip()
    try:
        image = Image.open(file_path).convert("RGB")  # Load and ensure RGB format
        prediction = classify_image(image)
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Error: Could not process the file. {e}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Live capture (from camera)")
    print("2. File input (provide image path)")
    choice = input("Enter your choice (1/2): ").strip()

    if choice == "1":
        live_capture()
    elif choice == "2":
        file_input()
    else:
        print("Invalid choice. Exiting.")
