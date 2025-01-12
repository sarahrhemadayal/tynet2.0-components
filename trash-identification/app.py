from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms, models
import io

# Initialize Flask app
app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 3),  # Biodegradable, Recyclable, Hazardous
    torch.nn.Softmax(dim=1)
)
model.load_state_dict(torch.load("ai/trash_classifier_best.pth", map_location=device))
model = model.to(device)
model.eval()

# Labels
labels = {0: "Biodegradable", 1: "Recyclable", 2: "Hazardous"}

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image):
    """Classify an image."""
    input_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted_class = torch.max(outputs, 1)
        return labels[predicted_class.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        prediction = classify_image(image)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
