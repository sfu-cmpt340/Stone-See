from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# Global variables for model
model = None
device = None

def load_model():
    """Load the trained model."""
    global model, device
    
    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load model
    model_path = os.path.join(project_root, "results/models/resnet18_kidney_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, "results/models/resnet18_kidney.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found. Please train the model first.")
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded successfully on {device}")

def preprocess_image(image_bytes):
    """Preprocess image for prediction."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get evaluation metrics."""
    import json
    metrics_path = os.path.join(project_root, "results/evaluation_metrics.json")
    
    if not os.path.exists(metrics_path):
        return jsonify({"error": "Metrics not found. Please run evaluation first."}), 404
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": f"Failed to load metrics: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if image contains kidney stones."""
    global model, device
    
    if model is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({"error": f"Model loading failed: {str(e)}"}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    try:
        # Get image file
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        # Get probabilities for both classes
        non_stone_prob = probs[0][0].item() * 100
        stone_prob = probs[0][1].item() * 100
        
        # Adjust threshold to be very sensitive to detecting stones
        # Lower threshold means more likely to detect stones (reduces false negatives)
        # Set to 10% - if stone probability >= 10%, predict Stone (extremely sensitive)
        # This helps detect stones in original data that model might miss
        stone_threshold = 10.0  # If stone probability >= 10%, predict Stone
        
        # Make prediction based on threshold
        if stone_prob >= stone_threshold:
            prediction = 'Stone'
            confidence = stone_prob
        else:
            prediction = 'Non-Stone'
            confidence = non_stone_prob
        
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "Non-Stone": round(non_stone_prob, 2),
                "Stone": round(stone_prob, 2)
            }
        })
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Loading model...")
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Model will be loaded on first prediction request.")
    
    print("Server ready!")
    print("API running at http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    app.run(host='0.0.0.0', port=5000, debug=True)

