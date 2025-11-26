import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

# Add project root to Python path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_model(model_path, device):
    """Load trained model."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocess image for prediction."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict_image(image_path, model_path="results/models/resnet18_kidney_best.pth"):
    """
    Predict if an image contains kidney stones.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
    
    Returns:
        prediction: 'Stone' or 'Non-Stone'
        confidence: Confidence score (0-100%)
    """
    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load model
    if not os.path.exists(model_path):
        # Try final model if best doesn't exist
        model_path = "results/models/resnet18_kidney.pth"
        if not os.path.exists(model_path):
            print(f"Error: Model not found. Please train the model first.")
            return None, None
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, device)
    
    # Preprocess image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None
    
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
    
    # Get results
    class_names = ['Non-Stone', 'Stone']
    prediction = class_names[pred.item()]
    confidence = probs[0][pred.item()].item() * 100
    
    return prediction, confidence


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python src/evaluation/predict_image.py <image_path>")
        print("Example: python src/evaluation/predict_image.py test_image.jpg")
        return
    
    image_path = sys.argv[1]
    prediction, confidence = predict_image(image_path)
    
    if prediction is not None:
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Image: {image_path}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}%")
        print("="*60)
    else:
        print("Prediction failed. Please check the image path and model file.")


if __name__ == "__main__":
    main()

