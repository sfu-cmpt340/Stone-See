#!/usr/bin/env python3
"""
Evaluate model specifically on original data to check performance.
This helps identify if model works well on original vs augmented data.
"""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.load_data import load_kidney_data

def load_trained_model(model_path, device):
    """Load a trained model."""
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

def main():
    """Evaluate model on original data."""
    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load original data
    print("\nLoading ORIGINAL dataset...")
    if not os.path.exists("data/Original_Dataset"):
        print("ERROR: Original_Dataset not found!")
        return
    
    _, _, test_loader = load_kidney_data("data/Original_Dataset")
    
    # Load trained model
    model_path = "results/models/resnet18_kidney_best.pth"
    if not os.path.exists(model_path):
        model_path = "results/models/resnet18_kidney.pth"
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
    
    print(f"\nLoading model from: {model_path}")
    model = load_trained_model(model_path, device)
    
    # Evaluate
    print("\nEvaluating on ORIGINAL data...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    
    print("\n" + "="*60)
    print("EVALUATION ON ORIGINAL DATA")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    
    print(f"\nPer-Class Metrics:")
    print(f"  Non-Stone:")
    print(f"    Precision: {precision_per_class[0]*100:.2f}%")
    print(f"    Recall:    {recall_per_class[0]*100:.2f}%")
    print(f"  Stone:")
    print(f"    Precision: {precision_per_class[1]*100:.2f}%")
    print(f"    Recall:    {recall_per_class[1]*100:.2f}%")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Non-Stone    Stone")
    print(f"True Non-Stone {cm[0][0]:<12} {cm[0][1]:<12}")
    print(f"True Stone     {cm[1][0]:<12} {cm[1][1]:<12}")
    
    # Check if Stone detection is working
    stone_recall = recall_per_class[1]
    if stone_recall < 0.5:
        print("\n⚠️  WARNING: Stone recall is low! Model is missing many stones.")
        print("   Consider retraining with higher class weights or lower threshold.")
    else:
        print(f"\n✓ Stone detection working (Recall: {stone_recall*100:.2f}%)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

