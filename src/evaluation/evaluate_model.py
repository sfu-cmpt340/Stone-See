import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

# Add project root to Python path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.load_data import load_kidney_data


def load_trained_model(model_path, device):
    """
    Load a trained model from a saved checkpoint.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load model on (CPU/GPU/MPS)
    
    Returns:
        model: Loaded model ready for evaluation
    """
    # Create the same model architecture as training
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Stone / Non-Stone
    model.to(device)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    return model


def evaluate_model(model, test_loader, device, class_names=['Non-Stone', 'Stone']):
    """
    Evaluate model on test set and calculate comprehensive metrics.
    
    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        class_names: Names of the classes
    
    Returns:
        metrics: Dictionary containing all evaluation metrics
        all_preds: All predictions made by the model
        all_labels: All true labels
        all_probs: All prediction probabilities
        fpr: False positive rates for ROC curve
        tpr: True positive rates for ROC curve
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []      # Store all predictions
    all_labels = []     # Store all true labels
    all_probs = []      # Store all prediction probabilities
    
    print("\nEvaluating model on test set...")
    
    # Process test data without calculating gradients (faster)
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            
            # Get model predictions
            outputs = model(images)
            
            # Convert outputs to probabilities (0 to 1)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predicted class (highest probability)
            _, preds = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays for easier calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ========== CALCULATE METRICS ==========
    
    # Overall accuracy - percentage of correct predictions
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Precision, Recall, F1-score (weighted average across classes)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Per-class metrics (for each class separately)
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Confusion matrix - shows correct vs incorrect predictions
    cm = confusion_matrix(all_labels, all_preds)
    
    # ROC curve and AUC (Area Under Curve) - measures model's ability to distinguish classes
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])  # Use probability of class 1 (Stone)
    roc_auc = auc(fpr, tpr)
    
    # Detailed classification report
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Store all metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc,
        'classification_report': report,
        'class_names': class_names
    }
    
    return metrics, all_preds, all_labels, all_probs, fpr, tpr


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        cm: Confusion matrix array
        class_names: Names of classes
        save_path: Where to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """
    Create and save a ROC curve visualization.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under ROC curve
        save_path: Where to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def main():
    """Main function to run model evaluation."""
    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load test data
    print("\nLoading test dataset...")
    # Try Augmented_Dataset first, fall back to data if not found
    data_path = "data/Augmented_Dataset" if os.path.exists("data/Augmented_Dataset") else "data"
    _, _, test_loader = load_kidney_data(data_path)
    
    # Load trained model
    model_path = "results/models/resnet18_kidney_best.pth"
    if not os.path.exists(model_path):
        # Try final model if best doesn't exist
        model_path = "results/models/resnet18_kidney.pth"
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first using: python src/models/train_model.py")
            return
    
    print(f"\nLoading model from: {model_path}")
    model = load_trained_model(model_path, device)
    
    # Evaluate model
    class_names = ['Non-Stone', 'Stone']
    metrics, all_preds, all_labels, all_probs, fpr, tpr = evaluate_model(
        model, test_loader, device, class_names
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision_per_class'][i]*100:.2f}%")
        print(f"    Recall:    {metrics['recall_per_class'][i]*100:.2f}%")
        print(f"    F1-Score:  {metrics['f1_per_class'][i]*100:.2f}%")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              {class_names[0]:<12} {class_names[1]:<12}")
    for i, class_name in enumerate(class_names):
        print(f"True {class_name:<8} {metrics['confusion_matrix'][i][0]:<12} {metrics['confusion_matrix'][i][1]:<12}")
    
    # Save visualizations
    os.makedirs("results/figures", exist_ok=True)
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, 
                         "results/figures/confusion_matrix.png")
    plot_roc_curve(fpr, tpr, metrics['roc_auc'], 
                   "results/figures/roc_curve.png")
    
    # Save metrics to JSON
    import json
    metrics_path = "results/evaluation_metrics.json"
    os.makedirs("results", exist_ok=True)
    
    # Convert numpy types to Python types for JSON
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'roc_auc': float(metrics['roc_auc']),
        'precision_per_class': [float(x) for x in metrics['precision_per_class']],
        'recall_per_class': [float(x) for x in metrics['recall_per_class']],
        'f1_per_class': [float(x) for x in metrics['f1_per_class']],
        'confusion_matrix': metrics['confusion_matrix'],
        'class_names': class_names
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    print("\nNext step: Generate report with: python src/evaluation/generate_report.py")


if __name__ == "__main__":
    main()

