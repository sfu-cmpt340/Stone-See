#!/usr/bin/env python3
"""
Compare model performance on augmented vs original datasets.
Generates comprehensive graphs and metrics for both datasets.
"""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def evaluate_on_dataset(model, data_loader, device, dataset_name):
    """Evaluate model on a dataset and return all metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"\nEvaluating on {dataset_name}...")
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluating {dataset_name}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
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
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }
    
    return metrics, all_preds, all_labels, all_probs


def plot_phase_comparison(all_results, save_dir):
    """Create graphs comparing all phases (Training, Validation, Test)."""
    os.makedirs(save_dir, exist_ok=True)
    
    phases = ['Training', 'Validation', 'Test']
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    # Prepare data
    aug_data = {metric: [] for metric in metrics_names}
    orig_data = {metric: [] for metric in metrics_names}
    
    for phase in phases:
        phase_lower = phase.lower()
        aug_key = f'augmented_{phase_lower}'
        orig_key = f'original_{phase_lower}'
        
        if aug_key in all_results and orig_key in all_results:
            aug_metrics = all_results[aug_key]
            orig_metrics = all_results[orig_key]
            
            aug_data['Accuracy'].append(aug_metrics['accuracy'] * 100)
            aug_data['Precision'].append(aug_metrics['precision'] * 100)
            aug_data['Recall'].append(aug_metrics['recall'] * 100)
            aug_data['F1-Score'].append(aug_metrics['f1_score'] * 100)
            aug_data['ROC-AUC'].append(aug_metrics['roc_auc'] * 100)
            
            orig_data['Accuracy'].append(orig_metrics['accuracy'] * 100)
            orig_data['Precision'].append(orig_metrics['precision'] * 100)
            orig_data['Recall'].append(orig_metrics['recall'] * 100)
            orig_data['F1-Score'].append(orig_metrics['f1_score'] * 100)
            orig_data['ROC-AUC'].append(orig_metrics['roc_auc'] * 100)
        else:
            # Fill with NaN if data not available
            for metric in metrics_names:
                aug_data[metric].append(np.nan)
                orig_data[metric].append(np.nan)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        x = np.arange(len(phases))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, aug_data[metric], width, label='Augmented', 
                      color='#667eea', alpha=0.8)
        bars2 = ax.bar(x + width/2, orig_data[metric], width, label='Original', 
                      color='#764ba2', alpha=0.8)
        
        ax.set_ylabel('Performance (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Across Phases', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Performance Comparison Across All Phases', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase_comparison_all_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/phase_comparison_all_metrics.png")


def plot_metrics_comparison(augmented_metrics, original_metrics, save_dir, phase=''):
    """Create comparison graphs for all metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    phase_suffix = f'_{phase.lower()}' if phase else ''
    
    # Prepare data
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    augmented_values = [
        augmented_metrics['accuracy'] * 100,
        augmented_metrics['precision'] * 100,
        augmented_metrics['recall'] * 100,
        augmented_metrics['f1_score'] * 100,
        augmented_metrics['roc_auc'] * 100
    ]
    original_values = [
        original_metrics['accuracy'] * 100,
        original_metrics['precision'] * 100,
        original_metrics['recall'] * 100,
        original_metrics['f1_score'] * 100,
        original_metrics['roc_auc'] * 100
    ]
    
    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, augmented_values, width, label='Augmented Dataset', 
                   color='#667eea', alpha=0.8)
    bars2 = ax.bar(x + width/2, original_values, width, label='Original Dataset', 
                   color='#764ba2', alpha=0.8)
    
    ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Augmented vs Original Dataset', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = f'metrics_comparison{phase_suffix}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/{filename}")


def plot_per_class_metrics(augmented_metrics, original_metrics, save_dir, phase=''):
    """Plot per-class metrics comparison."""
    os.makedirs(save_dir, exist_ok=True)
    
    phase_suffix = f'_{phase.lower()}' if phase else ''
    
    classes = ['Non-Stone', 'Stone']
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        aug_values = []
        orig_values = []
        
        if metric == 'Precision':
            aug_values = [x * 100 for x in augmented_metrics['precision_per_class']]
            orig_values = [x * 100 for x in original_metrics['precision_per_class']]
        elif metric == 'Recall':
            aug_values = [x * 100 for x in augmented_metrics['recall_per_class']]
            orig_values = [x * 100 for x in original_metrics['recall_per_class']]
        else:  # F1-Score
            aug_values = [x * 100 for x in augmented_metrics['f1_per_class']]
            orig_values = [x * 100 for x in original_metrics['f1_per_class']]
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, aug_values, width, label='Augmented', color='#667eea', alpha=0.8)
        bars2 = ax.bar(x + width/2, orig_values, width, label='Original', color='#764ba2', alpha=0.8)
        
        ax.set_ylabel('Performance (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} per Class', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontsize=9)
    
    title = f'Per-Class Metrics Comparison: Augmented vs Original{(" - " + phase) if phase else ""}'
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filename = f'per_class_metrics_comparison{phase_suffix}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/{filename}")


def plot_confusion_matrices(augmented_metrics, original_metrics, save_dir, phase=''):
    """Plot confusion matrices side by side."""
    os.makedirs(save_dir, exist_ok=True)
    
    phase_suffix = f'_{phase.lower()}' if phase else ''
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    class_names = ['Non-Stone', 'Stone']
    
    datasets = [
        ('Augmented Dataset', augmented_metrics['confusion_matrix'], axes[0]),
        ('Original Dataset', original_metrics['confusion_matrix'], axes[1])
    ]
    
    for title, cm, ax in datasets:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
    
    title = f'Confusion Matrices Comparison{(" - " + phase) if phase else ""}'
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filename = f'confusion_matrices_comparison{phase_suffix}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/{filename}")


def plot_roc_curves(augmented_metrics, original_metrics, save_dir, phase=''):
    """Plot ROC curves comparison."""
    os.makedirs(save_dir, exist_ok=True)
    
    phase_suffix = f'_{phase.lower()}' if phase else ''
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Augmented ROC
    fpr_aug = np.array(augmented_metrics['fpr'])
    tpr_aug = np.array(augmented_metrics['tpr'])
    auc_aug = augmented_metrics['roc_auc']
    ax.plot(fpr_aug, tpr_aug, color='#667eea', lw=2.5,
           label=f'Augmented Dataset (AUC = {auc_aug:.3f})')
    
    # Original ROC
    fpr_orig = np.array(original_metrics['fpr'])
    tpr_orig = np.array(original_metrics['tpr'])
    auc_orig = original_metrics['roc_auc']
    ax.plot(fpr_orig, tpr_orig, color='#764ba2', lw=2.5,
           label=f'Original Dataset (AUC = {auc_orig:.3f})')
    
    # Random classifier line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    title = f'ROC Curves Comparison: Augmented vs Original Dataset{(" - " + phase) if phase else ""}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'roc_curves_comparison{phase_suffix}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/{filename}")


def plot_training_history(save_dir):
    """Plot training history from saved JSON."""
    history_path = os.path.join(project_root, "results/models/resnet18_kidney_history.json")
    
    if not os.path.exists(history_path):
        print(f"Warning: Training history not found at {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history.get('epochs', list(range(1, len(history.get('train_loss', [])) + 1)))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'o-', label='Training Loss', color='#667eea', linewidth=2)
    ax.plot(epochs, history['val_loss'], 's-', label='Validation Loss', color='#764ba2', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax = axes[0, 1]
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], 's-', label='Validation Accuracy', 
               color='#764ba2', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax = axes[1, 0]
    if 'learning_rate' in history:
        ax.plot(epochs, history['learning_rate'], 'o-', label='Learning Rate', 
               color='#f093fb', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Combined metrics
    ax = axes[1, 1]
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], 's-', label='Validation Accuracy', 
               color='#764ba2', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Performance (%)', fontsize=11, fontweight='bold')
    ax.set_title('Training Progress', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_history.png")


def main():
    """Main function to compare model performance on both datasets."""
    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    model_path = os.path.join(project_root, "results/models/resnet18_kidney_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, "results/models/resnet18_kidney.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model not found. Please train the model first.")
            return
    
    print(f"\nLoading model from: {model_path}")
    model = load_trained_model(model_path, device)
    
    # Create results directory
    results_dir = os.path.join(project_root, "results/figures")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    # Evaluate on augmented dataset (train, val, test sets)
    if os.path.exists("data/Augmented_Dataset"):
        print("\n" + "="*60)
        print("EVALUATING ON AUGMENTED DATASET")
        print("="*60)
        train_loader_aug, val_loader_aug, test_loader_aug = load_kidney_data("data/Augmented_Dataset")
        
        # Test set
        aug_test_metrics, _, _, _ = evaluate_on_dataset(model, test_loader_aug, device, "Augmented Dataset (Test)")
        all_results['augmented_test'] = aug_test_metrics
        
        # Validation set
        aug_val_metrics, _, _, _ = evaluate_on_dataset(model, val_loader_aug, device, "Augmented Dataset (Validation)")
        all_results['augmented_val'] = aug_val_metrics
        
        # Training set
        aug_train_metrics, _, _, _ = evaluate_on_dataset(model, train_loader_aug, device, "Augmented Dataset (Training)")
        all_results['augmented_train'] = aug_train_metrics
        
        print(f"\nAugmented Dataset - Test Results:")
        print(f"  Accuracy:  {aug_test_metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {aug_test_metrics['precision']*100:.2f}%")
        print(f"  Recall:    {aug_test_metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {aug_test_metrics['f1_score']*100:.2f}%")
        print(f"  ROC-AUC:   {aug_test_metrics['roc_auc']:.3f}")
    
    # Evaluate on original dataset (train, val, test sets)
    if os.path.exists("data/Original_Dataset"):
        print("\n" + "="*60)
        print("EVALUATING ON ORIGINAL DATASET")
        print("="*60)
        train_loader_orig, val_loader_orig, test_loader_orig = load_kidney_data("data/Original_Dataset")
        
        # Test set
        orig_test_metrics, _, _, _ = evaluate_on_dataset(model, test_loader_orig, device, "Original Dataset (Test)")
        all_results['original_test'] = orig_test_metrics
        
        # Validation set
        orig_val_metrics, _, _, _ = evaluate_on_dataset(model, val_loader_orig, device, "Original Dataset (Validation)")
        all_results['original_val'] = orig_val_metrics
        
        # Training set
        orig_train_metrics, _, _, _ = evaluate_on_dataset(model, train_loader_orig, device, "Original Dataset (Training)")
        all_results['original_train'] = orig_train_metrics
        
        print(f"\nOriginal Dataset - Test Results:")
        print(f"  Accuracy:  {orig_test_metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {orig_test_metrics['precision']*100:.2f}%")
        print(f"  Recall:    {orig_test_metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {orig_test_metrics['f1_score']*100:.2f}%")
        print(f"  ROC-AUC:   {orig_test_metrics['roc_auc']:.3f}")
    
    # Generate comparison graphs for each phase
    print("\n" + "="*60)
    print("GENERATING COMPARISON GRAPHS")
    print("="*60)
    
    # Test set comparison
    if 'augmented_test' in all_results and 'original_test' in all_results:
        print("\nGenerating Test Set Comparison Graphs...")
        plot_metrics_comparison(all_results['augmented_test'], all_results['original_test'], 
                               results_dir, phase='Test')
        plot_per_class_metrics(all_results['augmented_test'], all_results['original_test'], 
                              results_dir, phase='Test')
        plot_confusion_matrices(all_results['augmented_test'], all_results['original_test'], 
                               results_dir, phase='Test')
        plot_roc_curves(all_results['augmented_test'], all_results['original_test'], 
                       results_dir, phase='Test')
    
    # Validation set comparison
    if 'augmented_val' in all_results and 'original_val' in all_results:
        print("\nGenerating Validation Set Comparison Graphs...")
        plot_metrics_comparison(all_results['augmented_val'], all_results['original_val'], 
                               results_dir, phase='Validation')
        plot_per_class_metrics(all_results['augmented_val'], all_results['original_val'], 
                              results_dir, phase='Validation')
        plot_confusion_matrices(all_results['augmented_val'], all_results['original_val'], 
                               results_dir, phase='Validation')
        plot_roc_curves(all_results['augmented_val'], all_results['original_val'], 
                       results_dir, phase='Validation')
    
    # Training set comparison
    if 'augmented_train' in all_results and 'original_train' in all_results:
        print("\nGenerating Training Set Comparison Graphs...")
        plot_metrics_comparison(all_results['augmented_train'], all_results['original_train'], 
                               results_dir, phase='Training')
        plot_per_class_metrics(all_results['augmented_train'], all_results['original_train'], 
                              results_dir, phase='Training')
        plot_confusion_matrices(all_results['augmented_train'], all_results['original_train'], 
                               results_dir, phase='Training')
        plot_roc_curves(all_results['augmented_train'], all_results['original_train'], 
                       results_dir, phase='Training')
    
    # Generate phase comparison graphs (all phases together)
    plot_phase_comparison(all_results, results_dir)
    
    # Plot training history
    plot_training_history(results_dir)
    
    # Save all metrics to JSON
    metrics_path = os.path.join(project_root, "results/dataset_comparison_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ All metrics saved to: {metrics_path}")
    print(f"✓ All graphs saved to: {results_dir}")
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

