import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
<<<<<<< HEAD
import os
import sys
import json
from datetime import datetime

# Add project root to Python path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.load_data import load_kidney_data, load_combined_kidney_data


def train_model(model, train_loader, val_loader, device, epochs=10, lr=5e-5, 
                save_dir="results/models", model_name="resnet18_kidney"):
    """
    Train the model with checkpointing and metrics tracking.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run training on (CPU, GPU, or MPS)
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save model checkpoints
        model_name: Name for saved model files
    
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    # Calculate class weights to handle imbalanced dataset
    # Stone class needs more weight to balance predictions
    # Using 1.5x weight (balanced - not too aggressive to maintain augmented performance)
    class_weights = torch.FloatTensor([1.0, 1.5]).to(device)  # Give Stone 1.5x more weight
    print(f"Using class weights: Non-Stone=1.0, Stone=1.5 (balanced for both datasets)")
    
    # Loss function with class weights - helps balance predictions
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer - updates model weights to reduce loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler - reduces learning rate when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Dictionary to store training history (loss and accuracy over time)
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': [],
        'epochs': []
    }
    
    # Track the best validation accuracy to save the best model
    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
    final_model_path = os.path.join(save_dir, f"{model_name}.pth")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}")
    
    # Training loop - one epoch = one pass through all training data
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)
        
        # ========== TRAINING PHASE ==========
        model.train()  # Set model to training mode
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Process each batch of training images
        for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
            # Move data to device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)
            
            # Zero out gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass - get predictions from model
            outputs = model(images)
            
            # Calculate loss (how wrong our predictions are)
            loss = criterion(outputs, labels)
            
            # Backward pass - calculate gradients
            loss.backward()
            
            # Update model weights
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)  # Get predicted class
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        # Calculate average training loss and accuracy
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        
        # ========== VALIDATION PHASE ==========
        model.eval()  # Set model to evaluation mode (no gradient calculation)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Don't calculate gradients during validation (saves memory and time)
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", unit="batch"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)
        history['epochs'].append(epoch + 1)
        
        print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save best model (highest validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': avg_val_loss,
            }, best_model_path)
            print(f"✓ New best model saved! (Val Accuracy: {val_acc:.2f}%)")
    
    # Save final model after all epochs
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'history': history,
    }, final_model_path)
    
    # Save training history to JSON file
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {history_path}")
    
    return model, history


def main():
    """Main function to run the training process."""
    # Select device - use GPU if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    else:
        device = torch.device("cpu")   # CPU
    
    print(f"Using device: {device}")
    
    # Load combined dataset - use both augmented and original for best performance
    print("\nLoading combined dataset (augmented + original)...")
    print("This ensures model performs well on both types of data.\n")
    
    # Check if both datasets exist
    has_augmented = os.path.exists("data/Augmented_Dataset")
    has_original = os.path.exists("data/Original_Dataset")
    
    if has_augmented and has_original:
        # Use combined dataset - best of both worlds!
        train_loader, val_loader, test_loader = load_combined_kidney_data(
            augmented_dir="data/Augmented_Dataset",
            original_dir="data/Original_Dataset"
        )
        print("\n✓ Training on COMBINED dataset (augmented + original)")
        print("  This maintains excellent performance on augmented data")
        print("  while improving performance on original data!")
    elif has_augmented:
        # Fall back to augmented only
        print("Using Augmented_Dataset only (original not found)")
        train_loader, val_loader, test_loader = load_kidney_data("data/Augmented_Dataset")
    elif has_original:
        # Fall back to original only
        print("Using Original_Dataset only (augmented not found)")
        train_loader, val_loader, test_loader = load_kidney_data("data/Original_Dataset")
    else:
        # Fall back to default
        print("Using default data folder")
        train_loader, val_loader, test_loader = load_kidney_data("data")
    
    # Initialize ResNet-18 model
    print("\nInitializing ResNet-18 model...")
    model = models.resnet18(weights=None)  # Start with random weights
    # Modify final layer for binary classification (2 classes: Stone / Non-Stone)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Train the model
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        device, 
        epochs=10,  # Number of training epochs (increased for better learning)
        lr=5e-5,    # Learning rate
        save_dir="results/models",
        model_name="resnet18_kidney"
    )
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run evaluation script to test on test set")
    print("2. Generate report with: python src/evaluation/generate_report.py")
=======

from src.preprocessing.load_data import load_kidney_data


def train_model(model, train_loader, val_loader, device, epochs=10, lr=5e-5):
    """Train the model using the given data loaders."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Average training loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Validation accuracy: {accuracy:.2f}%")

    return model


def main():
    # Select device (Apple Metal GPU if available)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    train_loader, val_loader, _ = load_kidney_data("data")

    # Initialize ResNet-18 model with random weights (no pretrained data)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Two classes: Stone / Non-Stone
    model.to(device)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, device, epochs=10, lr=5e-5)

    # Save the trained weights
    torch.save(trained_model.state_dict(), "results/models/resnet18_kidney.pth")
    print("Model saved to results/models/resnet18_kidney.pth")
>>>>>>> bb81204f4bae432736718559d00cea9bb8fa47e8


if __name__ == "__main__":
    main()
