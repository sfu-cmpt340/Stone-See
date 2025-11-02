import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

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


if __name__ == "__main__":
    main()
