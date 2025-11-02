# src/preprocessing/load_data.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_kidney_data(data_dir="data", image_size=224, batch_size=64, val_split=0.2):
    """
    Just read everything under data/ (Stone / Non-Stone subfolders),
    split into train/val/test, and return DataLoaders.
    """

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # simple random split
    total = len(dataset)
    val_size = int(total * val_split)
    test_size = val_size
    train_size = total - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    print(f"Total images: {total}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Loading all data")
    load_kidney_data("data")
