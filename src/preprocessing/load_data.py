# src/preprocessing/load_data.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import os

def load_combined_kidney_data(augmented_dir="data/Augmented_Dataset", 
                               original_dir="data/Original_Dataset",
                               image_size=224, batch_size=128, val_split=0.2):
    """
    Load and combine both augmented and original datasets for training.
    This helps model perform well on both types of data.
    """
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Validation/Test transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    datasets_list = []
    
    # Load augmented dataset if exists
    if os.path.exists(augmented_dir):
        aug_dataset = datasets.ImageFolder(root=augmented_dir, transform=None)
        datasets_list.append(aug_dataset)
        print(f"Loaded augmented dataset: {len(aug_dataset)} images")
    
    # Load original dataset if exists
    if os.path.exists(original_dir):
        orig_dataset = datasets.ImageFolder(root=original_dir, transform=None)
        datasets_list.append(orig_dataset)
        print(f"Loaded original dataset: {len(orig_dataset)} images")
    
    if not datasets_list:
        raise ValueError("Neither augmented nor original dataset found!")
    
    # Combine datasets
    combined_dataset = ConcatDataset(datasets_list)
    print(f"Combined dataset total: {len(combined_dataset)} images")
    
    # Split into train/val/test
    total = len(combined_dataset)
    val_size = int(total * val_split)
    test_size = val_size
    train_size = total - val_size - test_size
    train_indices, val_indices, test_indices = random_split(
        range(total), [train_size, val_size, test_size]
    )
    
    # Create datasets with appropriate transforms
    from torch.utils.data import Subset
    
    class TransformDataset:
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            # Handle ConcatDataset indexing
            if isinstance(self.dataset, ConcatDataset):
                dataset_idx = 0
                sample_idx = actual_idx
                for i, ds in enumerate(self.dataset.datasets):
                    if sample_idx < len(ds):
                        dataset_idx = i
                        break
                    sample_idx -= len(ds)
                img, label = self.dataset.datasets[dataset_idx][sample_idx]
            else:
                img, label = self.dataset[actual_idx]
            
            if self.transform:
                img = self.transform(img)
            return img, label
        
        def __len__(self):
            return len(self.indices)
    
    train_ds = TransformDataset(combined_dataset, train_indices, train_transform)
    val_ds = TransformDataset(combined_dataset, val_indices, val_transform)
    test_ds = TransformDataset(combined_dataset, test_indices, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader

def load_kidney_data(data_dir="data", image_size=224, batch_size=128, val_split=0.2):
    """
    Just read everything under data/ (Stone / Non-Stone subfolders),
    split into train/val/test, and return DataLoaders.
    """

    # Training transform with augmentation (helps model work on original data too)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Validation/Test transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset without transform first (we'll apply transforms separately)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=None)

    # Simple random split
    total = len(full_dataset)
    val_size = int(total * val_split)
    test_size = val_size
    train_size = total - val_size - test_size
    train_indices, val_indices, test_indices = random_split(
        range(total), [train_size, val_size, test_size]
    )
    
    # Create datasets with appropriate transforms
    from torch.utils.data import Subset
    
    class TransformDataset:
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            img, label = self.dataset[actual_idx]
            if self.transform:
                img = self.transform(img)
            return img, label
        
        def __len__(self):
            return len(self.indices)
    
    train_ds = TransformDataset(full_dataset, train_indices, train_transform)
    val_ds = TransformDataset(full_dataset, val_indices, val_transform)
    test_ds = TransformDataset(full_dataset, test_indices, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    print(f"Total images: {total}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Loading all data")
    load_kidney_data("data")
