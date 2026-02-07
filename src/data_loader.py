import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.model_selection import train_test_split
import numpy as np

class StanfordCarsDataset(Dataset):
    """
    Custom dataset for Stanford Cars dataset with the specified directory structure.
    
    Directory structure:
    dataset/
    ├── train/
    │   ├── class_001/
    │   │   ├── 1.jpg
    │   │   └── ...
    │   └── ...
    └── test/
        ├── class_001/
        │   ├── 1.jpg
        │   └── ...
        └── ...
    """
    
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 is_training: bool = True,
                 config=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory of the dataset (train or test)
            transform: Optional transforms to apply to images
            is_training: Whether this is training data
            config: Configuration object for dataset limits
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        self.config = config
        
        # Load image paths and labels
        self.image_paths, self.labels, self.class_to_idx = self._load_data()
        
        logging.info(f"Loaded {len(self.image_paths)} images from {root_dir}")
        logging.info(f"Number of classes: {len(self.class_to_idx)}")
    
    def _load_data(self) -> Tuple[List[str], List[int], Dict[str, int]]:
        """Load image paths and labels from directory structure."""
        image_paths = []
        labels = []
        class_to_idx = {}
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d))]
        class_dirs.sort()  # Ensure consistent ordering
        
        # Limit number of classes if config is provided
        if self.config:
            max_classes = self.config.get('dataset.num_classes', len(class_dirs))
            class_dirs = class_dirs[:max_classes]
            logging.info(f"Limited to {max_classes} classes")
        
        # Create class to index mapping
        for idx, class_name in enumerate(class_dirs):
            class_to_idx[class_name] = idx
        
        # Get max images per class from config
        max_images_per_class = None
        if self.config:
            max_images_per_class = self.config.get('dataset.max_images_per_class')
        
        # Load images from each class directory
        for class_name in class_dirs:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = class_to_idx[class_name]
            
            # Get all image files in the class directory
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            # Limit images per class if specified
            if max_images_per_class:
                image_files = image_files[:max_images_per_class]
                logging.info(f"Class {class_name}: limited to {len(image_files)} images")
            
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                image_paths.append(image_path)
                labels.append(class_idx)
        
        return image_paths, labels, class_to_idx
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        # Load image
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from index."""
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        return idx_to_class.get(idx, f"Unknown_{idx}")

def get_data_transforms(config) -> Dict[str, transforms.Compose]:
    """
    Create data transforms for training and validation.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary containing train and test transforms
    """
    image_size = config.get('dataset.image_size', [224, 224])
    
    # Lightweight training transforms with smart augmentation
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),  # Reduced
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # Better than random crop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test/Validation transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'test': test_transform
    }

def create_data_loaders(config) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create training and testing data loaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, test_loader, class_to_idx)
    """
    # Get data paths
    train_dir = config.get('dataset.train_dir', 'dataset/train')
    test_dir = config.get('dataset.test_dir', 'dataset/test')
    
    # Get transforms
    transforms_dict = get_data_transforms(config)
    
    # Create datasets
    train_dataset = StanfordCarsDataset(
        root_dir=train_dir,
        transform=transforms_dict['train'],
        is_training=True,
        config=config
    )
    
    test_dataset = StanfordCarsDataset(
        root_dir=test_dir,
        transform=transforms_dict['test'],
        is_training=False,
        config=config
    )
    
    # Validate that both datasets have the same number of classes
    if len(train_dataset.class_to_idx) != len(test_dataset.class_to_idx):
        logging.warning(f"Training classes: {len(train_dataset.class_to_idx)}, "
                       f"Test classes: {len(test_dataset.class_to_idx)}")
    
    # Get data loader parameters
    batch_size = config.get('dataset.batch_size', 32)
    num_workers = config.get('dataset.num_workers', 4)
    
    # Create data loaders with optimized settings
    if num_workers > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches
        )
    else:
        # Single-threaded loading
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
    
    logging.info(f"Created data loaders:")
    logging.info(f"  Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    logging.info(f"  Test: {len(test_loader)} batches, {len(test_dataset)} samples")
    
    return train_loader, test_loader, train_dataset.class_to_idx

def create_train_val_loaders(config, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create training and validation data loaders from the training directory.
    
    Args:
        config: Configuration object
        val_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader, class_to_idx)
    """
    # Get data paths
    train_dir = config.get('dataset.train_dir', 'dataset/train')
    
    # Get transforms
    transforms_dict = get_data_transforms(config)
    
    # Create full training dataset
    full_dataset = StanfordCarsDataset(
        root_dir=train_dir,
        transform=transforms_dict['train'],
        is_training=True,
        config=config
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Create stratified split to maintain class distribution
    train_indices, val_indices = train_test_split(
        range(dataset_size),
        test_size=val_size,
        stratify=full_dataset.labels,
        random_state=config.get('random_seed', 42)
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Update validation dataset transform (no augmentation)
    val_dataset.dataset.transform = transforms_dict['test']
    
    # Get data loader parameters
    batch_size = config.get('dataset.batch_size', 32)
    num_workers = config.get('dataset.num_workers', 4)
    
    # Create data loaders with optimized settings
    if num_workers > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches
        )
    else:
        # Single-threaded loading
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
    
    logging.info(f"Created train/val data loaders:")
    logging.info(f"  Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    logging.info(f"  Val: {len(val_loader)} batches, {len(val_dataset)} samples")
    
    return train_loader, val_loader, full_dataset.class_to_idx

def get_dataset_statistics(data_loader: DataLoader) -> Dict[str, np.ndarray]:
    """
    Calculate dataset statistics (mean, std) for normalization.
    
    Args:
        data_loader: Data loader to calculate statistics on
        
    Returns:
        Dictionary with 'mean' and 'std' arrays
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': mean.numpy(),
        'std': std.numpy()
    }
