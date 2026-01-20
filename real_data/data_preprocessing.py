import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


class ImagePreprocessing:
    """Preprocessing pipeline for normalizing flow models on image data."""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def dequantize(self, x):
        """Add uniform noise U(0,1) to discrete pixel values."""
        return x + torch.rand_like(x)
    
    def logit_transform(self, x):
        """Transform from [0, 256] to unbounded space."""
        # Scale to [0, 1] then apply logit with alpha smoothing
        x = x / 256.0
        x = self.alpha + (1 - self.alpha) * x
        return torch.logit(x)
    
    def __call__(self, x):
        """Apply dequantization and logit transform."""
        x = self.dequantize(x)
        x = self.logit_transform(x)
        return x


def get_mnist_loaders(batch_size=128, augment=False):
    """Get MNIST train/test dataloaders with preprocessing."""
    preprocess = ImagePreprocessing(alpha=0.05)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),  # Scale to [0, 255]
        preprocess
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_cifar10_loaders(batch_size=128, augment=True):
    """Get CIFAR-10 train/test dataloaders with preprocessing and augmentation."""
    preprocess = ImagePreprocessing(alpha=0.05)
    
    # Training transforms with optional augmentation
    train_transforms = [transforms.ToTensor()]
    if augment:
        train_transforms.insert(0, transforms.RandomHorizontalFlip())
    train_transforms.extend([
        transforms.Lambda(lambda x: x * 255),  # Scale to [0, 255]
        preprocess
    ])
    
    # Test transforms (no augmentation)
    test_transforms = [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
        preprocess
    ]
    
    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    # MNIST (no augmentation)
    mnist_train, mnist_test = get_mnist_loaders(batch_size=128, augment=False)
    
    # CIFAR-10 (with horizontal flips)
    cifar_train, cifar_test = get_cifar10_loaders(batch_size=128, augment=True)
    
    # Test the pipeline
    x, y = next(iter(mnist_train))
    print(f"MNIST batch shape: {x.shape}")
    print(f"MNIST value range: [{x.min():.2f}, {x.max():.2f}]")
    
    x, y = next(iter(cifar_train))
    print(f"CIFAR-10 batch shape: {x.shape}")
    print(f"CIFAR-10 value range: [{x.min():.2f}, {x.max():.2f}]")
    
    
    
    