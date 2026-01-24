"""
Version: 1.0.0
Data caching utility for AMF-VI experiments - Generate-Once-Cache-Forever approach.

CHANGELOG v1.0.0:
- Initial implementation of stratified train/val/test split caching
- Automatic cache management with seed-based reproducibility
- Memory-efficient loading with torch tensor support
- Metadata tracking for experiment provenance
"""

import os
import numpy as np
import torch
from pathlib import Path

# Import the actual data generator
try:
    from data.data_generator import generate_data
except ImportError:
    from data_generator import generate_data


class DataCache:
    """
    Manages cached train/val/test splits for reproducible AMF-VI experiments.
    
    Strategy: Generate large pool once, split into train/val/test, cache to disk.
    Subsequent runs load from cache instead of regenerating.
    """
    
    def __init__(self, cache_dir="./data/datasets", seed=2025):
        """
        Initialize data cache manager.
        
        Args:
            cache_dir: Directory to store cached .npz files
            seed: Global random seed for reproducibility (default: 2025)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Set global seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
    
    def _get_cache_path(self, dataset_name, n_samples):
        """
        Generate standardized cache filename.
        
        Format: {dataset}_n{samples}_seed{seed}.npz
        Example: banana_n100000_seed2025.npz
        """
        return self.cache_dir / f"{dataset_name}_n{n_samples}_seed{self.seed}.npz"
    
    def get_or_create_split(
        self, 
        dataset_name, 
        n_samples=100_000,
        split_ratios=(0.6, 0.2, 0.2)
    ):
        """
        Load cached split or generate new one.
        
        Args:
            dataset_name: Dataset name (e.g., 'banana', 'multimodal', 'BLR')
            n_samples: Total samples to generate
            split_ratios: (train, val, test) ratios (must sum to 1.0)
        
        Returns:
            dict with keys:
                - 'train': torch.Tensor [n_train, dim]
                - 'val': torch.Tensor [n_val, dim]
                - 'test': torch.Tensor [n_test, dim]
                - 'metadata': dict with generation info
        
        Raises:
            FileNotFoundError: If cache doesn't exist (no fallback)
            ValueError: If split_ratios invalid or data generation fails
        """
        # Validate split ratios
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(split_ratios)}")
        
        cache_path = self._get_cache_path(dataset_name, n_samples)
        
        # === CACHE HIT: Load from disk ===
        if cache_path.exists():
            print(f"📂 Loading cached split from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            
            # Verify cache integrity
            expected_keys = {'train', 'val', 'test', 'metadata'}
            if not expected_keys.issubset(data.keys()):
                raise ValueError(f"Corrupted cache file. Expected keys {expected_keys}, got {set(data.keys())}")
            
            metadata = data['metadata'].item()
            print(f"   ✅ Loaded: train={len(data['train'])}, val={len(data['val'])}, test={len(data['test'])}")
            print(f"   📋 Metadata: {metadata}")
            
            return {
                'train': torch.from_numpy(data['train']).float(),
                'val': torch.from_numpy(data['val']).float(),
                'test': torch.from_numpy(data['test']).float(),
                'metadata': metadata
            }
        
        # === CACHE MISS: Generate and save ===
        print(f"🔄 Cache miss - Generating fresh split for {dataset_name} (n={n_samples})")
        
        # Reset seeds for reproducible generation
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Generate full dataset
        print(f"   Generating {n_samples} samples...")
        try:
            full_data = generate_data(dataset_name, n_samples=n_samples)
        except Exception as e:
            raise ValueError(f"Failed to generate data for '{dataset_name}': {e}")
        
        # Convert to numpy for splitting
        if torch.is_tensor(full_data):
            full_data_np = full_data.cpu().numpy()
        else:
            full_data_np = np.asarray(full_data)
        
        # Calculate split sizes
        n_train = int(n_samples * split_ratios[0])
        n_val = int(n_samples * split_ratios[1])
        n_test = n_samples - n_train - n_val  # Remaining to avoid rounding errors
        
        print(f"   Splitting into train={n_train}, val={n_val}, test={n_test}")
        
        # Shuffle and split
        indices = np.random.permutation(n_samples)
        
        train_data = full_data_np[indices[:n_train]]
        val_data = full_data_np[indices[n_train:n_train+n_val]]
        test_data = full_data_np[indices[n_train+n_val:]]
        
        # Create metadata
        metadata = {
            'dataset_name': dataset_name,
            'n_samples': n_samples,
            'split_ratios': split_ratios,
            'seed': self.seed,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'data_shape': full_data_np.shape,
        }
        
        # Save to cache
        print(f"   💾 Saving to cache: {cache_path}")
        np.savez(
            cache_path,
            train=train_data,
            val=val_data,
            test=test_data,
            metadata=metadata
        )
        
        # Verify save
        if not cache_path.exists():
            raise IOError(f"Failed to save cache to {cache_path}")
        
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Cache saved successfully ({cache_size_mb:.2f} MB)")
        
        return {
            'train': torch.from_numpy(train_data).float(),
            'val': torch.from_numpy(val_data).float(),
            'test': torch.from_numpy(test_data).float(),
            'metadata': metadata
        }


# === Singleton instance ===
_data_cache = DataCache(seed=2025)


def get_split_data(dataset_name, n_samples=100_000, split_ratios=(0.6, 0.2, 0.2)):
    """
    Convenience function to get cached split data.
    
    Args:
        dataset_name: Dataset name
        n_samples: Total samples
        split_ratios: (train, val, test) split
    
    Returns:
        dict with 'train', 'val', 'test' torch tensors
    """
    return _data_cache.get_or_create_split(dataset_name, n_samples, split_ratios)


def clear_cache(dataset_name=None, cache_dir="./data/datasets"):
    """
    Clear cached files (for testing/debugging).
    
    Args:
        dataset_name: If provided, only clear this dataset. Otherwise clear all.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print("No cache directory found.")
        return
    
    if dataset_name:
        pattern = f"{dataset_name}_*.npz"
    else:
        pattern = "*.npz"
    
    files = list(cache_path.glob(pattern))
    
    if not files:
        print(f"No cache files found matching '{pattern}'")
        return
    
    print(f"🗑️  Clearing {len(files)} cache file(s)...")
    for f in files:
        f.unlink()
        print(f"   Deleted: {f.name}")
    
    print("✅ Cache cleared")


if __name__ == "__main__":
    # Test the caching system
    print("Testing DataCache...")
    
    # Test cache miss -> generate
    data = get_split_data('banana', n_samples=10000)
    print(f"\nFirst call: train shape = {data['train'].shape}")
    
    # Test cache hit -> load
    data2 = get_split_data('banana', n_samples=10000)
    print(f"Second call: train shape = {data2['train'].shape}")
    
    # Verify same data
    assert torch.allclose(data['train'], data2['train']), "Cache mismatch!"
    print("\n✅ Cache verification passed")