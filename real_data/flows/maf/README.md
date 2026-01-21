# Masked Autoregressive Flow (MAF) Implementation

**Version**: MAF-v1.0  
**Date**: 2025-01-21

## Overview

Implementation of Masked Autoregressive Flow (MAF) as described in:
> Papamakarios et al., "Masked Autoregressive Flow for Density Estimation", NeurIPS 2017

MAF is a normalizing flow that uses stacked MADE (Masked Autoencoder for Distribution Estimation) layers to model complex distributions.

## Architecture

### MADE Layer (`made.py`)
- Masked autoencoder with Gaussian conditionals
- Outputs μ(x₁:ᵢ₋₁) and α(x₁:ᵢ₋₁) for each dimension
- Binary masking enforces autoregressive property
- Two hidden layers (configurable)
- Alpha clamping [-10, 10] for numerical stability

### MAF Model (`maf.py`)
- Stack of K MADE layers
- Alternating orderings (natural → reversed → natural...)
- Batch normalization between layers
- Base distribution: N(0, I)

**Key Properties**:
- **Forward (density estimation)**: x → z in ONE pass per layer (fast)
- **Inverse (sampling)**: z → x in D sequential steps per layer (slow)

## Configuration

### MNIST
- Input: 784 dimensions (28×28 grayscale)
- Hidden: 512 units
- Layers: 5 MADE layers
- Degree assignment: Sequential

### CIFAR-10
- Input: 3072 dimensions (3×32×32 RGB)
- Hidden: 1024 units
- Layers: 10 MADE layers
- Degree assignment: Random (for high-dim efficiency)

## Usage

### Training

```bash
# Train both MNIST and CIFAR-10
python train.py --auto

# Train specific dataset
python train.py --dataset mnist --epochs 100
python train.py --dataset cifar10 --epochs 100

# Custom configuration
python train.py --dataset mnist --hidden_dim 512 --num_layers 5 --lr 1e-4
```

### Evaluation

```bash
# Evaluate MNIST
python evaluate.py --dataset mnist

# Evaluate CIFAR-10
python evaluate.py --dataset cifar10 --num_samples 64
```

## Results Structure

```
maf_results/
├── mnist/
│   ├── checkpoints/
│   │   ├── best.pth
│   │   └── latest.pth
│   ├── logs/
│   │   └── training.log
│   ├── samples/
│   │   └── mnist_samples.png
│   └── results_summary.txt
└── cifar10/
    ├── checkpoints/
    ├── logs/
    ├── samples/
    └── results_summary.txt
```

## Metrics

- **NLL (Negative Log-Likelihood)**: Primary training objective
- **BPD (Bits Per Dimension)**: NLL / (num_dims × log(2))

Lower is better for both metrics.

## Implementation Details

### Preprocessing
- Dequantization: Add U(0,1) noise to discrete pixels
- Logit transform: x → logit(α + (1-α)x) where α=0.05
- Uses existing `data_preprocessing.py`

### Numerical Stability
- Alpha clamping: α ∈ [-10, 10] to prevent exp overflow
- Gradient clipping: max_norm=5.0
- NaN/Inf detection with error logging

### Batch Normalization
- Tracks running statistics (μ, σ²)
- Invertible with tractable log-det
- Applied between MADE layers (not after last layer)

### Masking Strategy
- **Layer 1**: Natural order (0, 1, 2, ..., D-1)
- **Layer 2**: Reversed order (D-1, ..., 2, 1, 0)
- **Layer 3**: Natural order (repeating)
- Helps model capture dependencies in both directions

## Comparison with RealNVP

| Property | MAF | RealNVP |
|----------|-----|---------|
| Forward (x→z) | Fast (1 pass) | Fast (1 pass) |
| Inverse (z→x) | Slow (D passes) | Fast (1 pass) |
| Density eval | Fast | Fast |
| Sampling | Slow | Fast |
| Best for | Density estimation | Generation |
| Architecture | Autoregressive | Coupling layers |

**Recommendation**: Use MAF for density estimation tasks, RealNVP for generation.

## Files

- `__init__.py`: Package initializer
- `made.py`: MADE layer with masking (178 lines)
- `maf.py`: MAF model with batch norm (237 lines)
- `train.py`: Training script (272 lines)
- `evaluate.py`: Evaluation and sampling (215 lines)
- `README.md`: This file

## References

1. Papamakarios et al., "Masked Autoregressive Flow for Density Estimation", NeurIPS 2017
2. Germain et al., "MADE: Masked Autoencoder for Distribution Estimation", ICML 2015
3. Dinh et al., "Density estimation using Real NVP", ICLR 2017

## Troubleshooting

### Training Issues
- **NaN loss**: Reduce learning rate, check data preprocessing
- **Slow training**: Normal for MAF, consider reducing num_layers
- **Poor performance**: Increase hidden_dim or num_layers

### Memory Issues
- Reduce batch_size
- Use fewer layers or smaller hidden_dim
- For CIFAR-10, consider gradient checkpointing

## Version History

**v1.0 (2025-01-21)**:
- Initial implementation with MADE and MAF
- Batch normalization support
- Fixed reversed ordering strategy
- Alpha clamping for stability
- Training and evaluation scripts
- Auto-configuration for MNIST/CIFAR-10