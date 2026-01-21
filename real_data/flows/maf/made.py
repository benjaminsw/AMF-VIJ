# VERSION: MAF-v1.0
# FILE: made.py
# PURPOSE: Masked Autoencoder for Distribution Estimation (MADE) with Gaussian conditionals
# DATE: 2025-01-21

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE).
    
    Implements autoregressive property via masked connections.
    Outputs Gaussian parameters: μ(x₁:ᵢ₋₁) and α(x₁:ᵢ₋₁) for each dimension.
    
    Reference: Germain et al., "MADE: Masked Autoencoder for Distribution Estimation", ICML 2015
    """
    
    def __init__(self, input_dim, hidden_dim, num_hidden=2, ordering=None, use_random_degrees=False):
        """
        Args:
            input_dim: Dimensionality of input (e.g., 784 for MNIST)
            hidden_dim: Number of hidden units per layer
            num_hidden: Number of hidden layers (default: 2)
            ordering: Permutation of input dimensions (None = natural order)
            use_random_degrees: Use random degree assignment (for high-dim data like CIFAR-10)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.use_random_degrees = use_random_degrees
        
        # Set ordering (permutation of input dimensions)
        if ordering is None:
            self.ordering = torch.arange(input_dim)
        else:
            self.ordering = torch.tensor(ordering)
        
        # Build network
        layers = []
        dims = [input_dim] + [hidden_dim] * num_hidden + [input_dim * 2]  # *2 for μ and α
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after output layer
                layers.append(nn.ReLU())
        
        self.net = nn.ModuleList(layers)
        
        # Create masks
        self.masks = self._create_masks(dims)
        
        # Apply masks to weights
        self._apply_masks()
        
        logger.debug(f"MADE created: {input_dim}D, {num_hidden}x{hidden_dim} hidden, "
                    f"random_degrees={use_random_degrees}")
    
    def _create_masks(self, dims):
        """
        Create binary masks for enforcing autoregressive property.
        
        Following Germain et al. (2015) masking procedure:
        - Assign degrees to input (their position in ordering)
        - Assign degrees to hidden units (sequential or random)
        - Assign degrees to outputs (0 to D-1)
        - Connect unit i to unit j only if degree(i) <= degree(j)
        """
        masks = []
        
        # Input degrees: position in ordering (1 to D)
        input_degrees = self.ordering + 1  # +1 because degrees start at 1
        
        # Hidden layer degrees
        hidden_degrees = []
        for l in range(self.num_hidden):
            if self.use_random_degrees:
                # Random assignment (for CIFAR-10)
                # Ensure all degrees from 1 to input_dim appear
                degrees = torch.randint(1, self.input_dim + 1, (self.hidden_dim,))
            else:
                # Sequential assignment (for MNIST)
                # Ensure all degrees appear by cycling
                degrees = torch.arange(self.hidden_dim) % self.input_dim + 1
            
            hidden_degrees.append(degrees)
        
        # Output degrees: 0 to D-1 (each output i models p(x_i | x_{<i}))
        # For μ and α, we have 2*D outputs, but they share degrees
        output_degrees_mu = self.ordering  # 0 to D-1
        output_degrees_alpha = self.ordering  # 0 to D-1
        output_degrees = torch.cat([output_degrees_mu, output_degrees_alpha])
        
        # Create masks
        all_degrees = [input_degrees] + hidden_degrees + [output_degrees]
        
        for l in range(len(all_degrees) - 1):
            # Mask: connection from unit i to unit j exists if degree(i) <= degree(j)
            in_degrees = all_degrees[l]
            out_degrees = all_degrees[l + 1]
            
            # Broadcasting: [out_dim, 1] <= [1, in_dim]
            mask = (out_degrees.unsqueeze(1) >= in_degrees.unsqueeze(0)).float()
            
            masks.append(mask)
        
        return masks
    
    def _apply_masks(self):
        """Apply masks to weight matrices."""
        mask_idx = 0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # Register mask as buffer (not a parameter)
                self.register_buffer(f'mask_{mask_idx}', self.masks[mask_idx])
                mask_idx += 1
    
    def forward(self, x):
        """
        Forward pass with masked connections.
        
        Args:
            x: Input tensor [B, D]
            
        Returns:
            mu: Mean parameters [B, D]
            alpha: Log std parameters [B, D] (clamped for stability)
        """
        # Flatten if needed (for images)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        h = x
        mask_idx = 0
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # Apply masked weights: y = (W ⊙ M) @ x + b
                mask = getattr(self, f'mask_{mask_idx}')
                weight = layer.weight * mask
                h = nn.functional.linear(h, weight, layer.bias)
                mask_idx += 1
            else:
                # Activation (ReLU)
                h = layer(h)
        
        # Split output into μ and α
        mu, alpha = h.chunk(2, dim=1)
        
        # Clamp alpha for numerical stability
        alpha = torch.clamp(alpha, -10, 10)
        
        return mu, alpha
    
    def inverse(self, u):
        """
        Inverse pass: u -> x (sequential, slow).
        
        Args:
            u: Random numbers from N(0, I) [B, D]
            
        Returns:
            x: Generated data [B, D]
        """
        x = torch.zeros_like(u)
        
        for i in range(self.input_dim):
            # Get parameters for dimension i
            mu, alpha = self.forward(x)
            
            # Transform: x_i = u_i * exp(alpha_i) + mu_i
            x[:, i] = u[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
        
        return x


# CHANGELOG
"""
v1.0 (2025-01-21):
- MADE with Gaussian conditionals
- Masked connections via degree assignment
- Sequential (MNIST) and random (CIFAR-10) degree modes
- Two hidden layers (configurable)
- Alpha clamping [-10, 10] for stability
- Forward: x -> (μ, α) in one pass
- Inverse: u -> x in D sequential steps
"""