# VERSION: MAF-v1.0
# FILE: maf.py
# PURPOSE: Masked Autoregressive Flow - stack of MADE layers with batch norm
# DATE: 2025-01-21

import torch
import torch.nn as nn
import numpy as np
import logging

from made import MADE

logger = logging.getLogger(__name__)


class BatchNorm(nn.Module):
    """
    Batch normalization for normalizing flows.
    
    Tracks running statistics and is invertible with tractable Jacobian.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (not trained)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x, reverse=False):
        """
        Args:
            x: Input [B, D]
            reverse: If True, apply inverse transformation
            
        Returns:
            y: Normalized output [B, D]
            logdet: Log determinant [B]
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        if not reverse:
            # Forward: normalize
            if self.training:
                # Use batch statistics
                batch_mean = x.mean(0)
                batch_var = x.var(0, unbiased=False)
                
                # Update running statistics
                self.num_batches_tracked += 1
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                
                mean = batch_mean
                var = batch_var
            else:
                # Use running statistics
                mean = self.running_mean
                var = self.running_var
            
            # Normalize: y = γ * (x - μ) / σ + β
            std = torch.sqrt(var + self.eps)
            y = self.gamma * (x - mean) / std + self.beta
            
            # Log determinant: log |det J| = sum(log |γ / σ|)
            logdet = torch.sum(torch.log(torch.abs(self.gamma / std)))
            logdet = logdet.expand(x.size(0))  # [B]
            
            return y, logdet
        else:
            # Inverse: denormalize
            mean = self.running_mean
            var = self.running_var
            std = torch.sqrt(var + self.eps)
            
            # x = σ/γ * (y - β) + μ
            x = std / self.gamma * (x - self.beta) + mean
            
            # Log determinant (negative for inverse)
            logdet = -torch.sum(torch.log(torch.abs(self.gamma / std)))
            logdet = logdet.expand(x.size(0))
            
            return x, logdet


class MAF(nn.Module):
    """
    Masked Autoregressive Flow.
    
    Stack of MADE layers with batch normalization.
    Uses reversed ordering for each layer.
    
    Reference: Papamakarios et al., "Masked Autoregressive Flow for Density Estimation", NeurIPS 2017
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=5, num_hidden=2, 
                 use_random_degrees=False, use_batch_norm=True):
        """
        Args:
            input_dim: Dimensionality of input
            hidden_dim: Hidden units per MADE layer
            num_layers: Number of MADE layers to stack
            num_hidden: Hidden layers per MADE
            use_random_degrees: Use random degree assignment (for CIFAR-10)
            use_batch_norm: Use batch normalization between layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        # Build flow layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Alternate ordering: natural -> reversed -> natural -> ...
            if i % 2 == 0:
                ordering = None  # Natural order
            else:
                ordering = torch.arange(input_dim - 1, -1, -1)  # Reversed
            
            # MADE layer
            made = MADE(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_hidden=num_hidden,
                ordering=ordering,
                use_random_degrees=use_random_degrees
            )
            self.layers.append(made)
            
            # Batch normalization (except after last layer)
            if use_batch_norm and i < num_layers - 1:
                bn = BatchNorm(input_dim)
                self.layers.append(bn)
        
        logger.info(f"MAF created: {num_layers} layers, {hidden_dim}D hidden, "
                   f"batch_norm={use_batch_norm}, random_degrees={use_random_degrees}")
    
    def forward(self, x, reverse=False):
        """
        Forward or inverse pass through the flow.
        
        Args:
            x: Input tensor [B, D] or [B, C, H, W]
            reverse: If True, compute inverse (generation)
            
        Returns:
            z: Output tensor (latent if forward, data if reverse)
            logdet: Log determinant [B]
        """
        # Flatten if image
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        logdet = torch.zeros(x.size(0), device=x.device)
        
        if not reverse:
            # Forward: x -> z (fast, one pass through each MADE)
            for layer in self.layers:
                if isinstance(layer, MADE):
                    mu, alpha = layer(x)
                    # Transform: z = (x - μ) * exp(-α)
                    x = (x - mu) * torch.exp(-alpha)
                    # Log determinant: sum of -α
                    logdet += -alpha.sum(dim=1)
                elif isinstance(layer, BatchNorm):
                    x, ld = layer(x, reverse=False)
                    logdet += ld
            
            return x, logdet
        else:
            # Inverse: z -> x (slow, D sequential passes through each MADE)
            for layer in reversed(self.layers):
                if isinstance(layer, BatchNorm):
                    x, ld = layer(x, reverse=True)
                    logdet += ld
                elif isinstance(layer, MADE):
                    # Sequential generation: for each dim i, compute x_i = u_i * exp(α_i) + μ_i
                    for i in range(self.input_dim):
                        mu, alpha = layer(x)
                        x[:, i] = x[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
                    
                    # Log determinant for inverse
                    mu, alpha = layer(x)
                    logdet += alpha.sum(dim=1)
            
            # Reshape back if needed
            if len(original_shape) > 2:
                x = x.view(original_shape)
            
            return x, logdet
    
    def log_prob(self, x):
        """
        Compute log probability of data.
        
        Args:
            x: Data tensor [B, D] or [B, C, H, W]
            
        Returns:
            log_prob: Log probability [B]
        """
        z, logdet = self.forward(x, reverse=False)
        
        # Prior: N(0, I)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        
        # log p(x) = log p(z) + log |det J|
        log_px = log_pz + logdet
        
        return log_px
    
    def sample(self, num_samples, device='cuda'):
        """
        Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            samples: Generated data [num_samples, D]
        """
        # Sample from prior
        z = torch.randn(num_samples, self.input_dim, device=device)
        
        # Generate samples (slow, requires D passes through each MADE)
        with torch.no_grad():
            self.eval()
            samples, _ = self.forward(z, reverse=True)
        
        return samples


# CHANGELOG
"""
v1.0 (2025-01-21):
- Stack of MADE layers with alternating orderings
- Batch normalization between layers (optional)
- Forward: x -> z in one pass per layer (fast)
- Inverse: z -> x in D sequential steps per layer (slow)
- Log probability computation
- Sample generation from prior
"""