import torch
import torch.nn as nn
from typing import Tuple

class CouplingLayer(nn.Module):
    """NICE coupling layer implementation."""
    
    def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        
        # Register the mask as a buffer
        self.register_buffer('mask', mask)
        
        # Coupling function m(x) - input dimensions are the masked (unchanged) dimensions
        input_dim = int(mask.sum().item())
        output_dim = dim - input_dim
        
        if input_dim > 0 and output_dim > 0:
            self.coupling_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
            # Initialize to near-identity transformation
            nn.init.zeros_(self.coupling_net[-1].weight)
            nn.init.zeros_(self.coupling_net[-1].bias)
        else:
            self.coupling_net = None
    
    def forward(self, x):
        """Forward transformation: y1 = x1, y2 = x2 + m(x1)"""
        if self.coupling_net is None:
            return x
            
        # Extract masked (unchanged) and unmasked (to be transformed) parts
        x_masked = x * self.mask  # x1: input to coupling function
        x_unmasked = x * (1 - self.mask)  # x2: to be transformed
        
        # Get coupling function output
        coupling_input = x[:, self.mask.bool()]  # Only the masked dimensions
        coupling_output = self.coupling_net(coupling_input)
        
        # Apply additive coupling: y2 = x2 + m(x1)
        y = x_masked + x_unmasked  # Start with original
        y[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] + coupling_output
        
        return y
    
    def inverse(self, y):
        """Inverse transformation: x1 = y1, x2 = y2 - m(y1)"""
        if self.coupling_net is None:
            return y
            
        # Extract masked (unchanged) and unmasked (to be inverse transformed) parts
        y_masked = y * self.mask  # y1: unchanged
        
        # Get coupling function output using masked dimensions
        coupling_input = y[:, self.mask.bool()]
        coupling_output = self.coupling_net(coupling_input)
        
        # Apply inverse additive coupling: x2 = y2 - m(y1)
        x = y_masked + y * (1 - self.mask)  # Start with y
        x[:, ~self.mask.bool()] = y[:, ~self.mask.bool()] - coupling_output
        
        return x


class NICEFlow(nn.Module):
    """NICE: Non-linear Independent Components Estimation."""
    
    def __init__(self, dim: int, n_layers: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        
        # Create alternating masks similar to Real NVP
        self.masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            if dim == 2:
                # For 2D: alternate [1,0] and [0,1]
                mask[i % 2] = 1
            else:
                # For higher dimensions: alternate even/odd
                mask[i % 2::2] = 1
            
            self.register_buffer(f'mask_{i}', mask)
            self.masks.append(mask)
        
        # Create coupling layers
        self.coupling_layers = nn.ModuleList()
        for i in range(n_layers):
            layer = CouplingLayer(dim, self.masks[i], hidden_dim)
            self.coupling_layers.append(layer)
        
        # Diagonal scaling layer (learnable) - initialized to small values
        self.log_scale = nn.Parameter(torch.zeros(dim) * 0.01)
    
    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through NICE layers."""
        z = x
        
        # Apply coupling layers (each has Jacobian determinant of 1)
        for layer in self.coupling_layers:
            z = layer.forward(z)
        
        # Apply diagonal scaling
        z = z * torch.exp(self.log_scale)
        
        # Log determinant is sum of log scales (expanded for batch)
        log_det = self.log_scale.sum().expand(x.size(0))
        
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        # Inverse diagonal scaling
        x = z * torch.exp(-self.log_scale)
        
        # Apply inverse coupling layers in reverse order
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x)
        
        return x
    
    def log_prob(self, x):
        """Compute log probability under NICE model with standard Gaussian prior."""
        z, log_det = self.forward_and_log_det(x)
        
        # Standard Gaussian log probability in latent space
        log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * z.size(1) * torch.log(2 * torch.tensor(torch.pi, device=z.device))
        
        # Add log determinant term
        return log_prob_z + log_det
    
    def sample(self, n_samples):
        """Sample from NICE model."""
        device = next(self.parameters()).device
        
        # Sample from standard Gaussian in latent space
        z = torch.randn(n_samples, self.dim, device=device)
        
        # Transform to data space
        return self.inverse(z)