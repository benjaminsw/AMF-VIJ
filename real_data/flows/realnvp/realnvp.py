# VERSION: RNV-v1.0
# FILE: realnvp.py
# PURPOSE: Multi-scale RealNVP architecture
# DATE: 2025-01-20

import torch
import torch.nn as nn
import logging
from coupling_layers import (  # Changed: relative import
    AffineCouplingLayer,
    create_checkerboard_mask,
    create_channel_mask,
    squeeze_operation,
    unsqueeze_operation
)

logger = logging.getLogger(__name__)


class RealNVP(nn.Module):
    """
    Real Non-Volume Preserving (RealNVP) normalizing flow.
    
    Multi-scale architecture with checkerboard and channel-wise coupling.
    """
    
    def __init__(self, input_shape, hidden_channels=64, num_blocks=8, num_scales=3):
        """
        Args:
            input_shape: (C, H, W) tuple
            hidden_channels: Feature maps in ResNet (32 or 64)
            num_blocks: Residual blocks per coupling layer (4 or 8)
            num_scales: Number of multi-scale levels
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.num_scales = num_scales
        
        C, H, W = input_shape
        
        # Build coupling layers for each scale
        self.flows = nn.ModuleList()
        
        current_channels = C
        current_height = H
        current_width = W
        
        for scale in range(num_scales):
            # 3 coupling layers with checkerboard masks
            for i in range(3):
                mask = create_checkerboard_mask(current_height, current_width, 
                                               reverse=(i % 2 == 1))
                mask = mask.repeat(1, current_channels, 1, 1)
                
                self.flows.append(AffineCouplingLayer(
                    mask, current_channels, hidden_channels, num_blocks
                ))
            
            # Squeeze operation (except last scale)
            if scale < num_scales - 1:
                self.flows.append(SqueezeLayer())
                current_channels *= 4
                current_height //= 2
                current_width //= 2
                
                # 3 coupling layers with channel-wise masks
                for i in range(3):
                    mask = create_channel_mask(current_channels, reverse=(i % 2 == 1))
                    mask = mask.repeat(1, 1, current_height, current_width)
                    
                    self.flows.append(AffineCouplingLayer(
                        mask, current_channels, hidden_channels, num_blocks
                    ))
        
        # Final coupling layers (no squeeze after last scale)
        for i in range(4):
            mask = create_checkerboard_mask(current_height, current_width,
                                           reverse=(i % 2 == 1))
            mask = mask.repeat(1, current_channels, 1, 1)
            
            self.flows.append(AffineCouplingLayer(
                mask, current_channels, hidden_channels, num_blocks
            ))
        
        logger.info(f"RealNVP created: {len(self.flows)} coupling/squeeze layers")
        logger.info(f"Final shape: {current_channels}×{current_height}×{current_width}")
        
    def forward(self, x, reverse=False):
        """
        Forward/inverse pass through the flow.
        
        Args:
            x: Input tensor [B, C, H, W]
            reverse: If True, compute inverse (generation)
            
        Returns:
            z: Latent tensor (if forward) or generated sample (if reverse)
            logdet: Log determinant of Jacobian
        """
        logdet = 0
        
        if not reverse:
            # Forward: x -> z
            for flow in self.flows:
                x, logdet = flow(x, logdet, reverse=False)
            return x, logdet
        else:
            # Inverse: z -> x
            for flow in reversed(self.flows):
                x, logdet = flow(x, logdet, reverse=True)
            return x, logdet
    
    def log_prob(self, x):
        """
        Compute log probability of data.
        
        Args:
            x: Data tensor [B, C, H, W]
            
        Returns:
            log_prob: Log probability [B]
        """
        z, logdet = self.forward(x, reverse=False)
        
        # Prior: N(0, I)
        log_pz = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * 3.14159265359))).sum(dim=[1, 2, 3])
        
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
            samples: Generated images [num_samples, C, H, W]
        """
        C, H, W = self.input_shape
        
        # Compute final latent shape after all squeezes
        final_channels = C * (4 ** (self.num_scales - 1))
        final_height = H // (2 ** (self.num_scales - 1))
        final_width = W // (2 ** (self.num_scales - 1))
        
        # Sample from prior
        z = torch.randn(num_samples, final_channels, final_height, final_width).to(device)
        
        # Generate samples
        with torch.no_grad():
            samples, _ = self.forward(z, reverse=True)
        
        return samples


class SqueezeLayer(nn.Module):
    """Wrapper for squeeze operation to fit in nn.ModuleList."""
    
    def forward(self, x, logdet=0, reverse=False):
        if not reverse:
            return squeeze_operation(x), logdet
        else:
            return unsqueeze_operation(x), logdet


# CHANGELOG
"""
v1.0 (2025-01-20):
- Multi-scale architecture with alternating masks
- Checkerboard masking (pre-squeeze)
- Channel-wise masking (post-squeeze)
- Squeeze/unsqueeze operations between scales
- Sample generation via inverse pass
"""