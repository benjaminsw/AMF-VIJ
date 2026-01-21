# VERSION: RNV-v1.0
# FILE: coupling_layers.py
# PURPOSE: Affine coupling layers with checkerboard and channel-wise masking
# DATE: 2025-01-20

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer as described in RealNVP (Dinh et al., 2017).
    
    Forward: y = b⊙x + (1-b)⊙(x⊙exp(s(b⊙x)) + t(b⊙x))
    Inverse: x = b⊙y + (1-b)⊙((y-t(b⊙y))⊙exp(-s(b⊙y)))
    
    where b is a binary mask.
    """
    
    def __init__(self, mask, in_channels, hidden_channels, num_blocks=8):
        """
        Args:
            mask: Binary mask tensor (same shape as input)
            in_channels: Number of input channels
            hidden_channels: Number of hidden feature maps (e.g., 64)
            num_blocks: Number of residual blocks
        """
        super().__init__()
        self.register_buffer('mask', mask)
        
        # s and t are ResNets
        self.s_net = ResNet(in_channels, hidden_channels, num_blocks)
        self.t_net = ResNet(in_channels, hidden_channels, num_blocks)
        
    def forward(self, x, logdet=0, reverse=False):
        """
        Args:
            x: Input tensor [B, C, H, W]
            logdet: Log determinant accumulator
            reverse: If True, compute inverse
            
        Returns:
            y: Output tensor
            logdet: Updated log determinant
        """
        if not reverse:
            # Forward: x -> y
            x_masked = x * self.mask
            s = self.s_net(x_masked)
            t = self.t_net(x_masked)
            
            # Stabilize scale with tanh
            s = torch.tanh(s)  # Constrain to [-1, 1]
            
            y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
            
            # Log determinant: sum of log scales
            logdet = logdet + ((1 - self.mask) * s).sum(dim=[1, 2, 3])
            
            return y, logdet
        else:
            # Inverse: y -> x
            y_masked = x * self.mask
            s = self.s_net(y_masked)
            t = self.t_net(y_masked)
            
            s = torch.tanh(s)
            
            x = y_masked + (1 - self.mask) * ((x - t) * torch.exp(-s))
            
            # Log determinant is same (just negative for inverse)
            logdet = logdet - ((1 - self.mask) * s).sum(dim=[1, 2, 3])
            
            return x, logdet


class ResNet(nn.Module):
    """
    ResNet for scale (s) and translation (t) functions.
    Uses residual blocks with batch normalization.
    """
    
    def __init__(self, in_channels, hidden_channels, num_blocks):
        super().__init__()
        
        # Initial conv
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])
        
        # Output conv (back to input channels)
        self.output_conv = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        
        # Initialize output to zero (identity at initialization)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
        
    def forward(self, x):
        h = F.relu(self.input_conv(x))
        for block in self.blocks:
            h = block(h)
        return self.output_conv(h)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return x + h  # Residual connection


def create_checkerboard_mask(height, width, reverse=False):
    """
    Create checkerboard binary mask.
    
    Args:
        height, width: Spatial dimensions
        reverse: If True, flip the mask pattern
        
    Returns:
        mask: Binary tensor of shape [1, 1, H, W]
    """
    mask = torch.zeros(1, 1, height, width)
    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == 0:
                mask[0, 0, i, j] = 1
    
    if reverse:
        mask = 1 - mask
        
    return mask


def create_channel_mask(num_channels, reverse=False):
    """
    Create channel-wise binary mask (first half vs second half).
    
    Args:
        num_channels: Number of channels
        reverse: If True, flip the mask pattern
        
    Returns:
        mask: Binary tensor of shape [1, C, 1, 1]
    """
    mask = torch.zeros(1, num_channels, 1, 1)
    mask[0, :num_channels // 2, 0, 0] = 1
    
    if reverse:
        mask = 1 - mask
        
    return mask


def squeeze_operation(x):
    """
    Squeeze operation: s×s×c -> (s/2)×(s/2)×(4c).
    
    Args:
        x: Input tensor [B, C, H, W]
        
    Returns:
        Tensor [B, 4C, H/2, W/2]
    """
    B, C, H, W = x.shape
    
    # Reshape: [B, C, H, W] -> [B, C, H/2, 2, W/2, 2]
    x = x.view(B, C, H // 2, 2, W // 2, 2)
    
    # Permute: [B, C, H/2, 2, W/2, 2] -> [B, C, 2, 2, H/2, W/2]
    x = x.permute(0, 1, 3, 5, 2, 4)
    
    # Reshape: [B, C, 2, 2, H/2, W/2] -> [B, 4C, H/2, W/2]
    x = x.contiguous().view(B, C * 4, H // 2, W // 2)
    
    return x


def unsqueeze_operation(x):
    """
    Unsqueeze operation: (s/2)×(s/2)×(4c) -> s×s×c.
    Inverse of squeeze_operation.
    
    Args:
        x: Input tensor [B, 4C, H, W]
        
    Returns:
        Tensor [B, C, 2H, 2W]
    """
    B, C4, H, W = x.shape
    C = C4 // 4
    
    # Reshape: [B, 4C, H, W] -> [B, C, 2, 2, H, W]
    x = x.view(B, C, 2, 2, H, W)
    
    # Permute: [B, C, 2, 2, H, W] -> [B, C, H, 2, W, 2]
    x = x.permute(0, 1, 4, 2, 5, 3)
    
    # Reshape: [B, C, H, 2, W, 2] -> [B, C, 2H, 2W]
    x = x.contiguous().view(B, C, H * 2, W * 2)
    
    return x


# CHANGELOG
"""
v1.0 (2025-01-20):
- Initial implementation of affine coupling layers
- Checkerboard and channel-wise masking
- ResNet architecture for s/t functions
- Squeeze/unsqueeze operations
"""