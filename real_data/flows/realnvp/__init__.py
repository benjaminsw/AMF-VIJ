# VERSION: RNV-v1.0
# FILE: __init__.py
# PURPOSE: Package initializer for RealNVP module
# DATE: 2025-01-20

from .coupling_layers import (
    AffineCouplingLayer,
    create_checkerboard_mask,
    create_channel_mask,
    squeeze_operation,
    unsqueeze_operation
)
from .realnvp import RealNVP

__all__ = [
    'AffineCouplingLayer',
    'create_checkerboard_mask',
    'create_channel_mask',
    'squeeze_operation',
    'unsqueeze_operation',
    'RealNVP'
]

__version__ = '1.0.0'