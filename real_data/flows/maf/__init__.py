# VERSION: MAF-v1.0
# FILE: __init__.py
# PURPOSE: Package initializer for MAF module
# DATE: 2025-01-21

from .made import MADE
from .maf import MAF

__all__ = ['MADE', 'MAF']
__version__ = '1.0.0'

# CHANGELOG
"""
v1.0 (2025-01-21):
- Initial MAF implementation
- MADE with Gaussian conditionals and masking
- Batch normalization between layers
- Fixed reversed ordering strategy
- Numerical stability via alpha clamping
"""