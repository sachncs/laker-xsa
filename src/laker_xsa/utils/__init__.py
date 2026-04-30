"""
Utility functions for LAKER-XSA.

This package provides tensor operations, stability utilities, and
random seed management.
"""

from laker_xsa.utils.tensor_ops import (
    create_causal_mask,
    create_padding_mask,
    verify_tensor_shapes,
)
from laker_xsa.utils.stability import check_finite, clamp_tensor
from laker_xsa.utils.seed import set_seed, get_rng_states

__all__ = [
    "create_causal_mask",
    "create_padding_mask",
    "verify_tensor_shapes",
    "check_finite",
    "clamp_tensor",
    "set_seed",
    "get_rng_states",
]
