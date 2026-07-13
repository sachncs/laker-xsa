"""Utility functions for masks, validation, stability, and RNG state.

Tensor and stability helpers retain no module-local state. The seeding helpers
are intentionally stateful at the process level: they seed global Python,
NumPy, PyTorch, and optional CUDA generators and can change cuDNN settings.
"""

from __future__ import annotations

from laker_xsa.utils.tensor_ops import (
    create_causal_mask,
    create_padding_mask,
    verify_tensor_shapes,
)
from laker_xsa.utils.stability import check_finite, clamp_tensor
from laker_xsa.utils.seed import set_seed, get_rng_states, set_rng_states

__all__ = [
    "create_causal_mask",
    "create_padding_mask",
    "verify_tensor_shapes",
    "check_finite",
    "clamp_tensor",
    "set_seed",
    "get_rng_states",
    "set_rng_states",
]
