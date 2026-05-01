from __future__ import annotations

"""Attention module for LAKER-XSA.

Provides:
  - StandardMultiHeadAttention (baseline)
  - ExclusiveSelfAttention (XSA)
  - LakerAttention — breakthrough fused XSA+LAKER (v2)
  - LakerAttentionLayer — for Transformer block integration
  - Deprecated v1 classes (KernelAttentionRegression, FusedXSALAKERAttention)

Core utilities are in core.py; kernel functions in kernels.py.
"""

from laker_xsa.attention.core import (
    BaseMultiHeadAttention,
    QKVProjection,
    apply_mask,
    broadcast_mask,
    reshape_from_heads,
    reshape_to_heads,
    stable_clip,
)
from laker_xsa.attention.standard import StandardMultiHeadAttention
from laker_xsa.attention.xsa import ExclusiveSelfAttention
from laker_xsa.attention.kernels import AttentionKernel, compute_kernel_matrix
from laker_xsa.attention.laker import LakerAttention, LakerAttentionLayer
from laker_xsa.attention._legacy import (
    FusedXSALAKERAttention,
    KernelAttentionRegression,
    KernelFunction,
    LearnedPreconditioner,
)

__all__ = [
    # Core
    "BaseMultiHeadAttention",
    "QKVProjection",
    "apply_mask",
    "broadcast_mask",
    "reshape_from_heads",
    "reshape_to_heads",
    "stable_clip",
    # Kernels
    "AttentionKernel",
    "compute_kernel_matrix",
    # Modules
    "StandardMultiHeadAttention",
    "ExclusiveSelfAttention",
    "LakerAttention",
    "LakerAttentionLayer",
    # Deprecated (v1)
    "KernelAttentionRegression",
    "FusedXSALAKERAttention",
    "KernelFunction",
    "LearnedPreconditioner",
]
