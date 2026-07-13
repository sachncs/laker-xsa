"""Attention module for LAKER-XSA.

Public surface for the attention subpackage. Provides:

* ``StandardMultiHeadAttention`` — Vaswani-style scaled dot-product baseline
  used as a comparison reference.
* ``ExclusiveSelfAttention`` — XSA, which removes self-aligned components from
  attention outputs.
* ``LakerAttention`` — v2 module combining XSA-related diagonal/projection
  transformations with preconditioned kernel-regression inverse mixing.
* ``LakerAttentionLayer`` — thin ``nn.Module`` wrapper used to embed
  ``LakerAttention`` inside Transformer blocks.

Shared utilities (``BaseMultiHeadAttention``, ``QKVProjection``, mask helpers,
``reshape_*``) live in :mod:`laker_xsa.attention.core`; kernel implementations
live in :mod:`laker_xsa.attention.kernels` /
:mod:`laker_xsa.attention.functional`.

Backward compatibility:
    The legacy module paths ``attention_kernel``, ``standard_attention``,
    ``xsa_attention``, ``kernel_attention`` and ``fused_attention_v2`` remain
    importable. Each shim re-exports its canonical current or deprecated target;
    it does not implement an additional attention algorithm.

Deprecation:
    The v1 classes ``KernelFunction``, ``LearnedPreconditioner``,
    ``KernelAttentionRegression`` and ``FusedXSALAKERAttention`` (re-exported
    from :mod:`laker_xsa.attention._legacy`) emit ``DeprecationWarning`` at
    construction time and remain available for existing callers, benchmarks,
    and checkpoints. New code should use ``LakerAttention``.
"""

from __future__ import annotations

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
from laker_xsa.attention.kernels import AttentionKernel
from laker_xsa.attention.functional import compute_kernel_matrix
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
