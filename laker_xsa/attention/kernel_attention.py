"""Backward-compatibility shim for the ``kernel_attention`` module (v1).

Pre-v2 callers import the v1 LAKER kernel-attention classes from
``laker_xsa.attention.kernel_attention``. New code should use
:class:`~laker_xsa.attention.laker.LakerAttention` from
:mod:`laker_xsa.attention.laker` (or its compatibility alias
``FusedXSALAKERAttentionV2`` from
:mod:`laker_xsa.attention.fused_attention_v2`).

Each re-exported class still emits :class:`DeprecationWarning` from its
constructor — see :mod:`laker_xsa.attention._legacy` for the deprecation
boundary and a description of why v2 should be preferred.
"""

from __future__ import annotations

from laker_xsa.attention._legacy import (
    FusedXSALAKERAttention,
    KernelAttentionRegression,
    KernelFunction,
    LearnedPreconditioner,
)

__all__ = [
    "FusedXSALAKERAttention",
    "KernelAttentionRegression",
    "KernelFunction",
    "LearnedPreconditioner",
]
