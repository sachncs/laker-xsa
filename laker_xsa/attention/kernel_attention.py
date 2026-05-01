from __future__ import annotations

"""Backward-compatibility shim for kernel_attention module.

Re-exports deprecated v1 classes from laker_xsa.attention._legacy.
"""

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
