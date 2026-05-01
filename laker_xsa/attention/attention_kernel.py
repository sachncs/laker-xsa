from __future__ import annotations

"""Backward-compatibility shim for attention_kernel module.

Re-exports from laker_xsa.attention.kernels.
"""

from laker_xsa.attention.kernels import AttentionKernel, compute_kernel_matrix

__all__ = ["AttentionKernel", "compute_kernel_matrix"]
