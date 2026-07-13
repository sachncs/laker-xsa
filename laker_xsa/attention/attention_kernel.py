"""Backward-compatibility shim for the ``attention_kernel`` module.

Pre-v2 callers import the kernel from
``laker_xsa.attention.attention_kernel``; new code should import
:class:`AttentionKernel` and :func:`compute_kernel_matrix` directly from
:mod:`laker_xsa.attention.kernels` /
:mod:`laker_xsa.attention.functional`.

This shim only re-exports symbols; no logic lives here.
"""

from __future__ import annotations

from laker_xsa.attention.kernels import AttentionKernel, compute_kernel_matrix

__all__ = ["AttentionKernel", "compute_kernel_matrix"]
