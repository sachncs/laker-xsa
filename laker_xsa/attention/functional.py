"""Functional API for attention kernels.

This module provides stateless functional counterparts to the class-based
Modules in the attention subpackage. These follow the same design as
torch.nn.functional relative to torch.nn: the functional forms accept all
parameters explicitly and do not hold learnable state.

Functional form            Class-based form
-----------------          -----------------
compute_kernel_matrix     AttentionKernel.forward
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_kernel_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature: float = 1.0,
    normalize_qk: bool = True,
    symmetric: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute attention kernel: K_{ij} = exp(sim(q_i, k_j) / temperature).

    Stateless functional counterpart to AttentionKernel.forward().

    Args:
        q: Queries (..., seq_len, head_dim).
        k: Keys (..., seq_len, head_dim).
        temperature: Temperature for attention sharpness control.
        normalize_qk: L2-normalize Q/K before computing similarity.
        symmetric: Symmetrize output kernel.
        eps: Numerical stability term.

    Returns:
        Kernel matrix (..., seq_len, seq_len).
    """
    if normalize_qk:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1))
    else:
        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    scores = scores / temperature
    scores = torch.clamp(scores, -100.0, 100.0)
    kernel = torch.exp(scores)

    if symmetric:
        kernel = 0.5 * (kernel + kernel.transpose(-2, -1))

    return kernel + eps
