"""Attention kernel functions for LAKER-XSA.

Provides kernel functions that map (Q, K) pairs to positive kernel matrices.
The primary kernel is the exponential attention kernel:

    K_{ij} = exp(cosine(q_i, k_j) / temperature)

where Q and K are L2-normalized before computing similarity scores.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionKernel(nn.Module):
    """Bounded exponential attention kernel for LAKER attention.

    Computes K_{ij} = exp(cosine(q_i, k_j) / temperature).

    Unlike standard attention which uses unnormalized QK^T/sqrt(d) with softmax,
    this kernel L2-normalizes Q and K first, producing cosine similarities
    in [-1, 1]. The exponential preserves the LAKER kernel structure while
    the bounded input ensures numerical stability during iterative solving.

    Attributes:
        head_dim: Dimension per attention head.
        symmetric: If True, symmetrize K = (K + K^T)/2 for PSD guarantee.
        normalize_qk: If True, L2-normalize Q/K before computing scores.
        eps: Numerical stability constant added to kernel output.
        temperature: Effective temperature (derived from log_temperature param).
    """

    def __init__(
        self,
        head_dim: int,
        temperature: float = 1.0,
        symmetric: bool = False,
        learnable_temperature: bool = True,
        normalize_qk: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.symmetric = symmetric
        self.normalize_qk = normalize_qk
        self.eps = eps

        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(math.log(temperature))
            )
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(math.log(temperature))
            )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=0.05, max=100.0)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute K_{ij} = exp(sim(q_i, k_j) / temperature).

        Args:
            q: Queries (batch, num_heads, seq_len, head_dim).
            k: Keys (batch, num_heads, seq_len, head_dim).

        Returns:
            Kernel matrix (batch, num_heads, seq_len, seq_len)
            with values in [exp(-1/T) + eps, exp(1/T) + eps].
        """
        temp = self.temperature

        if self.normalize_qk:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            scores = torch.matmul(q, k.transpose(-2, -1))
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        scores = scores / temp
        scores = torch.clamp(scores, -100.0, 100.0)
        kernel = torch.exp(scores)

        if self.symmetric:
            kernel = 0.5 * (kernel + kernel.transpose(-2, -1))

        return kernel + self.eps


from laker_xsa.attention.functional import compute_kernel_matrix  # noqa: F401 — re-export
