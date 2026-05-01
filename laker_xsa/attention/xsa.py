"""Exclusive Self Attention (XSA).

Based on arXiv:2603.09078. XSA modifies attention to exclude self-components
from the output, forcing each token to aggregate only from other tokens.

Three exclusion strategies are supported:
  - subtract_projection: Remove self-value projection from output (default)
  - zero_diagonal: Zero the attention score diagonal before softmax
  - mask: Explicit boolean mask excluding self-positions
"""

from __future__ import annotations

import math
from typing import Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.core import BaseMultiHeadAttention, apply_mask


class XSAStrategy(Protocol):
    """Protocol for XSA exclusion strategies."""

    def __call__(
        self,
        scores: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply XSA exclusion to attention output.

        Args:
            scores: Pre-softmax attention scores (batch, num_heads, seq, seq).
            q, k, v: QKV tensors (batch, num_heads, seq, head_dim).
            mask: External mask or None.
            attn_output: Post-softmax weighted values (batch, num_heads, seq, head_dim).

        Returns:
            XSA-modified output (batch, num_heads, seq, head_dim).
        """
        ...


class XSAProjectionRemoval:
    """Remove the projection of attention output onto each token's own value vector.

    y_i^xsa = y_i - alpha * (y_i · v_i) / (v_i · v_i + eps) * v_i
    """

    def __init__(self, scale: nn.Parameter, eps: float) -> None:
        self.scale_param = scale
        self.epsilon = eps

    def __call__(
        self,
        scores: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        attn_output: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        dot = (attn_output * v).sum(dim=-1, keepdim=True)
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + self.epsilon
        coef = dot / v_norm_sq
        return attn_output - self.scale_param * coef * v


class XSAZeroDiagonal:
    """Zero the attention score diagonal before softmax.

    Self-scores become -inf, yielding zero weight after softmax.
    """

    def __call__(
        self,
        scores: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        attn_output: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return attn_output  # Diagonal already zeroed in scores; no post-processing needed


def build_xsa_strategy(
    mode: str, scale: nn.Parameter, eps: float
) -> XSAStrategy:
    if mode == "subtract_projection":
        return XSAProjectionRemoval(scale, eps)
    if mode == "zero_diagonal":
        return XSAZeroDiagonal()
    if mode == "mask":
        return XSAProjectionRemoval(scale, eps)  # Mask mode uses same output cleaning
    raise ValueError(f"Unknown xsa_mode: {mode}")


class ExclusiveSelfAttention(BaseMultiHeadAttention):
    """Exclusive Self Attention (XSA) module.

    Forces cross-token-only reasoning by removing self-aligned components
    from attention outputs.

    Example:
        >>> config = XSA_LAKER_Config(d_model=512, num_heads=8, xsa_mode="subtract_projection")
        >>> attn = ExclusiveSelfAttention(config)
        >>> x = torch.randn(2, 128, 512)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 128, 512])
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        super().__init__(config)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.xsa_mode = config.xsa_mode

        if config.xsa_mode == "subtract_projection":
            self.xsa_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("xsa_scale", torch.ones(1))

        self.strategy = build_xsa_strategy(
            config.xsa_mode, self.xsa_scale, config.eps
        )
        self.uses_diagonal_zeroing = (config.xsa_mode == "zero_diagonal")

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.uses_diagonal_zeroing:
            n = scores.shape[-1]
            diag = torch.eye(n, device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(diag, float("-inf"))

        if self.xsa_mode == "mask":
            n = scores.shape[-1]
            self_mask = torch.eye(n, device=scores.device, dtype=torch.bool).view(1, 1, n, n)
            if mask is None:
                mask = ~self_mask
            else:
                mask = mask & ~self_mask

        scores = apply_mask(scores, mask)
        weights = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            weights = self.dropout(weights)

        attn_output = torch.matmul(weights, v)
        return self.strategy(scores, q, k, v, mask, attn_output)
