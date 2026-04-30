"""
Standard Multi-Head Self-Attention implementation.

This module provides the baseline scaled dot-product attention mechanism
as described in "Attention Is All You Need" (Vaswani et al., 2017).

This serves as a reference implementation for comparing against the
XSA and LAKER variants.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from laker_xsa.config import XSA_LAKER_Config


class StandardMultiHeadAttention(nn.Module):
    """
    Standard scaled dot-product multi-head attention.

    This implementation follows the original Transformer attention mechanism:

    .. math::

        \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V

    The multi-head variant projects queries, keys, and values through separate
    linear heads and concatenates the results.

    Attributes:
        config: Configuration object containing hyperparameters.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        d_model: Total embedding dimension.
        scale: Scaling factor for attention scores (1 / sqrt(head_dim)).

    Input Shape:
        - Input: ``(batch, seq_len, d_model)``
        - Mask: ``(batch, seq_len, seq_len)`` or ``(batch, 1, seq_len, seq_len)``

    Output Shape:
        - Output: ``(batch, seq_len, d_model)``

    Example:
        >>> config = XSA_LAKER_Config(d_model=512, num_heads=8)
        >>> attn = StandardMultiHeadAttention(config)
        >>> x = torch.randn(2, 128, 512)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 128, 512])
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """
        Initialize standard multi-head attention.

        Args:
            config: Configuration object containing hyperparameters.
        """
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections: each has shape [d_model, d_model]
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # Dropout layer (only created if dropout > 0)
        self.dropout: Optional[nn.Dropout] = None
        if config.dropout > 0.0:
            self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for standard multi-head attention.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask. If provided, should have shape
                ``(batch, seq_len, seq_len)`` or ``(batch, 1, seq_len, seq_len)``.
                Positions with value 0 will be masked out (set to -inf before softmax).

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.

        Raises:
            RuntimeError: If input dimensions are invalid.
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V: each has shape (batch, seq_len, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape for multi-head attention:
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention scores:
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax normalization over keys dimension
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout to attention weights
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values:
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        out = torch.matmul(attn_weights, v)

        # Reshape back to (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # Output projection: (batch, seq_len, d_model)
        out = self.w_o(out)

        return out
