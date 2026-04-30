"""
Exclusive Self Attention (XSA) implementation.

Based on arXiv:2603.09078, this module implements attention that excludes
self-components from the output, forcing each token to aggregate only
from other tokens in the sequence.

The key mathematical operation is projection removal:

.. math::

    y_i^{\\text{XSA}} = y_i - \\frac{y_i \\cdot v_i}{v_i \\cdot v_i + \\epsilon} v_i

where :math:`y_i` is the attention output for token :math:`i` and :math:`v_i`
is the value vector for that token.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from laker_xsa.config import XSA_LAKER_Config


class ExclusiveSelfAttention(nn.Module):
    """
    Exclusive Self Attention (XSA) module.

    XSA modifies standard self-attention by removing the component of each
    token's output that aligns with its own value vector. This forces the
    attention mechanism to aggregate information only from OTHER tokens.

    The exclusion is implemented via orthogonal projection:

    .. math::

        \\text{proj}_{v_i}(y_i) = \\frac{y_i \\cdot v_i}{v_i \\cdot v_i + \\epsilon} v_i

        y_i^{\\text{XSA}} = y_i - \\text{proj}_{v_i}(y_i)

    This formulation ensures that after XSA, the output :math:`y_i^{\\text{XSA}}`
    is orthogonal to the token's own value :math:`v_i` (up to numerical precision).

    Attributes:
        config: Configuration object containing hyperparameters.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        d_model: Total embedding dimension.
        scale: Scaling factor for attention scores.
        xsa_mode: Method for excluding self-attention.

    Input Shape:
        - Input: ``(batch, seq_len, d_model)``
        - Mask: ``(batch, seq_len, seq_len)`` or ``(batch, 1, seq_len, seq_len)``

    Output Shape:
        - Output: ``(batch, seq_len, d_model)``

    Example:
        >>> config = XSA_LAKER_Config(d_model=512, num_heads=8, xsa_mode="subtract_projection")
        >>> attn = ExclusiveSelfAttention(config)
        >>> x = torch.randn(2, 128, 512)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 128, 512])
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """
        Initialize Exclusive Self Attention.

        Args:
            config: Configuration object containing hyperparameters.
        """
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.xsa_mode = config.xsa_mode

        # Linear projections
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # Dropout
        self.dropout: Optional[nn.Dropout] = None
        if config.dropout > 0.0:
            self.dropout = nn.Dropout(config.dropout)

        # Learnable scale for projection subtraction (only for subtract_projection mode)
        if config.xsa_mode == "subtract_projection":
            self.xsa_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("xsa_scale", torch.ones(1))

    def _subtract_projection(
        self,
        attn_output: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Subtract the projection of attention output onto each token's own value.

        This implements the core XSA operation:

        .. math::

            y_i^{\\text{XSA}} = y_i - \\alpha \\cdot \\frac{y_i \\cdot v_i}{v_i \\cdot v_i + \\epsilon} v_i

        where :math:`\\alpha` is a learnable scale parameter.

        Args:
            attn_output: Attention output tensor of shape
                ``(batch, num_heads, seq_len, head_dim)``.
            v: Value tensor of shape
                ``(batch, num_heads, seq_len, head_dim)``.

        Returns:
            Exclusive output tensor with self-aligned components removed,
            same shape as input.
        """
        # Compute dot product along head dimension:
        # (batch, num_heads, seq_len, head_dim) * (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, 1)
        dot = (attn_output * v).sum(dim=-1, keepdim=True)

        # Compute squared norm of values:
        # (batch, num_heads, seq_len, head_dim) * (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, 1)
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + self.config.eps

        # Compute projection coefficient:
        # (batch, num_heads, seq_len, 1) / (batch, num_heads, seq_len, 1)
        # -> (batch, num_heads, seq_len, 1)
        coef = dot / v_norm_sq

        # Compute projected component:
        # (batch, num_heads, seq_len, 1) * (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        projected = coef * v

        # Subtract to get exclusive output (with learnable scale)
        exclusive = attn_output - self.xsa_scale * projected

        return exclusive

    def _zero_diagonal_attention(
        self,
        attn_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero out the diagonal of attention score matrix before softmax.

        This prevents direct self-attention by setting self-scores to -inf,
        which becomes 0 after softmax.

        Args:
            attn_scores: Attention scores of shape
                ``(batch, num_heads, seq_len, seq_len)``.

        Returns:
            Modified attention scores with zero diagonal.
        """
        _, _, _, seq_len = attn_scores.shape

        # Create diagonal mask: (seq_len, seq_len) with True on diagonal
        diag_mask = torch.eye(seq_len, device=attn_scores.device, dtype=torch.bool)

        # Set diagonal to -inf (will become 0 after softmax)
        attn_scores = attn_scores.masked_fill(diag_mask, float("-inf"))

        return attn_scores

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Exclusive Self Attention.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask of shape
                ``(batch, seq_len, seq_len)`` or ``(batch, 1, seq_len, seq_len)``.

        Returns:
            Exclusive attention output of shape ``(batch, seq_len, d_model)``.
            The output for each token is orthogonal to (or has minimal projection
            onto) that token's own value vector.
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape for multi-head attention:
        # (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply XSA modification based on mode
        if self.xsa_mode == "zero_diagonal":
            # Zero diagonal before softmax
            attn_scores = self._zero_diagonal_attention(attn_scores)
        elif self.xsa_mode == "mask":
            # Create explicit mask excluding self
            self_mask = torch.eye(seq_len, device=x.device, dtype=torch.bool)
            self_mask = self_mask.view(1, 1, seq_len, seq_len)
            if mask is None:
                mask = ~self_mask  # Invert: True where allowed
            else:
                mask = mask & ~self_mask

        # Apply external mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Apply XSA post-processing for subtract_projection mode
        if self.xsa_mode == "subtract_projection":
            out = self._subtract_projection(out, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.w_o(out)

        return out
