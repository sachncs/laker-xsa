"""
Transformer block implementation with XSA + LAKER attention.

This module provides the complete Transformer block architecture including:
- Multi-head attention (standard, XSA, LAKER, or fused)
- Layer normalization
- Residual connections
- Feed-forward MLP
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention import (
    ExclusiveSelfAttention,
    LakerAttention,
    StandardMultiHeadAttention,
)
from laker_xsa.attention._legacy import (
    FusedXSALAKERAttention,
    KernelAttentionRegression,
)


class MLP(nn.Module):
    """
    Transformer feed-forward MLP block.

    Standard two-layer MLP with activation:

    .. math::

        \\text{MLP}(x) = W_2 \\cdot \\text{activation}(W_1 \\cdot x)

    Attributes:
        linear1: First linear layer (d_model -> d_ff).
        linear2: Second linear layer (d_ff -> d_model).
        dropout: Dropout layer (if dropout > 0).
        activation: Activation function (GELU or ReLU).

    Input Shape:
        - Input: ``(batch, seq_len, d_model)``

    Output Shape:
        - Output: ``(batch, seq_len, d_model)``
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        """
        Initialize MLP block.

        Args:
            d_model: Input/output dimension.
            d_ff: Hidden dimension.
            dropout: Dropout probability.
            activation: Activation function ('gelu' or 'relu').
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout: Optional[nn.Dropout] = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor, shape ``(batch, seq_len, d_model)``.

        Returns:
            Output tensor, shape ``(batch, seq_len, d_model)``.
        """
        x = self.linear1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear2(x)
        return x


class XSALAKERTransformerBlock(nn.Module):
    """
    Complete Transformer block with configurable attention.

    Architecture (pre-norm):

    .. code-block:: text

        x -> LayerNorm -> Attention -> Dropout -> x + attn_out
          -> LayerNorm -> MLP -> Dropout -> x + mlp_out

    The attention can be:
    - Standard multi-head self-attention
    - Exclusive Self Attention (XSA)
    - Kernel attention regression (LAKER)
    - Fused XSA + LAKER

    Attributes:
        config: Configuration object.
        norm1: Layer normalization before attention.
        norm2: Layer normalization before MLP.
        attention: Attention module (configurable).
        mlp: Feed-forward MLP block.
        dropout: Dropout for residual connections.

    Input Shape:
        - Input: ``(batch, seq_len, d_model)``
        - Mask: ``(batch, seq_len, seq_len)`` or ``(batch, 1, seq_len, seq_len)``

    Output Shape:
        - Output: ``(batch, seq_len, d_model)``
    """

    def __init__(
        self,
        config: XSA_LAKER_Config,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        activation: Literal["gelu", "relu"] = "gelu",
        attention_type: Literal["standard", "xsa", "kernel", "fused", "fused_v2"] = "fused_v2",
    ) -> None:
        """
        Initialize Transformer block.

        Args:
            config: Configuration object.
            d_ff: Feed-forward dimension. Defaults to 4 * d_model.
            dropout: Dropout probability.
            activation: MLP activation function.
            attention_type: Type of attention to use.
        """
        super().__init__()
        self.config = config

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.eps)

        # Select attention type
        if attention_type == "standard":
            self.attention = StandardMultiHeadAttention(config)
        elif attention_type == "xsa":
            self.attention = ExclusiveSelfAttention(config)
        elif attention_type == "kernel":
            self.attention = KernelAttentionRegression(config)
        elif attention_type == "fused":
            self.attention = FusedXSALAKERAttention(config)
        elif attention_type == "fused_v2":
            self.attention = LakerAttention(config)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # MLP
        d_ff = d_ff if d_ff is not None else config.d_model * 4
        self.mlp = MLP(config.d_model, d_ff, dropout, activation)

        # Dropout for residuals
        self.dropout: Optional[nn.Dropout] = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Transformer block.

        Args:
            x: Input tensor, shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask.

        Returns:
            Output tensor, shape ``(batch, seq_len, d_model)``.
        """
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, mask)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)
        x = x + attn_out  # Residual connection

        # Pre-norm MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        if self.dropout is not None:
            mlp_out = self.dropout(mlp_out)
        x = x + mlp_out  # Residual connection

        return x
