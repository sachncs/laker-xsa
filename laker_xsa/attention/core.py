"""Core abstractions and shared utilities for multi-head attention.

Provides:
  - Tensor reshape utilities for multi-head format conversion
  - Shared QKV projection module
  - Mask application with proper broadcasting
  - Abstract base class with template-method pattern
  - Input validation and logging
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, cast

import torch
from torch import nn

from laker_xsa.config import XSA_LAKER_Config

logger = logging.getLogger(__name__)

# Numerical bounds for clipping intermediate tensors
TENSOR_CLIP_ABS = 1e6


def reshape_to_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reshape from (batch, seq_len, d_model) to (batch, num_heads, seq_len, head_dim)."""
    batch, seq_len, _ = x.shape
    return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def reshape_from_heads(x: torch.Tensor) -> torch.Tensor:
    """Reshape from (batch, num_heads, seq_len, head_dim) to (batch, seq_len, d_model)."""
    batch, _, seq_len, _ = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, -1)


def broadcast_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ensure mask has compatible shape with attention scores.

    Expands a (batch, seq_len, seq_len) mask to (batch, 1, seq_len, seq_len)
    if the target has a head dimension.
    """
    if mask.dim() == 3 and target.dim() == 4:
        return mask.unsqueeze(1)
    return mask


def apply_mask(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    mask_fill_value: float = float("-inf"),
) -> torch.Tensor:
    """Apply an attention mask to scores, filling masked positions.

    Args:
        scores: (batch, num_heads, seq_len, seq_len).
        mask: (batch, seq_len, seq_len) or broadcastable shape.
            Positions where mask == 0 are filled with mask_fill_value.
        mask_fill_value: Value for masked positions (default -inf for softmax).

    Returns:
        Masked scores, same shape as input.
    """
    if mask is None:
        return scores

    mask_expanded = broadcast_mask(mask, scores)
    return scores.masked_fill(mask_expanded == 0, mask_fill_value)


def stable_clip(tensor: torch.Tensor, bound: float = TENSOR_CLIP_ABS) -> torch.Tensor:
    """Clip tensor values for numerical stability during iterative solves."""
    return torch.clamp(tensor, -bound, bound)


class QKVProjection(nn.Module):
    """Shared Q, K, V linear projections for multi-head attention.

    Wraps three independent linear layers (no bias, per standard practice).
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        super().__init__()
        d_model = config.d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input to Q, K, V tensors."""
        return self.w_q(x), self.w_k(x), self.w_v(x)


class BaseMultiHeadAttention(nn.Module, ABC):
    """Abstract base for multi-head attention with template-method pattern.

    Subclasses implement compute_attention(), receiving pre-projected
    and reshaped Q, K, V tensors in (batch, num_heads, seq_len, head_dim) format.
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = cast(int, config.head_dim)
        self.d_model = config.d_model

        self.qkv_proj = QKVProjection(config)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout: Optional[nn.Dropout] = None
        if config.dropout > 0.0:
            self.dropout = nn.Dropout(config.dropout)

    def validate_input(self, x: torch.Tensor) -> None:
        """Check input tensor shape and finiteness."""
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, d_model), got shape {x.shape}"
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Input dim {x.shape[-1]} != d_model {self.d_model}")
        if not torch.isfinite(x).all():
            logger.warning("Non-finite values detected in attention input; clamping.")
            x.clamp_(-TENSOR_CLIP_ABS, TENSOR_CLIP_ABS)

    @abstractmethod
    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute multi-head attention from already-projected Q, K, V.

        All tensors have shape (batch, num_heads, seq_len, head_dim).

        Returns:
            Attention output in (batch, num_heads, seq_len, head_dim).
        """

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass: project, compute attention, and output projection."""
        self.validate_input(x)

        q_raw, k_raw, v_raw = self.qkv_proj(x)

        q = reshape_to_heads(q_raw, self.num_heads, self.head_dim)
        k = reshape_to_heads(k_raw, self.num_heads, self.head_dim)
        v = reshape_to_heads(v_raw, self.num_heads, self.head_dim)

        out_heads = self.compute_attention(q, k, v, mask)

        out = reshape_from_heads(out_heads)

        return cast(torch.Tensor, self.w_o(out))
