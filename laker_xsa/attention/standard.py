"""Standard Multi-Head Self-Attention.

Reference implementation of scaled dot-product attention as described in
"Attention Is All You Need" (Vaswani et al., 2017). Serves as the baseline
for comparing XSA and LAKER variants.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.core import BaseMultiHeadAttention, apply_mask


class StandardMultiHeadAttention(BaseMultiHeadAttention):
    """Standard scaled dot-product multi-head attention.

    Implements Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.

    Example:
        >>> config = XSA_LAKER_Config(d_model=512, num_heads=8)
        >>> attn = StandardMultiHeadAttention(config)
        >>> x = torch.randn(2, 128, 512)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 128, 512])
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        super().__init__(config)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = apply_mask(scores, mask)
        weights = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            weights = self.dropout(weights)
        return torch.matmul(weights, v)
