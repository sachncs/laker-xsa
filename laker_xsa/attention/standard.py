"""Standard Multi-Head Self-Attention.

Reference implementation of scaled dot-product attention as described in
"Attention Is All You Need" (Vaswani et al., 2017). Serves as the baseline
for comparing XSA and LAKER variants.

The math is the textbook form,
``Attention(Q, K, V) = softmax(Q K^T / sqrt(head_dim)) V``,
broadcast independently over heads and batches. Masked positions are
filled with ``-inf``; their softmax probability is zero when the row contains
at least one finite score, while a fully masked row produces NaNs. If
``dropout`` is configured, it is applied to the post-softmax weights.
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

    Implements
    ``Attention(Q, K, V) = softmax(Q K^T / sqrt(head_dim)) V``
    per head, with the QKV/Output projections and head reshape handled by the
    :class:`~laker_xsa.attention.core.BaseMultiHeadAttention` base class.

    Example:
        >>> config = XSA_LAKER_Config(d_model=512, num_heads=8)
        >>> attn = StandardMultiHeadAttention(config)
        >>> x = torch.randn(2, 128, 512)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 128, 512])
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """Initialise the standard scaled dot-product attention.

        Args:
            config: :class:`laker_xsa.config.XSA_LAKER_Config` consumed
                by the base class. Stores ``1 / sqrt(head_dim)`` as
                ``self.scale`` for the per-head score scaling.

        Side Effects:
            Allocates the Q/K/V/output projection layers and
            optional dropout via the base class.
        """
        super().__init__(config)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute scaled dot-product attention per head.

        Args:
            q: Queries ``(batch, num_heads, seq_len, head_dim)``.
            k: Keys ``(batch, num_heads, seq_len, head_dim)``.
            v: Values ``(batch, num_heads, seq_len, head_dim)``.
            mask: Optional mask broadcastable to
                ``(batch, num_heads, seq_len, seq_len)``; masked entries are
                filled with ``-inf`` before softmax.

        Returns:
            Weighted value tensor of shape
            ``(batch, num_heads, seq_len, head_dim)`` to be merged by the
            base class.

        Raises:
            RuntimeError: Propagated from :func:`torch.matmul`,
                :func:`torch.nn.functional.softmax`, or
                :func:`apply_mask` for incompatible shapes, dtypes,
                or devices.

        Numerical notes:
            A masked score is ``-inf`` and therefore has zero softmax weight
            when its row also contains a finite score. If every entry in a row
            is masked, softmax receives only ``-inf`` values and returns NaNs.
            Dropout, when configured, is applied after softmax, so retained row
            sums need not remain one.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = apply_mask(scores, mask)
        weights = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            weights = self.dropout(weights)
        return torch.matmul(weights, v)
