"""Exclusive Self Attention (XSA).

Based on the XSA reference already cited by the repository
(``arXiv:2603.09078``). Depending on ``xsa_mode``, this implementation either
subtracts a regularized projection onto the corresponding value vector, masks
the score diagonal, or does both.

Three exclusion strategies are supported (selected via
``XSA_LAKER_Config.xsa_mode``):

* ``"subtract_projection"`` (default) — softmax attention is run normally,
  then a learnably scaled, epsilon-regularized projection of the output onto
  the corresponding value vector is subtracted.
* ``"zero_diagonal"`` — the attention score diagonal is masked to ``-inf``
  before softmax. This yields zero self-weight if another finite score remains
  in the row; a fully excluded row yields NaNs.
* ``"mask"`` — combines an explicit diagonal-excluding boolean mask with
  projection subtraction using a fixed scale of one.

The attention computation itself remains
``Attention(Q, K, V) = softmax(Q K^T / sqrt(head_dim)) V`` in the
``zero_diagonal`` and ``mask`` modes; only the scoring / weighting and output
post-processing depend on the configured mode.
"""

from __future__ import annotations

import math
from typing import Optional, Protocol

import torch
from torch import nn
import torch.nn.functional as F

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.core import BaseMultiHeadAttention, apply_mask


class XSAStrategy(Protocol):
    """Protocol for XSA exclusion strategies.

    A strategy takes the post-softmax weighted values (already incorporating
    any explicit mask and diagonal zeroing handled by
    :meth:`ExclusiveSelfAttention.compute_attention`) and optionally cleans
    the output to remove self-aligned components.
    """

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
            scores: Pre-softmax attention scores of shape
                ``(batch, num_heads, seq_len, seq_len)``.
            q: Queries ``(batch, num_heads, seq_len, head_dim)``.
            k: Keys ``(batch, num_heads, seq_len, head_dim)``.
            v: Values ``(batch, num_heads, seq_len, head_dim)``.
            mask: External mask broadcastable to scores, or ``None``.
            attn_output: Post-softmax weighted values of shape
                ``(batch, num_heads, seq_len, head_dim)``.

        Returns:
            XSA-modified attention output of shape
            ``(batch, num_heads, seq_len, head_dim)``.
        """
        ...  # pylint: disable=unnecessary-ellipsis


class XSAProjectionRemoval:
    """Strategy: remove the projection of the output onto each token's value.

    Implements (per token ``i``)

        y_i^xsa = y_i - alpha * (y_i · v_i) / (v_i · v_i + eps) * v_i

    where ``alpha`` is the scale (``xsa_scale``). The added ``eps`` makes this
    a regularized projection subtraction rather than exact orthogonal
    projection; it can also alter components contributed by other tokens that
    happen to align with ``v_i``.
    """

    def __init__(self, scale: nn.Parameter, eps: float) -> None:
        """Store the scale parameter and stability epsilon.

        Args:
            scale: ``nn.Parameter`` of shape ``(1,)`` (the ``xsa_scale``)
                controlling the projection subtraction. A value of ``1``
                applies the full regularized subtraction; other values scale
                it and are not constrained to any range.
            eps: Stability term added inside the ``v``-norm squared
                denominator to avoid division by zero when a value vector
                is (near) zero.
        """
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
        """Apply projection removal to ``attn_output``.

        Args:
            scores, q, k, mask: Unused; accepted to fit the
                :class:`XSAStrategy` protocol.
            v: Per-token value vectors used as projection directions.
            attn_output: Output of softmax-weighted values of shape
                ``(batch, num_heads, seq_len, head_dim)``.

        Returns:
            Result of the scaled, epsilon-regularized subtraction. Unless
            ``epsilon == 0`` and ``scale == 1``, this need not be exactly
            orthogonal to ``v``.
        """
        dot = (attn_output * v).sum(dim=-1, keepdim=True)
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + self.epsilon
        coef = dot / v_norm_sq
        return attn_output - self.scale_param * coef * v


class XSAZeroDiagonal:
    """Strategy complementing :attr:`ExclusiveSelfAttention.uses_diagonal_zeroing`.

    When ``xsa_mode == "zero_diagonal"``, the score diagonal has already been
    filled with ``-inf`` before softmax in
    :meth:`ExclusiveSelfAttention.compute_attention`. Its softmax weight is
    zero when the row contains at least one finite score; a fully excluded
    row instead produces NaNs. This strategy performs no further
    post-processing.
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
        """Return ``attn_output`` unchanged.

        Args:
            scores, q, k, v, mask: Unused; present to satisfy the
                :class:`XSAStrategy` protocol.
            attn_output: Output of softmax-weighted values.

        Returns:
            ``attn_output`` verbatim.
        """
        return attn_output


def build_xsa_strategy(mode: str, scale: nn.Parameter, eps: float) -> XSAStrategy:
    """Build the XSA exclusion strategy matching ``mode``.

    Args:
        mode: One of ``"subtract_projection"``, ``"zero_diagonal"`` or
            ``"mask"`` (matches :class:`XSA_LAKER_Config.xsa_mode`).
        scale: Tensor controlling projection-subtraction strength. It is a
            trainable ``nn.Parameter`` in ``"subtract_projection"`` mode, a
            fixed buffer in ``"mask"`` mode, and ignored in
            ``"zero_diagonal"`` mode.
        eps: Numerical stability term used in the projection-removal
            denominator.

    Returns:
        An :class:`XSAStrategy` instance.

    Raises:
        ValueError: If ``mode`` is not one of the three recognised values.
    """
    if mode == "subtract_projection":
        return XSAProjectionRemoval(scale, eps)
    if mode == "zero_diagonal":
        return XSAZeroDiagonal()
    if mode == "mask":
        return XSAProjectionRemoval(scale, eps)  # Mask mode uses same output cleaning
    raise ValueError(f"Unknown xsa_mode: {mode}")


class ExclusiveSelfAttention(BaseMultiHeadAttention):
    """Exclusive Self Attention (XSA) module.

    Applies one of the configured score/output transformations. The Q/K/V/output
    projections and head reshape are inherited from
    :class:`~laker_xsa.attention.core.BaseMultiHeadAttention`; only the
    per-head attention computation differs from the standard variant.

    Example:
        >>> config = XSA_LAKER_Config(d_model=512, num_heads=8, xsa_mode="subtract_projection")
        >>> attn = ExclusiveSelfAttention(config)
        >>> x = torch.randn(2, 128, 512)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 128, 512])
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """Initialise the XSA attention module.

        Args:
            config: :class:`laker_xsa.config.XSA_LAKER_Config`. The
                ``xsa_mode`` field selects the exclusion strategy; the
                ``eps`` field is forwarded to the projection-removal
                strategy when relevant.

        Side Effects:
            Allocates the Q/K/V/output projection layers and
            optional dropout via the base class, builds the selected
            :class:`XSAStrategy`, and allocates either a trainable
            ``xsa_scale`` parameter (``xsa_mode ==
            "subtract_projection"``) or a non-trainable ``xsa_scale``
            buffer of ones (other modes, kept for state-dict
            compatibility).
        """
        super().__init__(config)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.xsa_mode = config.xsa_mode

        if config.xsa_mode == "subtract_projection":
            self.xsa_scale = nn.Parameter(torch.ones(1))
        else:
            # ``xsa_scale`` is trainable only for ``subtract_projection``.
            # ``mask`` mode still uses this value as a fixed scale of one;
            # ``zero_diagonal`` ignores it.
            self.register_buffer("xsa_scale", torch.ones(1))

        self.strategy = build_xsa_strategy(config.xsa_mode, self.xsa_scale, config.eps)
        self.uses_diagonal_zeroing = config.xsa_mode == "zero_diagonal"

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute attention with XSA self-exclusion.

        Args:
            q: Queries ``(batch, num_heads, seq_len, head_dim)``.
            k: Keys ``(batch, num_heads, seq_len, head_dim)``.
            v: Values ``(batch, num_heads, seq_len, head_dim)``.
            mask: Optional attention mask. Outside ``"mask"`` mode it follows
                :func:`apply_mask` broadcasting semantics. In ``"mask"`` mode
                it is combined with a 4-D boolean self-mask before
                :func:`apply_mask`; a 4-D boolean mask is the unambiguous input
                shape. Fully excluded rows produce NaNs under softmax.

        Returns:
            XSA-modified attention output of shape
            ``(batch, num_heads, seq_len, head_dim)``.

        Raises:
            RuntimeError: Propagated from :func:`torch.matmul`,
                :func:`torch.nn.functional.softmax`, or
                :func:`apply_mask` for incompatible shapes, dtypes,
                or devices.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.uses_diagonal_zeroing:
            n = scores.shape[-1]
            diag = torch.eye(n, device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(diag, float("-inf"))

        if self.xsa_mode == "mask":
            n = scores.shape[-1]
            self_mask = torch.eye(n, device=scores.device, dtype=torch.bool).view(
                1, 1, n, n
            )
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
