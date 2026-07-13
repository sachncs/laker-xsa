"""Core abstractions and shared utilities for multi-head attention.

Provides:

* :func:`reshape_to_heads` / :func:`reshape_from_heads` — convert between the
  ``(batch, seq_len, d_model)`` layout produced by Linear projections and the
  ``(batch, num_heads, seq_len, head_dim)`` layout expected by attention math.
* :class:`QKVProjection` — three independent bias-free linear projections used
  by subclasses of :class:`BaseMultiHeadAttention`.
* :func:`apply_mask` / :func:`broadcast_mask` — masked filling and limited
  3-D-to-4-D mask expansion.
* :func:`stable_clip` — symmetric value clamp used to keep iterative solves
  bounded.
* :class:`BaseMultiHeadAttention` — abstract base implementing the projection
  boilerplate via the template-method pattern; subclasses only override
  :meth:`~BaseMultiHeadAttention.compute_attention`.
* Input validation in :class:`BaseMultiHeadAttention` checks rank, width, and
  non-finite values before projection.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, cast

import torch
from torch import nn

from laker_xsa.config import XSA_LAKER_Config

logger = logging.getLogger(__name__)

# Clamp bound used by ``stable_clip`` and non-finite input handling.
TENSOR_CLIP_ABS = 1e6


def reshape_to_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reshape a projected tensor into per-head layout.

    Args:
        x: Tensor of shape ``(batch, seq_len, d_model)``. The last dimension
            must equal ``num_heads * head_dim``.
        num_heads: Number of attention heads.
        head_dim: Per-head feature dimension.

    Returns:
        Tensor of shape ``(batch, num_heads, seq_len, head_dim)`` suitable for
        batched attention matmuls. The view + transpose is non-contiguous; do
        not assume the returned tensor is contiguous in memory.

    Raises:
        ValueError: If ``x`` is not three-dimensional and its shape cannot be
            unpacked as ``(batch, seq_len, width)``.
        RuntimeError: If ``view`` cannot reshape the tensor to the requested
            number and width of heads.
    """
    batch, seq_len, _ = x.shape
    return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def reshape_from_heads(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`reshape_to_heads`; merges heads back into ``d_model``.

    Args:
        x: Tensor of shape ``(batch, num_heads, seq_len, head_dim)``.

    Returns:
        Contiguous tensor of shape ``(batch, seq_len, d_model)`` where
        ``d_model = num_heads * head_dim``. The transpose + ``contiguous()``
        call triggers a memory copy which is safe under autograd but not free.

    Raises:
        ValueError: If ``x`` is not four-dimensional and its shape cannot be
            unpacked as per-head layout.
        RuntimeError: If the final ``view`` cannot merge the head dimensions.
    """
    batch, _, seq_len, _ = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, -1)


def broadcast_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ensure a 3-D mask broadcasts against an attention-score tensor.

    Args:
        mask: ``(batch, seq_len, seq_len)`` mask (1 means keep, 0 means fill)
            or any tensor already compatible with ``target``.
        target: Score tensor of shape ``(batch, num_heads, seq_len, seq_len)``
            (or another rank) used to decide whether to insert a head axis.

    Returns:
        Mask with an inserted singleton at position 1 when ``mask.dim() == 3``
        and ``target.dim() == 4``; otherwise ``mask`` is returned unchanged.
    """
    if mask.dim() == 3 and target.dim() == 4:
        return mask.unsqueeze(1)
    return mask


def apply_mask(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    mask_fill_value: float = float("-inf"),
) -> torch.Tensor:
    """Replace positions where ``mask == 0`` in a score tensor.

    The default fill value of ``-inf`` is intended for scores passed to
    softmax. This is a replacement operation, not addition of a numeric mask.

    Args:
        scores: ``(batch, num_heads, seq_len, seq_len)`` raw or scaled scores.
        mask: ``(batch, seq_len, seq_len)`` or already-broadcastable mask,
            or ``None`` to return scores unchanged.
        mask_fill_value: Value inserted at masked positions.

    Returns:
        Masked scores with the same shape and dtype as ``scores``.

    Raises:
        RuntimeError: Propagated from
            :func:`torch.Tensor.masked_fill` for incompatible shapes
            or dtypes.
    """
    if mask is None:
        return scores

    mask_expanded = broadcast_mask(mask, scores)
    return scores.masked_fill(mask_expanded == 0, mask_fill_value)


def stable_clip(tensor: torch.Tensor, bound: float = TENSOR_CLIP_ABS) -> torch.Tensor:
    """Symmetric value clamp used inside iterative solves.

    Clamps each entry to ``[-bound, bound]``. NaNs remain NaN under
    ``torch.clamp``.

    Args:
        tensor: Tensor of arbitrary shape (typically ``alpha`` or residual).
        bound: Absolute value beyond which entries are clipped.

    Returns:
        A clamped tensor with the same shape and dtype as the input. The
        operation does not modify ``tensor`` in-place.

    Raises:
        RuntimeError: Propagated from :func:`torch.clamp` for
            incompatible dtypes (e.g. non-floating tensors).
    """
    return torch.clamp(tensor, -bound, bound)


class QKVProjection(nn.Module):
    """Shared Q, K, V linear projections for multi-head attention.

    Wraps three independent ``nn.Linear`` layers with ``bias=False``. Weights
    are managed by PyTorch's parameter system and can be initialised externally
    (``LakerAttention.init_weights`` does so).
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """Initialise the Q, K, V linear projections.

        Args:
            config: :class:`laker_xsa.config.XSA_LAKER_Config` whose
                ``d_model`` field drives every projection width.

        Side Effects:
            Allocates three independent
            :class:`torch.nn.Linear` layers (``w_q``, ``w_k``,
            ``w_v``), each ``d_model -> d_model`` with ``bias=False``.
            Initial weights follow PyTorch's default initialization;
            external modules (e.g. :meth:`LakerAttention.init_weights`)
            may reinitialise them.
        """
        super().__init__()
        d_model = config.d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input to Q, K, V tensors.

        Args:
            x: Token embeddings of shape ``(batch, seq_len, d_model)``.

        Returns:
            ``(q, k, v)`` each of shape ``(batch, seq_len, d_model)``; reshape
            into per-head layout with :func:`reshape_to_heads` before the
            attention op.

        Raises:
            RuntimeError: Propagated from :class:`torch.nn.Linear` for
                incompatible shapes, dtypes, or devices.
        """
        return self.w_q(x), self.w_k(x), self.w_v(x)


class BaseMultiHeadAttention(nn.Module, ABC):
    """Abstract base for multi-head attention with template-method pattern.

    Centralises Q/K/V projection, dropout configuration, the output Linear and
    input validation so subclasses can focus solely on the per-head attention
    computation. Subclasses must implement :meth:`compute_attention`, which
    receives the pre-projected, per-head Q/K/V tensors and returns the per-head
    output.

    The forward signature is fixed at ``forward(x, mask=None)`` and is shared
    by every attention module in this subpackage, which allows them to be used
    interchangeably inside Transformer blocks.

    Attributes:
        config: ``XSA_LAKER_Config`` driving projection widths and dropout.
        num_heads: Number of attention heads.
        head_dim: Per-head feature dimension.
        d_model: Configured input/output width. A conflicting explicit
            ``head_dim`` is not validated here and can cause reshape failure.
        qkv_proj: Module producing ``(q, k, v)`` of shape
            ``(batch, seq_len, d_model)``.
        w_o: Output projection applied after head merging.
        dropout: Optional ``nn.Dropout`` applied inside
            :meth:`compute_attention` for variants that use softmax weights.
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """Initialise the base multi-head attention.

        Args:
            config: :class:`laker_xsa.config.XSA_LAKER_Config` consumed
                by the base class. ``head_dim`` is read via
                :func:`cast` (assumed already validated by
                :meth:`XSA_LAKER_Config.__post_init__`).

        Side Effects:
            Allocates :class:`QKVProjection`, the output linear
            :attr:`w_o`, and (when ``config.dropout > 0.0``) an
            :class:`torch.nn.Dropout` stored on :attr:`dropout`.
        """
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
        """Check input rank and width and clamp infinities in place.

        Args:
            x: Candidate input tensor.

        Raises:
            ValueError: If ``x`` is not 3-D or its last dimension does not
                match ``self.d_model``.
            RuntimeError: If the in-place non-finite clamp is rejected, for
                example because ``x`` is a leaf tensor requiring gradients.

        Side Effects:
            When non-finite entries are detected, logs a warning and attempts
            an in-place clamp to ``[-TENSOR_CLIP_ABS, TENSOR_CLIP_ABS]``.
            Infinities become finite bounds; NaNs remain NaN.
        """
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

        Args:
            q: Queries of shape ``(batch, num_heads, seq_len, head_dim)``.
            k: Keys of shape ``(batch, num_heads, seq_len, head_dim)``.
            v: Values of shape ``(batch, num_heads, seq_len, head_dim)``.
            mask: Optional mask broadcastable to
                ``(batch, num_heads, seq_len, seq_len)``; ``None`` if no
                masking should be applied.

        Returns:
            Per-head attention output of shape
            ``(batch, num_heads, seq_len, head_dim)``; the base class merges
            heads and applies ``w_o`` afterward.
        """

    def forward(  # noqa: D401
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Project, compute attention, merge heads and apply output projection.

        Args:
            x: Token embeddings of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask; semantics depend on the subclass.

        Returns:
            Tensor of shape ``(batch, seq_len, d_model)`` ready to feed into
            the next Transformer sublayer.

        Raises:
            ValueError: Propagated from :meth:`validate_input` when
                ``x`` is not 3-D or its width does not match
                ``self.d_model``.
            RuntimeError: Propagated from :meth:`compute_attention`
                or the downstream matmul/resize operations for
                incompatible shapes, dtypes, or devices.
        """
        self.validate_input(x)

        q_raw, k_raw, v_raw = self.qkv_proj(x)

        q = reshape_to_heads(q_raw, self.num_heads, self.head_dim)
        k = reshape_to_heads(k_raw, self.num_heads, self.head_dim)
        v = reshape_to_heads(v_raw, self.num_heads, self.head_dim)

        out_heads = self.compute_attention(q, k, v, mask)

        out = reshape_from_heads(out_heads)

        return cast(torch.Tensor, self.w_o(out))
