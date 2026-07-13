"""Transformer block with XSA + LAKER attention.

The block uses the pre-norm residual pattern:

.. code-block:: text

    x = x + Dropout(Attention(LayerNorm1(x)))
    x = x + Dropout(MLP(LayerNorm2(x)))

Normalizing before each sublayer controls the scale presented to the selected
attention and feed-forward implementations. Attention type is fixed when the
block is constructed.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn
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
    """Position-wise feed-forward MLP.

    A two-layer, bias-free feed-forward network with a GELU or ReLU
    activation. Optional dropout is applied between the activation and second
    projection.

    .. math::

        \\text{MLP}(x) = W_2 \\cdot \\text{activation}(W_1 \\cdot x)

    Attributes:
        linear1: First linear layer mapping ``d_model`` to ``d_ff``.
        linear2: Second linear layer mapping ``d_ff`` back to
            ``d_model``. Both layers are bias-free.
        dropout: Optional dropout applied between the activation and
            the second projection. ``None`` when ``dropout == 0`` so
            the forward pass can skip the call entirely.
        activation: Callable applied to the hidden activations. Either
            :func:`torch.nn.functional.gelu` or
            :func:`torch.nn.functional.relu`, selected once at
            construction.

    Tensor Shapes:
        * Input: ``(..., d_model)``.
        * Hidden: ``(..., d_ff)``.
        * Output: ``(..., d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        """Initialize the MLP.

        Args:
            d_model: Input and output feature dimension.
            d_ff: Hidden feature dimension. Larger values increase
                capacity and compute roughly linearly.
            dropout: Dropout probability applied after the activation.
                ``0.0`` disables dropout and the corresponding module
                is not allocated.
            activation: ``"gelu"`` selects GELU; any other runtime string,
                including the annotated ``"relu"``, selects ReLU.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout: Optional[nn.Dropout] = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to the input.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of shape ``(batch, seq_len, d_model)`` after the
            two linear projections, activation, and (optional)
            dropout.
        """
        x = self.linear1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear2(x)
        return x


class XSALAKERTransformerBlock(nn.Module):
    """Single Transformer block with configurable attention.

    The block follows the pre-norm residual pattern:

    .. code-block:: text

        x = x + Dropout(Attention(LayerNorm1(x)))
        x = x + Dropout(MLP(LayerNorm2(x)))

    The pre-norm order is part of the architecture; the implementation does not
    claim or check that normalization improves the conditioning of a particular
    attention kernel.

    The attention module is chosen once at construction time and is
    *not* re-evaluated per forward pass. The supported modes are:

    * ``"standard"`` - :class:`StandardMultiHeadAttention` (the
      softmax-attention baseline used for ablations).
    * ``"xsa"`` - :class:`ExclusiveSelfAttention` (XSA only, no
      kernel solve).
    * ``"kernel"`` - :class:`KernelAttentionRegression` (the legacy
      LAKER path; kept for backward compatibility).
    * ``"fused"`` - :class:`FusedXSALAKERAttention` (the v1 fusion).
    * ``"fused_v2"`` (default) - :class:`LakerAttention`, using the
      configured v2 kernel, preconditioner mode, and PCG-style solve.

    The ``use_fused`` field on :class:`XSA_LAKER_Config` is not consulted;
    ``attention_type`` alone selects the module.

    Attributes:
        config: The shared :class:`XSA_LAKER_Config` instance. Stored
            on the block for introspection and for sub-modules that
            re-read it.
        norm1: LayerNorm applied to the block input before the
            attention sub-layer. Uses the config's ``eps``.
        norm2: LayerNorm applied to the block input before the MLP
            sub-layer.
        attention: The selected attention module. Concrete type
            depends on ``attention_type``; one of
            :class:`StandardMultiHeadAttention`,
            :class:`ExclusiveSelfAttention`,
            :class:`KernelAttentionRegression`,
            :class:`FusedXSALAKERAttention`, or
            :class:`LakerAttention`.
        mlp: The position-wise feed-forward network.
        dropout: Optional dropout applied to both the attention
            output and the MLP output before the residual addition.
            ``None`` when ``dropout == 0``.

    Tensor Shapes:
        * Input ``x``:  ``(batch, seq_len, d_model)``.
        * Input ``mask``: ``(batch, seq_len, seq_len)`` or
          ``(batch, 1, seq_len, seq_len)``; passed through to the
          attention module which interprets it.
        * Output: ``(batch, seq_len, d_model)``.
    """

    def __init__(
        self,
        config: XSA_LAKER_Config,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        activation: Literal["gelu", "relu"] = "gelu",
        attention_type: Literal[
            "standard", "xsa", "kernel", "fused", "fused_v2"
        ] = "fused_v2",
    ) -> None:
        """Initialize the Transformer block.

        Args:
            config: Shared configuration. ``d_model`` and ``eps`` are
                used to build the LayerNorms; the rest is forwarded
                to the attention module.
            d_ff: Hidden dimension of the MLP. ``None`` (the default)
                selects ``4 * d_model`` - the standard Transformer
                ratio. Block attention itself does not depend on
                ``d_ff``; this only sizes the MLP.
            dropout: Dropout probability applied to both the
                attention output and the MLP output before each
                residual addition. ``0.0`` (the default) disables
                dropout; the corresponding module is not allocated
                and the residual path is taken verbatim.
            activation: MLP activation. ``"gelu"`` (default) or
                ``"relu"``.
            attention_type: Which attention module to instantiate.
                See the class docstring for the full enumeration. The
                default ``"fused_v2"`` selects
                :class:`laker_xsa.attention.LakerAttention`, the v2 path.

        Raises:
            ValueError: If ``attention_type`` is not one of the
                recognized values.
        """
        super().__init__()
        self.config = config

        self.norm1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.eps)

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

        d_ff = d_ff if d_ff is not None else config.d_model * 4
        self.mlp = MLP(config.d_model, d_ff, dropout, activation)

        self.dropout: Optional[nn.Dropout] = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the block to the input sequence.

        Runs the pre-norm attention sub-layer, adds the residual,
        runs the pre-norm MLP sub-layer, and adds the second
        residual. Dropout (when configured) is applied to each
        sub-layer's output before the residual addition.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask forwarded to the attention
                module. The block does not interpret the mask itself;
                it is passed through verbatim. May be ``None`` for
                fully-visible sequences.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, mask)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)
        x = x + attn_out

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        if self.dropout is not None:
            mlp_out = self.dropout(mlp_out)
        x = x + mlp_out

        return x

    attention: nn.Module
