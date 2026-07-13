"""Transformer building blocks and full encoder model.

This package contains the model-level components that compose
attention modules into a complete Transformer:

* :class:`MLP` - the position-wise feed-forward network used inside
  each Transformer block.
* :class:`XSALAKERTransformerBlock` - a single pre-norm block that
  wraps a configurable attention module (standard, XSA, kernel
  regression, or the fused v2 attention).
* :class:`XSALAKERTransformer` - a stack of such blocks plus optional
  token/positional embeddings and a vocabulary projection.
"""

from __future__ import annotations

from laker_xsa.model.transformer_block import MLP, XSALAKERTransformerBlock
from laker_xsa.model.full_model import XSALAKERTransformer

__all__ = [
    "MLP",
    "XSALAKERTransformerBlock",
    "XSALAKERTransformer",
]
