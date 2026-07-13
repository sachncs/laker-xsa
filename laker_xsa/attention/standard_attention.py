"""Backward-compatibility shim for the ``standard_attention`` module.

Pre-v2 callers import the baseline attention module from
``laker_xsa.attention.standard_attention``; new code should import
:class:`StandardMultiHeadAttention` directly from
:mod:`laker_xsa.attention.standard`.

This shim only re-exports the symbol; no logic lives here.
"""

from __future__ import annotations

from laker_xsa.attention.standard import StandardMultiHeadAttention

__all__ = ["StandardMultiHeadAttention"]
