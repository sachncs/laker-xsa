"""Backward-compatibility shim for the ``xsa_attention`` module.

Pre-v2 callers import XSA from ``laker_xsa.attention.xsa_attention``; new
code should import :class:`ExclusiveSelfAttention` directly from
:mod:`laker_xsa.attention.xsa`.

This shim only re-exports the symbol; no logic lives here.
"""

from __future__ import annotations

from laker_xsa.attention.xsa import ExclusiveSelfAttention

__all__ = ["ExclusiveSelfAttention"]
