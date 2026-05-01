from __future__ import annotations

"""Backward-compatibility shim for standard_attention module.

Re-exports from laker_xsa.attention.standard.
"""

from laker_xsa.attention.standard import StandardMultiHeadAttention

__all__ = ["StandardMultiHeadAttention"]
