from __future__ import annotations

"""Backward-compatibility shim for xsa_attention module.

Re-exports from laker_xsa.attention.xsa.
"""

from laker_xsa.attention.xsa import ExclusiveSelfAttention

__all__ = ["ExclusiveSelfAttention"]
