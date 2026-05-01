from __future__ import annotations

"""Backward-compatibility shim for fused_attention_v2 module.

Re-exports from laker_xsa.attention.laker.
"""

from laker_xsa.attention.laker import LakerAttention, LakerAttentionLayer

# Backward-compat aliases
FusedXSALAKERAttentionV2 = LakerAttention
XSALAKERAttentionV2 = LakerAttentionLayer

__all__ = ["FusedXSALAKERAttentionV2", "XSALAKERAttentionV2"]
