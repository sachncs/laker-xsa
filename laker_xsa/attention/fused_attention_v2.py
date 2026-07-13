"""Backward-compatibility shim for the ``fused_attention_v2`` module.

Pre-v2 callers import the v2 fused module under the names
``FusedXSALAKERAttentionV2`` and ``XSALAKERAttentionV2``. Both map onto the
current implementations:

* ``FusedXSALAKERAttentionV2`` — alias for
  :class:`~laker_xsa.attention.laker.LakerAttention`.
* ``XSALAKERAttentionV2`` — alias for
  :class:`~laker_xsa.attention.laker.LakerAttentionLayer` (the
  Transformer-block-friendly wrapper).

New code should import :class:`LakerAttention` and
:class:`LakerAttentionLayer` directly from
:mod:`laker_xsa.attention.laker`.
"""

from __future__ import annotations

from laker_xsa.attention.laker import LakerAttention, LakerAttentionLayer

# Backward-compat aliases — same classes, v2-style names retained for
# callers that were written against the intermediate v2 release.
FusedXSALAKERAttentionV2 = LakerAttention
XSALAKERAttentionV2 = LakerAttentionLayer

__all__ = ["FusedXSALAKERAttentionV2", "XSALAKERAttentionV2"]
