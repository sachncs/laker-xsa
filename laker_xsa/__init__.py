from __future__ import annotations

"""
LAKER-XSA: Fused Exclusive Self Attention and LAKER Kernel Attention.

This package implements a breakthrough Transformer attention mechanism that fuses:

1. Exclusive Self Attention (XSA): Removes self-aligned components to force
   context-only aggregation (arXiv:2603.09078)

2. LAKER Kernel Attention: Frames attention as kernel ridge regression with
   learned preconditioning via CCCP for improved conditioning (arXiv:2604.25138)

The v2 fusion (LakerAttention) is a novel, unpublished combination
that solves two fundamental failure modes of standard attention:
- Self-bias (tokens copy themselves) — fixed by XSA
- Spectral collapse (eigenvalue decay) — fixed by LAKER kernel inverse

Example:
    from laker_xsa import XSA_LAKER_Config, LakerAttention

    config = XSA_LAKER_Config(d_model=512, num_heads=8)
    attn = LakerAttention(config)
    x = torch.randn(2, 128, 512)
    out = attn(x)
"""

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention import (
    AttentionKernel,
    BaseMultiHeadAttention,
    ExclusiveSelfAttention,
    LakerAttention,
    LakerAttentionLayer,
    StandardMultiHeadAttention,
)
from laker_xsa.attention._legacy import (
    FusedXSALAKERAttention,
    KernelAttentionRegression,
)
from laker_xsa.attention.fused_attention_v2 import (
    FusedXSALAKERAttentionV2,
    XSALAKERAttentionV2,
)
from laker_xsa.model.transformer_block import XSALAKERTransformerBlock
from laker_xsa.model.full_model import XSALAKERTransformer
from laker_xsa.utils.stability import check_finite
from laker_xsa.utils.tensor_ops import create_causal_mask

__all__ = [
    "XSA_LAKER_Config",
    # Core abstractions
    "BaseMultiHeadAttention",
    # Attention modules
    "StandardMultiHeadAttention",
    "ExclusiveSelfAttention",
    "LakerAttention",
    "LakerAttentionLayer",
    # Backward-compat
    "KernelAttentionRegression",
    "FusedXSALAKERAttention",
    "FusedXSALAKERAttentionV2",
    "XSALAKERAttentionV2",
    # Kernels
    "AttentionKernel",
    # Model
    "XSALAKERTransformerBlock",
    "XSALAKERTransformer",
    # Utils
    "check_finite",
    "create_causal_mask",
]

__version__ = "0.2.0"
