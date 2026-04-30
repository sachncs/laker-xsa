"""
LAKER-XSA: Fused Exclusive Self Attention and LAKER-style Kernel Attention.

This package implements a Transformer attention mechanism that fuses:

1. Exclusive Self Attention (XSA): Removes self-aligned components to force
   context-only aggregation (arXiv:2603.09078)

2. LAKER-style Kernel Attention: Treats attention as kernel regression with
   learned preconditioning for improved conditioning (arXiv:2604.25138)

Example usage:

    from laker_xsa import XSA_LAKER_Config, FusedXSALAKERAttention

    config = XSA_LAKER_Config(
        d_model=512,
        num_heads=8,
        num_iterations=10,
        preconditioner_rank=32,
    )

    attn = FusedXSALAKERAttention(config)
    x = torch.randn(2, 128, 512)
    out = attn(x)
"""

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.standard_attention import StandardMultiHeadAttention
from laker_xsa.attention.xsa_attention import ExclusiveSelfAttention
from laker_xsa.attention.kernel_attention import KernelAttentionRegression
from laker_xsa.attention.kernel_attention import FusedXSALAKERAttention
from laker_xsa.model.transformer_block import XSALAKERTransformerBlock
from laker_xsa.model.full_model import XSALAKERTransformer
from laker_xsa.utils.stability import check_finite
from laker_xsa.utils.tensor_ops import create_causal_mask

__all__ = [
    "XSA_LAKER_Config",
    "StandardMultiHeadAttention",
    "ExclusiveSelfAttention",
    "KernelAttentionRegression",
    "FusedXSALAKERAttention",
    "XSALAKERTransformerBlock",
    "XSALAKERTransformer",
    "check_finite",
    "create_causal_mask",
]

__version__ = "0.1.0"
