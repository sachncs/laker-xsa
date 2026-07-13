"""Public API for LAKER-XSA attention, solver, and Transformer modules.

The package exposes standard scaled dot-product attention, Exclusive Self
Attention (XSA), and the v2 :class:`LakerAttention` implementation. XSA modes
modify score diagonals and/or subtract a regularized output projection. The
LAKER path builds an exponential attention kernel and applies a configurable
preconditioner with an iterative linear-system solve. The implementation cites
the XSA (arXiv:2603.09078) and LAKER (arXiv:2604.25138) references elsewhere in
the repository.

Subpackages separate attention primitives, iterative solvers, Transformer model
composition, training helpers, benchmarks, utilities, and command-line entry
points. Deprecated v1 kernel-regression classes remain exported for checkpoint
and import compatibility; new integrations should use :class:`LakerAttention`.

Example:
    >>> import torch
    >>> from laker_xsa import LakerAttention, XSA_LAKER_Config
    >>> config = XSA_LAKER_Config(d_model=512, num_heads=8)
    >>> output = LakerAttention(config)(torch.randn(2, 128, 512))
    >>> output.shape
    torch.Size([2, 128, 512])
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
    "BaseMultiHeadAttention",
    "StandardMultiHeadAttention",
    "ExclusiveSelfAttention",
    "LakerAttention",
    "LakerAttentionLayer",
    "KernelAttentionRegression",
    "FusedXSALAKERAttention",
    "FusedXSALAKERAttentionV2",
    "XSALAKERAttentionV2",
    "AttentionKernel",
    "XSALAKERTransformerBlock",
    "XSALAKERTransformer",
    "check_finite",
    "create_causal_mask",
]

__version__ = "0.2.3"
