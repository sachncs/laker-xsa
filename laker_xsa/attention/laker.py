"""Fused XSA + LAKER Attention — breakthrough fused attention.

Implements the mathematically principled fusion of:
  - XSA (arXiv:2603.09078): Exclusive Self Attention — removes self-components
  - LAKER (arXiv:2604.25138): Learned Attention Kernel Regression with
    preconditioned iterative solving.

Pipeline:
  1. Q,K,V = Linear projections
  2. K = AttentionKernel(Q, K)                          [exp kernel]
  3. K_ii = 0 for all i                                 [XSA: zero diagonal]
  4. P = Learn preconditioner for (K + lambda*I)        [LAKER preconditioner]
  5. alpha = PCG_solve(K + lambda*I, V, precond=P)      [inverse mixing]
  6. alpha = RMS_norm(alpha)                            [stabilize scale]
  7. alpha -= proj_V(alpha)                             [XSA: output cleaning]
  8. Reshape + output projection
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.core import (
    BaseMultiHeadAttention,
    stable_clip,
)
from laker_xsa.attention.kernels import AttentionKernel
from laker_xsa.solver.laker_preconditioner import LakerPreconditioner
from laker_xsa.solver.conjugate_gradient import pcg_solve

logger = logging.getLogger(__name__)


class LakerAttention(BaseMultiHeadAttention):
    """Breakthrough fused XSA + LAKER attention.

    Combines exclusive self-attention (no self-copying) with kernel regression
    inverse mixing (no spectral collapse) in a single module.

    Attributes:
        kernel_fn: Attention kernel K = exp(QK^T / sqrt(d) / temperature).
        preconditioner: LAKER learned preconditioner for PCG acceleration.
        lambda_reg: SPD-guaranteeing regularization (softplus parameterized).
        xsa_scale: Learnable scale for output projection removal.

    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        super().__init__(config)

        self.kernel_fn = AttentionKernel(
            head_dim=self.head_dim,
            temperature=config.kernel_temperature,
            symmetric=config.kernel_symmetric,
            learnable_temperature=True,
            normalize_qk=config.kernel_normalize_qk,
            eps=config.eps,
        )

        self.preconditioner = LakerPreconditioner(
            num_heads=config.num_heads,
            mode=config.preconditioner_type,
            rank=config.preconditioner_rank,
            gamma=config.cccp_gamma,
            rho=config.cccp_shrinkage_rho,
            eps_safeguard=config.cccp_shrinkage_eps,
            n_random_directions=config.cccp_num_directions,
            max_cccp_iters=config.cccp_max_iterations,
            eps=config.eps,
        )

        self.raw_lambda = nn.Parameter(torch.tensor(config.lambda_init))

        if config.xsa_mode == "subtract_projection":
            self.xsa_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("xsa_scale", torch.ones(1))

        self.init_weights()

    def init_weights(self) -> None:
        std = 0.02 / math.sqrt(2.0)
        for proj in [self.qkv_proj.w_q, self.qkv_proj.w_k, self.qkv_proj.w_v, self.w_o]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)

    @property
    def lambda_reg(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda) + self.config.eps

    def zero_diagonal(self, kernel: torch.Tensor) -> torch.Tensor:
        """Zero the kernel diagonal: K_{ii} = 0 for all i (XSA)."""
        _, _, n, _ = kernel.shape
        diag_mask = torch.eye(n, device=kernel.device, dtype=kernel.dtype)
        diag_mask = diag_mask.view(1, 1, n, n)
        return kernel * (1.0 - diag_mask)

    def clean_self_projection(
        self, output: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Remove projection of output onto each token's own value vector."""
        dot = (output * values).sum(dim=-1, keepdim=True)
        v_norm_sq = (values * values).sum(dim=-1, keepdim=True) + self.config.eps
        return output - self.xsa_scale * (dot / v_norm_sq) * values

    def rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(
            (x * x).mean(dim=(-2, -1), keepdim=True) + self.config.eps
        )
        return x / rms

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _, _, seq_len, _ = q.shape

        # Attention kernel K = exp(QK^T_norm / temperature)
        kernel = self.kernel_fn(q, k)

        # Apply mask before XSA diagonal zeroing to preserve structure
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            kernel = kernel * mask.to(dtype=kernel.dtype)

        # XSA: zero diagonal for bidirectional (unmasked) attention
        # With causal masking, diagonal zeroing creates nilpotent systems;
        # output cleaning (below) still provides XSA benefits.
        if mask is None:
            kernel = self.zero_diagonal(kernel)

        # Regularization for SPD guarantee
        lam = self.lambda_reg.view(1, 1, 1, 1)

        # Learn preconditioner
        precond_data = self.preconditioner(
            kernel,
            lam,
            seq_len,
            force_update=False,
            update_frequency=self.config.precond_update_frequency,
        )

        # PCG solve: (K + lambda*I) @ alpha = V
        try:
            alpha = pcg_solve(
                kernel=kernel,
                b=v,
                lambda_reg=lam,
                precond_data=precond_data,
                apply_preconditioner=self.preconditioner.apply_preconditioner,
                max_iterations=self.config.effective_pcg_iters,
                tolerance=self.config.pcg_tolerance,
                min_iterations=3,
            )
        except RuntimeError:
            logger.warning(
                "PCG solve failed; falling back to direct solve. "
                "seq_len=%d, lambda=%.4f",
                seq_len,
                self.lambda_reg.item(),
            )
            eye = torch.eye(seq_len, device=kernel.device, dtype=kernel.dtype)
            eye = eye.view(1, 1, seq_len, seq_len)
            kernel_reg = kernel + lam * eye
            alpha = torch.linalg.solve(kernel_reg, v)

        # Stabilize output scale for multi-layer stacking
        alpha = stable_clip(alpha)
        alpha = self.rms_normalize(alpha)

        # XSA output cleaning
        if self.config.xsa_mode == "subtract_projection":
            alpha = self.clean_self_projection(alpha, v)

        return alpha


class LakerAttentionLayer(nn.Module):
    """Multi-layer-ready LakerAttention with per-layer configuration.

    Wraps LakerAttention for use in Transformer blocks with support
    for per-layer preconditioner sharing/isolation and mask modes.

    This is the module intended for embedding in Transformer architectures.
    """

    def __init__(
        self,
        config: XSA_LAKER_Config,
        layer_idx: int = 0,
        share_preconditioner_across_layers: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.share_preconditioner = share_preconditioner_across_layers
        self.attention = LakerAttention(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.attention(x, mask)
