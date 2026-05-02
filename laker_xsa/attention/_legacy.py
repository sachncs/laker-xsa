"""Legacy v1 kernel attention module (deprecated).

This module contains the original v1 implementation of LAKER kernel attention.
It is kept for backward compatibility with existing code and benchmarks.

For new work, use LakerAttention from laker.py, which implements the correct
LAKER preconditioner (CCCP-based) with PCG solver and proper XSA fusion.

Key issues with v1 (fixed in v2 / laker.py):
  - Uses wrong kernel (RBF/linear/cosine instead of exp(QK^T/sqrt(d)))
  - Preconditioner is position-embedding-based, not CCCP spectral learning
  - Uses fixed-iteration Richardson instead of PCG with convergence monitoring
  - Missing all LAKER safeguards (shrinkage, trace normalization, eps guard)
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from laker_xsa.config import XSA_LAKER_Config

_DEPRECATION_MSG = (
    "{} is deprecated. Use LakerAttention from laker_xsa.attention.laker "
    "for current LAKER kernel regression with correct exp(QK^T/sqrt(d)) kernel, "
    "CCCP preconditioner, and PCG solve."
)


class KernelFunction(nn.Module):
    """[DEPRECATED] Kernel function for v1 kernel attention.

    Use AttentionKernel from laker_xsa.attention.kernels instead.
    """

    def __init__(
        self,
        kernel_type: Literal["rbf", "linear", "cosine", "exp_attention"],
        eps: float = 1e-6,
    ) -> None:
        warnings.warn(
            _DEPRECATION_MSG.format("KernelFunction"),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self.kernel_type = kernel_type
        self.eps = eps

        if kernel_type in ("rbf", "exp_attention"):
            self.bandwidth = nn.Parameter(torch.tensor(1.0))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.kernel_type in ("rbf", "exp_attention"):
            return self.rbf_kernel(q, k)
        if self.kernel_type == "linear":
            return self.linear_kernel(q, k)
        if self.kernel_type == "cosine":
            return self.cosine_kernel(q, k)
        raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def rbf_kernel(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        bw = F.softplus(self.bandwidth) + self.eps
        q_norm_sq = (q * q).sum(dim=-1, keepdim=True)
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True)
        dist_sq = (
            q_norm_sq
            + k_norm_sq.transpose(-2, -1)
            - 2 * torch.matmul(q, k.transpose(-2, -1))
        )
        dist_sq = torch.clamp(dist_sq, min=0.0)
        return torch.exp(-dist_sq / (2.0 * bw * bw))

    def linear_kernel(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q, k.transpose(-2, -1)) + 1.0

    def cosine_kernel(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        return torch.matmul(q_norm, k_norm.transpose(-2, -1)) + 1.0


class LearnedPreconditioner(nn.Module):
    """[DEPRECATED] Position-embedding-based preconditioner (v1).

    Use LakerPreconditioner from laker_xsa.solver.laker_preconditioner instead.
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        warnings.warn(
            _DEPRECATION_MSG.format("LearnedPreconditioner"),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.rank = config.preconditioner_rank

        self.diag_scale = nn.Parameter(torch.ones(1, config.num_heads, 1))

        if self.rank is not None and self.rank > 0:
            self.max_positions = 2048
            self.pos_embedding = nn.Parameter(
                torch.randn(self.max_positions, self.rank) * 0.02
            )
            self.head_proj = nn.Parameter(
                torch.randn(config.num_heads, self.rank, self.rank) * 0.02
            )
        else:
            self.register_buffer("pos_embedding", torch.empty(0, 0))
            self.register_buffer("head_proj", torch.empty(0, 0, 0))

        self.reg = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, kernel_diag: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch = kernel_diag.shape[0]

        diag_precond = F.softplus(kernel_diag) * self.diag_scale + self.reg

        lr_precond: Optional[torch.Tensor] = None
        if self.rank is not None and self.rank > 0 and self.pos_embedding.numel() > 0:
            pos_emb = self.pos_embedding[:seq_len]
            lr_precond_base = torch.matmul(pos_emb.unsqueeze(0), self.head_proj)
            lr_precond = lr_precond_base.unsqueeze(0).expand(batch, -1, -1, -1)

        return diag_precond, lr_precond

    def apply_precondition(
        self,
        residual: torch.Tensor,
        diag_precond: torch.Tensor,
        lr_precond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        precond = residual * diag_precond.unsqueeze(-1)

        if lr_precond is not None:
            lr_t_r = torch.matmul(lr_precond.transpose(2, 3), residual)
            precond = precond + torch.matmul(lr_precond, lr_t_r)

        return precond


class KernelAttentionRegression(nn.Module):
    """[DEPRECATED] v1 kernel regression with Richardson iteration.

    Use LakerAttention from laker_xsa.attention.laker instead.
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        warnings.warn(
            _DEPRECATION_MSG.format("KernelAttentionRegression"),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.num_iterations = config.num_iterations

        self.kernel_fn = KernelFunction(config.kernel_type, config.eps)
        self.preconditioner = LearnedPreconditioner(config)

        self.lambda_init = config.lambda_init
        self.lambda_reg = nn.Parameter(torch.tensor(config.lambda_init))

        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

    def solve_system(
        self,
        kernel: torch.Tensor,
        values: torch.Tensor,
        diag_precond: torch.Tensor,
        lr_precond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = values.shape
        device = values.device

        alpha = torch.zeros_like(values)
        lambda_reg = F.softplus(self.lambda_reg) + self.config.eps

        kernel_reg = kernel.clone()
        eye = torch.eye(seq_len, device=device, dtype=kernel.dtype)
        for b in range(batch):
            for h in range(num_heads):
                kernel_reg[b, h] = kernel_reg[b, h] + eye * lambda_reg

        with torch.no_grad():
            min_eig = estimate_min_eigval(kernel_reg)
            if min_eig < 0:
                warnings.warn(
                    f"Kernel matrix has negative eigenvalues (min={min_eig:.4f}). "
                    f"Increasing lambda may help."
                )

        clip_abs = self.config.clip_abs
        for _ in range(self.num_iterations):
            k_alpha = torch.matmul(kernel_reg, alpha)
            residual = values - k_alpha
            precond_residual = self.preconditioner.apply_precondition(
                residual, diag_precond, lr_precond
            )
            alpha = alpha + precond_residual
            alpha = torch.clamp(alpha, -clip_abs, clip_abs)

        return alpha

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        kernel = self.kernel_fn(q, k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            kernel = kernel * mask

        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag_precond, lr_precond = self.preconditioner(kernel_diag, seq_len)

        alpha = self.solve_system(kernel, v, diag_precond, lr_precond)

        out = torch.matmul(kernel, alpha)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return cast(torch.Tensor, self.w_o(out))


class FusedXSALAKERAttention(nn.Module):
    """[DEPRECATED] v1 fused XSA + LAKER attention.

    Use LakerAttention from laker_xsa.attention.laker instead.
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        warnings.warn(
            _DEPRECATION_MSG.format("FusedXSALAKERAttention"),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.kernel_fn = KernelFunction(config.kernel_type, config.eps)
        self.preconditioner = LearnedPreconditioner(config)

        self.lambda_init = config.lambda_init
        self.lambda_reg = nn.Parameter(torch.tensor(config.lambda_init))

        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

        if config.xsa_mode == "subtract_projection":
            self.xsa_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("xsa_scale", torch.ones(1))

    def apply_xsa_to_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        _, _, seq_len, _ = kernel.shape
        diag_mask = torch.eye(seq_len, device=kernel.device, dtype=kernel.dtype)
        diag_mask = diag_mask.view(1, 1, seq_len, seq_len)
        return kernel * (1.0 - diag_mask)

    def solve_system(
        self,
        kernel: torch.Tensor,
        values: torch.Tensor,
        diag_precond: torch.Tensor,
        lr_precond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, num_heads, seq_len, _ = values.shape
        device = values.device

        alpha = torch.zeros_like(values)
        lambda_reg = F.softplus(self.lambda_reg) + self.config.eps

        kernel_reg = kernel.clone()
        eye = torch.eye(seq_len, device=device, dtype=kernel.dtype)
        for b in range(batch):
            for h in range(num_heads):
                kernel_reg[b, h] = kernel_reg[b, h] + eye * lambda_reg

        clip_abs = self.config.clip_abs
        for _ in range(self.config.num_iterations):
            k_alpha = torch.matmul(kernel_reg, alpha)
            residual = values - k_alpha
            precond_residual = self.preconditioner.apply_precondition(
                residual, diag_precond, lr_precond
            )
            alpha = alpha + precond_residual
            alpha = torch.clamp(alpha, -clip_abs, clip_abs)

        return alpha

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = (
            self.w_q(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.w_k(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.w_v(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kernel = self.kernel_fn(q, k)
        kernel = self.apply_xsa_to_kernel(kernel)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            kernel = kernel * mask

        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag_precond, lr_precond = self.preconditioner(kernel_diag, seq_len)

        alpha = self.solve_system(kernel, v, diag_precond, lr_precond)
        out = torch.matmul(kernel, alpha)

        if self.config.xsa_mode == "subtract_projection":
            v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + self.config.eps
            out_dot_v = (out * v).sum(dim=-1, keepdim=True)
            coef = out_dot_v / v_norm_sq
            out = out - self.xsa_scale * coef * v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return cast(torch.Tensor, self.w_o(out))


def estimate_min_eigval(kernel: torch.Tensor) -> float:
    """Estimate minimum eigenvalue via eigendecomposition on first batch/head."""
    try:
        eigs = torch.linalg.eigvalsh(kernel[0, 0])
        return float(eigs.min().item())
    except (torch.linalg.LinAlgError, RuntimeError):
        return float("nan")
