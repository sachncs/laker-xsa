"""
Kernel Attention Regression with LAKER-style preconditioning.

This module implements attention as kernel ridge regression, following the
approach described in arXiv:2604.25138 (LAKER).

The key formulation treats attention as solving a linear system:

.. math::

    (K + \\lambda I) \\alpha = V

    \\text{output} = K \\alpha

where :math:`K` is a kernel matrix computed from queries and keys,
:math:`\\lambda` is a regularization parameter, and :math:`\\alpha` is solved
using preconditioned Richardson iteration.

The fused XSA + LAKER variant additionally excludes self-attention by zeroing
the kernel diagonal.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from laker_xsa.config import XSA_LAKER_Config


class KernelFunction(nn.Module):
    """
    Kernel function for attention-based kernel regression.

    Computes a positive definite kernel matrix between query and key vectors.
    The kernel matrix K has entries:

    .. math::

        K_{ij} = k(q_i, k_j)

    Supported kernel types:
    - RBF: :math:`k(x, y) = \\exp(-\\|x - y\\|^2 / (2\\sigma^2))`
    - Linear: :math:`k(x, y) = x \\cdot y + 1`
    - Cosine: :math:`k(x, y) = \\cos(x, y) + 1`

    Attributes:
        kernel_type: Type of kernel function.
        eps: Numerical stability epsilon.
        bandwidth: Learnable bandwidth parameter (RBF only).

    Input Shape:
        - Queries: ``(batch, num_heads, seq_len, head_dim)``
        - Keys: ``(batch, num_heads, seq_len, head_dim)``

    Output Shape:
        - Kernel matrix: ``(batch, num_heads, seq_len, seq_len)``
    """

    def __init__(
        self,
        kernel_type: Literal["rbf", "linear", "cosine"],
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize kernel function.

        Args:
            kernel_type: Type of kernel ('rbf', 'linear', or 'cosine').
            eps: Numerical stability epsilon.
        """
        super().__init__()
        self.kernel_type = kernel_type
        self.eps = eps

        # Learnable bandwidth for RBF kernel
        if kernel_type == "rbf":
            self.bandwidth = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute kernel matrix between queries and keys.

        Args:
            q: Queries of shape ``(batch, num_heads, seq_len, head_dim)``.
            k: Keys of shape ``(batch, num_heads, seq_len, head_dim)``.

        Returns:
            Kernel matrix of shape ``(batch, num_heads, seq_len, seq_len)``.
        """
        if self.kernel_type == "rbf":
            return self._rbf_kernel(q, k)
        elif self.kernel_type == "linear":
            return self._linear_kernel(q, k)
        elif self.kernel_type == "cosine":
            return self._cosine_kernel(q, k)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _rbf_kernel(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        RBF (Gaussian) kernel.

        .. math::

            K_{ij} = \\exp\\left(-\\frac{\\|q_i - k_j\\|^2}{2\\sigma^2}\\right)

        Computed efficiently using:
        :math:`\\|q - k\\|^2 = \\|q\\|^2 + \\|k\\|^2 - 2q \\cdot k`

        Args:
            q: Queries.
            k: Keys.

        Returns:
            RBF kernel matrix.
        """
        # Ensure positive bandwidth: softplus(bandwidth) + eps
        bw = F.softplus(self.bandwidth) + self.eps

        # Compute squared norms: (batch, num_heads, seq_len, 1)
        q_norm_sq = (q * q).sum(dim=-1, keepdim=True)
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True)

        # Compute squared distances using broadcasting:
        # q_norm_sq: (..., seq_len, 1), k_norm_sq.T: (..., 1, seq_len)
        # Result: (..., seq_len, seq_len)
        dist_sq = q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * torch.matmul(q, k.transpose(-2, -1))

        # Clamp to non-negative (numerical issues can cause small negatives)
        dist_sq = torch.clamp(dist_sq, min=0.0)

        # RBF kernel: exp(-dist_sq / (2 * bw^2))
        kernel = torch.exp(-dist_sq / (2.0 * bw * bw))

        return kernel

    def _linear_kernel(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Linear kernel.

        .. math::

            K_{ij} = q_i \\cdot k_j + 1

        Args:
            q: Queries.
            k: Keys.

        Returns:
            Linear kernel matrix.
        """
        return torch.matmul(q, k.transpose(-2, -1)) + 1.0

    def _cosine_kernel(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity kernel.

        .. math::

            K_{ij} = \\cos(q_i, k_j) + 1 = \\frac{q_i \\cdot k_j}{\\|q_i\\| \\|k_j\\|} + 1

        The +1 ensures the kernel is positive (range [0, 2]).

        Args:
            q: Queries.
            k: Keys.

        Returns:
            Cosine kernel matrix.
        """
        # L2 normalize along head dimension
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)

        # Cosine similarity + 1
        return torch.matmul(q_norm, k_norm.transpose(-2, -1)) + 1.0


class LearnedPreconditioner(nn.Module):
    """
    Learned preconditioner for the kernel attention system.

    The kernel attention problem requires solving:

    .. math::

        (K + \\lambda I) \\alpha = V

    A preconditioner :math:`P \\approx (K + \\lambda I)^{-1}` accelerates
    iterative solving. We use a low-rank + diagonal parameterization:

    .. math::

        P = \\text{diag}(d) + U U^T

    where:
    - :math:`d` is a learned per-head diagonal scaling
    - :math:`U` is a low-rank factor generated from position embeddings

    This design allows the preconditioner to:
    1. Scale residuals per-token (diagonal part)
    2. Capture cross-token correlations (low-rank part)

    Attributes:
        num_heads: Number of attention heads.
        rank: Rank of the low-rank preconditioner factor.
        diag_scale: Learned diagonal scaling parameter.
        pos_embedding: Position embeddings for generating low-rank factors.
        head_proj: Head-specific projection for low-rank factors.
        reg: Regularization term for numerical stability.

    Input Shape:
        - kernel_diag: ``(batch, num_heads, seq_len)``

    Output Shape:
        - diag_precond: ``(batch, num_heads, seq_len)``
        - lr_precond: ``(batch, num_heads, seq_len, rank)`` or None
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """
        Initialize learned preconditioner.

        Args:
            config: Configuration object containing hyperparameters.
        """
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.rank = config.preconditioner_rank

        # Diagonal preconditioner: learned per-head scale
        # Shape: (1, num_heads, 1)
        self.diag_scale = nn.Parameter(torch.ones(1, config.num_heads, 1))

        # Low-rank factor generator (if rank specified)
        if self.rank is not None and self.rank > 0:
            # Position embeddings for generating per-token factors
            # Shape: (max_positions, rank)
            self.max_positions = 2048
            self.pos_embedding = nn.Parameter(
                torch.randn(self.max_positions, self.rank) * 0.02
            )
            # Head-specific projection
            # Shape: (num_heads, rank, rank)
            self.head_proj = nn.Parameter(
                torch.randn(config.num_heads, self.rank, self.rank) * 0.02
            )
        else:
            self.register_buffer("pos_embedding", torch.empty(0, 0))
            self.register_buffer("head_proj", torch.empty(0, 0, 0))

        # Regularization for numerical stability
        self.reg = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        kernel_diag: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute preconditioner parameters.

        Args:
            kernel_diag: Diagonal of kernel matrix, shape
                ``(batch, num_heads, seq_len)``.
            seq_len: Sequence length for position embeddings.

        Returns:
            Tuple containing:
            - diag_precond: Diagonal preconditioner, shape
              ``(batch, num_heads, seq_len)``
            - lr_precond: Low-rank preconditioner factor, shape
              ``(batch, num_heads, seq_len, rank)`` or None
        """
        batch = kernel_diag.shape[0]
        device = kernel_diag.device

        # Compute diagonal preconditioner from kernel diagonal
        # softplus ensures positivity
        # Shape: (batch, num_heads, seq_len)
        diag_precond = F.softplus(kernel_diag) * self.diag_scale + self.reg

        # Generate low-rank factor from position embeddings
        lr_precond: Optional[torch.Tensor] = None
        if self.rank is not None and self.rank > 0 and self.pos_embedding.numel() > 0:
            # Get position embeddings for current sequence length
            # Shape: (seq_len, rank)
            pos_emb = self.pos_embedding[:seq_len]

            # Apply head-specific projection
            # pos_emb.unsqueeze(0): (1, seq_len, rank)
            # head_proj: (num_heads, rank, rank)
            # Result: (num_heads, seq_len, rank)
            lr_precond_base = torch.matmul(pos_emb.unsqueeze(0), self.head_proj)

            # Expand to batch dimension
            # Shape: (batch, num_heads, seq_len, rank)
            lr_precond = lr_precond_base.unsqueeze(0).expand(batch, -1, -1, -1)

        return diag_precond, lr_precond

    def apply_precondition(
        self,
        residual: torch.Tensor,
        diag_precond: torch.Tensor,
        lr_precond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply preconditioner to a residual vector.

        Computes :math:`P \\cdot r` where :math:`P = \\text{diag}(d) + U U^T`.

        Args:
            residual: Residual tensor of shape
                ``(batch, num_heads, seq_len, head_dim)``.
            diag_precond: Diagonal preconditioner of shape
                ``(batch, num_heads, seq_len)``.
            lr_precond: Low-rank factor of shape
                ``(batch, num_heads, seq_len, rank)`` or None.

        Returns:
            Preconditioned residual, same shape as input.
        """
        # Diagonal part: element-wise multiplication
        # Broadcast diag_precond from (batch, num_heads, seq_len) to
        # (batch, num_heads, seq_len, head_dim)
        precond = residual * diag_precond.unsqueeze(-1)

        # Low-rank part: (U @ U^T) @ r = U @ (U^T @ r)
        if lr_precond is not None:
            # Compute U^T @ r:
            # lr_precond.T: (batch, num_heads, rank, seq_len)
            # residual: (batch, num_heads, seq_len, head_dim)
            # Result: (batch, num_heads, rank, head_dim)
            lr_t_r = torch.matmul(lr_precond.transpose(2, 3), residual)

            # Compute U @ (U^T @ r):
            # lr_precond: (batch, num_heads, seq_len, rank)
            # lr_t_r: (batch, num_heads, rank, head_dim)
            # Result: (batch, num_heads, seq_len, head_dim)
            precond = precond + torch.matmul(lr_precond, lr_t_r)

        return precond


class KernelAttentionRegression(nn.Module):
    """
    Kernel attention formulated as kernel ridge regression.

    Standard attention computes:

    .. math::

        \\text{output} = \\text{softmax}(QK^T / \\sqrt{d}) V

    The kernel regression formulation instead solves:

    .. math::

        (K + \\lambda I) \\alpha = V

        \\text{output} = K \\alpha

    where :math:`K` is a positive definite kernel matrix. This is equivalent
    to kernel ridge regression with regularization parameter :math:`\\lambda`.

    The system is solved using preconditioned Richardson iteration, which is
    fully differentiable through the unrolled iterations.

    Attributes:
        config: Configuration object.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        d_model: Total embedding dimension.
        num_iterations: Number of Richardson iterations.
        kernel_fn: Kernel function module.
        preconditioner: Learned preconditioner module.
        lambda_reg: Learnable regularization parameter.

    Input Shape:
        - Input: ``(batch, seq_len, d_model)``
        - Mask: ``(batch, seq_len, seq_len)`` or ``(batch, 1, seq_len, seq_len)``

    Output Shape:
        - Output: ``(batch, seq_len, d_model)``
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """
        Initialize kernel attention regression.

        Args:
            config: Configuration object containing hyperparameters.
        """
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.num_iterations = config.num_iterations

        # Kernel function
        self.kernel_fn = KernelFunction(config.kernel_type, config.eps)

        # Preconditioner
        self.preconditioner = LearnedPreconditioner(config)

        # Learnable regularization parameter (softplus ensures positivity)
        self.lambda_init = config.lambda_init
        self.lambda_reg = nn.Parameter(torch.tensor(config.lambda_init))

        # Linear projections
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

    def _solve_system(
        self,
        kernel: torch.Tensor,
        values: torch.Tensor,
        diag_precond: torch.Tensor,
        lr_precond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Solve the kernel system using preconditioned Richardson iteration.

        Solves: :math:`(K + \\lambda I) \\alpha = V`

        We use Richardson iteration instead of Conjugate Gradient because:
        1. CG requires inner products that vary with sequence length
        2. Richardson is simpler to make fully differentiable
        3. With good preconditioning, Richardson converges quickly

        The iteration is:

        .. math::

            \\alpha_{t+1} = \\alpha_t + P \\cdot (V - (K + \\lambda I) \\alpha_t)

        Args:
            kernel: Kernel matrix of shape
                ``(batch, num_heads, seq_len, seq_len)``.
            values: Value tensor of shape
                ``(batch, num_heads, seq_len, head_dim)``.
            diag_precond: Diagonal preconditioner of shape
                ``(batch, num_heads, seq_len)``.
            lr_precond: Low-rank preconditioner factor of shape
                ``(batch, num_heads, seq_len, rank)`` or None.

        Returns:
            Solution tensor :math:`\\alpha` of shape
            ``(batch, num_heads, seq_len, head_dim)``.
        """
        batch, num_heads, seq_len, head_dim = values.shape
        device = values.device

        # Initialize solution to zero
        alpha = torch.zeros_like(values)

        # Compute regularization (softplus ensures positivity)
        lambda_reg = F.softplus(self.lambda_reg) + self.config.eps

        # Add regularization to kernel diagonal: K_reg = K + lambda * I
        # We add lambda to the diagonal for each batch and head
        kernel_reg = kernel.clone()

        # Create identity matrix and scale by lambda
        eye = torch.eye(seq_len, device=device, dtype=kernel.dtype)
        for b in range(batch):
            for h in range(num_heads):
                kernel_reg[b, h] = kernel_reg[b, h] + eye * lambda_reg

        # Preconditioned Richardson iteration
        for _ in range(self.num_iterations):
            # Compute residual: r = V - K_reg @ alpha
            # Matrix-vector multiply: (..., seq_len, seq_len) @ (..., seq_len, head_dim)
            k_alpha = torch.matmul(kernel_reg, alpha)
            residual = values - k_alpha

            # Apply preconditioner: P @ r
            precond_residual = self.preconditioner.apply_precondition(
                residual, diag_precond, lr_precond
            )

            # Update solution
            alpha = alpha + precond_residual

            # Clip for numerical stability
            alpha = torch.clamp(alpha, -1e6, 1e6)

        return alpha

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for kernel attention regression.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask of shape
                ``(batch, seq_len, seq_len)`` or ``(batch, 1, seq_len, seq_len)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute kernel matrix K: (batch, num_heads, seq_len, seq_len)
        kernel = self.kernel_fn(q, k)

        # Apply mask if provided (zero out masked entries)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension
            kernel = kernel * mask

        # Compute preconditioner parameters from kernel diagonal
        # kernel_diag: (batch, num_heads, seq_len)
        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag_precond, lr_precond = self.preconditioner(kernel_diag, seq_len)

        # Solve (K + lambda*I) @ alpha = V
        alpha = self._solve_system(kernel, v, diag_precond, lr_precond)

        # Compute output: K @ alpha
        out = torch.matmul(kernel, alpha)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.w_o(out)

        return out


class FusedXSALAKERAttention(nn.Module):
    """
    Fused Exclusive Self Attention + LAKER Kernel Attention.

    This module combines XSA (which excludes self-attention) with LAKER
    (kernel regression with learned preconditioning).

    The fusion works as follows:
    1. Compute kernel matrix K from Q, K projections
    2. Apply XSA by zeroing the diagonal of K (no self-attention)
    3. Solve the kernel regression system with learned preconditioning
    4. Optionally apply additional projection-based exclusion

    Mathematical formulation:

    .. math::

        K_{ij} = \\text{kernel}(q_i, k_j) \\quad \\text{for } i \\neq j, \\quad K_{ii} = 0

        \\alpha = (K + \\lambda I)^{-1} V \\quad \\text{(preconditioned solve)}

        \\text{output} = K \\alpha

    This differs from standard attention:
    - No softmax normalization (kernel regression naturally normalizes)
    - Explicit self-exclusion (diagonal zeroed, not just masked by softmax)
    - Iterative solve enables truncation for long sequences

    Attributes:
        config: Configuration object.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        d_model: Total embedding dimension.
        kernel_fn: Kernel function module.
        preconditioner: Learned preconditioner module.
        lambda_reg: Learnable regularization parameter.
        xsa_scale: Learnable scale for post-projection XSA.

    Input Shape:
        - Input: ``(batch, seq_len, d_model)``
        - Mask: ``(batch, seq_len, seq_len)`` or ``(batch, 1, seq_len, seq_len)``

    Output Shape:
        - Output: ``(batch, seq_len, d_model)``
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """
        Initialize fused XSA + LAKER attention.

        Args:
            config: Configuration object containing hyperparameters.
        """
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Kernel function
        self.kernel_fn = KernelFunction(config.kernel_type, config.eps)

        # Preconditioner
        self.preconditioner = LearnedPreconditioner(config)

        # Learnable regularization
        self.lambda_init = config.lambda_init
        self.lambda_reg = nn.Parameter(torch.tensor(config.lambda_init))

        # Linear projections
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # Learnable scale for post-projection XSA
        if config.xsa_mode == "subtract_projection":
            self.xsa_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("xsa_scale", torch.ones(1))

    def _apply_xsa_to_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        """
        Apply XSA by zeroing the diagonal of the kernel matrix.

        This prevents any token from attending to itself in the kernel
        regression formulation.

        Args:
            kernel: Kernel matrix of shape
                ``(batch, num_heads, seq_len, seq_len)``.

        Returns:
            Kernel matrix with zero diagonal.
        """
        _, _, seq_len, _ = kernel.shape

        # Create mask with zero diagonal
        diag_mask = torch.eye(seq_len, device=kernel.device, dtype=kernel.dtype)
        diag_mask = diag_mask.view(1, 1, seq_len, seq_len)

        # Zero out diagonal: K_ij = 0 if i == j
        kernel = kernel * (1.0 - diag_mask)

        return kernel

    def _solve_system(
        self,
        kernel: torch.Tensor,
        values: torch.Tensor,
        diag_precond: torch.Tensor,
        lr_precond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Solve kernel system using preconditioned Richardson iteration.

        Solves: :math:`(K + \\lambda I) \\alpha = V`

        Args:
            kernel: Kernel matrix of shape
                ``(batch, num_heads, seq_len, seq_len)``.
            values: Value tensor of shape
                ``(batch, num_heads, seq_len, head_dim)``.
            diag_precond: Diagonal preconditioner of shape
                ``(batch, num_heads, seq_len)``.
            lr_precond: Low-rank factor of shape
                ``(batch, num_heads, seq_len, rank)`` or None.

        Returns:
            Solution :math:`\\alpha` of shape
            ``(batch, num_heads, seq_len, head_dim)``.
        """
        batch, num_heads, seq_len, _ = values.shape
        device = values.device

        # Initialize solution
        alpha = torch.zeros_like(values)

        # Regularization (softplus ensures positivity)
        lambda_reg = F.softplus(self.lambda_reg) + self.config.eps

        # Regularize kernel: K_reg = K + lambda * I
        kernel_reg = kernel.clone()
        eye = torch.eye(seq_len, device=device, dtype=kernel.dtype)
        for b in range(batch):
            for h in range(num_heads):
                kernel_reg[b, h] = kernel_reg[b, h] + eye * lambda_reg

        # Richardson iteration
        for _ in range(self.config.num_iterations):
            # Residual: r = V - K_reg @ alpha
            k_alpha = torch.matmul(kernel_reg, alpha)
            residual = values - k_alpha

            # Preconditioned update
            precond_residual = self.preconditioner.apply_precondition(
                residual, diag_precond, lr_precond
            )

            alpha = alpha + precond_residual
            alpha = torch.clamp(alpha, -1e6, 1e6)

        return alpha

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for fused XSA + LAKER attention.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask of shape
                ``(batch, seq_len, seq_len)`` or ``(batch, 1, seq_len, seq_len)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute kernel matrix
        kernel = self.kernel_fn(q, k)

        # Apply XSA: zero diagonal (exclude self-attention)
        kernel = self._apply_xsa_to_kernel(kernel)

        # Apply external mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            kernel = kernel * mask

        # Compute preconditioner
        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag_precond, lr_precond = self.preconditioner(kernel_diag, seq_len)

        # Solve kernel system
        alpha = self._solve_system(kernel, v, diag_precond, lr_precond)

        # Compute output
        out = torch.matmul(kernel, alpha)

        # Optional: additional XSA projection subtraction (post-hoc)
        if self.config.xsa_mode == "subtract_projection":
            # Subtract projection of output onto own value
            v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + self.config.eps
            out_dot_v = (out * v).sum(dim=-1, keepdim=True)
            coef = out_dot_v / v_norm_sq
            out = out - self.xsa_scale * coef * v

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.w_o(out)

        return out
