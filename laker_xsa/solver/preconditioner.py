"""
Preconditioner implementations for iterative solvers.

This module provides the LearnedPreconditioner class used in LAKER-style
kernel attention. The preconditioner approximates the inverse of the
kernel matrix to accelerate iterative solving.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from laker_xsa.config import XSA_LAKER_Config


class LearnedPreconditioner(nn.Module):
    """
    Learned preconditioner for kernel attention systems.

    For a linear system :math:`Ax = b`, a preconditioner :math:`P \\approx A^{-1}`
    transforms the system to :math:`PAx = Pb`, which has better conditioning
    and converges faster under iterative methods.

    We use a low-rank + diagonal parameterization:

    .. math::

        P = \\text{diag}(d) + U U^T

    where:
    - :math:`d` is learned per-head diagonal scaling
    - :math:`U` is a low-rank factor generated from position embeddings

    Attributes:
        num_heads: Number of attention heads.
        rank: Rank of low-rank preconditioner factor.
        diag_scale: Learned diagonal scaling.
        reg: Regularization term for stability.

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
            config: Configuration object with hyperparameters.
        """
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.rank = config.preconditioner_rank

        # Diagonal scaling: (1, num_heads, 1)
        self.diag_scale = nn.Parameter(torch.ones(1, config.num_heads, 1))

        # Low-rank generator
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

        # Regularization
        self.reg = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        kernel_diag: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute preconditioner parameters.

        Args:
            kernel_diag: Kernel diagonal, shape ``(batch, num_heads, seq_len)``.
            seq_len: Sequence length.

        Returns:
            Tuple of (diagonal preconditioner, low-rank factor).
        """
        batch = kernel_diag.shape[0]

        # Diagonal from kernel diagonal + learned scale
        diag_precond = F.softplus(kernel_diag) * self.diag_scale + self.reg

        # Low-rank factor
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
        """
        Apply preconditioner: compute P @ residual.

        Args:
            residual: Residual tensor, shape ``(batch, num_heads, seq_len, head_dim)``.
            diag_precond: Diagonal preconditioner, shape ``(batch, num_heads, seq_len)``.
            lr_precond: Low-rank factor or None.

        Returns:
            Preconditioned residual, same shape as input.
        """
        # Diagonal part
        precond = residual * diag_precond.unsqueeze(-1)

        # Low-rank part: (U @ U^T) @ r = U @ (U^T @ r)
        if lr_precond is not None:
            lr_t_r = torch.matmul(lr_precond.transpose(2, 3), residual)
            precond = precond + torch.matmul(lr_precond, lr_t_r)

        return precond
