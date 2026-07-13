"""Legacy-style position-based learned preconditioner.

This module contains a standalone ``LearnedPreconditioner`` exported from
:mod:`laker_xsa.solver`. It mirrors the parameterization embedded separately in
:mod:`laker_xsa.attention._legacy`; the deprecated attention classes do not
import this copy. Its learned diagonal scale and regularizer are unconstrained,
so the resulting preconditioner is not guaranteed to meet PCG's positivity
assumptions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.functional import softplus

from laker_xsa.config import XSA_LAKER_Config


class LearnedPreconditioner(nn.Module):
    """Legacy low-rank + diagonal preconditioner.

    The construction is ``P = diag(d) + U U^T``. The low-rank term is
    positive-semidefinite for any real ``U``; however, ``diag_scale`` and
    ``reg`` are unconstrained, so ``d`` can become negative and the complete
    preconditioner is not guaranteed to be positive-definite.

    Attributes:
        num_heads: Configured number of attention heads.
        rank: Configured low-rank dimension. ``None`` or ``0`` disables
            the low-rank factor.
        diag_scale: Learnable scaling of shape ``(1, num_heads, 1)`` with
            no sign constraint.
        reg: Learnable scalar added to the diagonal preconditioner; its
            sign is not constrained.
        pos_embedding, head_proj: Learnable position embedding and head
            projection, only allocated when ``rank > 0``.
        max_positions: Fixed value ``2048`` when the low-rank component is
            enabled. A larger ``seq_len`` truncates the returned low-rank basis
            to 2048 rows while the diagonal retains ``seq_len`` rows; applying
            both to the same residual then fails with incompatible shapes.
        config: Stored :class:`laker_xsa.config.XSA_LAKER_Config`.

    Input Shape:
        - ``kernel_diag``: ``(batch, num_heads, seq_len)``.

    Output Shape:
        - ``diag_precond``: ``(batch, num_heads, seq_len)``.
        - ``lr_precond``: ``(batch, num_heads, min(seq_len, 2048), rank)`` or
          ``None`` if the low-rank component is disabled.
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """Initialize the learned preconditioner.

        Args:
            config: :class:`laker_xsa.config.XSA_LAKER_Config` providing
                ``num_heads``, ``preconditioner_rank``, and other
                hyperparameters.

        Side Effects:
            Allocates learnable parameters and registered buffers. When the
            low-rank factor is enabled, random parameter initialization advances
            PyTorch's global RNG state.
        """
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
        self,
        kernel_diag: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute preconditioner parameters from the kernel diagonal.

        Args:
            kernel_diag: Kernel diagonal entries with shape
                ``(batch, num_heads, seq_len)``.
            seq_len: Number of position rows requested for the low-rank factor.
                Values above ``2048`` return only the available 2048 rows; the
                mismatch becomes an error only when that factor is later
                applied to a ``seq_len``-row residual.

        Returns:
            Tuple ``(diag_precond, lr_precond)``:

            - ``diag_precond`` with shape
              ``(batch, num_heads, seq_len)``. Computed as
              ``softplus(kernel_diag) * diag_scale + reg``; this
              construction does not enforce non-negativity because
              ``diag_scale`` and ``reg`` are unconstrained.
            - ``lr_precond`` with shape
              ``(batch, num_heads, min(seq_len, 2048), rank)`` when the
              low-rank component is active; ``None`` otherwise. For supported
              lengths this is ``(batch, num_heads, seq_len, rank)``.

        Notes:
            - The diagonal multiplication is broadcast: ``diag_scale``
              has shape ``(1, num_heads, 1)`` and is multiplied
              element-wise with ``softplus(kernel_diag)`` of shape
              ``(batch, num_heads, seq_len)``.
            - ``softplus(kernel_diag)`` is differentiable w.r.t.
              ``kernel_diag``. The unconstrained ``diag_scale`` and
              ``reg`` parameters are also differentiable but their
              optimization can drive the produced diagonal negative.
            - ``reg`` is added as a Python-style scalar broadcast.
        """
        batch = kernel_diag.shape[0]

        # Broadcasting: diag_scale has shape (1, num_heads, 1) and
        # broadcasts against (batch, num_heads, seq_len) kernel_diag.
        diag_precond = softplus(kernel_diag) * self.diag_scale + self.reg

        lr_precond: Optional[torch.Tensor] = None
        if self.rank is not None and self.rank > 0 and self.pos_embedding.numel() > 0:
            pos_emb = self.pos_embedding[:seq_len]
            lr_precond_base = torch.matmul(pos_emb.unsqueeze(0), self.head_proj)
            # Broadcasting: share the same (num_heads, seq_len, rank)
            # basis across the batch dimension.
            lr_precond = lr_precond_base.unsqueeze(0).expand(batch, -1, -1, -1)

        return diag_precond, lr_precond

    def apply_precondition(
        self,
        residual: torch.Tensor,
        diag_precond: torch.Tensor,
        lr_precond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the preconditioner: compute ``P r``.

        Implements the factored form
        ``r -> diag(d) * r + U * (U^T r)`` so that ``U U^T`` is not formed
        explicitly.

        Args:
            residual: Residual tensor with shape
                ``(batch, num_heads, seq_len, head_dim)``.
            diag_precond: Diagonal preconditioner with shape
                ``(batch, num_heads, seq_len)``.
            lr_precond: Low-rank factor with shape
                ``(batch, num_heads, seq_len, rank)``, or ``None`` to
                skip the low-rank component.

        Returns:
            Preconditioned residual with the same shape and dtype as
            ``residual``.

        Raises:
            RuntimeError: Propagated from elementwise operations or matrix
                multiplication when shapes, dtypes, or devices are incompatible;
                this includes an overlength low-rank factor applied to a longer
                residual.

        Notes:
            - The diagonal contribution broadcasts across the trailing
              ``head_dim`` axis via ``unsqueeze(-1)``.
            - The low-rank contribution is ``matmul(lr_precond,
              matmul(lr_precond.transpose(2, 3), residual))``. The inner
              contraction is over the rank axis, the outer over the
              sequence axis. The transpose is required by the leading
              ``n`` axis before the inner contraction.
            - When ``lr_precond`` is ``None`` only the diagonal
              contribution is applied.
        """
        precond = residual * diag_precond.unsqueeze(-1)

        if lr_precond is not None:
            lr_t_r = torch.matmul(lr_precond.transpose(2, 3), residual)
            precond = precond + torch.matmul(lr_precond, lr_t_r)

        return precond
