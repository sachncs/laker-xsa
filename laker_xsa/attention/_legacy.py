"""Legacy v1 kernel attention module (deprecated).

This module preserves the original v1 LAKER kernel-attention implementation
so that older training scripts, benchmarks and checkpoints keep loading. Every
public class in this module emits a :class:`DeprecationWarning` when
constructed; new work should instead use
:class:`~laker_xsa.attention.laker.LakerAttention`.

Backward compatibility / deprecation boundary:
    * ``KernelFunction`` — kept only for legacy kernels (``"rbf"``,
      ``"linear"``, ``"cosine"``) and the misnamed ``"exp_attention"``
      alias that the v1 module mapped onto RBF. Superseded by
      :class:`~laker_xsa.attention.kernels.AttentionKernel`.
    * ``LearnedPreconditioner`` — the v1 position-embedding-based
      preconditioner. Superseded by
      :class:`~laker_xsa.solver.laker_preconditioner.LakerPreconditioner`.
    * ``KernelAttentionRegression`` — v1 attention module using Richardson
      iteration and ``KernelFunction`` / ``LearnedPreconditioner``. Superseded
      by :class:`~laker_xsa.attention.laker.LakerAttention`.
    * ``FusedXSALAKERAttention`` — v1 attempt at fusing XSA-style diagonal
      zeroing with the kernel regression. Superseded by
      :class:`~laker_xsa.attention.laker.LakerAttention`.
    * ``estimate_min_eigval`` — internal diagnostic helper used by
      ``KernelAttentionRegression.solve_system``.

Key differences from v2:
    * ``KernelFunction`` uses RBF, linear, or shifted-cosine kernels; its
      ``"exp_attention"`` option is an alias for RBF. The v2 path constructs
      :class:`AttentionKernel` instead.
    * The preconditioner uses learned position factors rather than the v2 CCCP
      or fast parameterizations.
    * The iterative solve is fixed-budget Richardson rather than PCG.
    * It does not perform CCCP shrinkage or trace normalization. Its learned
      ``diag_scale`` and ``reg`` are unconstrained, so its effective diagonal
      is not guaranteed positive.
    * The fused class zeros the kernel diagonal before applying any mask. A
      typical causal mask then produces a nonsymmetric lower-triangular
      kernel; adding positive diagonal regularization generally makes that
      triangular system nonsingular but does not make it SPD.

The classes in this module are re-exported as a compatibility shim from
:mod:`laker_xsa.attention.kernel_attention` so that pre-v2 import paths
``from laker_xsa.attention.kernel_attention import ...`` continue to resolve.
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

    .. deprecated::
        Superseded by
        :class:`~laker_xsa.attention.kernels.AttentionKernel`. The
        ``"exp_attention"`` literal here is a v1 alias for the RBF
        implementation; it is not the exponential dot-product kernel used by
        :class:`~laker_xsa.attention.kernels.AttentionKernel`.

    Emits :class:`DeprecationWarning` from the constructor.
    """

    def __init__(
        self,
        kernel_type: Literal["rbf", "linear", "cosine", "exp_attention"],
        eps: float = 1e-6,
    ) -> None:
        """Initialize the selected v1 kernel.

        Args:
            kernel_type: ``"rbf"`` and ``"exp_attention"`` both select the
                same RBF implementation; ``"linear"`` and ``"cosine"`` select
                their named implementations. Validation is deferred to
                :meth:`forward`.
            eps: Value added to the softplus-derived RBF bandwidth.

        Side Effects:
            Emits :class:`DeprecationWarning`. RBF modes allocate a learnable
            scalar ``bandwidth`` parameter.
        """
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
        """Dispatch to the selected v1 kernel.

        Args:
            q: Queries of shape ``(..., seq_len, head_dim)``.
            k: Keys of shape ``(..., seq_len, head_dim)``.

        Returns:
            Kernel matrix with shape ``(..., seq_len, seq_len)`` whose
            specific value range depends on ``self.kernel_type``.

        Raises:
            ValueError: If ``self.kernel_type`` is not one of the four
                recognised literals.
        """
        if self.kernel_type in ("rbf", "exp_attention"):
            return self.rbf_kernel(q, k)
        if self.kernel_type == "linear":
            return self.linear_kernel(q, k)
        if self.kernel_type == "cosine":
            return self.cosine_kernel(q, k)
        raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def rbf_kernel(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute the v1 squared-exponential kernel.

        Uses ``exp(-||q_i-k_j||^2 / (2*bw^2))`` with
        ``bw = softplus(bandwidth) + eps``. Expanded squared distances are
        clamped at zero before exponentiation.

        Args:
            q: Query features shaped ``(..., query_len, head_dim)``.
            k: Key features shaped ``(..., key_len, head_dim)``.

        Returns:
            Kernel shaped ``(..., query_len, key_len)``.
        """
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
        """Compute the biased linear kernel ``q @ k.T + 1``.

        Args:
            q: Query features shaped ``(..., query_len, head_dim)``.
            k: Key features shaped ``(..., key_len, head_dim)``.

        Returns:
            Kernel shaped ``(..., query_len, key_len)``.
        """
        return torch.matmul(q, k.transpose(-2, -1)) + 1.0

    def cosine_kernel(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute shifted cosine similarity ``normalize(q) @ normalize(k).T + 1``.

        Args:
            q: Query features shaped ``(..., query_len, head_dim)``.
            k: Key features shaped ``(..., key_len, head_dim)``.

        Returns:
            Kernel shaped ``(..., query_len, key_len)`` with mathematical range
            ``[0, 2]`` for finite real inputs, subject to rounding.
        """
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        return torch.matmul(q_norm, k_norm.transpose(-2, -1)) + 1.0


class LearnedPreconditioner(nn.Module):
    """[DEPRECATED] Position-embedding-based preconditioner (v1).

    .. deprecated::
        Superseded by
        :class:`~laker_xsa.solver.laker_preconditioner.LakerPreconditioner`,
        which provides CCCP, fast low-rank-plus-diagonal, diagonal, and identity
        modes.

    Implements ``P = diag(d) + U U^T`` where:

    * ``d = softplus(kernel_diag) * diag_scale + reg`` — an unconstrained
      learned affine transformation of a positive softplus value; and
    * ``U = pos_embedding[:seq_len] @ head_proj`` — a position-dependent
      low-rank factor shared across the batch.

    Emits :class:`DeprecationWarning` from the constructor.
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """Initialize the v1 position-based preconditioner.

        Args:
            config: Supplies ``num_heads`` and ``preconditioner_rank``.

        Side Effects:
            Emits :class:`DeprecationWarning`, allocates parameters/buffers, and
            advances PyTorch RNG state when low-rank factors are initialized.
        """
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
        """Compute the v1 preconditioner factors for the kernel system.

        Args:
            kernel_diag: Diagonal of the kernel matrix of shape
                ``(batch, num_heads, seq_len)``.
            seq_len: Sequence length used to slice the position embedding.

        Returns:
            Tuple ``(diag_precond, lr_precond)`` where:

            * ``diag_precond`` is ``(batch, num_heads, seq_len)`` — the
              diagonal preconditioner.
            * ``lr_precond`` is
              ``(batch, num_heads, min(seq_len, 2048), rank)`` or ``None``.
              Overlength factors fail when later applied to longer residuals.
        """
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
        """Apply ``P @ residual`` in the v1 diagonal + low-rank form.

        Args:
            residual: ``(batch, num_heads, seq_len, head_dim)`` residual.
            diag_precond: ``(batch, num_heads, seq_len)`` diagonal factor.
            lr_precond: Optional ``(batch, num_heads, seq_len, rank)`` low-
                rank factor; ignored when ``None``.

        Returns:
            Preconditioned residual of the same shape as ``residual``.
        """
        precond = residual * diag_precond.unsqueeze(-1)

        if lr_precond is not None:
            lr_t_r = torch.matmul(lr_precond.transpose(2, 3), residual)
            precond = precond + torch.matmul(lr_precond, lr_t_r)

        return precond


class KernelAttentionRegression(nn.Module):
    """[DEPRECATED] v1 kernel regression attention with Richardson iteration.

    .. deprecated::
        Superseded by :class:`~laker_xsa.attention.laker.LakerAttention`, which
        uses an exponential attention kernel, configurable v2 preconditioning,
        and the PCG-style recurrence.

    The forward pass projects ``x`` through four Linear layers, builds a
    kernel with :class:`KernelFunction`, masks it, computes ``diag_precond``
    and ``lr_precond`` from :class:`LearnedPreconditioner`, solves the
    regularised system with fixed-iteration Richardson, and finally applies
    the kernel matrix to the solution ``alpha``.

    Emits :class:`DeprecationWarning` from the constructor.
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """Initialize deprecated kernel-regression attention.

        Args:
            config: Supplies projection sizes, v1 kernel type, preconditioner
                rank, Richardson budget, regularization, and clamp values.

        Side Effects:
            Emits deprecation warnings for this module and its nested v1 kernel
            and preconditioner, allocates learnable layers/parameters, and
            advances PyTorch RNG state during initialization.
        """
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
        """Solve ``(K + lambda*I) @ alpha = V`` with preconditioned Richardson.

        Args:
            kernel: ``(batch, num_heads, seq_len, seq_len)`` kernel matrix.
            values: ``(batch, num_heads, seq_len, head_dim)`` value vectors
                ``V``.
            diag_precond: ``(batch, num_heads, seq_len)`` diagonal factor.
            lr_precond: Optional ``(batch, num_heads, seq_len, rank)`` low-
                rank factor.

        Returns:
            ``alpha`` of shape ``(batch, num_heads, seq_len, head_dim)``.

        Side Effects:
            Computes a first-slice ``eigvalsh`` diagnostic under
            ``torch.no_grad()`` and emits a warning when its returned minimum is
            negative. For nonsymmetric input that value is not a valid spectrum
            of the original matrix.
        """
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
        """Project, build the v1 kernel and solve the regression.

        Args:
            x: ``(batch, seq_len, d_model)`` token embeddings.
            mask: Optional mask. A 3-D mask receives a singleton head axis;
                any other shape is multiplied with the kernel using PyTorch
                broadcasting rules.

        Returns:
            ``(batch, seq_len, d_model)`` attention output.

        Raises:
            ValueError: If ``x.shape`` cannot be unpacked as a 3-D tensor.
            RuntimeError: Propagated from projections, reshaping, masking,
                preconditioning, or matrix multiplication.
        """
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
    """[DEPRECATED] v1 attempt at fusing XSA with LAKER kernel regression.

    .. deprecated::
        Superseded by
        :class:`~laker_xsa.attention.laker.LakerAttention`. This v1 class
        zeros the diagonal before applying any external mask and uses its
        Richardson-based kernel regression.

    Emits :class:`DeprecationWarning` from the constructor.
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """Initialize deprecated fused v1 attention.

        Args:
            config: Supplies projection, v1 kernel, XSA, Richardson,
                preconditioner, regularization, and clamp settings.

        Side Effects:
            Emits deprecation warnings for this class and nested v1 modules,
            allocates parameters/buffers, and advances PyTorch RNG state.
        """
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
        """Zero the kernel diagonal (v1 XSA; unconditional).

        This helper does not inspect an external mask. If a causal mask is
        applied later, the resulting kernel is typically lower triangular with
        a zero diagonal; the solver subsequently adds ``lambda * I``.

        Args:
            kernel: ``(batch, num_heads, seq_len, seq_len)`` kernel.

        Returns:
            New tensor with ``kernel[..., i, i] == 0`` everywhere.
        """
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
        """Solve the regularized v1 system with fixed Richardson updates.

        Args:
            kernel: Kernel shaped ``(batch, num_heads, seq_len, seq_len)``.
            values: Right-hand side shaped
                ``(batch, num_heads, seq_len, head_dim)``.
            diag_precond: Diagonal payload shaped
                ``(batch, num_heads, seq_len)``.
            lr_precond: Optional factor shaped
                ``(batch, num_heads, seq_len, rank)``.

        Returns:
            Final fixed-budget iterate with the same shape as ``values``. No
            convergence status is returned; non-finite values can propagate.

        Raises:
            RuntimeError: Propagated from tensor operations for incompatible
                shapes, dtypes, or devices.
        """
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
        """Forward pass — same shape semantics as :class:`KernelAttentionRegression`.

        Args:
            x: ``(batch, seq_len, d_model)`` token embeddings.
            mask: Optional mask broadcastable to
                ``(batch, num_heads, seq_len, seq_len)``.

        Returns:
            ``(batch, seq_len, d_model)`` attention output.

        Raises:
            ValueError: If ``x.shape`` cannot be unpacked as a 3-D tensor.
            RuntimeError: Propagated from projections, reshaping, masking,
                preconditioning, or matrix multiplication.
        """
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
    """Estimate the minimum eigenvalue of the first ``(batch, head)`` slice.

    ``torch.linalg.eigvalsh`` is called directly on ``kernel[0, 0]``. This is
    meaningful as an eigenvalue diagnostic only when that slice is symmetric;
    the function does not check symmetry.

    Args:
        kernel: ``(batch, num_heads, seq_len, seq_len)`` tensor. Only the
            slice ``kernel[0, 0]`` is used.

    Returns:
        The minimum value returned by ``torch.linalg.eigvalsh`` for
        ``kernel[0, 0]`` as a Python float, or ``float('nan')`` if
        ``torch.linalg.eigvalsh`` raises ``LinAlgError`` or ``RuntimeError``.
    """
    try:
        eigs = torch.linalg.eigvalsh(kernel[0, 0])
        return float(eigs.min().item())
    except (torch.linalg.LinAlgError, RuntimeError):
        return float("nan")
