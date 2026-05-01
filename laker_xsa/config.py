"""
Configuration module for LAKER-XSA attention.

This module defines the configuration dataclass that controls all hyperparameters
for the XSA and LAKER attention mechanisms, including the fused v2 breakthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class XSA_LAKER_Config:
    """
    Configuration for fused XSA + LAKER attention.

    Controls Exclusive Self Attention (XSA) and LAKER kernel regression
    with learned preconditioning.

    Attributes:
        d_model: Input/output embedding dimension.
        num_heads: Number of attention heads.
        head_dim: Per-head dimension. Defaults to d_model // num_heads.
        dropout: Attention dropout probability.
        eps: Numerical stability epsilon.
        lambda_init: Initial regularization for kernel system.
        kernel_type: Kernel function for attention.
        xsa_mode: Self-exclusion method.
        use_fused: Use fused XSA+LAKER (or standard attention).
        seed: Random seed.

        # LAKER preconditioner parameters
        preconditioner_type: 'cccp' for full CCCP, 'fast' for gradient-based,
            'diagonal' for Jacobi-style, 'none' for no preconditioning.
        preconditioner_rank: Low-rank dimension for fast preconditioner.
        cccp_num_directions: Random directions N_r for CCCP angular sampling.
        cccp_max_iterations: Maximum CCCP fixed-point iterations.
        cccp_gamma: Nuclear norm regularization weight.
        cccp_shrinkage_rho: Initial isotropic shrinkage strength.
        cccp_shrinkage_eps: Safeguard epsilon for denominator.

        # PCG solver parameters
        pcg_max_iterations: Maximum PCG iterations.
        pcg_tolerance: Relative residual tolerance for early stopping.
        num_iterations: Legacy alias for pcg_max_iterations.

        # Preconditioner update
        precond_update_frequency: Update preconditioner every N forward passes
            (1 = every pass, 0 = never update after first).

        # Attention kernel parameters
        kernel_temperature: Temperature scaling inside exp(). Higher = sharper.
        kernel_symmetric: If True, symmetrize the attention kernel.
        kernel_normalize_qk: If True, L2-normalize Q/K before computing scores.
        clip_abs: Absolute value clamp for numerical stability in iterative solves.
    """

    d_model: int
    num_heads: int
    head_dim: Optional[int] = None
    dropout: float = 0.0
    eps: float = 1e-6
    lambda_init: float = 3.0
    kernel_type: Literal["exp_attention", "rbf", "linear", "cosine"] = "exp_attention"
    xsa_mode: Literal["subtract_projection", "zero_diagonal", "mask"] = (
        "subtract_projection"
    )
    use_fused: bool = True
    seed: Optional[int] = None

    # LAKER preconditioner
    preconditioner_type: Literal["cccp", "fast", "diagonal", "none"] = "fast"
    preconditioner_rank: Optional[int] = 32
    cccp_num_directions: int = 64
    cccp_max_iterations: int = 20
    cccp_gamma: float = 0.1
    cccp_shrinkage_rho: float = 0.01
    cccp_shrinkage_eps: float = 1e-8

    # PCG solver (training: 10-20 iters; inference: 30-50)
    pcg_max_iterations: int = 20
    pcg_tolerance: float = 1e-2
    num_iterations: int = 10

    # Preconditioner update cadence
    precond_update_frequency: int = 1

    # Attention kernel
    kernel_temperature: float = 1.0
    kernel_symmetric: bool = False
    kernel_normalize_qk: bool = True

    # Numerical bounds
    clip_abs: float = 1e6

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.d_model // self.num_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        valid_kernels = ("exp_attention", "rbf", "linear", "cosine")
        if self.kernel_type not in valid_kernels:
            raise ValueError(
                f"kernel_type must be one of {valid_kernels}, "
                f"got '{self.kernel_type}'"
            )

        valid_xsa = ("subtract_projection", "zero_diagonal", "mask")
        if self.xsa_mode not in valid_xsa:
            raise ValueError(
                f"xsa_mode must be one of {valid_xsa}, got '{self.xsa_mode}'"
            )

        valid_precond = ("cccp", "fast", "diagonal", "none")
        if self.preconditioner_type not in valid_precond:
            raise ValueError(
                f"preconditioner_type must be one of {valid_precond}, "
                f"got '{self.preconditioner_type}'"
            )

        if self.pcg_max_iterations < 1:
            raise ValueError(f"pcg_max_iterations must be >= 1")
        if self.num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1")
        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError(f"dropout must be in [0,1]")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive")
        if self.lambda_init < 0:
            raise ValueError(f"lambda_init must be non-negative")

    @property
    def effective_pcg_iters(self) -> int:
        """Use pcg_max_iterations, falling back to num_iterations for compat."""
        return self.pcg_max_iterations
