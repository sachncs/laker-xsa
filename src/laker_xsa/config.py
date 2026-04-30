"""
Configuration module for LAKER-XSA attention.

This module defines the configuration dataclass that controls all hyperparameters
for the XSA and LAKER attention mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class XSA_LAKER_Config:
    """
    Configuration for the fused XSA + LAKER attention module.

    This configuration controls both the Exclusive Self Attention (XSA)
    and LAKER-style kernel regression components.

    Attributes:
        d_model: Dimension of input/output embeddings. Must be divisible by
            num_heads.
        num_heads: Number of attention heads.
        head_dim: Dimension per head. If None, defaults to d_model // num_heads.
        dropout: Dropout probability for attention weights. Default: 0.0.
        eps: Numerical stability epsilon for division and normalization.
            Default: 1e-6.
        num_iterations: Number of iterations for the preconditioned Richardson
            solver. Default: 10.
        preconditioner_rank: Rank of the low-rank preconditioner. If None,
            uses diagonal-only preconditioner. Default: None.
        kernel_type: Type of kernel function. Options: 'rbf', 'linear', 'cosine'.
            Default: 'rbf'.
        xsa_mode: Method for excluding self-attention. Options:
            - 'subtract_projection': Remove projection of output onto own value
            - 'zero_diagonal': Zero the diagonal of attention scores
            - 'mask': Use explicit binary mask
            Default: 'subtract_projection'.
        use_fused: If True, use the fused XSA + LAKER implementation.
            If False, use standard attention. Default: True.
            Note: This flag is provided for API compatibility; for explicit
            control, use attention_type in TransformerBlock instead.
        solver_tolerance: Convergence tolerance for iterative solver.
            Default: 1e-6.
        lambda_init: Initial value for learnable regularization parameter.
            Default: 0.1.
        seed: Random seed for reproducibility. If None, no seed is set.
            Default: None.

    Raises:
        ValueError: If d_model is not divisible by num_heads.
        ValueError: If an invalid kernel_type or xsa_mode is provided.

    Example:
        >>> config = XSA_LAKER_Config(
        ...     d_model=512,
        ...     num_heads=8,
        ...     num_iterations=10,
        ...     preconditioner_rank=32,
        ...     kernel_type="rbf",
        ... )
    """

    d_model: int
    num_heads: int
    head_dim: Optional[int] = None
    dropout: float = 0.0
    eps: float = 1e-6
    num_iterations: int = 10
    preconditioner_rank: Optional[int] = None
    kernel_type: Literal["rbf", "linear", "cosine"] = "rbf"
    xsa_mode: Literal["subtract_projection", "zero_diagonal", "mask"] = (
        "subtract_projection"
    )
    use_fused: bool = True
    solver_tolerance: float = 1e-6
    lambda_init: float = 0.1
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.head_dim is None:
            self.head_dim = self.d_model // self.num_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.kernel_type not in ("rbf", "linear", "cosine"):
            raise ValueError(
                f"kernel_type must be 'rbf', 'linear', or 'cosine', "
                f"got '{self.kernel_type}'"
            )

        if self.xsa_mode not in ("subtract_projection", "zero_diagonal", "mask"):
            raise ValueError(
                f"xsa_mode must be 'subtract_projection', 'zero_diagonal', "
                f"or 'mask', got '{self.xsa_mode}'"
            )

        if self.num_iterations < 1:
            raise ValueError(
                f"num_iterations must be at least 1, got {self.num_iterations}"
            )

        if self.preconditioner_rank is not None and self.preconditioner_rank < 1:
            raise ValueError(
                f"preconditioner_rank must be at least 1 if specified, "
                f"got {self.preconditioner_rank}"
            )

        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError(
                f"dropout must be in [0, 1], got {self.dropout}"
            )

        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

        if self.solver_tolerance <= 0:
            raise ValueError(
                f"solver_tolerance must be positive, got {self.solver_tolerance}"
            )

        if self.lambda_init < 0:
            raise ValueError(
                f"lambda_init must be non-negative, got {self.lambda_init}"
            )

        if self.seed is not None and self.seed < 0:
            raise ValueError(
                f"seed must be non-negative, got {self.seed}"
            )
