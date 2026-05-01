"""Functional API for linear system solvers.

This module provides stateless functional counterparts to the class-based
solvers. These follow the same design as torch.nn.functional relative to
torch.nn: the functional forms accept all parameters explicitly and do not
hold state.

Functional form            Class-based form
-----------------          -----------------
apply_kernel_operator     (operator application used by pcg_solve)
"""

from __future__ import annotations

import torch


def apply_kernel_operator(
    kernel: torch.Tensor,
    x: torch.Tensor,
    lambda_reg: torch.Tensor,
) -> torch.Tensor:
    """Apply the regularized kernel operator: (K + lambda*I) @ x.

    Args:
        kernel: (batch, num_heads, n, n) kernel matrix.
        x: (batch, num_heads, n, head_dim) vector.
        lambda_reg: Scalar or broadcastable tensor.

    Returns:
        (batch, num_heads, n, head_dim) result.
    """
    Kx = torch.matmul(kernel, x)
    return Kx + lambda_reg * x
