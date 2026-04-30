"""
Conjugate Gradient solver for kernel attention systems.

This module provides a Conjugate Gradient (CG) implementation as an
alternative to Richardson iteration. CG typically converges in fewer
iterations but is more complex to implement differentiably.

NOTE: This implementation is provided for completeness but is NOT used
by default in the fused attention modules. The Richardson iteration is
preferred because:
1. It is simpler to make fully differentiable
2. With good preconditioning, convergence is adequate
3. It has more stable gradients for deep networks
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


def matvec_with_kernel(
    kernel: torch.Tensor,
    x: torch.Tensor,
    lambda_reg: float,
) -> torch.Tensor:
    """
    Compute (K + lambda * I) @ x.

    Args:
        kernel: Kernel matrix, shape ``(batch, num_heads, seq_len, seq_len)``.
        x: Vector, shape ``(batch, num_heads, seq_len, head_dim)``.
        lambda_reg: Regularization parameter.

    Returns:
        Result of matrix-vector product, same shape as x.
    """
    # K @ x
    kx = torch.matmul(kernel, x)
    # + lambda * x (diagonal addition)
    return kx + lambda_reg * x


def conjugate_gradient_solve(
    kernel: torch.Tensor,
    b: torch.Tensor,
    lambda_reg: float,
    x0: Optional[torch.Tensor] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Solve (K + lambda * I) @ x = b using Conjugate Gradient.

    This is a standard CG implementation with optional preconditioning.
    The algorithm is:

    .. math::

        r_0 = b - A x_0

        z_0 = P r_0  (preconditioning)

        p_0 = z_0

        \\text{for } k = 0, 1, ...:

            \\alpha_k = \\frac{r_k^T z_k}{p_k^T A p_k}

            x_{k+1} = x_k + \\alpha_k p_k

            r_{k+1} = r_k - \\alpha_k A p_k

            z_{k+1} = P r_{k+1}

            \\beta_k = \\frac{r_{k+1}^T z_{k+1}}{r_k^T z_k}

            p_{k+1} = z_{k+1} + \\beta_k p_k

    Args:
        kernel: Kernel matrix, shape ``(batch, num_heads, seq_len, seq_len)``.
        b: Right-hand side, shape ``(batch, num_heads, seq_len, head_dim)``.
        lambda_reg: Regularization parameter.
        x0: Initial guess. If None, starts from zero.
        max_iterations: Maximum number of CG iterations.
        tolerance: Convergence tolerance for residual norm.
        preconditioner: Optional preconditioning function P @ r.

    Returns:
        Tuple of (solution x, iterations used).
    """
    batch, num_heads, seq_len, head_dim = b.shape
    device = b.device

    # Initialize x
    if x0 is not None:
        x = x0.clone()
    else:
        x = torch.zeros_like(b)

    # Initial residual: r = b - A @ x
    Ax = matvec_with_kernel(kernel, x, lambda_reg)
    r = b - Ax

    # Apply preconditioner if provided
    if preconditioner is not None:
        z = preconditioner(r)
    else:
        z = r.clone()

    # Initial search direction
    p = z.clone()

    # Compute initial residual norm for convergence check
    rs_old = (r * z).sum(dim=(1, 2, 3), keepdim=True)

    iterations_completed = 0

    for iteration in range(max_iterations):
        # Compute A @ p
        Ap = matvec_with_kernel(kernel, p, lambda_reg)

        # Step size: alpha = (r^T z) / (p^T A p)
        pAp = (p * Ap).sum(dim=(1, 2, 3), keepdim=True)
        alpha = rs_old / (pAp + 1e-10)

        # Update solution: x = x + alpha * p
        x = x + alpha * p

        # Update residual: r = r - alpha * A p
        r = r - alpha * Ap

        # Apply preconditioner
        if preconditioner is not None:
            z = preconditioner(r)
        else:
            z = r.clone()

        # Compute new residual norm
        rs_new = (r * z).sum(dim=(1, 2, 3), keepdim=True)

        # Check convergence
        residual_norm = torch.sqrt((r * r).sum(dim=(1, 2, 3), keepdim=True))
        converged = (residual_norm < tolerance).all()

        iterations_completed = iteration + 1

        if converged:
            break

        # Compute beta and update search direction
        beta = rs_new / (rs_old + 1e-10)
        p = z + beta * p

        rs_old = rs_new

    return x, iterations_completed
