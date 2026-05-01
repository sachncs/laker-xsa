"""
Preconditioned Conjugate Gradient (PCG) solver for kernel attention.

Implements Algorithm 1 lines 15-24 from arXiv:2604.25138 (LAKER).
Solves (K + lambda*I) @ alpha = V with learned preconditioning.

Also provides Richardson iteration as a baseline option.

Key improvements over v1:
- Proper PCG with direction conjugacy (quadratic convergence)
- Convergence monitoring with relative residual
- Adaptive early stopping
- Full differentiability through unrolled iterations
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


from laker_xsa.solver.functional import apply_kernel_operator  # noqa: F401 — re-export


def pcg_solve(
    kernel: torch.Tensor,
    b: torch.Tensor,
    lambda_reg: torch.Tensor,
    precond_data=None,
    apply_preconditioner: Optional[Callable] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
    min_iterations: int = 3,
    x0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Solve (K + lambda*I) @ x = b using Preconditioned Conjugate Gradient.

    Algorithm (from LAKER paper):

        x_0 = 0
        r_0 = b
        z_0 = P @ r_0          (precondition)
        p_0 = z_0
        for k = 0, 1, ...:
            delta_k = (r_k^T @ z_k) / (p_k^T @ A @ p_k)
            x_{k+1} = x_k + delta_k * p_k
            r_{k+1} = r_k - delta_k * A @ p_k
            if ||r_{k+1}|| / ||b|| <= tol: break
            z_{k+1} = P @ r_{k+1}
            beta_k = (r_{k+1}^T @ z_{k+1}) / (r_k^T @ z_k)
            p_{k+1} = z_{k+1} + beta_k * p_k

    PCG converges quadratically for well-conditioned systems and linearly
    for ill-conditioned ones. With the LAKER preconditioner reducing
    condition numbers by ~1000x, convergence is dramatically faster.

    Args:
        kernel: (batch, num_heads, n, n) kernel matrix.
        b: (batch, num_heads, n, head_dim) right-hand side (values V).
        lambda_reg: Scalar or broadcastable regularization.
        precond_data: Preconditioner data from LakerPreconditioner.
        apply_preconditioner: Function (residual, precond_data) -> precond_residual.
        max_iterations: Maximum PCG steps.
        tolerance: Relative residual tolerance for early stopping.
        min_iterations: Minimum iterations before checking convergence.
        x0: Initial guess (None = zeros).

    Returns:
        Solution alpha: (batch, num_heads, n, head_dim).
    """
    batch, num_heads, n, head_dim = b.shape

    # Initial guess
    if x0 is not None:
        x = x0
    else:
        x = torch.zeros_like(b)

    # Initial residual: r_0 = b - A @ x_0
    Ax = apply_kernel_operator(kernel, x, lambda_reg)
    r = b - Ax

    # b_norm for relative residual (per sample, for convergence check)
    # Use a scalar estimate for efficiency
    b_norm = torch.sqrt((b * b).sum(dim=(-2, -1), keepdim=True))

    # Preconditioned residual: z_0 = P @ r_0
    if apply_preconditioner is not None and precond_data is not None:
        z = apply_preconditioner(r, precond_data)
    else:
        z = r

    # Initial search direction: p_0 = z_0
    p = z

    # r_k^T @ z_k (scalar per batch/head, used for beta)
    # Sum over seq and head_dim
    rz_old = (r * z).sum(dim=(-2, -1), keepdim=True)

    for iteration in range(max_iterations):
        # A @ p_k = (K + lambda*I) @ p_k
        Ap = apply_kernel_operator(kernel, p, lambda_reg)

        # delta_k = (r_k^T @ z_k) / (p_k^T @ A @ p_k)
        pAp = (p * Ap).sum(dim=(-2, -1), keepdim=True)
        delta = rz_old / (pAp + 1e-12)

        # x_{k+1} = x_k + delta_k * p_k
        x = x + delta * p

        # r_{k+1} = r_k - delta_k * A @ p_k
        r = r - delta * Ap

        # Check convergence: relative residual ||r|| / ||b||
        if iteration >= min_iterations - 1:
            r_norm = torch.sqrt((r * r).sum(dim=(-2, -1), keepdim=True))
            rel_residual = r_norm / (b_norm + 1e-12)

            if (rel_residual < tolerance).all():
                # At least one more iteration for safety
                if iteration >= min_iterations:
                    break

        # Precondition: z_{k+1} = P @ r_{k+1}
        if apply_preconditioner is not None and precond_data is not None:
            z = apply_preconditioner(r, precond_data)
        else:
            z = r

        # beta_k = (r_{k+1}^T @ z_{k+1}) / (r_k^T @ z_k)
        rz_new = (r * z).sum(dim=(-2, -1), keepdim=True)
        beta = rz_new / (rz_old + 1e-12)

        # p_{k+1} = z_{k+1} + beta_k * p_k
        p = z + beta * p

        rz_old = rz_new

        # Numerical safety: clamp to prevent explosion
        x = torch.clamp(x, -1e6, 1e6)

    return x


def richardson_solve(
    kernel: torch.Tensor,
    b: torch.Tensor,
    lambda_reg: torch.Tensor,
    precond_data=None,
    apply_preconditioner: Optional[Callable] = None,
    num_iterations: int = 10,
    omega: float = 1.0,
) -> torch.Tensor:
    """
    Solve (K + lambda*I) @ x = b using preconditioned Richardson iteration.

    x_{t+1} = x_t + omega * P @ (b - A @ x_t)

    Simpler than PCG but converges linearly. Provided for baseline comparison.

    Args:
        kernel: (batch, num_heads, n, n).
        b: (batch, num_heads, n, head_dim).
        lambda_reg: Regularization.
        precond_data: Preconditioner data.
        apply_preconditioner: Preconditioner application function.
        num_iterations: Fixed number of iterations.
        omega: Step size (1.0 = full step).

    Returns:
        Solution alpha: (batch, num_heads, n, head_dim).
    """
    x = torch.zeros_like(b)

    for _ in range(num_iterations):
        Ax = apply_kernel_operator(kernel, x, lambda_reg)
        residual = b - Ax

        if apply_preconditioner is not None and precond_data is not None:
            update = apply_preconditioner(residual, precond_data)
        else:
            update = residual

        x = x + omega * update
        x = torch.clamp(x, -1e6, 1e6)

    return x
