"""Preconditioned Conjugate Gradient (PCG) and Richardson iterations.

Both functions apply the regularized operator

.. math::

    A(x) = Kx + \\lambda x

in batches. :func:`pcg_solve` implements the PCG-style recurrence already
associated in this repository with LAKER (``arXiv:2604.25138``), and
:func:`richardson_solve` implements a fixed number of preconditioned
Richardson updates.

Classical PCG requires a symmetric positive-definite operator and a compatible
SPD preconditioner. This module does not validate symmetry, definiteness,
finite values, or convergence. Adding a positive ``lambda`` is insufficient to
establish those properties for an arbitrary nonsymmetric or indefinite
``kernel``.

For each batch/head pair, reductions combine both the sequence and
``head_dim`` axes, so all right-hand-side columns share each scalar PCG step.
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
    """Solve :math:`(K + \\lambda I)\\,x = b` by Preconditioned Conjugate Gradient.

    Implements the PCG-style recurrence identified in the repository as
    Algorithm 1, lines 15-24 of the LAKER reference
    (``arXiv:2604.25138``), with added denominator constants, iterate clamping,
    and residual-based early stopping. The recurrence has PCG's usual
    guarantees only when its operator and preconditioner meet the required
    symmetry and positive-definiteness assumptions; this function does not
    check them.

    Algorithm (per (batch, head)):

    .. code-block:: text

        x_0 = x0 (default 0)
        r_0 = b - A @ x_0
        z_0 = P @ r_0
        p_0 = z_0
        rTz_old = r_0^T @ z_0
        for k = 0, 1, ...:
            Ap   = A @ p_k
            pAp  = p_k^T @ Ap
            delta = rTz_old / (pAp + 1e-12)
            x_{k+1} = x_k + delta * p_k
            r_{k+1} = r_k - delta * Ap
            if k >= min_iterations - 1 and ||r_{k+1}|| / ||b|| <= tol: break
            z_{k+1} = P @ r_{k+1}
            rTz_new = r_{k+1}^T @ z_{k+1}
            beta    = rTz_new / (rTz_old + 1e-12)
            p_{k+1} = z_{k+1} + beta * p_k
            rTz_old = rTz_new

    Applicability:

    - Classical finite-step PCG theory assumes an SPD system and compatible
      SPD preconditioning, plus exact arithmetic for the ``n``-step result.
    - This implementation does not enforce those assumptions and does not
      raise merely because residuals stagnate, grow, or become non-finite.
    - It can return after ``max_iterations`` without satisfying ``tolerance``;
      no convergence status is returned.

    Args:
        kernel: Matrix with shape ``(batch, num_heads, n, n)``. PCG requires
            the regularized operator derived from it to be SPD, but this is not
            checked.
        b: Right-hand side with shape
            ``(batch, num_heads, n, head_dim)``. This is the values
            tensor ``V`` in the LAKER pipeline.
        lambda_reg: Regularization coefficient. Same broadcasting rules
            as :func:`apply_kernel_operator`; should be a Python float,
            a 0-d tensor, or broadcastable to ``(1, 1, 1, 1)``.
        precond_data: Opaque payload passed unchanged to
            ``apply_preconditioner``. ``None`` selects the identity-preconditioned
            recurrence.
        apply_preconditioner: Callable
            ``(residual, precond_data) -> preconditioned_residual``.
            Typically ``LakerPreconditioner.apply_preconditioner``.
            ``None`` disables preconditioning.
        max_iterations: Loop cap. No validation is performed; a value below
            one skips the loop and returns the initial guess.
        tolerance: Early-stopping threshold on the per-sample relative
            residual :math:`\\|r\\|_2 / \\|b\\|_2`. A scalar
            ``(rel_residual < tolerance).all()`` test is used so every
            sample must satisfy the tolerance before stopping.
        min_iterations: Iteration index gate for convergence checks. With the
            defaults, the first check occurs after three updates. If that check
            passes, one more update is attempted because breaking additionally
            requires ``iteration >= min_iterations``. Values are not
            validated.
        x0: Optional initial guess with the same shape and dtype as
            ``b``. ``None`` (default) initializes with zeros.

    Returns:
        The final iterate with the same shape as ``b``. It may be unconverged
        or non-finite; the API returns no status or residual.

    Raises:
        RuntimeError: Tensor operations may raise, for example on incompatible
            shapes, dtypes, or devices. Numerical nonconvergence and NaNs do
            not automatically raise.

    Side Effects:
        The solver does not modify tensor arguments in place. A caller-supplied
        ``apply_preconditioner`` callback may have its own side effects.

    Notes:
        - All reductions are taken over the trailing ``(n, head_dim)``
          axes with ``keepdim=True`` so per-(batch, head) scalars are
          broadcast against the full iterate.
        - The constants added to the ``delta`` and ``beta`` denominators are
          exactly ``1e-12``. They perturb the recurrence and do not ensure a
          meaningful step when a denominator is near zero.
        - ``x`` is clamped to ``[-1e6, 1e6]`` after direction updates, except
          when the loop breaks before reaching that statement. Clamping does
          not remove NaNs.
        - Tensor operations remain in the autograd graph unless the caller
          disables gradient tracking. Early stopping means the number of
          unrolled iterations can vary with the data.
        - The early-stop test requires all batch/head residuals to satisfy the
          threshold. The separate systems share loop termination but not their
          matrix-vector products or scalar updates.

    Complexity:
        - Each iteration performs one dense kernel application with time
          :math:`O(\\text{batch} \\cdot \\text{num\\_heads} \\cdot n^2 \\cdot d)`,
          plus the configured preconditioner application and
          :math:`O(\\text{batch} \\cdot \\text{num\\_heads} \\cdot n \\cdot d)`
          vector operations.
        - Excluding inputs and autograd retention, live work tensors require
          :math:`O(\\text{batch} \\cdot \\text{num\\_heads} \\cdot n \\cdot d)`
          storage. The kernel itself is
          :math:`O(\\text{batch} \\cdot \\text{num\\_heads} \\cdot n^2)`, and
          retaining the unrolled autograd graph increases memory with the
          executed iteration count.
    """
    _batch, _num_heads, _n, _head_dim = b.shape

    if x0 is not None:
        x = x0
    else:
        x = torch.zeros_like(b)

    Ax = apply_kernel_operator(kernel, x, lambda_reg)
    r = b - Ax

    # Batched per-(batch, head) L2 norm with keepdim for broadcasting.
    b_norm = torch.sqrt((b * b).sum(dim=(-2, -1), keepdim=True))

    if apply_preconditioner is not None and precond_data is not None:
        z = apply_preconditioner(r, precond_data)
    else:
        z = r

    p = z

    # Per-(batch, head) inner product r^T z, reduced across all RHS columns.
    rz_old = (r * z).sum(dim=(-2, -1), keepdim=True)

    for iteration in range(max_iterations):
        Ap = apply_kernel_operator(kernel, p, lambda_reg)

        pAp = (p * Ap).sum(dim=(-2, -1), keepdim=True)
        # Small fixed denominator perturbation; it does not validate curvature
        # or prevent non-finite arithmetic.
        delta = rz_old / (pAp + 1e-12)

        x = x + delta * p

        r = r - delta * Ap

        # Convergence check is gated by min_iterations so the iterate is
        # updated at least that many times even when the residual drops
        # below the tolerance immediately.
        if iteration >= min_iterations - 1:
            r_norm = torch.sqrt((r * r).sum(dim=(-2, -1), keepdim=True))
            rel_residual = r_norm / (b_norm + 1e-12)

            if (rel_residual < tolerance).all():
                # The additional index condition delays the break by one update
                # when the first permitted check succeeds.
                if iteration >= min_iterations:
                    break

        if apply_preconditioner is not None and precond_data is not None:
            z = apply_preconditioner(r, precond_data)
        else:
            z = r

        rz_new = (r * z).sum(dim=(-2, -1), keepdim=True)
        # Small fixed denominator perturbation; NaNs and near-breakdown are not
        # detected separately.
        beta = rz_new / (rz_old + 1e-12)

        p = z + beta * p

        rz_old = rz_new

        # Clamp after the direction update. A convergence break above skips
        # this clamp, and NaNs remain NaN.
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
    """Solve :math:`(K + \\lambda I)\\,x = b` by preconditioned Richardson iteration.

    The fixed-point update is

    .. math::

        x_{t+1} = x_t + \\omega\\, P\\,(b - A\\,x_t)

    where :math:`P` is the supplied preconditioner (``P = I`` when no
    preconditioner is provided). The iteration converges only when the
    iteration matrix :math:`I - \\omega P A` has spectral radius below one;
    this function does not check that condition.

    Unlike :func:`pcg_solve`, this routine has no residual test and always runs
    the requested Python ``range(num_iterations)``.

    Args:
        kernel: Kernel matrix with shape
            ``(batch, num_heads, n, n)``.
        b: Right-hand side with shape
            ``(batch, num_heads, n, head_dim)``.
        lambda_reg: Regularization coefficient broadcastable to
            ``(1, 1, 1, 1)``.
        precond_data: Opaque preconditioner payload. ``None`` disables
            preconditioning.
        apply_preconditioner: Callable
            ``(residual, precond_data) -> preconditioned_residual``.
            ``None`` disables preconditioning.
        num_iterations: Number passed to ``range``. Zero or a negative value
            performs no updates and returns zeros; no validation is performed.
        omega: Step size multiplying each preconditioned residual. No
            convergence-compatible range is enforced.

    Returns:
        Final iterate with the same shape as ``b``. It may be unconverged or
        non-finite.

    Raises:
        RuntimeError: Propagated from :func:`apply_kernel_operator` or
            from the preconditioner application when shapes, dtypes,
            or devices are incompatible. The function does not
            otherwise raise on numerical issues; non-finite iterates
            are returned as-is.

    Side Effects:
        The solver does not modify tensor arguments in place. A caller-supplied
        preconditioner callback may have its own side effects.

    Notes:
        - The iterate is clamped to ``[-1e6, 1e6]`` after each update; NaNs
          remain NaN.
        - There is no convergence monitoring or minimum-iteration guard.

    Complexity:
        - Time: :math:`O(\\text{num\\_iterations} \\cdot \\text{batch}
          \\cdot \\text{num\\_heads} \\cdot n^2 \\cdot d)` for the dense
          kernel applications, plus the preconditioner application
          each iteration (mode-dependent cost).
        - Memory: :math:`O(\\text{batch} \\cdot \\text{num\\_heads}
          \\cdot n \\cdot d)` for the iterate and the residual.
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
        # Numerical safeguard: clamp the iterate to prevent runaway values
        # when omega is too large or the system is poorly conditioned.
        x = torch.clamp(x, -1e6, 1e6)

    return x
