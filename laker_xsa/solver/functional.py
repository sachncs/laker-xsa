"""Functional operator for the regularized kernel linear system.

This module provides stateless functional counterparts to the class-based
solvers in :mod:`laker_xsa.solver`. The design follows the same pattern
as :mod:`torch.nn.functional` relative to :mod:`torch.nn`: the functional
forms accept all parameters explicitly and do not hold ``nn.Module`` state.

Functional form            Used by
-----------------          -----------------------------------------
apply_kernel_operator     ``pcg_solve`` / ``richardson_solve`` inner loop.
"""

from __future__ import annotations

import torch


def apply_kernel_operator(
    kernel: torch.Tensor,
    x: torch.Tensor,
    lambda_reg: torch.Tensor,
) -> torch.Tensor:
    """Apply the regularized kernel operator :math:`A(x) = (K + \\lambda I)\\,x`.

    This is implemented as ``kernel @ x + lambda_reg * x`` rather than by
    materializing ``K + lambda I``. It therefore avoids allocating a separate
    regularized kernel matrix; matrix-multiplication workspace remains backend
    dependent.

    Args:
        kernel: Kernel matrix with shape
            ``(batch, num_heads, n, n)``. The last two dims are
            interpreted as the rows and columns of a per-(batch, head)
            matrix.
        x: Right-hand side with shape
            ``(batch, num_heads, n, head_dim)``. Treated as a stack of
            ``head_dim`` vectors per (batch, head).
        lambda_reg: Regularization coefficient. At runtime, any scalar or
            tensor broadcastable to ``x`` can be used; incompatible shapes,
            devices, or dtypes are left to PyTorch to reject.

    Returns:
        torch.Tensor: Result of ``K @ x + lambda_reg * x`` with shape
        ``(batch, num_heads, n, head_dim)`` matching ``x``.

    Raises:
        RuntimeError: Propagated from the underlying ``torch.matmul``
            or elementwise add for incompatible shapes, dtypes, or
            devices. The function does not validate inputs.

    Notes:
        - The function performs no in-place modification of its inputs.
        - The dense ``matmul`` contributes
          :math:`O(n^2 \\cdot d)` arithmetic per batch/head and produces an
          output-shaped ``Kx`` tensor.
        - Gradients follow ordinary PyTorch autograd rules for tensor operands.

    Complexity:
        - Time: :math:`O(\\text{batch} \\cdot \\text{num\\_heads} \\cdot n^2 \\cdot d)`.
        - Memory beyond the inputs is output-sized,
          :math:`O(\\text{batch} \\cdot \\text{num\\_heads} \\cdot n \\cdot d)`,
          plus backend-dependent matrix-multiplication workspace.
    """
    Kx = torch.matmul(kernel, x)
    return Kx + lambda_reg * x
