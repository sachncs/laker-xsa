"""Conditioning analysis for kernel attention.

This module measures safeguarded singular-value ratios for deprecated v1 kernel
matrices and applies the separate v1 position-based preconditioner to random
residuals. The metrics do not cover v2 ``LakerAttention``, which uses
``AttentionKernel`` and ``LakerPreconditioner``.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from laker_xsa.attention._legacy import KernelFunction, LearnedPreconditioner
from laker_xsa.config import XSA_LAKER_Config


def compute_kernel_condition_number(
    kernel: torch.Tensor,
    lambda_reg: float = 0.1,
) -> torch.Tensor:
    """Compute the condition number of the regularised kernel matrix.

    The ratio of largest to smallest singular value measures sensitivity of the
    regularized linear system. The implementation adds ``1e-10`` to the
    smallest singular value, so its finite estimate is a safeguarded proxy
    rather than the exact condition number near singularity.

    Args:
        kernel: Kernel matrix with shape
            ``(batch, num_heads, seq_len, seq_len)``.
        lambda_reg: Tikhonov regularisation coefficient added to the
            diagonal before measuring conditioning. Use ``0.0`` to probe
            the raw kernel.

    Returns:
        Tensor shaped ``(batch, num_heads)`` containing the per-slice
        safeguarded SVD ratio. If SVD raises ``RuntimeError``, the
        function instead returns the entrywise ``L1`` norm of each regularized
        slice. The fallback is not a condition number and preserves only the
        result shape, not the meaning or device: the successful path constructs
        a CPU tensor, while the fallback remains on ``kernel.device``.

    Side Effects:
        Does not mutate ``kernel``. On accelerator input, successful per-slice
        ``.item()`` calls synchronize and transfer scalar results to CPU.

    Complexity:
        ``O(batch * num_heads * seq_len ** 3)`` dominated by per-slice
        ``torch.svd``; the fallback path is ``O(batch * num_heads * seq_len ** 2)``.
    """
    batch, num_heads, seq_len, _ = kernel.shape

    # Tikhonov regularisation: add lambda_reg * I on every (batch, head)
    # slice, using explicit Python loops over the batch/head axes.
    kernel_reg = kernel.clone()
    eye = torch.eye(seq_len, device=kernel.device, dtype=kernel.dtype)
    for b in range(batch):
        for h in range(num_heads):
            kernel_reg[b, h] = kernel_reg[b, h] + eye * lambda_reg

    # Singular values support nonsymmetric kernels; only the values are needed.
    # Per-slice ``.item()`` transfers each scalar to Python/CPU.
    try:
        condition_numbers = []
        for b in range(batch):
            for h in range(num_heads):
                K = kernel_reg[b, h]
                s = torch.svd(K, compute_uv=False).S
                # 1e-10 floor guards against exact-zero sigma_min that would
                # otherwise produce inf/NaN that propagates into reports.
                cond = s.max() / (s.min() + 1e-10)
                condition_numbers.append(cond.item())

        return torch.tensor(condition_numbers).view(batch, num_heads)

    except RuntimeError:
        # Preserve the result shape if SVD raises, but change the metric to the
        # entrywise L1 norm. Callers cannot distinguish this path by shape.
        fallback = kernel_reg.abs().sum(dim=(-2, -1))
        return fallback


def compute_conditioning_metrics(
    config: XSA_LAKER_Config,
    seq_len: int = 128,
    num_samples: int = 10,
) -> Dict[str, Any]:
    """Compute a battery of kernel-conditioning diagnostics for v1 attention.

    For each of ``num_samples`` random ``(Q, K)`` pairs the function:

        1. Builds the kernel ``K = kernel_fn(Q, K)`` via the v1
           ``KernelFunction``.
        2. Records the raw condition number (no regularisation) and the
           Tikhonov-regularised condition number (``lambda_reg=0.1``).
        3. Routes the kernel diagonal through the v1
           ``LearnedPreconditioner`` and applies the preconditioner to a
           random residual, recording the post-precondition residual norm
           as a loose proxy for how aggressively the preconditioner
           reshapes error modes.

    Args:
        config: LAKER-XSA configuration. ``config.head_dim`` must be set
            (``config.__post_init__`` fills it from
            ``d_model // num_heads`` if left as ``None``, so in practice it
            is always set for a freshly constructed config).
        seq_len: Sequence length of the random ``Q, K`` tensors.
        num_samples: Number passed to ``range``. Non-positive values leave all
            metric lists empty and fail during aggregation; one yields ``NaN``
            for unbiased standard deviations.

    Returns:
        A dictionary with these scalar metrics and a sub-dict describing
        the configuration used:

        * ``raw_condition_mean`` / ``raw_condition_std`` — statistics of the
          un-regularised kernel condition number.
        * ``regularized_condition_mean`` / ``regularized_condition_std`` —
          statistics of the kernel condition number with
          ``lambda_reg=0.1``.
        * ``preconditioned_residual_norm`` — mean L2 norm of the residual
          after one application of the v1 preconditioner.
        * ``config`` — echo of ``kernel_type``, ``seq_len``, ``num_samples``.

    Raises:
        ValueError: If ``config.head_dim`` is ``None`` at call time
            (checked inside the sampling loop, so only when
            ``num_samples >= 1``).
        ZeroDivisionError: If ``num_samples`` is non-positive; the loop never
            runs and final averages divide by zero.

    Side Effects:
        Constructs deprecated v1 modules, which emit ``DeprecationWarning``,
        and advances PyTorch CPU or CUDA RNG state while initializing modules
        and random samples. Modules are moved to CUDA whenever it is available.

    Complexity:
        ``O(num_samples * (seq_len ** 3))`` dominated by the per-sample SVD
        used in :func:`compute_kernel_condition_number`.

    Note:
        With ``num_samples == 1``, ``torch.std`` uses its default unbiased
        correction and the ``*_std`` entries are ``NaN``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Intentionally instantiate the deprecated v1 path so the diagnostics
    # describe exactly the kernel/preconditioner combination used by the
    # legacy kernel-regression benchmarks.
    kernel_fn = KernelFunction(config.kernel_type).to(device)
    preconditioner = LearnedPreconditioner(config).to(device)

    raw_conditions = []
    regularized_conditions = []
    preconditioned_residuals = []

    for _ in range(num_samples):
        # Guard against stale configs that haven't been through __post_init__.
        if config.head_dim is None:
            raise ValueError("config.head_dim must be set for conditioning metrics")
        head_dim = int(config.head_dim)
        q = torch.randn(1, config.num_heads, seq_len, head_dim, device=device)
        k = torch.randn(1, config.num_heads, seq_len, head_dim, device=device)

        kernel = kernel_fn(q, k)

        # Raw kernel condition number (no Tikhonov regularisation).
        raw_cond = compute_kernel_condition_number(kernel, lambda_reg=0.0)
        raw_conditions.append(raw_cond.mean().item())

        # Regularised condition number (default Tikhonov coefficient).
        reg_cond = compute_kernel_condition_number(kernel, lambda_reg=0.1)
        regularized_conditions.append(reg_cond.mean().item())

        # Preconditioner effect: feed the diagonal through the v1 learned
        # preconditioner and apply it to a random residual. The output norm
        # is a rough proxy for how much the preconditioner reshapes the
        # residual; it is *not* a convergence guarantee.
        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag_precond, lr_precond = preconditioner(kernel_diag, seq_len)

        # Test preconditioner on random residual.
        residual = torch.randn_like(q)
        precond_residual = preconditioner.apply_precondition(
            residual, diag_precond, lr_precond
        )

        preconditioned_residuals.append(precond_residual.norm().item())

    return {
        "raw_condition_mean": sum(raw_conditions) / len(raw_conditions),
        "raw_condition_std": torch.tensor(raw_conditions).std().item(),
        "regularized_condition_mean": sum(regularized_conditions)
        / len(regularized_conditions),
        "regularized_condition_std": torch.tensor(regularized_conditions).std().item(),
        "preconditioned_residual_norm": sum(preconditioned_residuals)
        / len(preconditioned_residuals),
        "config": {
            "kernel_type": config.kernel_type,
            "seq_len": seq_len,
            "num_samples": num_samples,
        },
    }
