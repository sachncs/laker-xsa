"""
Conditioning analysis for kernel attention.

This module provides tools for analyzing the conditioning of the
kernel matrix and the effect of preconditioning.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention._legacy import KernelFunction, LearnedPreconditioner


def compute_kernel_condition_number(
    kernel: torch.Tensor,
    lambda_reg: float = 0.1,
) -> torch.Tensor:
    """
    Compute condition number of regularized kernel matrix.

    The condition number is the ratio of largest to smallest eigenvalue.
    A high condition number indicates an ill-conditioned matrix that
    is difficult to solve iteratively.

    Args:
        kernel: Kernel matrix, shape ``(batch, num_heads, seq_len, seq_len)``.
        lambda_reg: Regularization parameter.

    Returns:
        Condition number for each batch and head.
    """
    batch, num_heads, seq_len, _ = kernel.shape

    # Add regularization
    kernel_reg = kernel.clone()
    eye = torch.eye(seq_len, device=kernel.device, dtype=kernel.dtype)
    for b in range(batch):
        for h in range(num_heads):
            kernel_reg[b, h] = kernel_reg[b, h] + eye * lambda_reg

    # Compute eigenvalues (symmetric matrix, so real eigenvalues)
    # Using SVD for numerical stability
    try:
        # For each batch and head
        condition_numbers = []
        for b in range(batch):
            for h in range(num_heads):
                K = kernel_reg[b, h]
                # Singular values
                s = torch.svd(K, compute_uv=False).S
                # Condition number
                cond = s.max() / (s.min() + 1e-10)
                condition_numbers.append(cond.item())

        return torch.tensor(condition_numbers).view(batch, num_heads)

    except RuntimeError:
        # SVD may fail for large matrices; use trace-based estimate
        trace = kernel_reg.abs().sum(dim=(-2, -1))
        return trace


def compute_conditioning_metrics(
    config: XSA_LAKER_Config,
    seq_len: int = 128,
    num_samples: int = 10,
) -> Dict[str, Any]:
    """
    Compute conditioning metrics for kernel attention.

    Evaluates:
    - Raw kernel condition number
    - Condition number with regularization
    - Effect of learned preconditioner

    Args:
        config: Configuration object.
        seq_len: Sequence length for evaluation.
        num_samples: Number of random samples to average over.

    Returns:
        Dictionary with conditioning metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kernel_fn = KernelFunction(config.kernel_type).to(device)
    preconditioner = LearnedPreconditioner(config).to(device)

    raw_conditions = []
    regularized_conditions = []
    preconditioned_residuals = []

    for _ in range(num_samples):
        # Random Q, K
        if config.head_dim is None:
            raise ValueError("config.head_dim must be set for conditioning metrics")
        head_dim = int(config.head_dim)
        q = torch.randn(1, config.num_heads, seq_len, head_dim, device=device)
        k = torch.randn(1, config.num_heads, seq_len, head_dim, device=device)

        # Compute kernel
        kernel = kernel_fn(q, k)

        # Raw condition number (no regularization)
        raw_cond = compute_kernel_condition_number(kernel, lambda_reg=0.0)
        raw_conditions.append(raw_cond.mean().item())

        # Regularized condition number
        reg_cond = compute_kernel_condition_number(kernel, lambda_reg=0.1)
        regularized_conditions.append(reg_cond.mean().item())

        # Preconditioner effect
        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag_precond, lr_precond = preconditioner(kernel_diag, seq_len)

        # Test preconditioner on random residual
        residual = torch.randn_like(q)
        precond_residual = preconditioner.apply_precondition(
            residual, diag_precond, lr_precond
        )

        # Measure residual reduction (approximate)
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
