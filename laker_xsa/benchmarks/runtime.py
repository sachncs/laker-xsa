"""
Runtime profiling for LAKER-XSA attention.

This module provides detailed runtime profiling including
memory usage and iteration convergence analysis.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention._legacy import FusedXSALAKERAttention


def runtime_profile(
    module: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """
    Profile runtime of a module.

    Args:
        module: Module to profile.
        input_tensor: Input tensor.
        num_warmup: Number of warmup runs.
        num_runs: Number of timing runs.

    Returns:
        Dictionary with timing statistics.
    """
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = module(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Forward pass timing
    forward_times: List[float] = []
    module.eval()
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = module(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)

    # Backward pass timing
    backward_times: List[float] = []
    module.train()
    input_tensor.requires_grad_(True)
    for _ in range(num_runs):
        start = time.perf_counter()
        out = module(input_tensor)
        out.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start)
        input_tensor.grad = None

    # Memory profiling
    memory_mb: Optional[float] = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        input_tensor.requires_grad_(True)
        out = module(input_tensor)
        out.sum().backward()
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Compute statistics
    def stats(times: List[float]) -> Dict[str, float]:
        times_ms = [t * 1000 for t in times]
        return {
            "mean_ms": sum(times_ms) / len(times_ms),
            "std_ms": torch.tensor(times_ms).std().item(),
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
        }

    result: Dict[str, Any] = {
        "forward": stats(forward_times),
        "backward": stats(backward_times),
    }

    if memory_mb is not None:
        result["memory_mb"] = memory_mb

    return result


def profile_iterations(
    config: XSA_LAKER_Config,
    seq_len: int = 128,
    max_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Profile convergence of iterative solver.

    Measures residual norm at each iteration to analyze convergence.

    Args:
        config: Configuration object.
        seq_len: Sequence length.
        max_iterations: Maximum iterations to profile.

    Returns:
        Dictionary with convergence data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attn = FusedXSALAKERAttention(config).to(device)
    attn.eval()

    x = torch.randn(1, seq_len, config.d_model, device=device)

    # Get internal tensors
    with torch.no_grad():
        q = attn.w_q(x)
        k = attn.w_k(x)
        v = attn.w_v(x)

        q = q.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
        k = k.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
        v = v.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)

        kernel = attn.kernel_fn(q, k)
        kernel = attn.apply_xsa_to_kernel(kernel)

        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag_precond, lr_precond = attn.preconditioner(kernel_diag, seq_len)

        lambda_reg = (
            torch.nn.functional.softplus(attn.lambda_reg) + config.eps
        )  # pylint: disable=not-callable

        kernel_reg = kernel.clone()
        eye = torch.eye(seq_len, device=device, dtype=kernel.dtype)
        for b in range(1):
            for h in range(config.num_heads):
                kernel_reg[b, h] = kernel_reg[b, h] + eye * lambda_reg

        # Track residual over iterations
        alpha = torch.zeros_like(v)
        residuals: List[float] = []

        for _ in range(max_iterations):
            k_alpha = torch.matmul(kernel_reg, alpha)
            residual = v - k_alpha
            res_norm = residual.norm().item()
            residuals.append(res_norm)

            precond_residual = attn.preconditioner.apply_precondition(
                residual, diag_precond, lr_precond
            )
            alpha = alpha + precond_residual

    return {
        "iterations": list(range(1, max_iterations + 1)),
        "residual_norms": residuals,
        "initial_residual": residuals[0],
        "final_residual": residuals[-1],
        "reduction_factor": residuals[0] / (residuals[-1] + 1e-10),
    }
