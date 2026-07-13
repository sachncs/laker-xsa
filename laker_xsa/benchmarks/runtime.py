"""Runtime profiling for LAKER-XSA attention.

This module provides two profilers:

* :func:`runtime_profile` — forward / backward wall-clock timing and
  peak GPU memory for a generic ``nn.Module``.
* :func:`profile_iterations` — per-iteration residual norm of the
  Richardson-style fixed-point solver used by the deprecated v1
  ``FusedXSALAKERAttention``.

Runtime-synchronisation behaviour:

* On CUDA, ``torch.cuda.synchronize()`` is called before the timing
  loop starts and after every timed forward/backward call so that
  ``time.perf_counter()`` measures device completion, not just kernel
  launch overhead.
* On CPU (or when CUDA is unavailable) no explicit synchronisation is
  performed; timings therefore include only the CPU-side wall-clock
  cost.
* GPU memory is only reported when CUDA is available; the
  ``memory_mb`` key is omitted from the result otherwise.

Important methodological notes:

* :func:`profile_iterations` mirrors the fixed-budget Richardson loop used by
  deprecated v1 ``FusedXSALAKERAttention``. It does not profile the v2 PCG
  implementation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from laker_xsa.attention._legacy import FusedXSALAKERAttention
from laker_xsa.config import XSA_LAKER_Config

_SOFTPLUS = nn.Softplus()


def runtime_profile(
    module: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Profile forward / backward runtime and peak GPU memory of a module.

    Args:
        module: The ``nn.Module`` to profile. Its ``forward`` must accept
            ``input_tensor`` with shape ``(batch, seq_len, d_model)``.
        input_tensor: The input tensor. Its ``requires_grad`` flag is
            mutated by this function: it is set to ``True`` before the
            backward-timing and memory blocks and is left ``True`` on
            exit (it is not restored). The tensor is not moved to
            another device.
        num_warmup: Number of un-timed forward passes used to amortise
            cuBLAS workspace selection, caching, and lazy
            initialisation costs.
        num_runs: Number of timed passes. Statistics are computed over
            this many samples. With ``num_runs == 0`` the statistics
            helper divides by zero (and ``min``/``max`` of an empty list
            raise ``ValueError``); with ``num_runs == 1`` the reported
            ``std_ms`` is ``NaN`` (unbiased std of a single sample).

    Returns:
        A dictionary with two sub-dictionaries ``"forward"`` and
        ``"backward"``, each containing ``mean_ms``, ``std_ms``,
        ``min_ms``, ``max_ms``. The ``"backward"`` timings wrap both the
        forward call and ``out.sum().backward()`` for each iteration, so
        they measure forward+backward, not backward in isolation. When
        CUDA is available, a top-level ``"memory_mb"`` key holds the peak
        GPU memory allocated during a single forward+backward.

    Side Effects:
        * Leaves ``module`` in training mode after timing.
        * Leaves ``input_tensor.requires_grad`` set to ``True`` and clears only
          ``input_tensor.grad`` between samples. Parameter gradients accumulate
          across backward samples and the optional memory pass.
        * Resets and reads the process-wide CUDA peak-memory counter when CUDA
          is available.
        * Synchronizes CUDA after every timed call.

    Raises:
        ZeroDivisionError: If ``num_runs`` is zero while statistics are reduced.
        ValueError: If no timing samples are available to ``min`` or ``max``.
        RuntimeError: Propagated from module, autograd, or CUDA operations.

    Limitations:
        Warmup covers forward only. CUDA ``memory_mb`` is the allocator's
        absolute peak after resetting its peak counter, not incremental memory
        attributable only to this call.

    Complexity:
        Each timed pass is ``O(cost(forward(module, input_tensor)))``.
        The memory pass allocates a single additional forward+backward
        graph on top of that.
    """
    # Warm up module and backend caches; durations are discarded.
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = module(input_tensor)

    if torch.cuda.is_available():
        # Make sure prior async work is done before the timed window.
        torch.cuda.synchronize()

    # Collect individual no-grad forward timings.
    forward_times: List[float] = []
    module.eval()
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = module(input_tensor)
            if torch.cuda.is_available():
                # Per-iteration sync so each interval measures device
                # completion rather than kernel-launch latency.
                torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)

    # Collect complete forward-and-backward timings.
    backward_times: List[float] = []
    module.train()
    input_tensor.requires_grad_(True)
    for _ in range(num_runs):
        start = time.perf_counter()
        out = module(input_tensor)
        # Sum-to-scalar backward: ensures every leaf in the autograd
        # graph receives a gradient, matching typical training behaviour.
        out.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start)
        # Reset only the input leaf's grad; parameter grads accumulate by
        # design and will be cleared by the next ``zero_grad()`` in a
        # real optimiser loop.
        input_tensor.grad = None

    # Record CUDA allocator peak memory for one additional pass.
    memory_mb: Optional[float] = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        input_tensor.requires_grad_(True)
        out = module(input_tensor)
        out.sum().backward()
        # Convert bytes -> MiB.
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

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
    """Profile the per-iteration residual norm of the v1 Richardson solver.

    Re-implements the inner loop of the v1 ``FusedXSALAKERAttention``
    ``solve_system`` (deprecated). Both the setup phase (Q/K/V
    projections, kernel construction, preconditioner build) and the
    iterative loop run inside a single ``torch.no_grad()`` block, so the
    residual norms reflect pure forward-pass behaviour and are not
    entangled with the autograd tape.

    This helper duplicates the v1 Richardson loop for inspection. The v2
    :class:`~laker_xsa.attention.laker.LakerAttention` uses a different
    PCG-style recurrence and is not profiled by this function.

    Args:
        config: LAKER-XSA configuration. ``config.head_dim`` must resolve
            to a positive integer (post-init fills it from
            ``d_model // num_heads``).
        seq_len: Sequence length of the synthetic single-batch
            ``(1, seq_len, d_model)`` input.
        max_iterations: Number of Richardson iterations to run. On each
            iteration the current residual is recorded *before* the
            preconditioner update is applied, so ``residual_norms`` holds
            one entry per iteration measured at the iterate from the end
            of the previous step. With ``max_iterations == 0`` no
            residual is recorded and the ``residuals[0]`` / ``residuals[-1]``
            accesses below raise ``IndexError``.

    Returns:
        A dictionary with:

        * ``iterations`` — list ``[1, 2, ..., max_iterations]``.
        * ``residual_norms`` — list of per-iteration L2 norms of the
          residual ``|| V - (K_reg @ alpha) ||`` recorded before each
          update. The first entry is taken at ``alpha == 0`` and is
          therefore ``||V||``.
        * ``initial_residual`` — same as ``residuals[0]`` (the
          ``alpha == 0`` residual).
        * ``final_residual`` — same as ``residuals[-1]`` (recorded before
          the last update, not after it).
        * ``reduction_factor`` — ratio
          ``residuals[0] / (residuals[-1] + 1e-10)``. It is not a convergence
          test and does not include the final update's post-update residual.

    Raises:
        IndexError: If ``max_iterations`` is non-positive, because no residual
            is available for the summary's first/last indexing.
        RuntimeError: Propagated from tensor operations for incompatible
            configuration, shapes, devices, or dtypes.

    Side Effects:
        Allocates a deprecated v1 attention module and random tensors on the
        automatically selected device. Construction emits deprecation warnings.
        No autograd graph or parameter gradients are created.

    Assumptions:
        Single batch (``batch=1``) — the inner regularisation loop runs
        over ``num_heads`` only.

    Complexity:
        ``O(max_iterations * num_heads * seq_len ** 2 * head_dim)``
        dominated by the ``matmul(kernel_reg, alpha)`` call (``kernel_reg``
        is dense ``(seq_len, seq_len)`` and ``alpha`` is
        ``(seq_len, head_dim)`` per head).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Deprecated v1 fused XSA + LAKER attention. This is the legacy
    # Richardson-profiling path, intentionally retained.
    attn = FusedXSALAKERAttention(config).to(device)
    attn.eval()

    x = torch.randn(1, seq_len, config.d_model, device=device)

    # Setup phase: extract the Q/K/V projections and build the kernel,
    # exactly mirroring the v1 module's pre-solve pipeline.
    with torch.no_grad():
        q = attn.w_q(x)
        k = attn.w_k(x)
        v = attn.w_v(x)

        # Reshape ``(B, S, D)`` -> ``(B, H, S, d_h)``.
        q = q.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
        k = k.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
        v = v.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)

        kernel = attn.kernel_fn(q, k)
        kernel = attn.apply_xsa_to_kernel(kernel)

        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag_precond, lr_precond = attn.preconditioner(kernel_diag, seq_len)

        # softplus keeps ``lambda_reg`` strictly positive as a ridge
        # term; the ``config.eps`` floor matches the v1 module's
        # stabiliser. Note this does not by itself guarantee an SPD
        # system when the kernel is nonsymmetric.
        lambda_reg = _SOFTPLUS(attn.lambda_reg) + config.eps

        kernel_reg = kernel.clone()
        eye = torch.eye(seq_len, device=device, dtype=kernel.dtype)
        # Batch size is 1 in this profiler; iterate over heads only.
        for b in range(1):
            for h in range(config.num_heads):
                kernel_reg[b, h] = kernel_reg[b, h] + eye * lambda_reg

        # Track residual over Richardson iterations.
        alpha = torch.zeros_like(v)
        residuals: List[float] = []

        for _ in range(max_iterations):
            # Apply the regularised kernel to the current iterate.
            k_alpha = torch.matmul(kernel_reg, alpha)
            residual = v - k_alpha
            # ``.item()`` forces synchronisation and gives a true scalar.
            res_norm = residual.norm().item()
            residuals.append(res_norm)

            # Apply v1 learned preconditioner (diagonal + low-rank).
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
