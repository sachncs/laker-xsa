#!/usr/bin/env python3
"""Benchmark script for LAKER-XSA models.

Runs runtime and peak-memory benchmarks comparing different attention
variants and writes a JSON report to disk.

Usage:
    python -m laker_xsa.cli.benchmark --output results.json

Important methodological notes:

* The ``kernel`` and ``fused`` arms currently instantiate deprecated v1
  ``KernelAttentionRegression`` and ``FusedXSALAKERAttention`` rather than
  :class:`~laker_xsa.attention.laker.LakerAttention`. Results from those arms
  do not characterize the v2 implementation.

* :func:`benchmark_attention` uses ``time.perf_counter`` for wall-clock
  timing and calls ``torch.cuda.synchronize()`` before and after each
  timed loop (not per iteration) on CUDA so timings measure device
  completion, not kernel-launch latency. CPU runs do not synchronise.

* Memory is only reported when CUDA is available; on CPU the
  ``memory_mb`` slot is set to ``0.0``.

* The ``--cuda`` flag is a hint/acknowledgement that is printed about
  when CUDA is unavailable but does **not** override the device
  selection (which always prefers CUDA when available). The defaults
  already encode this behaviour; the flag is informational.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import torch

from laker_xsa.attention import (
    ExclusiveSelfAttention,
    StandardMultiHeadAttention,
)
from laker_xsa.attention._legacy import (
    FusedXSALAKERAttention,
    KernelAttentionRegression,
)
from laker_xsa.config import XSA_LAKER_Config


def benchmark_attention(
    attn_module: torch.nn.Module,
    x: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark forward and backward pass of an attention module.

    Args:
        attn_module: Attention module to benchmark. Its ``forward`` must
            accept ``x`` with shape ``(batch, seq_len, d_model)``.
        x: Input tensor of shape ``(batch, seq_len, d_model)``. The
            function mutates its ``requires_grad`` flag (it is set to
            ``True`` for backward timing and left set).
        num_runs: Number of timed iterations per pass.
        warmup_runs: Number of *un-timed* forward passes used to amortise
            cuBLAS-workspace selection and cache warm-up costs.

    Returns:
        A dictionary with three keys:

        * ``forward_ms`` — mean forward pass wall-clock time, in ms.
        * ``backward_ms`` — mean forward+backward wall-clock time, in ms.
        * ``memory_mb`` — peak GPU memory allocated in MiB, or ``0.0`` on
          CPU.

    Side Effects:
        * Synchronizes CUDA before and after each timed loop.
        * Leaves ``x.requires_grad`` set and accumulates parameter gradients
          during backward timing and the memory pass; only ``x.grad`` is reset.
        * Resets and reads CUDA peak-memory statistics when CUDA is available.
        * Runs the module in its existing training/evaluation mode and may
          mutate any state the module normally updates during ``forward``.

    Raises:
        ZeroDivisionError: If ``num_runs`` is zero.
        RuntimeError: Propagated from module execution or CUDA operations.

    Note:
        Each backward sample includes both forward graph construction and
        ``backward``. The warmup loop exercises forward only.

    Complexity:
        ``O((warmup_runs + 2 * num_runs) * cost(module.forward(x)))``.
    """
    # Warm up module and backend caches in the same autograd mode as callers.
    for _ in range(warmup_runs):
        _ = attn_module(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time forward calls without autograd.
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = attn_module(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start) / num_runs * 1000  # ms

    # Time complete forward-and-backward calls.
    x.requires_grad_(True)
    start = time.perf_counter()
    for _ in range(num_runs):
        out = attn_module(x)
        # Sum-to-scalar backward ensures every leaf in the autograd graph
        # receives a gradient, matching typical training behaviour.
        out.sum().backward()
        # Reset only the input leaf; parameter grads accumulate by design.
        x.grad = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    backward_time = (time.perf_counter() - start) / num_runs * 1000  # ms

    # Record CUDA allocator peak memory for one additional pass.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        x.requires_grad_(True)
        out = attn_module(x)
        out.sum().backward()
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0.0

    return {
        "forward_ms": round(forward_time, 3),
        "backward_ms": round(backward_time, 3),
        "memory_mb": round(memory_mb, 2),
    }


def run_benchmarks(
    d_model: int = 512,
    num_heads: int = 8,
    seq_lens: Optional[List[int]] = None,
    num_runs: int = 50,
) -> Dict[str, Any]:
    """Run the full benchmark suite across the four attention variants.

    Evaluates, per sequence length, four arms:

        * ``"standard"`` — :class:`~laker_xsa.attention.StandardMultiHeadAttention`.
        * ``"xsa"`` — :class:`~laker_xsa.attention.ExclusiveSelfAttention`.
        * ``"kernel"`` — deprecated v1
          :class:`~laker_xsa.attention._legacy.KernelAttentionRegression`.
        * ``"fused"`` — deprecated v1
          :class:`~laker_xsa.attention._legacy.FusedXSALAKERAttention`.

    The v2 :class:`~laker_xsa.attention.laker.LakerAttention` is not included,
    so the output should not be presented as a v2 LAKER benchmark.

    Args:
        d_model: Model (embedding) dimension used for every arm.
        num_heads: Number of attention heads; ``d_model`` must be
            divisible by ``num_heads``.
        seq_lens: Sequence lengths to benchmark. Defaults to
            ``[64, 128, 256, 512]`` when not provided.
        num_runs: Number of timed passes per pass direction.

    Returns:
        A dictionary with three keys:

        * ``"config"`` — echo of the benchmark configuration.
        * ``"attention_types"`` — list of arm names actually evaluated.
        * ``"results"`` — ``{seq_len: {arm_name: benchmark_attention(...)}}``.

    Side Effects:
        Instantiates four modules per sequence length and moves them to
        ``cuda`` if available, else ``cpu``. Writes progress text to
        :data:`sys.stdout` via ``print``.

    Limitations:
        Each arm receives ten untimed warmup forwards through the default
        ``warmup_runs`` argument. Parameter gradients are not cleared between
        backward samples, and CUDA memory is the allocator's absolute peak
        after resetting its counter rather than an incremental model-only
        measurement.

    Complexity:
        ``O(|seq_lens| * num_runs * sum(arm_cost(seq_len)))``. The v1
        ``kernel`` and ``fused`` arms scale ``O(seq_len ** 2)`` due to
        materialising the full kernel matrix.
    """
    if seq_lens is None:
        seq_lens = [64, 128, 256, 512]

    # ``num_iterations=10`` and ``preconditioner_rank=d_model // 16`` are
    # hard-coded here; they are not exposed as caller-facing knobs.
    config = XSA_LAKER_Config(
        d_model=d_model,
        num_heads=num_heads,
        num_iterations=10,
        preconditioner_rank=d_model // 16,
    )

    results: Dict[str, Any] = {
        "config": {
            "d_model": d_model,
            "num_heads": num_heads,
            "seq_lens": seq_lens,
            "num_runs": num_runs,
        },
        "attention_types": ["standard", "xsa", "kernel", "fused"],
        "results": {},
    }

    for seq_len in seq_lens:
        # Visible progress for the CLI; intentionally uses ``print``
        # rather than the logging module because this is an interactive
        # entry point.
        print(f"\nBenchmarking seq_len={seq_len}...")
        x = torch.randn(
            4, seq_len, d_model, device="cuda" if torch.cuda.is_available() else "cpu"
        )

        results["results"][seq_len] = {}

        # Standard attention.
        attn_std = StandardMultiHeadAttention(config)
        if torch.cuda.is_available():
            attn_std = attn_std.cuda()
        results["results"][seq_len]["standard"] = benchmark_attention(
            attn_std, x, num_runs
        )
        print(
            f"  Standard: {results['results'][seq_len]['standard']['forward_ms']:.2f}ms"
        )

        attn_xsa = ExclusiveSelfAttention(config)
        if torch.cuda.is_available():
            attn_xsa = attn_xsa.cuda()
        results["results"][seq_len]["xsa"] = benchmark_attention(attn_xsa, x, num_runs)
        print(f"  XSA: {results['results'][seq_len]['xsa']['forward_ms']:.2f}ms")

        # Kernel attention (deprecated v1 — intentional).
        attn_kernel = KernelAttentionRegression(config)
        if torch.cuda.is_available():
            attn_kernel = attn_kernel.cuda()
        results["results"][seq_len]["kernel"] = benchmark_attention(
            attn_kernel, x, num_runs
        )
        print(f"  Kernel: {results['results'][seq_len]['kernel']['forward_ms']:.2f}ms")

        # Fused XSA + LAKER (deprecated v1 — intentional).
        attn_fused = FusedXSALAKERAttention(config)
        if torch.cuda.is_available():
            attn_fused = attn_fused.cuda()
        results["results"][seq_len]["fused"] = benchmark_attention(
            attn_fused, x, num_runs
        )
        print(f"  Fused: {results['results'][seq_len]['fused']['forward_ms']:.2f}ms")

    return results


def main() -> None:
    """CLI entry point for the runtime benchmark.

    Parses command-line arguments, runs :func:`run_benchmarks`, writes
    the result dictionary to the path given by ``--output`` as JSON, and
    prints a human-readable summary table to :data:`sys.stdout`.

    argparse behaviour:
        Calling :func:`argparse.ArgumentParser.parse_args` inside this
        function will raise :class:`SystemExit` on invalid flags and on
        ``--help``. Callers embedding this in unit tests should be aware
        that :func:`parse_args` may exit the interpreter.

    Side Effects:
        Writes a JSON file at ``args.output``. Writes progress and
        summary text to :data:`sys.stdout` via ``print``. May also
        write to :data:`sys.stderr` indirectly via ``argparse`` error
        messages.

    Assumptions:
        * ``args.cuda`` is informational only — the device is always
          ``cuda`` when available, ``cpu`` otherwise. The flag does
          **not** override this default.
        * The output directory for ``args.output`` must already exist;
          ``open(args.output, "w")`` will fail otherwise.
    """
    parser = argparse.ArgumentParser(description="Benchmark LAKER-XSA attention")
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Model dimension",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of timing runs",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available",
    )

    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        # Acknowledgement only — the device selection in ``run_benchmarks``
        # already falls back to CPU in this case.
        print("CUDA not available, running on CPU")

    print("Running LAKER-XSA benchmarks...")
    print(f"  d_model: {args.d_model}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  num_runs: {args.num_runs}")

    results = run_benchmarks(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_runs=args.num_runs,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for seq_len, data in results["results"].items():
        print(f"\nSequence length: {seq_len}")
        print("-" * 40)
        for attn_type, metrics in data.items():
            print(
                f"  {attn_type:10s}: {metrics['forward_ms']:7.2f}ms forward, "
                f"{metrics['backward_ms']:7.2f}ms backward"
            )


if __name__ == "__main__":
    main()
