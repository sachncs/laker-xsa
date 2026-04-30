#!/usr/bin/env python3
"""
Benchmark script for LAKER-XSA models.

Runs performance benchmarks comparing different attention variants.

Usage:
    python -m laker_xsa.cli.benchmark --output results.json
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List

import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.standard_attention import StandardMultiHeadAttention
from laker_xsa.attention.xsa_attention import ExclusiveSelfAttention
from laker_xsa.attention.kernel_attention import (
    KernelAttentionRegression,
    FusedXSALAKERAttention,
)


def benchmark_attention(
    attn_module: torch.nn.Module,
    x: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    Benchmark forward and backward pass of attention module.

    Args:
        attn_module: Attention module to benchmark.
        x: Input tensor.
        num_runs: Number of timing runs.
        warmup_runs: Number of warmup runs.

    Returns:
        Dictionary with timing statistics.
    """
    # Warmup
    for _ in range(warmup_runs):
        _ = attn_module(x)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Forward pass timing
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = attn_module(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = (time.perf_counter() - start) / num_runs * 1000  # ms

    # Backward pass timing
    x.requires_grad_(True)
    start = time.perf_counter()
    for _ in range(num_runs):
        out = attn_module(x)
        out.sum().backward()
        x.grad = None
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    backward_time = (time.perf_counter() - start) / num_runs * 1000  # ms

    # Memory (approximate)
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
    seq_lens: List[int] = None,
    num_runs: int = 50,
) -> Dict[str, Any]:
    """
    Run full benchmark suite.

    Args:
        d_model: Model dimension.
        num_heads: Number of heads.
        seq_lens: List of sequence lengths to test.
        num_runs: Number of timing runs.

    Returns:
        Dictionary with all benchmark results.
    """
    if seq_lens is None:
        seq_lens = [64, 128, 256, 512]

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
        print(f"\nBenchmarking seq_len={seq_len}...")
        x = torch.randn(4, seq_len, d_model, device="cuda" if torch.cuda.is_available() else "cpu")

        results["results"][seq_len] = {}

        # Standard attention
        attn_std = StandardMultiHeadAttention(config)
        if torch.cuda.is_available():
            attn_std = attn_std.cuda()
        results["results"][seq_len]["standard"] = benchmark_attention(attn_std, x, num_runs)
        print(f"  Standard: {results['results'][seq_len]['standard']['forward_ms']:.2f}ms")

        # XSA
        attn_xsa = ExclusiveSelfAttention(config)
        if torch.cuda.is_available():
            attn_xsa = attn_xsa.cuda()
        results["results"][seq_len]["xsa"] = benchmark_attention(attn_xsa, x, num_runs)
        print(f"  XSA: {results['results'][seq_len]['xsa']['forward_ms']:.2f}ms")

        # Kernel attention
        attn_kernel = KernelAttentionRegression(config)
        if torch.cuda.is_available():
            attn_kernel = attn_kernel.cuda()
        results["results"][seq_len]["kernel"] = benchmark_attention(attn_kernel, x, num_runs)
        print(f"  Kernel: {results['results'][seq_len]['kernel']['forward_ms']:.2f}ms")

        # Fused XSA + LAKER
        attn_fused = FusedXSALAKERAttention(config)
        if torch.cuda.is_available():
            attn_fused = attn_fused.cuda()
        results["results"][seq_len]["fused"] = benchmark_attention(attn_fused, x, num_runs)
        print(f"  Fused: {results['results'][seq_len]['fused']['forward_ms']:.2f}ms")

    return results


def main() -> None:
    """Main benchmark entry point."""
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

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for seq_len, data in results["results"].items():
        print(f"\nSequence length: {seq_len}")
        print("-" * 40)
        for attn_type, metrics in data.items():
            print(f"  {attn_type:10s}: {metrics['forward_ms']:7.2f}ms forward, "
                  f"{metrics['backward_ms']:7.2f}ms backward")


if __name__ == "__main__":
    main()
