#!/usr/bin/env python3
"""
Run all benchmarks for LAKER-XSA.

This script runs the full benchmark suite and saves results.

Usage:
    python -m examples.run_benchmarks --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.benchmarks.long_context import long_context_benchmark
from laker_xsa.benchmarks.conditioning import compute_conditioning_metrics
from laker_xsa.benchmarks.runtime import runtime_profile, profile_iterations


def main() -> None:
    """Run full benchmark suite."""
    parser = argparse.ArgumentParser(description="Run LAKER-XSA benchmarks")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model dimension for benchmarks",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (fewer iterations)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LAKER-XSA Benchmark Suite")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    config = XSA_LAKER_Config(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_iterations=10,
        preconditioner_rank=args.d_model // 16,
    )

    results: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "config": {
            "d_model": args.d_model,
            "num_heads": args.num_heads,
        },
        "benchmarks": {},
    }

    num_trials = 2 if args.quick else 3
    seq_lens = [64, 128] if args.quick else [64, 128, 256]

    # Long-context benchmark
    print("\n" + "-" * 40)
    print("1. Long-Context Scaling Benchmark")
    print("-" * 40)
    lc_results = long_context_benchmark(
        d_model=args.d_model,
        num_heads=args.num_heads,
        seq_lens=seq_lens,
        num_trials=num_trials,
    )
    results["benchmarks"]["long_context"] = lc_results
    print("  Complete!")

    # Conditioning benchmark
    print("\n" + "-" * 40)
    print("2. Conditioning Analysis Benchmark")
    print("-" * 40)
    cond_results = compute_conditioning_metrics(
        config,
        seq_len=128,
        num_samples=num_trials,
    )
    results["benchmarks"]["conditioning"] = cond_results
    print(f"  Raw condition number: {cond_results['raw_condition_mean']:.2f}")
    print(
        f"  Regularized condition number: {cond_results['regularized_condition_mean']:.2f}"
    )
    print("  Complete!")

    # Runtime benchmark
    print("\n" + "-" * 40)
    print("3. Runtime Profile Benchmark")
    print("-" * 40)
    from laker_xsa.attention.kernel_attention import FusedXSALAKERAttention

    attn = FusedXSALAKERAttention(config).to(device)
    x = torch.randn(4, 128, args.d_model, device=device)

    runtime_results = runtime_profile(attn, x, num_warmup=5, num_runs=20)
    results["benchmarks"]["runtime"] = runtime_results
    print(f"  Forward: {runtime_results['forward']['mean_ms']:.2f}ms")
    print(f"  Backward: {runtime_results['backward']['mean_ms']:.2f}ms")
    print("  Complete!")

    # Iteration convergence
    print("\n" + "-" * 40)
    print("4. Iteration Convergence Benchmark")
    print("-" * 40)
    iter_results = profile_iterations(config, seq_len=128, max_iterations=30)
    results["benchmarks"]["iterations"] = iter_results
    print(f"  Initial residual: {iter_results['initial_residual']:.4f}")
    print(f"  Final residual: {iter_results['final_residual']:.4f}")
    print(f"  Reduction factor: {iter_results['reduction_factor']:.2f}x")
    print("  Complete!")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("All benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
