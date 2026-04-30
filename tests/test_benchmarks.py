"""
Benchmark tests for LAKER-XSA.

This module provides smoke tests for the benchmark modules to ensure
they run correctly. Full benchmarks should be run separately.
"""

from __future__ import annotations

import pytest
import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.benchmarks.long_context import long_context_benchmark
from laker_xsa.benchmarks.conditioning import compute_conditioning_metrics
from laker_xsa.benchmarks.runtime import runtime_profile, profile_iterations


class TestLongContextBenchmark:
    """Tests for long-context benchmark."""

    def test_long_context_runs(self) -> None:
        """Test long-context benchmark completes."""
        results = long_context_benchmark(
            d_model=64,
            num_heads=4,
            seq_lens=[32, 64],
            num_trials=2,
        )

        assert "config" in results
        assert "results" in results
        assert len(results["results"]) == 2  # Two sequence lengths

    def test_long_context_metrics(self) -> None:
        """Test long-context benchmark produces metrics."""
        results = long_context_benchmark(
            d_model=64,
            num_heads=4,
            seq_lens=[32],
            num_trials=2,
        )

        for seq_len, data in results["results"].items():
            for attn_type, metrics in data.items():
                assert "accuracy" in metrics
                assert "loss" in metrics
                assert 0.0 <= metrics["accuracy"] <= 1.0


class TestConditioningBenchmark:
    """Tests for conditioning benchmark."""

    def test_conditioning_metrics_runs(self) -> None:
        """Test conditioning metrics computation."""
        config = XSA_LAKER_Config(
            d_model=64,
            num_heads=4,
            kernel_type="rbf",
        )

        metrics = compute_conditioning_metrics(config, seq_len=32, num_samples=3)

        assert "raw_condition_mean" in metrics
        assert "regularized_condition_mean" in metrics
        assert metrics["raw_condition_mean"] > 0
        assert metrics["regularized_condition_mean"] > 0

    def test_conditioning_improvement(self) -> None:
        """Test regularization improves conditioning."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)

        metrics = compute_conditioning_metrics(config, seq_len=32, num_samples=3)

        # Regularization should reduce condition number
        assert metrics["regularized_condition_mean"] <= metrics["raw_condition_mean"]


class TestRuntimeBenchmark:
    """Tests for runtime benchmark."""

    def test_runtime_profile_runs(self) -> None:
        """Test runtime profiling completes."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        from laker_xsa.attention.kernel_attention import FusedXSALAKERAttention

        attn = FusedXSALAKERAttention(config)
        x = torch.randn(2, 32, config.d_model)

        profile = runtime_profile(attn, x, num_warmup=2, num_runs=5)

        assert "forward" in profile
        assert "backward" in profile
        assert profile["forward"]["mean_ms"] > 0
        assert profile["backward"]["mean_ms"] > 0

    def test_profile_iterations_runs(self) -> None:
        """Test iteration profiling completes."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)

        profile = profile_iterations(config, seq_len=32, max_iterations=20)

        assert "iterations" in profile
        assert "residual_norms" in profile
        assert len(profile["iterations"]) == 20
        assert len(profile["residual_norms"]) == 20

    def test_iterations_converge(self) -> None:
        """Test that iterations reduce residual."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)

        profile = profile_iterations(config, seq_len=32, max_iterations=30)

        # Final residual should be less than initial
        assert profile["residual_norms"][-1] < profile["residual_norms"][0]
        assert profile["reduction_factor"] > 1.0
