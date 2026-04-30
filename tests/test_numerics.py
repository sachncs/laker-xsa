"""
Numerical stability tests for LAKER-XSA.

This module tests numerical stability under edge cases including
very long sequences, extreme values, and ill-conditioned inputs.
"""

from __future__ import annotations

import pytest
import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.standard_attention import StandardMultiHeadAttention
from laker_xsa.attention.xsa_attention import ExclusiveSelfAttention
from laker_xsa.attention.kernel_attention import FusedXSALAKERAttention
from laker_xsa.utils.stability import check_finite, clamp_tensor


class TestNumericalStability:
    """Test numerical stability of attention modules."""

    def test_standard_attention_stability(self) -> None:
        """Test standard attention with extreme values."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        attn = StandardMultiHeadAttention(config)
        attn.eval()

        # Large values
        x_large = torch.randn(2, 32, config.d_model) * 100
        output = attn(x_large)
        assert check_finite(output, "large input output", raise_error=False)

        # Small values
        x_small = torch.randn(2, 32, config.d_model) * 1e-6
        output = attn(x_small)
        assert check_finite(output, "small input output", raise_error=False)

    def test_xsa_stability(self) -> None:
        """Test XSA with extreme values."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        attn = ExclusiveSelfAttention(config)
        attn.eval()

        # Large values
        x_large = torch.randn(2, 32, config.d_model) * 100
        output = attn(x_large)
        assert check_finite(output, "large input output", raise_error=False)

    def test_fused_stability(self) -> None:
        """Test fused attention with extreme values."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        attn = FusedXSALAKERAttention(config)
        attn.eval()

        # Large values
        x_large = torch.randn(2, 32, config.d_model) * 100
        output = attn(x_large)
        assert check_finite(output, "large input output", raise_error=False)

    def test_long_sequence_stability(self) -> None:
        """Test stability with long sequences."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        attn = FusedXSALAKERAttention(config)
        attn.eval()

        # Long sequence
        x = torch.randn(1, 512, config.d_model)
        output = attn(x)
        assert check_finite(output, "long sequence output", raise_error=False)

    def test_very_long_sequence(self) -> None:
        """Test with very long sequence (may stress iterative solver)."""
        config = XSA_LAKER_Config(
            d_model=64,
            num_heads=4,
            num_iterations=20,  # More iterations for convergence
        )
        attn = FusedXSALAKERAttention(config)
        attn.eval()

        x = torch.randn(1, 1024, config.d_model)
        output = attn(x)
        assert check_finite(output, "very long sequence output", raise_error=False)


class TestClampTensor:
    """Test tensor clamping utility."""

    def test_clamp_both_bounds(self) -> None:
        """Test clamping with both bounds."""
        x = torch.tensor([-1e10, -1.0, 0.0, 1.0, 1e10])
        result = clamp_tensor(x, min_val=-2.0, max_val=2.0)
        assert result.min() >= -2.0
        assert result.max() <= 2.0

    def test_clamp_min_only(self) -> None:
        """Test clamping with only min bound."""
        x = torch.tensor([-1e10, -1.0, 0.0, 1.0, 1e10])
        result = clamp_tensor(x, min_val=-2.0)
        assert result.min() >= -2.0

    def test_clamp_max_only(self) -> None:
        """Test clamping with only max bound."""
        x = torch.tensor([-1e10, -1.0, 0.0, 1.0, 1e10])
        result = clamp_tensor(x, max_val=2.0)
        assert result.max() <= 2.0

    def test_clamp_no_op(self) -> None:
        """Test clamping with no bounds is no-op."""
        x = torch.randn(10)
        result = clamp_tensor(x)
        assert torch.allclose(result, x)


class TestCheckFinite:
    """Test finite checking utility."""

    def test_check_finite_pass(self) -> None:
        """Test check_finite passes for finite tensor."""
        x = torch.randn(10, 10)
        assert check_finite(x, raise_error=False)

    def test_check_finite_nan(self) -> None:
        """Test check_finite detects NaN."""
        x = torch.randn(10, 10)
        x[0, 0] = float("nan")
        assert not check_finite(x, raise_error=False)

    def test_check_finite_inf(self) -> None:
        """Test check_finite detects Inf."""
        x = torch.randn(10, 10)
        x[0, 0] = float("inf")
        assert not check_finite(x, raise_error=False)

    def test_check_finite_neg_inf(self) -> None:
        """Test check_finite detects -Inf."""
        x = torch.randn(10, 10)
        x[0, 0] = float("-inf")
        assert not check_finite(x, raise_error=False)

    def test_check_finite_raises(self) -> None:
        """Test check_finite raises on non-finite."""
        x = torch.randn(10, 10)
        x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            check_finite(x, raise_error=True)


class TestDeterminism:
    """Test deterministic behavior."""

    def test_deterministic_output(self) -> None:
        """Test same input produces same output."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4, seed=42)
        attn = FusedXSALAKERAttention(config)
        attn.eval()

        x = torch.randn(2, 32, config.d_model)

        with torch.no_grad():
            output1 = attn(x)
            output2 = attn(x)

        assert torch.allclose(output1, output2)

    def test_seed_reproducibility(self) -> None:
        """Test seed produces reproducible results."""
        config1 = XSA_LAKER_Config(d_model=64, num_heads=4)
        config2 = XSA_LAKER_Config(d_model=64, num_heads=4)

        torch.manual_seed(42)
        attn1 = FusedXSALAKERAttention(config1)

        torch.manual_seed(42)
        attn2 = FusedXSALAKERAttention(config2)

        # Parameters should be identical
        for (n1, p1), (n2, p2) in zip(
            attn1.named_parameters(), attn2.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Parameter {n1} differs"
