"""
Tests for attention modules.

This module tests the core attention implementations including
standard, XSA, kernel, and fused variants.
"""

from __future__ import annotations

import pytest
import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.standard import StandardMultiHeadAttention
from laker_xsa.attention.xsa import ExclusiveSelfAttention
from laker_xsa.attention._legacy import (
    KernelAttentionRegression,
    FusedXSALAKERAttention,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning"
)


@pytest.fixture
def config() -> XSA_LAKER_Config:
    """Create test configuration."""
    return XSA_LAKER_Config(
        d_model=64,
        num_heads=4,
        head_dim=16,
        dropout=0.0,
        eps=1e-6,
        num_iterations=10,
        preconditioner_rank=4,
        kernel_type="rbf",
        xsa_mode="subtract_projection",
    )


@pytest.fixture
def sample_input(config: XSA_LAKER_Config) -> torch.Tensor:
    """Create sample input tensor."""
    return torch.randn(2, 32, config.d_model)


class TestStandardMultiHeadAttention:
    """Tests for StandardMultiHeadAttention."""

    def test_output_shape(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that output shape matches input shape."""
        attn = StandardMultiHeadAttention(config)
        output = attn(sample_input)
        assert output.shape == sample_input.shape

    def test_output_finite(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that output contains only finite values."""
        attn = StandardMultiHeadAttention(config)
        output = attn(sample_input)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that gradients flow through module."""
        attn = StandardMultiHeadAttention(config)
        x = sample_input.clone().requires_grad_(True)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_with_mask(self, config: XSA_LAKER_Config) -> None:
        """Test attention with causal mask."""
        attn = StandardMultiHeadAttention(config)
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, config.d_model)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = ~mask
        output = attn(x, mask=mask.unsqueeze(0))
        assert output.shape == x.shape


class TestExclusiveSelfAttention:
    """Tests for ExclusiveSelfAttention."""

    def test_output_shape(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that output shape matches input shape."""
        attn = ExclusiveSelfAttention(config)
        output = attn(sample_input)
        assert output.shape == sample_input.shape

    def test_output_finite(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that output contains only finite values."""
        attn = ExclusiveSelfAttention(config)
        output = attn(sample_input)
        assert torch.isfinite(output).all()

    def test_xsa_exclusion(self, config: XSA_LAKER_Config) -> None:
        """Test that XSA excludes self-components from output."""
        cfg = XSA_LAKER_Config(
            d_model=64, num_heads=4, head_dim=16,
            dropout=0.0, xsa_mode="subtract_projection",
        )
        attn = ExclusiveSelfAttention(cfg)
        attn.eval()

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, cfg.d_model)

        with torch.no_grad():
            v = attn.qkv_proj.w_v(x)
            v = v.view(batch, seq_len, cfg.num_heads, cfg.head_dim).transpose(1, 2)
            output = attn(x)
            output = output.view(batch, seq_len, cfg.num_heads, cfg.head_dim).transpose(1, 2)

            for i in range(seq_len):
                out_i = output[:, :, i, :]
                v_i = v[:, :, i, :]
                cos_sim = torch.nn.functional.cosine_similarity(out_i, v_i, dim=-1)
                assert cos_sim.abs().mean() < 0.5

    def test_zero_diagonal_mode(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test XSA with zero_diagonal mode."""
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4, head_dim=16, dropout=0.0, xsa_mode="zero_diagonal")
        attn = ExclusiveSelfAttention(cfg)
        output = attn(sample_input)
        assert output.shape == sample_input.shape
        assert torch.isfinite(output).all()

    def test_mask_mode(self) -> None:
        """Test XSA with explicit mask mode."""
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4, head_dim=16, dropout=0.0, xsa_mode="mask")
        attn = ExclusiveSelfAttention(cfg)
        x = torch.randn(2, 32, cfg.d_model)
        output = attn(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()


class TestKernelAttentionRegression:
    """Tests for KernelAttentionRegression (deprecated v1)."""

    def test_output_shape(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that output shape matches input shape."""
        attn = KernelAttentionRegression(config)
        output = attn(sample_input)
        assert output.shape == sample_input.shape

    def test_output_finite(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that output contains only finite values."""
        attn = KernelAttentionRegression(config)
        output = attn(sample_input)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that gradients flow through module."""
        attn = KernelAttentionRegression(config)
        x = sample_input.clone().requires_grad_(True)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_different_kernels(self, sample_input: torch.Tensor) -> None:
        """Test different kernel types."""
        for kernel_type in ["rbf", "linear", "cosine"]:
            config = XSA_LAKER_Config(
                d_model=64, num_heads=4, kernel_type=kernel_type,
            )
            attn = KernelAttentionRegression(config)
            output = attn(sample_input)
            assert output.shape == sample_input.shape
            assert torch.isfinite(output).all()


class TestFusedXSALAKERAttention:
    """Tests for FusedXSALAKERAttention (deprecated v1)."""

    def test_output_shape(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that output shape matches input shape."""
        attn = FusedXSALAKERAttention(config)
        output = attn(sample_input)
        assert output.shape == sample_input.shape

    def test_output_finite(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that output contains only finite values."""
        attn = FusedXSALAKERAttention(config)
        output = attn(sample_input)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self, config: XSA_LAKER_Config, sample_input: torch.Tensor) -> None:
        """Test that gradients flow through module."""
        attn = FusedXSALAKERAttention(config)
        x = sample_input.clone().requires_grad_(True)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_xsa_diagonal_zeroed(self, config: XSA_LAKER_Config) -> None:
        """Test that XSA zeros the kernel diagonal."""
        attn = FusedXSALAKERAttention(config)
        attn.eval()

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)

        with torch.no_grad():
            q = attn.w_q(x)
            k = attn.w_k(x)
            q = q.view(batch, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, config.num_heads, config.head_dim).transpose(1, 2)

            kernel = attn.kernel_fn(q, k)
            kernel = attn.apply_xsa_to_kernel(kernel)

            diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
            assert (diag.abs() < 1e-6).all()
