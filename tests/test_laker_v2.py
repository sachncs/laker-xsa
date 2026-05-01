"""Tests for flagship LakerAttention v2 and AttentionKernel.

Covers LakerAttention, LakerAttentionLayer, LakerPreconditioner, and
AttentionKernel comprehensively.
"""

from __future__ import annotations

import pytest
import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.laker import LakerAttention, LakerAttentionLayer
from laker_xsa.attention.kernels import AttentionKernel, compute_kernel_matrix
from laker_xsa.solver.laker_preconditioner import LakerPreconditioner


@pytest.fixture
def config() -> XSA_LAKER_Config:
    return XSA_LAKER_Config(d_model=64, num_heads=4, dropout=0.0, eps=1e-6)


# ---------------------------------------------------------------------------
# AttentionKernel
# ---------------------------------------------------------------------------

class TestAttentionKernel:
    """Tests for AttentionKernel module."""

    def test_output_shape(self) -> None:
        kernel = AttentionKernel(head_dim=16)
        q = torch.randn(2, 4, 32, 16)
        k = torch.randn(2, 4, 32, 16)
        out = kernel(q, k)
        assert out.shape == (2, 4, 32, 32)

    def test_output_finite(self) -> None:
        kernel = AttentionKernel(head_dim=16)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 4, 16, 16)
        out = kernel(q, k)
        assert torch.isfinite(out).all()

    def test_temperature_property(self) -> None:
        kernel = AttentionKernel(head_dim=16, temperature=2.0)
        t = kernel.temperature.item()
        assert 0.05 <= t <= 100.0

    def test_temperature_clamped(self) -> None:
        kernel = AttentionKernel(head_dim=16, temperature=200.0)
        t = kernel.temperature.item()
        assert t <= 100.0

    def test_learnable_temperature(self) -> None:
        kernel = AttentionKernel(head_dim=16, learnable_temperature=True)
        assert isinstance(kernel.log_temperature, torch.nn.Parameter)

    def test_fixed_temperature(self) -> None:
        kernel = AttentionKernel(head_dim=16, learnable_temperature=False)
        assert not isinstance(kernel.log_temperature, torch.nn.Parameter)

    def test_symmetric_mode(self) -> None:
        kernel = AttentionKernel(head_dim=16, symmetric=True)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 4, 16, 16)
        K = kernel(q, k)
        assert torch.allclose(K, K.transpose(-2, -1), atol=1e-5)

    def test_dot_product_mode(self) -> None:
        kernel = AttentionKernel(head_dim=16, normalize_qk=False)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 4, 16, 16)
        out = kernel(q, k)
        assert out.shape == (2, 4, 16, 16)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        kernel = AttentionKernel(head_dim=16, learnable_temperature=True)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 4, 16, 16)
        K = kernel(q, k)
        loss = K.sum()
        loss.backward()
        assert kernel.log_temperature.grad is not None

    def test_kernel_values_bounded(self) -> None:
        kernel = AttentionKernel(head_dim=16, temperature=1.0)
        q = torch.randn(2, 4, 32, 16)
        k = torch.randn(2, 4, 32, 16)
        K = kernel(q, k)
        # Kernel values should be positive (exp + eps)
        assert (K > 0).all()


# ---------------------------------------------------------------------------
# LakerAttention (v2)
# ---------------------------------------------------------------------------

class TestLakerAttention:
    """Tests for LakerAttention v2."""

    def test_output_shape(self, config: XSA_LAKER_Config) -> None:
        attn = LakerAttention(config)
        attn.eval()
        x = torch.randn(2, 32, config.d_model)
        out = attn(x)
        assert out.shape == x.shape

    def test_output_finite(self, config: XSA_LAKER_Config) -> None:
        attn = LakerAttention(config)
        attn.eval()
        x = torch.randn(2, 32, config.d_model)
        out = attn(x)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, config: XSA_LAKER_Config) -> None:
        attn = LakerAttention(config)
        attn.train()
        x = torch.randn(2, 32, config.d_model, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_parameter_gradients(self, config: XSA_LAKER_Config) -> None:
        attn = LakerAttention(config)
        attn.train()
        x = torch.randn(2, 32, config.d_model)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has NaN/Inf"

    def test_lambda_reg_positive(self, config: XSA_LAKER_Config) -> None:
        attn = LakerAttention(config)
        assert attn.lambda_reg.item() > 0

    def test_zero_diagonal(self, config: XSA_LAKER_Config) -> None:
        attn = LakerAttention(config)
        kernel = torch.randn(2, 4, 16, 16)
        result = attn.zero_diagonal(kernel)
        diag = torch.diagonal(result, dim1=-2, dim2=-1)
        assert (diag.abs() < 1e-6).all()

    def test_clean_self_projection(self, config: XSA_LAKER_Config) -> None:
        attn = LakerAttention(config)
        output = torch.randn(2, 32, 64)
        values = torch.randn(2, 32, 64)
        cleaned = attn.clean_self_projection(output, values)
        assert cleaned.shape == output.shape
        assert torch.isfinite(cleaned).all()

    def test_rms_normalize(self, config: XSA_LAKER_Config) -> None:
        attn = LakerAttention(config)
        x = torch.randn(2, 4, 32, 16)
        normed = attn.rms_normalize(x)
        assert normed.shape == x.shape
        assert torch.isfinite(normed).all()
        # RMS should be approximately 1 per sample
        rms = torch.sqrt((normed * normed).mean(dim=(-2, -1)))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_deterministic_eval(self, config: XSA_LAKER_Config) -> None:
        attn = LakerAttention(config)
        attn.eval()
        x = torch.randn(2, 32, config.d_model)
        with torch.no_grad():
            out1 = attn(x)
            out2 = attn(x)
        assert torch.allclose(out1, out2)

    def test_with_causal_mask(self, config: XSA_LAKER_Config) -> None:
        attn = LakerAttention(config)
        attn.eval()
        x = torch.randn(2, 16, config.d_model)
        mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
        mask = ~mask
        out = attn(x, mask=mask.unsqueeze(0))
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_configs_with_preconditioner_types(self) -> None:
        for mode in ["fast", "diagonal", "none"]:
            cfg = XSA_LAKER_Config(
                d_model=64, num_heads=4, preconditioner_type=mode,
            )
            attn = LakerAttention(cfg)
            attn.eval()
            x = torch.randn(2, 16, cfg.d_model)
            out = attn(x)
            assert out.shape == x.shape

    def test_zero_diagonal_xsa_mode(self) -> None:
        cfg = XSA_LAKER_Config(
            d_model=64, num_heads=4, xsa_mode="zero_diagonal",
        )
        attn = LakerAttention(cfg)
        attn.eval()
        x = torch.randn(2, 16, cfg.d_model)
        out = attn(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# LakerAttentionLayer
# ---------------------------------------------------------------------------

class TestLakerAttentionLayer:
    """Tests for LakerAttentionLayer."""

    def test_forwards_to_attention(self, config: XSA_LAKER_Config) -> None:
        layer = LakerAttentionLayer(config, layer_idx=0)
        layer.eval()
        x = torch.randn(2, 32, config.d_model)
        out = layer(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_layer_idx_stored(self, config: XSA_LAKER_Config) -> None:
        layer = LakerAttentionLayer(config, layer_idx=3)
        assert layer.layer_idx == 3

    def test_share_preconditioner_flag(self, config: XSA_LAKER_Config) -> None:
        layer = LakerAttentionLayer(
            config, layer_idx=0, share_preconditioner_across_layers=True,
        )
        assert layer.share_preconditioner is True

    def test_with_mask(self, config: XSA_LAKER_Config) -> None:
        layer = LakerAttentionLayer(config, layer_idx=0)
        layer.eval()
        x = torch.randn(2, 16, config.d_model)
        mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
        mask = ~mask
        out = layer(x, mask=mask.unsqueeze(0))
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# LakerPreconditioner
# ---------------------------------------------------------------------------

class TestLakerPreconditioner:
    """Tests for LakerPreconditioner v2."""

    @pytest.fixture
    def sample_kernel(self) -> torch.Tensor:
        n = 16
        A = torch.randn(2, 4, n, n)
        # Build PSD kernel: A @ A^T ensures SPD-like structure
        kernel = torch.matmul(A, A.transpose(-2, -1))
        kernel = kernel / kernel.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        return kernel

    def test_fast_mode_output(self, sample_kernel: torch.Tensor) -> None:
        precond = LakerPreconditioner(num_heads=4, mode="fast", rank=8)
        lam = torch.tensor(0.1)
        data = precond(sample_kernel, lam, 16)
        assert data is not None
        diag, lr = data
        assert diag.shape == (2, 4, 16)
        assert not torch.isnan(diag).any()

    def test_diagonal_mode_output(self, sample_kernel: torch.Tensor) -> None:
        precond = LakerPreconditioner(num_heads=4, mode="diagonal")
        lam = torch.tensor(0.1)
        data = precond(sample_kernel, lam, 16)
        assert data.shape == (2, 4, 16)

    def test_none_mode_output(self, sample_kernel: torch.Tensor) -> None:
        precond = LakerPreconditioner(num_heads=4, mode="none")
        lam = torch.tensor(0.1)
        data = precond(sample_kernel, lam, 16)
        assert data is None

    def test_apply_preconditioner_fast(self) -> None:
        precond = LakerPreconditioner(num_heads=4, mode="fast", rank=4)
        A = torch.randn(2, 4, 16, 16)
        kernel = torch.matmul(A, A.transpose(-2, -1))
        kernel = kernel / kernel.max()
        lam = torch.tensor(0.1)
        data = precond(kernel, lam, 16)
        residual = torch.randn(2, 4, 16, 8)
        out = precond.apply_preconditioner(residual, data)
        assert out.shape == residual.shape
        assert torch.isfinite(out).all()

    def test_apply_preconditioner_diagonal(self) -> None:
        precond = LakerPreconditioner(num_heads=4, mode="diagonal")
        A = torch.randn(2, 4, 16, 16)
        kernel = torch.matmul(A, A.transpose(-2, -1))
        kernel = kernel / kernel.max()
        lam = torch.tensor(0.1)
        data = precond(kernel, lam, 16)
        residual = torch.randn(2, 4, 16, 8)
        out = precond.apply_preconditioner(residual, data)
        assert out.shape == residual.shape

    def test_apply_preconditioner_none(self) -> None:
        precond = LakerPreconditioner(num_heads=4, mode="none")
        residual = torch.randn(2, 4, 16, 8)
        out = precond.apply_preconditioner(residual, None)
        assert torch.equal(out, residual)

    def test_step_counter_increments(self, sample_kernel: torch.Tensor) -> None:
        precond = LakerPreconditioner(num_heads=4, mode="fast", rank=4)
        lam = torch.tensor(0.1)
        assert precond.step_counter.item() == 0
        precond(sample_kernel, lam, 16)
        assert precond.step_counter.item() == 1

    def test_caching_behavior(self, sample_kernel: torch.Tensor) -> None:
        precond = LakerPreconditioner(num_heads=4, mode="fast", rank=4)
        lam = torch.tensor(0.1)
        data1 = precond(sample_kernel, lam, 16, force_update=True)
        # force_update=False uses cache when step_counter % update_frequency != 0
        # Since update_frequency defaults to 1, we pass update_frequency=0 to prevent recompute
        data2 = precond(sample_kernel, lam, 16, force_update=False, update_frequency=0)
        # Cached values should be equal
        diag1, lr1 = data1
        diag2, lr2 = data2
        assert torch.allclose(diag1, diag2)

    def test_fast_preconditioner_no_lr(self) -> None:
        precond = LakerPreconditioner(num_heads=4, mode="fast", rank=0)
        A = torch.randn(2, 4, 16, 16)
        kernel = torch.matmul(A, A.transpose(-2, -1))
        lam = torch.tensor(0.1)
        data = precond(kernel, lam, 16)
        diag, lr = data
        assert lr is None


# ---------------------------------------------------------------------------
# compute_kernel_matrix functional
# ---------------------------------------------------------------------------

class TestFunctionalKernel:
    """Tests for stateless compute_kernel_matrix."""

    def test_normalized_kernel_shape(self) -> None:
        q = torch.randn(2, 32, 16)
        k = torch.randn(2, 32, 16)
        K = compute_kernel_matrix(q, k, normalize_qk=True)
        assert K.shape == (2, 32, 32)

    def test_dot_product_kernel_shape(self) -> None:
        q = torch.randn(2, 32, 16)
        k = torch.randn(2, 32, 16)
        K = compute_kernel_matrix(q, k, normalize_qk=False)
        assert K.shape == (2, 32, 32)

    def test_symmetric_kernel(self) -> None:
        q = torch.randn(2, 32, 16)
        k = torch.randn(2, 32, 16)
        K = compute_kernel_matrix(q, k, symmetric=True)
        assert torch.allclose(K, K.transpose(-2, -1))

    def test_temperature_effect(self) -> None:
        q = torch.randn(2, 32, 16)
        k = torch.randn(2, 32, 16)
        K1 = compute_kernel_matrix(q, k, temperature=1.0)
        K2 = compute_kernel_matrix(q, k, temperature=10.0)
        # Higher temperature = sharper = different values
        assert not torch.allclose(K1, K2)

    def test_output_finite(self) -> None:
        q = torch.randn(2, 64, 16)
        k = torch.randn(2, 64, 16)
        K = compute_kernel_matrix(q, k)
        assert torch.isfinite(K).all()
