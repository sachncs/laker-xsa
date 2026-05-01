"""Tests for functional API (stateless functions).

Tests compute_kernel_matrix and apply_kernel_operator,
the thin functional wrappers over class-based Modules.
"""

from __future__ import annotations

import pytest
import torch

from laker_xsa.attention.functional import compute_kernel_matrix
from laker_xsa.solver.functional import apply_kernel_operator


class TestComputeKernelMatrix:
    """Tests for compute_kernel_matrix functional API."""

    def test_shape_2d(self) -> None:
        q = torch.randn(16, 8)
        k = torch.randn(16, 8)
        K = compute_kernel_matrix(q, k)
        assert K.shape == (16, 16)

    def test_shape_3d(self) -> None:
        q = torch.randn(2, 16, 8)
        k = torch.randn(2, 16, 8)
        K = compute_kernel_matrix(q, k)
        assert K.shape == (2, 16, 16)

    def test_shape_4d(self) -> None:
        q = torch.randn(2, 4, 16, 8)
        k = torch.randn(2, 4, 16, 8)
        K = compute_kernel_matrix(q, k)
        assert K.shape == (2, 4, 16, 16)

    def test_finite_values(self) -> None:
        q = torch.randn(4, 32, 16)
        k = torch.randn(4, 32, 16)
        K = compute_kernel_matrix(q, k)
        assert torch.isfinite(K).all()

    def test_positive_values(self) -> None:
        q = torch.randn(4, 32, 16)
        k = torch.randn(4, 32, 16)
        K = compute_kernel_matrix(q, k)
        # exp + eps is always positive
        assert (K > 0).all()

    @pytest.mark.parametrize("normalize_qk", [True, False])
    def test_normalize_modes(self, normalize_qk: bool) -> None:
        q = torch.randn(2, 16, 8)
        k = torch.randn(2, 16, 8)
        K = compute_kernel_matrix(q, k, normalize_qk=normalize_qk)
        assert K.shape == (2, 16, 16)
        assert torch.isfinite(K).all()

    @pytest.mark.parametrize("symmetric", [True, False])
    def test_symmetric_modes(self, symmetric: bool) -> None:
        q = torch.randn(2, 16, 8)
        k = torch.randn(2, 16, 8)
        K = compute_kernel_matrix(q, k, symmetric=symmetric)
        if symmetric:
            assert torch.allclose(K, K.transpose(-2, -1))

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 5.0, 20.0])
    def test_temperature_values(self, temperature: float) -> None:
        q = torch.randn(2, 16, 8)
        k = torch.randn(2, 16, 8)
        K = compute_kernel_matrix(q, k, temperature=temperature)
        assert torch.isfinite(K).all()

    def test_different_qk_shapes(self) -> None:
        q = torch.randn(2, 16, 8)
        k = torch.randn(2, 32, 8)
        K = compute_kernel_matrix(q, k)
        assert K.shape == (2, 16, 32)


class TestApplyKernelOperator:
    """Tests for apply_kernel_operator functional API."""

    def test_shape_matches_input(self) -> None:
        kernel = torch.randn(2, 4, 32, 32)
        x = torch.randn(2, 4, 32, 16)
        lam = torch.tensor(0.1)
        result = apply_kernel_operator(kernel, x, lam)
        assert result.shape == x.shape

    def test_finite_output(self) -> None:
        kernel = torch.randn(2, 4, 32, 32)
        x = torch.randn(2, 4, 32, 16)
        lam = torch.tensor(0.1)
        result = apply_kernel_operator(kernel, x, lam)
        assert torch.isfinite(result).all()

    def test_identity_kernel_zero_lambda(self) -> None:
        n = 8
        kernel = torch.eye(n).unsqueeze(0).unsqueeze(0).expand(1, 1, -1, -1)
        x = torch.randn(1, 1, n, 4)
        lam = torch.tensor(0.0)
        result = apply_kernel_operator(kernel, x, lam)
        # With identity kernel and lambda=0, result should equal x
        assert torch.allclose(result, x, atol=1e-5)

    def test_lambda_effect(self) -> None:
        kernel = torch.eye(8).unsqueeze(0).unsqueeze(0)
        x = torch.randn(1, 1, 8, 4)
        r0 = apply_kernel_operator(kernel, x, torch.tensor(0.0))
        r2 = apply_kernel_operator(kernel, x, torch.tensor(2.0))
        # Adding lambda*I should produce different results
        assert (r0 != r2).any()

    def test_broadcast_lambda(self) -> None:
        kernel = torch.randn(2, 4, 16, 16)
        x = torch.randn(2, 4, 16, 8)
        lam = torch.tensor(1.0).view(1, 1, 1, 1)
        result = apply_kernel_operator(kernel, x, lam)
        assert result.shape == x.shape

    def test_zero_kernel(self) -> None:
        kernel = torch.zeros(2, 4, 16, 16)
        x = torch.randn(2, 4, 16, 8)
        lam = torch.tensor(1.0)
        result = apply_kernel_operator(kernel, x, lam)
        assert torch.allclose(result, x, atol=1e-5)  # K=0, lam=1 → result = x
