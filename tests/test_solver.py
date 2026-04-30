"""
Tests for solver modules.

This module tests the preconditioner and conjugate gradient solver.
"""

from __future__ import annotations

import pytest
import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.solver.preconditioner import LearnedPreconditioner
from laker_xsa.solver.conjugate_gradient import conjugate_gradient_solve, matvec_with_kernel


@pytest.fixture
def config() -> XSA_LAKER_Config:
    """Create test configuration."""
    return XSA_LAKER_Config(
        d_model=64,
        num_heads=4,
        head_dim=16,
        preconditioner_rank=4,
    )


class TestLearnedPreconditioner:
    """Tests for LearnedPreconditioner."""

    def test_output_shapes(self, config: XSA_LAKER_Config) -> None:
        """Test preconditioner output shapes."""
        precond = LearnedPreconditioner(config)
        batch, seq_len = 2, 32

        kernel_diag = torch.randn(batch, config.num_heads, seq_len)
        diag_precond, lr_precond = precond(kernel_diag, seq_len)

        assert diag_precond.shape == (batch, config.num_heads, seq_len)
        assert lr_precond is not None
        assert lr_precond.shape == (batch, config.num_heads, seq_len, config.preconditioner_rank)

    def test_output_positive(self, config: XSA_LAKER_Config) -> None:
        """Test that diagonal preconditioner is positive."""
        precond = LearnedPreconditioner(config)
        batch, seq_len = 2, 32

        kernel_diag = torch.randn(batch, config.num_heads, seq_len)
        diag_precond, _ = precond(kernel_diag, seq_len)

        assert (diag_precond > 0).all()

    def test_apply_precondition(self, config: XSA_LAKER_Config) -> None:
        """Test applying preconditioner to residual."""
        precond = LearnedPreconditioner(config)
        batch, seq_len = 2, 32

        kernel_diag = torch.randn(batch, config.num_heads, seq_len)
        diag_precond, lr_precond = precond(kernel_diag, seq_len)

        residual = torch.randn(batch, config.num_heads, seq_len, config.head_dim)
        precond_residual = precond.apply_precondition(residual, diag_precond, lr_precond)

        assert precond_residual.shape == residual.shape
        assert torch.isfinite(precond_residual).all()

    def test_no_low_rank(self) -> None:
        """Test preconditioner without low-rank factor."""
        config = XSA_LAKER_Config(
            d_model=64,
            num_heads=4,
            preconditioner_rank=None,
        )
        precond = LearnedPreconditioner(config)
        batch, seq_len = 2, 32

        kernel_diag = torch.randn(batch, config.num_heads, seq_len)
        diag_precond, lr_precond = precond(kernel_diag, seq_len)

        assert diag_precond.shape == (batch, config.num_heads, seq_len)
        assert lr_precond is None


class TestConjugateGradient:
    """Tests for conjugate gradient solver."""

    def test_cg_convergence(self) -> None:
        """Test CG converges on simple system."""
        batch, num_heads, seq_len, head_dim = 1, 2, 16, 8

        # Create positive definite kernel
        A = torch.randn(batch, num_heads, seq_len, seq_len)
        kernel = torch.matmul(A, A.transpose(-2, -1))  # K = A @ A^T is PSD

        b = torch.randn(batch, num_heads, seq_len, head_dim)

        x, iterations = conjugate_gradient_solve(
            kernel, b, lambda_reg=0.1, max_iterations=100, tolerance=1e-6
        )

        # Check solution quality
        residual = b - matvec_with_kernel(kernel, x, lambda_reg=0.1)
        residual_norm = residual.norm().item()

        assert residual_norm < 1.0  # Should converge reasonably
        assert iterations <= 100

    def test_cg_with_preconditioner(self) -> None:
        """Test CG with preconditioner."""
        batch, num_heads, seq_len, head_dim = 1, 2, 16, 8

        A = torch.randn(batch, num_heads, seq_len, seq_len)
        kernel = torch.matmul(A, A.transpose(-2, -1))
        b = torch.randn(batch, num_heads, seq_len, head_dim)

        # Simple diagonal preconditioner
        def precond(r: torch.Tensor) -> torch.Tensor:
            return r * 0.1

        x, iterations = conjugate_gradient_solve(
            kernel, b, lambda_reg=0.1,
            max_iterations=100, tolerance=1e-6,
            preconditioner=precond
        )

        assert torch.isfinite(x).all()

    def test_cg_zero_initial(self) -> None:
        """Test CG with zero initial guess."""
        batch, num_heads, seq_len, head_dim = 1, 1, 8, 4

        kernel = torch.eye(seq_len).unsqueeze(0).unsqueeze(0) * 2.0
        b = torch.ones(batch, num_heads, seq_len, head_dim)

        x, _ = conjugate_gradient_solve(kernel, b, lambda_reg=0.0, x0=None)

        # For K = 2I and b = 1, solution should be x = 0.5
        expected = torch.ones_like(b) * 0.5
        assert torch.allclose(x, expected, atol=1e-4)
