"""Tests for configuration validation (XSA_LAKER_Config)."""

from __future__ import annotations

import pytest

from laker_xsa.config import XSA_LAKER_Config


class TestConfigConstruction:
    """Tests for valid config construction."""

    def test_valid_default_config(self) -> None:
        cfg = XSA_LAKER_Config(d_model=128, num_heads=4)
        assert cfg.d_model == 128
        assert cfg.num_heads == 4
        assert cfg.head_dim == 32  # auto: 128/4
        assert cfg.effective_pcg_iters == 20

    def test_explicit_head_dim(self) -> None:
        cfg = XSA_LAKER_Config(d_model=128, num_heads=8, head_dim=32)
        assert cfg.head_dim == 32

    def test_auto_head_dim(self) -> None:
        cfg = XSA_LAKER_Config(d_model=256, num_heads=8)
        assert cfg.head_dim == 32

    def test_all_preconditioner_types(self) -> None:
        for ptype in ["cccp", "fast", "diagonal", "none"]:
            cfg = XSA_LAKER_Config(d_model=64, num_heads=4, preconditioner_type=ptype)
            assert cfg.preconditioner_type == ptype

    def test_all_xsa_modes(self) -> None:
        for mode in ["subtract_projection", "zero_diagonal", "mask"]:
            cfg = XSA_LAKER_Config(d_model=64, num_heads=4, xsa_mode=mode)
            assert cfg.xsa_mode == mode

    def test_all_kernel_types(self) -> None:
        for kt in ["exp_attention", "rbf", "linear", "cosine"]:
            cfg = XSA_LAKER_Config(d_model=64, num_heads=4, kernel_type=kt)
            assert cfg.kernel_type == kt

    def test_effective_pcg_iters_property(self) -> None:
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4, pcg_max_iterations=30)
        assert cfg.effective_pcg_iters == 30


class TestConfigValidationErrors:
    """Tests for config validation."""

    def test_d_model_not_divisible(self) -> None:
        with pytest.raises(ValueError, match="must be divisible"):
            XSA_LAKER_Config(d_model=65, num_heads=4)

    def test_invalid_kernel_type(self) -> None:
        with pytest.raises(ValueError, match="kernel_type"):
            XSA_LAKER_Config(d_model=64, num_heads=4, kernel_type="invalid")

    def test_invalid_xsa_mode(self) -> None:
        with pytest.raises(ValueError, match="xsa_mode"):
            XSA_LAKER_Config(d_model=64, num_heads=4, xsa_mode="invalid")

    def test_invalid_preconditioner_type(self) -> None:
        with pytest.raises(ValueError, match="preconditioner_type"):
            XSA_LAKER_Config(d_model=64, num_heads=4, preconditioner_type="invalid")

    def test_pcg_max_iterations_zero(self) -> None:
        with pytest.raises(ValueError, match="pcg_max_iterations"):
            XSA_LAKER_Config(d_model=64, num_heads=4, pcg_max_iterations=0)

    def test_num_iterations_zero(self) -> None:
        with pytest.raises(ValueError, match="num_iterations"):
            XSA_LAKER_Config(d_model=64, num_heads=4, num_iterations=0)

    def test_dropout_out_of_range_low(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            XSA_LAKER_Config(d_model=64, num_heads=4, dropout=-0.1)

    def test_dropout_out_of_range_high(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            XSA_LAKER_Config(d_model=64, num_heads=4, dropout=1.1)

    def test_eps_zero(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            XSA_LAKER_Config(d_model=64, num_heads=4, eps=0.0)

    def test_eps_negative(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            XSA_LAKER_Config(d_model=64, num_heads=4, eps=-0.1)

    def test_lambda_init_negative(self) -> None:
        with pytest.raises(ValueError, match="lambda_init"):
            XSA_LAKER_Config(d_model=64, num_heads=4, lambda_init=-1.0)


class TestConfigDefaults:
    """Tests for config default values."""

    def test_defaults(self) -> None:
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4)
        assert cfg.dropout == 0.0
        assert cfg.eps == 1e-6
        assert cfg.lambda_init == 3.0
        assert cfg.kernel_type == "exp_attention"
        assert cfg.xsa_mode == "subtract_projection"
        assert cfg.preconditioner_type == "fast"
        assert cfg.kernel_temperature == 1.0
        assert cfg.kernel_symmetric is False
        assert cfg.kernel_normalize_qk is True
        assert cfg.precond_update_frequency == 1
        assert cfg.pcg_tolerance == 1e-2
        assert cfg.clip_abs == 1e6
