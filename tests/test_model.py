"""Tests for model components (MLP, block, full model)."""

from __future__ import annotations

import pytest
import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.model.transformer_block import MLP, XSALAKERTransformerBlock
from laker_xsa.model.full_model import XSALAKERTransformer

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestMLP:
    """Tests for MLP feed-forward block."""

    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    def test_output_shape(self, activation: str) -> None:
        mlp = MLP(d_model=64, d_ff=256, activation=activation)
        x = torch.randn(2, 32, 64)
        out = mlp(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    def test_output_finite(self, activation: str) -> None:
        mlp = MLP(d_model=64, d_ff=256, activation=activation)
        x = torch.randn(2, 32, 64)
        out = mlp(x)
        assert torch.isfinite(out).all()

    def test_dropout_off(self) -> None:
        mlp = MLP(d_model=64, d_ff=256, dropout=0.0)
        assert mlp.dropout is None

    def test_dropout_on(self) -> None:
        mlp = MLP(d_model=64, d_ff=256, dropout=0.1)
        assert mlp.dropout is not None

    def test_gradient_flow(self) -> None:
        mlp = MLP(d_model=64, d_ff=256)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_dropout_train_vs_eval(self) -> None:
        mlp = MLP(d_model=64, d_ff=256, dropout=0.5)
        x = torch.randn(2, 16, 64)
        mlp.train()
        out_train = mlp(x)
        assert out_train.shape == x.shape


class TestTransformerBlock:
    """Tests for XSALAKERTransformerBlock."""

    def test_fused_v2_block(self) -> None:
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4, dropout=0.0)
        block = XSALAKERTransformerBlock(cfg, d_ff=256, attention_type="fused_v2")
        block.eval()
        x = torch.randn(2, 32, 64)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_standard_block(self) -> None:
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4, dropout=0.0)
        block = XSALAKERTransformerBlock(cfg, d_ff=128, attention_type="standard")
        block.eval()
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_xsa_block(self) -> None:
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4, dropout=0.0)
        block = XSALAKERTransformerBlock(cfg, d_ff=128, attention_type="xsa")
        block.eval()
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize(
        "attn_type", ["standard", "xsa", "kernel", "fused", "fused_v2"]
    )
    def test_all_attention_types(self, attn_type: str) -> None:
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4, dropout=0.0)
        block = XSALAKERTransformerBlock(cfg, d_ff=128, attention_type=attn_type)
        block.eval()
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_invalid_attention_type(self) -> None:
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4)
        with pytest.raises(ValueError, match="Unknown attention type"):
            XSALAKERTransformerBlock(cfg, attention_type="imaginary")

    def test_with_dropout(self) -> None:
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4, dropout=0.0)
        block = XSALAKERTransformerBlock(cfg, d_ff=128, dropout=0.1)
        assert block.dropout is not None

    def test_gradient_flow_fused_v2(self) -> None:
        cfg = XSA_LAKER_Config(d_model=64, num_heads=4, dropout=0.0)
        block = XSALAKERTransformerBlock(cfg, d_ff=128, attention_type="fused_v2")
        block.train()
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestFullModel:
    """Tests for XSALAKERTransformer."""

    @pytest.fixture
    def config(self) -> XSA_LAKER_Config:
        return XSA_LAKER_Config(d_model=64, num_heads=4, dropout=0.0)

    def test_token_id_path(self, config: XSA_LAKER_Config) -> None:
        model = XSALAKERTransformer(config, num_layers=2, vocab_size=500)
        model.eval()
        x = torch.randint(0, 500, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, 500)

    def test_embedding_path(self, config: XSA_LAKER_Config) -> None:
        model = XSALAKERTransformer(config, num_layers=2, vocab_size=None)
        model.eval()
        x = torch.randn(2, 32, config.d_model)
        out = model(x)
        assert out.shape == x.shape

    def test_token_path_validation(self, config: XSA_LAKER_Config) -> None:
        model = XSALAKERTransformer(config, num_layers=2, vocab_size=500)
        model.eval()
        with pytest.raises(ValueError, match="2D"):
            model(torch.randn(2, 32, config.d_model))

    def test_output_logits_finite(self, config: XSA_LAKER_Config) -> None:
        model = XSALAKERTransformer(config, num_layers=2, vocab_size=500)
        model.eval()
        x = torch.randint(0, 500, (2, 32))
        out = model(x)
        assert torch.isfinite(out).all()

    def test_with_causal_mask(self, config: XSA_LAKER_Config) -> None:
        model = XSALAKERTransformer(config, num_layers=2, vocab_size=None)
        model.eval()
        x = torch.randn(2, 16, config.d_model)
        mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
        mask = ~mask
        out = model(x, mask=mask.unsqueeze(0))
        assert out.shape == x.shape

    def test_gradient_flow_token_path(self, config: XSA_LAKER_Config) -> None:
        model = XSALAKERTransformer(config, num_layers=1, vocab_size=100)
        model.train()
        x = torch.randint(0, 100, (2, 16))
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN in {name}"
