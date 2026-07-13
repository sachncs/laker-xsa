"""Gradient-flow regression tests for the LAKER-XSA stack.

A single ``output.sum().backward()`` call must produce finite input
gradients for Standard, XSA, deprecated v1 :class:`FusedXSALAKERAttention`,
and v2 :class:`LakerAttention`. The v1 fused attention's parameter
gradients are also checked for finiteness; multiple training-style
backward passes on the full model must stay NaN/Inf-free.

The deprecated v1 ``FusedXSALAKERAttention`` emits
:class:`DeprecationWarning` on import; the module-level ``pytestmark =
pytest.mark.filterwarnings("ignore::DeprecationWarning")`` suppresses
those so the warning does not appear in test output.
"""

from __future__ import annotations

import pytest
import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.standard import StandardMultiHeadAttention
from laker_xsa.attention.xsa import ExclusiveSelfAttention
from laker_xsa.attention._legacy import FusedXSALAKERAttention
from laker_xsa.attention.laker import LakerAttention
from laker_xsa.model.transformer_block import XSALAKERTransformerBlock
from laker_xsa.model.full_model import XSALAKERTransformer

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def config() -> XSA_LAKER_Config:
    """Small ``(d_model=64, num_heads=4)`` test config with ``dropout=0``."""
    return XSA_LAKER_Config(d_model=64, num_heads=4, dropout=0.0)


class TestAttentionGradients:
    """Test gradient flow through attention modules."""

    def test_standard_attention_gradients(self, config: XSA_LAKER_Config) -> None:
        """Test standard attention gradients."""
        attn = StandardMultiHeadAttention(config)
        attn.train()

        x = torch.randn(2, 32, config.d_model, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.shape == x.shape

    def test_xsa_gradients(self, config: XSA_LAKER_Config) -> None:
        """Test XSA gradients."""
        attn = ExclusiveSelfAttention(config)
        attn.train()

        x = torch.randn(2, 32, config.d_model, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_fused_gradients(self, config: XSA_LAKER_Config) -> None:
        """Test fused v1 attention gradients."""
        attn = FusedXSALAKERAttention(config)
        attn.train()

        x = torch.randn(2, 32, config.d_model, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_v1_parameter_gradients(self, config: XSA_LAKER_Config) -> None:
        """Test that all v1 parameters receive gradients."""
        attn = FusedXSALAKERAttention(config)
        attn.train()

        x = torch.randn(2, 32, config.d_model)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        for name, param in attn.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.isfinite(
                param.grad
            ).all(), f"Parameter {name} has non-finite gradient"

    def test_laker_attention_gradients(self, config: XSA_LAKER_Config) -> None:
        """Test LakerAttention (v2) gradients."""
        attn = LakerAttention(config)
        attn.train()

        x = torch.randn(2, 32, config.d_model, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


class TestBlockGradients:
    """Test gradient flow through Transformer block."""

    def test_block_gradients(self, config: XSA_LAKER_Config) -> None:
        """Test Transformer block gradients."""
        block = XSALAKERTransformerBlock(config, d_ff=256)
        block.train()

        x = torch.randn(2, 32, config.d_model, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_block_parameter_gradients(self, config: XSA_LAKER_Config) -> None:
        """Test all block parameters receive gradients."""
        block = XSALAKERTransformerBlock(config, d_ff=256)
        block.train()

        x = torch.randn(2, 32, config.d_model)
        output = block(x)
        loss = output.sum()
        loss.backward()

        for name, param in block.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


class TestModelGradients:
    """Test gradient flow through full model."""

    def test_model_gradients_with_vocab(self, config: XSA_LAKER_Config) -> None:
        """Test model gradients with vocabulary."""
        model = XSALAKERTransformer(config, num_layers=2, vocab_size=500)
        model.train()

        x = torch.randint(0, 500, (2, 32))
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert model.token_embedding.weight.grad is not None
        assert torch.isfinite(model.token_embedding.weight.grad).all()

    def test_model_no_nan_after_multiple_steps(self, config: XSA_LAKER_Config) -> None:
        """Test no NaN after multiple backward passes."""
        model = XSALAKERTransformer(config, num_layers=2, vocab_size=500)
        model.train()

        for _ in range(5):
            x = torch.randint(0, 500, (2, 32))
            output = model(x)
            loss = output.sum()
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), f"NaN in {name} gradient"

            model.zero_grad()
