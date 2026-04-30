"""
Gradient flow tests for LAKER-XSA.

This module verifies that gradients flow correctly through all components.
"""

from __future__ import annotations

import pytest
import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.standard_attention import StandardMultiHeadAttention
from laker_xsa.attention.xsa_attention import ExclusiveSelfAttention
from laker_xsa.attention.kernel_attention import FusedXSALAKERAttention
from laker_xsa.model.transformer_block import XSALAKERTransformerBlock
from laker_xsa.model.full_model import XSALAKERTransformer


class TestAttentionGradients:
    """Test gradient flow through attention modules."""

    def test_standard_attention_gradients(self) -> None:
        """Test standard attention gradients."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        attn = StandardMultiHeadAttention(config)
        attn.train()

        x = torch.randn(2, 32, config.d_model, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.shape == x.shape

    def test_xsa_gradients(self) -> None:
        """Test XSA gradients."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        attn = ExclusiveSelfAttention(config)
        attn.train()

        x = torch.randn(2, 32, config.d_model, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_fused_gradients(self) -> None:
        """Test fused attention gradients."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        attn = FusedXSALAKERAttention(config)
        attn.train()

        x = torch.randn(2, 32, config.d_model, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_parameter_gradients(self) -> None:
        """Test that all parameters receive gradients."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        attn = FusedXSALAKERAttention(config)
        attn.train()

        x = torch.randn(2, 32, config.d_model)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        for name, param in attn.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"


class TestBlockGradients:
    """Test gradient flow through Transformer block."""

    def test_block_gradients(self) -> None:
        """Test Transformer block gradients."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        block = XSALAKERTransformerBlock(config, d_ff=256)
        block.train()

        x = torch.randn(2, 32, config.d_model, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_block_parameter_gradients(self) -> None:
        """Test all block parameters receive gradients."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
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

    def test_model_gradients_with_vocab(self) -> None:
        """Test model gradients with vocabulary."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        model = XSALAKERTransformer(
            config,
            num_layers=4,
            vocab_size=500,
        )
        model.train()

        x = torch.randint(0, 500, (2, 32))
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check embedding gradients
        assert model.token_embedding.weight.grad is not None
        assert torch.isfinite(model.token_embedding.weight.grad).all()

    def test_model_no_nan_after_multiple_steps(self) -> None:
        """Test no NaN after multiple backward passes."""
        config = XSA_LAKER_Config(d_model=64, num_heads=4)
        model = XSALAKERTransformer(
            config,
            num_layers=4,
            vocab_size=500,
        )
        model.train()

        for _ in range(5):
            x = torch.randint(0, 500, (2, 32))
            output = model(x)
            loss = output.sum()
            loss.backward()

            # Check for NaN
            for name, param in model.named_parameters():
                assert torch.isfinite(param.grad).all(), f"NaN in {name} gradient"

            # Zero gradients for next step
            model.zero_grad()
