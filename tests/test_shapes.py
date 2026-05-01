"""
Shape verification tests for LAKER-XSA.

This module tests that all components produce correct output shapes
across various input configurations.
"""

from __future__ import annotations

import pytest
import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.standard import StandardMultiHeadAttention
from laker_xsa.attention.xsa import ExclusiveSelfAttention
from laker_xsa.attention._legacy import FusedXSALAKERAttention
from laker_xsa.model.transformer_block import XSALAKERTransformerBlock
from laker_xsa.model.full_model import XSALAKERTransformer
from laker_xsa.utils.tensor_ops import verify_tensor_shapes

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestAttentionShapes:
    """Test attention module shapes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("seq_len", [16, 32, 64, 128])
    def test_standard_attention_shapes(
        self, batch_size: int, seq_len: int
    ) -> None:
        """Test standard attention output shapes."""
        config = XSA_LAKER_Config(d_model=128, num_heads=4)
        attn = StandardMultiHeadAttention(config)

        x = torch.randn(batch_size, seq_len, config.d_model)
        output = attn(x)

        verify_tensor_shapes(output, (batch_size, seq_len, config.d_model), "output")

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_xsa_shapes(self, batch_size: int, seq_len: int) -> None:
        """Test XSA output shapes."""
        config = XSA_LAKER_Config(d_model=128, num_heads=4)
        attn = ExclusiveSelfAttention(config)

        x = torch.randn(batch_size, seq_len, config.d_model)
        output = attn(x)

        verify_tensor_shapes(output, (batch_size, seq_len, config.d_model), "output")

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_fused_shapes(self, batch_size: int, seq_len: int) -> None:
        """Test fused attention output shapes."""
        config = XSA_LAKER_Config(d_model=128, num_heads=4)
        attn = FusedXSALAKERAttention(config)

        x = torch.randn(batch_size, seq_len, config.d_model)
        output = attn(x)

        verify_tensor_shapes(output, (batch_size, seq_len, config.d_model), "output")


class TestBlockShapes:
    """Test Transformer block shapes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_transformer_block_shapes(
        self, batch_size: int, seq_len: int
    ) -> None:
        """Test Transformer block output shapes."""
        config = XSA_LAKER_Config(d_model=128, num_heads=4)
        block = XSALAKERTransformerBlock(config, d_ff=512)

        x = torch.randn(batch_size, seq_len, config.d_model)
        output = block(x)

        verify_tensor_shapes(output, (batch_size, seq_len, config.d_model), "output")


class TestModelShapes:
    """Test full model shapes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_model_with_vocab_shapes(
        self, batch_size: int, seq_len: int
    ) -> None:
        """Test model with vocabulary output shapes."""
        config = XSA_LAKER_Config(d_model=128, num_heads=4)
        model = XSALAKERTransformer(
            config,
            num_layers=4,
            vocab_size=1000,
            max_seq_len=512,
        )

        x = torch.randint(0, 1000, (batch_size, seq_len))
        output = model(x)

        verify_tensor_shapes(output, (batch_size, seq_len, 1000), "output")

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_model_embedding_shapes(
        self, batch_size: int, seq_len: int
    ) -> None:
        """Test model without vocabulary (embedding output)."""
        config = XSA_LAKER_Config(d_model=128, num_heads=4)
        model = XSALAKERTransformer(
            config,
            num_layers=4,
            vocab_size=None,
        )

        x = torch.randn(batch_size, seq_len, config.d_model)
        output = model(x)

        verify_tensor_shapes(output, (batch_size, seq_len, config.d_model), "output")
