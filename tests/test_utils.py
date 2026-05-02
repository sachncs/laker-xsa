"""Tests for utility functions (masks, shapes, seeds)."""

from __future__ import annotations

import pytest
import torch

from laker_xsa.utils.tensor_ops import (
    create_causal_mask,
    create_padding_mask,
    verify_tensor_shapes,
)
from laker_xsa.utils.seed import set_seed, get_rng_states, set_rng_states


class TestCreateCausalMask:
    """Tests for create_causal_mask."""

    @pytest.mark.parametrize("seq_len", [1, 4, 16, 64])
    def test_shape(self, seq_len: int) -> None:
        mask = create_causal_mask(seq_len)
        assert mask.shape == (1, seq_len, seq_len)

    def test_lower_triangular(self) -> None:
        mask = create_causal_mask(4)
        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        assert torch.equal(mask[0], expected)

    def test_on_device(self) -> None:
        if torch.cuda.is_available():
            mask = create_causal_mask(4, device=torch.device("cuda"))
            assert mask.device.type == "cuda"


class TestCreatePaddingMask:
    """Tests for create_padding_mask."""

    def test_shape(self) -> None:
        padding = torch.tensor([[True, False, False, True]])
        mask = create_padding_mask(padding)
        assert mask.shape == (1, 1, 1, 4)

    def test_padding_positions_false(self) -> None:
        padding = torch.tensor([[True, False, False, True]])
        mask = create_padding_mask(padding)
        # Position 0 is padding → should be False (ignored)
        assert not mask[0, 0, 0, 0].item()
        # Position 1 is valid → should be True
        assert mask[0, 0, 0, 1].item()

    def test_no_padding(self) -> None:
        padding = torch.tensor([[False, False, False, False]])
        mask = create_padding_mask(padding)
        assert mask.all()

    def test_all_padding(self) -> None:
        padding = torch.tensor([[True, True, True, True]])
        mask = create_padding_mask(padding)
        assert not mask.any()

    def test_batch_multiple(self) -> None:
        padding = torch.tensor(
            [
                [True, False, False],
                [False, True, False],
            ]
        )
        mask = create_padding_mask(padding)
        assert mask.shape == (2, 1, 1, 3)

    def test_invalid_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            create_padding_mask(torch.randn(1, 1, 4).bool())


class TestVerifyTensorShapes:
    """Tests for verify_tensor_shapes."""

    def test_matching_shapes(self) -> None:
        x = torch.randn(2, 128, 512)
        assert verify_tensor_shapes(x, (2, 128, 512)) is True

    def test_with_None_wildcards(self) -> None:
        x = torch.randn(2, 128, 512)
        assert verify_tensor_shapes(x, (None, 128, 512)) is True
        assert verify_tensor_shapes(x, (None, None, None)) is True
        assert verify_tensor_shapes(x, (2, None, 512)) is True

    def test_mismatch_dim_count_raises(self) -> None:
        x = torch.randn(2, 128, 512)
        with pytest.raises(ValueError, match="dimensions"):
            verify_tensor_shapes(x, (2, 128))

    def test_mismatch_dim_value_raises(self) -> None:
        x = torch.randn(2, 128, 512)
        with pytest.raises(ValueError, match="dimension"):
            verify_tensor_shapes(x, (2, 64, 512))

    def test_None_does_not_trigger_mismatch(self) -> None:
        x = torch.randn(5, 100, 200)
        assert verify_tensor_shapes(x, (None, None, 200)) is True


class TestSeedFunctions:
    """Tests for seed utilities."""

    def test_set_seed_no_error(self) -> None:
        set_seed(42)

    def test_get_set_rng_round_trip(self) -> None:
        set_seed(42)
        torch.randn(5)
        state = get_rng_states()

        # Generate more values from this state
        a2 = torch.randn(5)

        # Restore and re-generate: should match a2, not a
        set_rng_states(state)
        a2_again = torch.randn(5)
        assert torch.allclose(a2, a2_again)

    def test_get_rng_states_has_required_keys(self) -> None:
        state = get_rng_states()
        assert "python" in state
        assert "numpy" in state
        assert "torch" in state

    def test_set_rng_states_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            set_rng_states({"numpy": None, "torch": None})

    def test_reproducible_with_same_seed(self) -> None:
        set_seed(123)
        x1 = torch.randn(10)
        set_seed(123)
        x2 = torch.randn(10)
        assert torch.allclose(x1, x2)
