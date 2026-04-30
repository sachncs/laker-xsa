"""
Tensor operations for LAKER-XSA.

This module provides common tensor manipulation utilities including
mask creation and shape verification.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create causal (autoregressive) attention mask.

    The mask has shape ``(1, seq_len, seq_len)`` with ``True`` for positions
    that can attend (current and past) and ``False`` for future positions.

    For position i, only positions 0, 1, ..., i are visible.

    Args:
        seq_len: Sequence length.
        device: Device to create mask on. If None, uses CPU.

    Returns:
        Boolean mask tensor of shape ``(1, seq_len, seq_len)``.

    Example:
        >>> mask = create_causal_mask(4)
        >>> mask[0]
        tensor([[ True, False, False, False],
                [ True,  True, False, False],
                [ True,  True,  True, False],
                [ True,  True,  True,  True]])
    """
    if device is None:
        device = torch.device("cpu")

    # Upper triangular matrix (above diagonal = future)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )

    # Invert: True where can attend (lower triangular including diagonal)
    return (~mask).unsqueeze(0)


def create_padding_mask(
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert padding mask to attention mask.

    The input padding_mask has ``True`` for padding positions that should
    be ignored. The output attention mask has ``True`` for valid positions.

    Args:
        padding_mask: Boolean tensor of shape ``(batch, seq_len)`` where
            ``True`` indicates padding.

    Returns:
        Attention mask of shape ``(batch, 1, 1, seq_len)`` where ``True``
        indicates valid (non-padding) positions.

    Example:
        >>> padding = torch.tensor([[True, False, False, True]])
        >>> attn_mask = create_padding_mask(padding)
        >>> attn_mask.shape
        torch.Size([1, 1, 1, 4])
    """
    if padding_mask.dim() != 2:
        raise ValueError(
            f"padding_mask must be 2D (batch, seq_len), got {padding_mask.dim()}D"
        )

    # Invert: True for valid positions
    # Add dimensions: (batch, seq_len) -> (batch, 1, 1, seq_len)
    return (~padding_mask).unsqueeze(1).unsqueeze(2)


def verify_tensor_shapes(
    x: torch.Tensor,
    expected_shape: Tuple[Union[int, None], ...],
    name: str = "tensor",
) -> bool:
    """
    Verify tensor shapes match expected.

    Uses ``None`` in expected_shape to indicate flexible dimensions.

    Args:
        x: Tensor to check.
        expected_shape: Expected shape tuple. Use ``None`` for flexible dims.
        name: Name for error messages.

    Returns:
        ``True`` if shapes match.

    Raises:
        ValueError: If shapes don't match.

    Example:
        >>> x = torch.randn(2, 128, 512)
        >>> verify_tensor_shapes(x, (None, 128, 512), "input")  # Returns True
        >>> verify_tensor_shapes(x, (2, 64, 512), "input")  # Raises ValueError
    """
    if len(x.shape) != len(expected_shape):
        raise ValueError(
            f"{name} has {len(x.shape)} dimensions, "
            f"expected {len(expected_shape)}. "
            f"Got shape {x.shape}, expected pattern {expected_shape}"
        )

    for i, (actual, expected) in enumerate(zip(x.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} dimension {i} is {actual}, expected {expected}. "
                f"Full shape: {x.shape}, expected pattern: {expected_shape}"
            )

    return True
