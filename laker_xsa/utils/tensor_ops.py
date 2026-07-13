"""Tensor manipulation helpers for LAKER-XSA.

This module creates boolean masks using the convention ``True`` for allowed
positions and ``False`` for blocked positions. Consumers either replace blocked
attention scores with a fill value such as ``-inf`` or multiply the mask into a
kernel matrix. The shape verifier supports ``None`` as a wildcard dimension.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a causal (lower-triangular) attention mask.

    The returned mask has shape ``(1, seq_len, seq_len)`` with
    ``True`` for positions that are allowed to attend (the
    current position and all earlier positions) and ``False``
    for future positions. The leading singleton dimension
    broadcasts naturally against attention scores of shape
    ``(batch, heads, seq_len, seq_len)``.

    The function never materializes a Python loop; the mask is
    built with :func:`torch.triu` over a boolean ``ones`` matrix
    and the diagonal offset is set to ``1`` to keep the diagonal
    itself visible.

    Args:
        seq_len: Sequence length. Zero produces an empty ``(1, 0, 0)`` mask;
            negative values are rejected by the underlying tensor allocation.
        device: Device on which to allocate the mask. ``None``
            (the default) places the mask on CPU; the caller is
            expected to move it to the model's device before use.

    Returns:
        Boolean tensor of shape ``(1, seq_len, seq_len)`` on the
        requested device. Entries ``mask[0, i, j]`` are ``True``
        iff ``j <= i``.

    Raises:
        RuntimeError: If ``seq_len`` is negative or ``device`` cannot be used
            for the allocation.

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

    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )

    return (~mask).unsqueeze(0)


def create_padding_mask(
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert a padding mask into an attention-compatible mask.

    The convention is inverted on the way in and unsqueezed
    twice on the way out:

    * **Input**: ``(batch, seq_len)`` boolean tensor with
      ``True`` for padding positions.
    * **Output**: ``(batch, 1, 1, seq_len)`` boolean tensor with
      ``True`` for *valid* (non-padding) positions. The two
      leading singleton dimensions broadcast against attention
      scores of shape ``(batch, heads, seq_len, seq_len)``.

    The output mask is boolean for boolean input. Consumers apply it either by
    filling blocked pre-softmax scores (normally with ``-inf``) or by
    multiplying it into a kernel matrix; it is not an additive logit mask.

    Args:
        padding_mask: Expected to be a boolean tensor of shape
            ``(batch, seq_len)`` where ``True`` indicates padding. The
            function validates rank but not dtype; integer inputs undergo
            bitwise inversion rather than boolean negation.

    Returns:
        Attention mask of shape ``(batch, 1, 1, seq_len)``. For the expected
        boolean input, ``True`` indicates a valid non-padding position.

    Raises:
        ValueError: If ``padding_mask`` is not 2D.
        TypeError: If bitwise inversion is unsupported for the input dtype.

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

    return (~padding_mask).unsqueeze(1).unsqueeze(2)


def verify_tensor_shapes(
    x: torch.Tensor,
    expected_shape: Tuple[Union[int, None], ...],
    name: str = "tensor",
) -> bool:
    """Verify that a tensor matches an expected shape pattern.

    The ``expected_shape`` argument is a tuple where each entry
    is either a concrete ``int`` (must match) or ``None``
    (wildcard, any size accepted). This is more flexible than
    ``x.shape == expected`` for cases where only some axes are
    known statically - for example, the batch dimension or a
    sequence length that may vary.

    The function checks two things, in order:

    1. The rank of ``x`` matches the rank of ``expected_shape``.
    2. For every axis, the size matches when ``expected_shape``
       specifies a concrete integer.

    Args:
        x: Tensor to check.
        expected_shape: Expected shape pattern. ``None`` entries
            are treated as wildcards.
        name: Human-readable name used in error messages to
            identify which tensor failed. Defaults to
            ``"tensor"``.

    Returns:
        ``True`` if the shape matches. The return value is
        mostly for call-site convenience; the function also
        raises on failure.

    Raises:
        ValueError: If the rank does not match, or if any
            concrete axis size does not match. The error
            message includes both the actual shape and the
            expected pattern.

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
