"""
Numerical stability utilities for LAKER-XSA.

This module provides functions for checking tensor validity and
applying numerical safeguards.
"""

from __future__ import annotations

import torch


def check_finite(
    x: torch.Tensor,
    name: str = "tensor",
    raise_error: bool = True,
) -> bool:
    """
    Check if tensor contains only finite values.

    Detects NaN and Inf values that can arise from numerical instabilities
    in attention computation or iterative solving.

    Args:
        x: Tensor to check.
        name: Name for error messages.
        raise_error: If ``True``, raise ``ValueError`` on non-finite values.
            If ``False``, return ``False``.

    Returns:
        ``True`` if all values are finite.

    Raises:
        ValueError: If ``raise_error=True`` and non-finite values found.

    Example:
        >>> x = torch.randn(10, 10)
        >>> check_finite(x)  # Returns True
        >>> x[0, 0] = float('inf')
        >>> check_finite(x, raise_error=False)  # Returns False
    """
    is_finite = torch.isfinite(x).all()

    if not is_finite:
        if raise_error:
            # Count different types of non-finite values
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isposinf(x).sum().item()
            neg_inf_count = torch.isneginf(x).sum().item()

            raise ValueError(
                f"{name} contains non-finite values: "
                f"{nan_count} NaNs, {inf_count} +Infs, {neg_inf_count} -Infs. "
                f"Shape: {x.shape}"
            )
        return False

    return True


def clamp_tensor(
    x: torch.Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "tensor",
) -> torch.Tensor:
    """
    Clamp tensor values to a specified range.

    Used to prevent numerical overflow/underflow in attention computation.

    Args:
        x: Tensor to clamp.
        min_val: Minimum value. If ``None``, no lower bound.
        max_val: Maximum value. If ``None``, no upper bound.
        name: Name for logging.

    Returns:
        Clamped tensor (in-place if possible).

    Example:
        >>> x = torch.tensor([-1e10, 0.0, 1e10])
        >>> clamp_tensor(x, min_val=-1e6, max_val=1e6)
        tensor([-1.0000e+06,  0.0000e+00,  1.0000e+06])
    """
    if min_val is None and max_val is None:
        return x

    if min_val is not None and max_val is not None:
        if min_val > max_val:
            raise ValueError(
                f"clamp_tensor: min_val ({min_val}) > max_val ({max_val})"
            )

    return torch.clamp(x, min_val if min_val is not None else float("-inf"),
                       max_val if max_val is not None else float("inf"))
