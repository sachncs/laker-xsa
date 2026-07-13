"""Standalone finite-value detection and tensor-clamping helpers.

These utilities are exported for callers that want explicit numerical checks.
The current attention and solver implementations use their own local clamps and
do not automatically invoke these functions.
"""

from __future__ import annotations

from typing import Optional

import torch


def check_finite(
    x: torch.Tensor,
    name: str = "tensor",
    raise_error: bool = True,
) -> bool:
    """Check that a tensor contains only finite values.

    The check is a single :func:`torch.isfinite` reduction across
    the entire tensor. On failure, the function can either raise
    a :class:`ValueError` with a precise breakdown (number of
    ``NaN``s, ``+Inf``s, and ``-Inf``s) or simply return
    ``False`` for non-raising callers. The breakdown is computed
    only on the error path, so the hot path is a single reduction
    regardless of the answer.

    Args:
        x: Tensor to check. Any dtype is accepted, but the check
            is most meaningful for floating-point tensors; integer
            tensors are always finite.
        name: Human-readable name used in error messages to
            identify which tensor failed. Defaults to
            ``"tensor"``.
        raise_error: When ``True`` (the default), raise
            :class:`ValueError` on the first non-finite sample.
            When ``False``, return ``False`` and stay silent -
            useful for callers that prefer to log and continue.

    Returns:
        ``True`` if every entry of ``x`` is finite. ``False`` if
        any entry is non-finite and ``raise_error`` is ``False``.

    Raises:
        ValueError: If ``raise_error`` is ``True`` and ``x``
            contains at least one ``NaN``, ``+Inf``, or
            ``-Inf``. The message reports the per-kind counts
            and the tensor's shape.

    Example:
        >>> x = torch.randn(10, 10)
        >>> check_finite(x)  # Returns True
        >>> x[0, 0] = float('inf')
        >>> check_finite(x, raise_error=False)  # Returns False
    """
    is_finite = torch.isfinite(x).all()

    if not is_finite:
        if raise_error:
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
    name: str = "tensor",  # pylint: disable=unused-argument
) -> torch.Tensor:
    """Clamp a tensor to a closed interval.

    A small wrapper over :func:`torch.clamp` that treats
    ``None`` for either bound as "no bound on that side". When
    both bounds are ``None`` the function returns ``x`` itself,
    with no copy. The returned tensor is a fresh tensor whenever
    at least one bound is supplied.

    The ``name`` argument is accepted for API symmetry with
    :func:`check_finite` and is not used inside the function -
    it exists so callers can swap the two helpers without
    changing their call sites.

    Args:
        x: Tensor to clamp.
        min_val: Lower bound. ``None`` means "no lower bound".
        max_val: Upper bound. ``None`` means "no upper bound".
        name: Name for logging. Currently unused; preserved for
            API symmetry.

    Returns:
        A tensor of the same shape and dtype as ``x`` with values
        clipped to ``[min_val, max_val]``. The result shares
        storage with ``x`` only when no clamp is applied; any
        actual clamping produces a new tensor.

    Raises:
        ValueError: If both ``min_val`` and ``max_val`` are
            supplied and ``min_val > max_val``.

    Example:
        >>> x = torch.tensor([-1e10, 0.0, 1e10])
        >>> clamp_tensor(x, min_val=-1e6, max_val=1e6)
        tensor([-1.0000e+06,  0.0000e+00,  1.0000e+06])
    """
    if min_val is None and max_val is None:
        return x

    if min_val is not None and max_val is not None:
        if min_val > max_val:
            raise ValueError(f"clamp_tensor: min_val ({min_val}) > max_val ({max_val})")

    return torch.clamp(
        x,
        min_val if min_val is not None else float("-inf"),
        max_val if max_val is not None else float("inf"),
    )
