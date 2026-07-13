"""Functional (stateless) API for attention kernels.

This module provides stateless functional counterparts to the class-based
modules in the attention subpackage. They follow the same design as
``torch.nn.functional`` relative to ``torch.nn``: the functional forms accept
all parameters explicitly and do not hold learnable state, which makes them
convenient for ad-hoc analysis, scripting and unit tests.

Functional form            Class-based form
-----------------          -----------------
``compute_kernel_matrix``  :meth:`AttentionKernel.forward`
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_kernel_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature: float = 1.0,
    normalize_qk: bool = True,
    symmetric: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute an exponential attention kernel matrix.

    The returned entries are ``exp(sim(q_i, k_j) / temperature) + eps``.
    This is the stateless counterpart to :meth:`AttentionKernel.forward`.
    Two similarity modes are supported:

    * ``normalize_qk=True`` (default): Q and K are L2-normalised along the
      feature dimension, so scores are cosine similarities in ``[-1, 1]``.
    * ``normalize_qk=False``: raw dot products are scaled by
      ``1 / sqrt(head_dim)`` (Vaswani-style) and then divided by ``temperature``.

    Scores are clipped to ``[-100, 100]`` before exponentiation. This does not
    prevent ``exp(100)`` from overflowing in lower-precision dtypes. Neither
    optional symmetric averaging nor entrywise positivity implies positive
    semidefiniteness, and negative ``eps`` can make entries non-positive.

    Args:
        q: Queries with shape ``(..., query_len, head_dim)``.
        k: Keys with broadcast-compatible leading dimensions and shape
            ``(..., key_len, head_dim)``.
        temperature: Score divisor. It is not validated; zero or negative
            values can produce non-finite or inverted scores.
        normalize_qk: If ``True``, use cosine similarity; otherwise use
            scaled-dot similarity.
        symmetric: If ``True``, return ``(K + K^T) / 2``. This requires equal
            query/key lengths and enforces symmetry, not PSD.
        eps: Scalar added to every kernel entry; its sign is not validated.

    Returns:
        Kernel matrix of shape ``(..., query_len, key_len)``. Non-finite inputs,
        zero temperature, and dtype overflow can produce non-finite outputs.

    Raises:
        ZeroDivisionError: If the raw-dot branch receives ``head_dim == 0``.
        RuntimeError: Propagated from :func:`torch.matmul`,
            :func:`torch.exp`, or :func:`torch.nn.functional.normalize`
            for incompatible shapes, dtypes, or devices.
    """
    if normalize_qk:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1))
    else:
        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    scores = scores / temperature
    scores = torch.clamp(scores, -100.0, 100.0)
    kernel = torch.exp(scores)

    if symmetric:
        kernel = 0.5 * (kernel + kernel.transpose(-2, -1))

    return kernel + eps
