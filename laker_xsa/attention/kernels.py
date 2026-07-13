"""Exponential attention-kernel implementation used by LAKER v2.

The module computes ``exp(similarity / temperature) + eps`` with optional Q/K
normalization and optional symmetric averaging. Scores are clamped before
exponentiation, but ``exp(100)`` can still overflow in lower-precision dtypes.
The constructor does not validate temperature or ``eps``; non-positive
initial temperatures fail in ``math.log`` or can produce non-finite state, and
negative ``eps`` can invalidate entrywise positivity.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
import torch.nn.functional as F


class AttentionKernel(nn.Module):
    """Exponential attention kernel used by ``LakerAttention``.

    Computes ``exp(sim(q_i, k_j) / temperature)``, optionally averages the
    result with its transpose, then adds ``eps``. Symmetry and entrywise
    positivity do not imply positive semidefiniteness. Entrywise positivity
    itself requires a non-negative-enough ``eps`` and finite exponentiation.

    Two similarity modes are supported and mirror the classical attention
    variants:

    * ``normalize_qk=True`` (default) — Q and K are L2-normalised along the
      feature dimension, so scores are cosine similarities in ``[-1, 1]``.
    * ``normalize_qk=False`` — raw scaled-dot scores, divided by
      ``sqrt(head_dim)`` to match Vaswani-style attention scaling.

    Attributes:
        head_dim: Feature dimension per attention head.
        symmetric: When ``True``, returns ``(K + K^T) / 2``. This requires a
            square score matrix and enforces symmetry, not positive
            semidefiniteness.
        normalize_qk: When ``True``, L2-normalise Q/K before scoring.
        eps: Scalar added to every kernel entry. The constructor does not
            validate its sign.
        log_temperature: Log-domain learnable (or buffer) temperature used to
            enforce ``temperature ∈ [0.05, 100]`` via the :attr:`temperature`
            property.
    """

    def __init__(
        self,
        head_dim: int,
        temperature: float = 1.0,
        symmetric: bool = False,
        learnable_temperature: bool = True,
        normalize_qk: bool = True,
        eps: float = 1e-6,
    ) -> None:
        """Initialise the kernel.

        Args:
            head_dim: Per-head feature dimension; required for the
                scaled-dot branch and stored on the module.
            temperature: Initial value passed to ``math.log``. It must be
                positive for normal operation; the constructor relies on
                ``math.log`` rather than validating it explicitly.
            symmetric: Whether to average ``K`` with its transpose.
            learnable_temperature: When ``True``, store temperature as an
                ``nn.Parameter``; otherwise it is a non-trainable buffer.
            normalize_qk: When ``True``, use cosine similarity; otherwise
                use scaled dot-product.
            eps: Scalar added to the output; its sign is not validated.

        Raises:
            ValueError: If ``temperature`` is zero or negative and
                ``math.log`` rejects it.

        Side Effects:
            Allocates :attr:`log_temperature` as an ``nn.Parameter`` when
            learnable, otherwise as a registered buffer. The stored value is
            ``math.log(temperature)``.
        """
        super().__init__()
        self.head_dim = head_dim
        self.symmetric = symmetric
        self.normalize_qk = normalize_qk
        self.eps = eps

        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temperature", torch.tensor(math.log(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        """Effective temperature derived from :attr:`log_temperature`.

        The clamp ``[0.05, 100]`` bounds the effective temperature. It does not
        modify the underlying ``log_temperature`` parameter.

        Returns:
            Scalar 0-d :class:`torch.Tensor` of dtype matching
            ``log_temperature``, equal to
            ``exp(log_temperature).clamp(min=0.05, max=100.0)``.
        """
        return torch.exp(self.log_temperature).clamp(min=0.05, max=100.0)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute ``K_{ij} = exp(sim(q_i, k_j) / temperature)``.

        Args:
            q: Queries of shape ``(batch, num_heads, seq_len, head_dim)``.
            k: Keys of shape ``(batch, num_heads, seq_len, head_dim)``.

        Returns:
            Kernel matrix of shape
            ``(batch, num_heads, query_len, key_len)``. Symmetric mode requires
            equal query and key lengths.

        Numerical Notes:
            With normalized finite Q/K and positive ``eps``, mathematical
            pre-rounding values lie in
            ``[exp(-1/T) + eps, exp(1/T) + eps]``. In the raw-dot branch,
            clamping scores to ``100`` does not prevent ``exp`` overflow in
            dtypes whose finite logarithmic range is smaller (for example,
            float32). Non-finite inputs are not sanitized.

        Raises:
            RuntimeError: Propagated from :func:`torch.matmul`,
                :func:`torch.exp`, or :func:`torch.nn.functional.normalize`
                for incompatible shapes, dtypes, or devices.

        Side Effects:
            None; the module is stateless apart from the parameter/buffer.
        """
        temp = self.temperature

        if self.normalize_qk:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            scores = torch.matmul(q, k.transpose(-2, -1))
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        scores = scores / temp
        scores = torch.clamp(scores, -100.0, 100.0)
        kernel = torch.exp(scores)

        if self.symmetric:
            kernel = 0.5 * (kernel + kernel.transpose(-2, -1))

        return kernel + self.eps


__all__ = ["AttentionKernel"]


def compute_kernel_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
    normalize_qk: bool = True,
    symmetric: bool = False,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Stateless helper retained for legacy imports of ``compute_kernel_matrix``.

    New code should instantiate :class:`AttentionKernel` directly (it is stateful
    and supports a learnable temperature). This shim constructs a non-learnable
    :class:`AttentionKernel` with the requested ``temperature``/``eps`` and
    invokes it once.

    Args:
        q: Queries with shape ``(..., query_len, head_dim)``.
        k: Keys with shape ``(..., key_len, head_dim)``. Symmetric mode
            requires ``key_len == query_len``.
        normalize_qk: Cosine (default) or scaled-dot similarity.
        symmetric: Average the matrix with its transpose when ``True``;
            enforces symmetry but not positive semidefiniteness.
        temperature: Scalar passed to ``math.log`` at construction.
        eps: Scalar added to every result entry.

    Returns:
        Kernel matrix with shape ``(..., query_len, key_len)``.

    Raises:
        ValueError: If ``temperature`` is invalid for ``math.log``.
        RuntimeError: Propagated from the underlying
            :class:`AttentionKernel` for incompatible shapes, dtypes,
            or devices.
    """
    head_dim = int(q.shape[-1])
    kernel = AttentionKernel(
        head_dim=head_dim,
        temperature=temperature,
        symmetric=symmetric,
        learnable_temperature=False,
        normalize_qk=normalize_qk,
        eps=eps,
    )
    return cast(torch.Tensor, kernel(q, k))
