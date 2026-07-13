"""Loss functions for training LAKER-XSA models.

This module provides :func:`label_smoothing_cross_entropy`, a
drop-in replacement for
:func:`torch.nn.functional.cross_entropy` that supports label
smoothing and ``ignore_index`` masking, the two features the rest
of the training loop relies on.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def label_smoothing_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.1,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Cross-entropy loss with label smoothing and ignore-index support.

    Computes the negative log-likelihood of the targets using a
    smoothed target distribution. The smoothed distribution is the
    convex combination of the one-hot target and a uniform
    distribution over the vocabulary:

    .. math::

        p(y|x) = (1 - \\epsilon) \\cdot \\text{one\\_hot}(y)
               + \\epsilon / V

    where :math:`\\epsilon` is ``smoothing`` and :math:`V` is
    ``logits.size(-1)``. The resulting per-position loss is

    .. math::

        \\ell(x, y) = (1 - \\epsilon) \\cdot \\text{NLL}(p, y)
                    - (\\epsilon / V) \\cdot \\sum_{v} \\log p_v

    Labels equal to ``ignore_index`` are excluded when at least one valid label
    exists. If all labels are ignored, the implementation returns the
    unfiltered mean instead; ignored labels are clamped to class zero for the
    gathered NLL in that fallback. Negative labels other than ``ignore_index``
    are likewise clamped to zero rather than rejected.

    Args:
        logits: Model output. Either a 2D tensor of shape
            ``(N, vocab_size)`` or a 3D tensor of shape
            ``(batch, seq_len, vocab_size)``. In the 3D case the
            leading two dimensions are flattened.
        labels: Target labels matching the flattened leading dimensions of
            ``logits``. Entries equal to ``ignore_index`` are excluded when at
            least one valid label remains.
        smoothing: Label smoothing factor. Must lie in ``[0, 1)``.
            ``0.0`` recovers the standard (unsmoothed) NLL loss.
        ignore_index: Label value to ignore. Defaults to ``-1``,
            the conventional value used by PyTorch losses. Labels
            equal to ``ignore_index`` contribute neither to the
            numerator nor to the denominator of the mean.

    Returns:
        Scalar loss. When every label equals ``ignore_index``, this is the
        mean over the unfiltered, class-zero-clamped per-position losses rather
        than a zero loss or an empty mean.

    Raises:
        ValueError: If ``smoothing`` is outside ``[0, 1)``.
        RuntimeError: Propagated from reshape, log-softmax, or gather when
            shapes, devices, or dtypes are incompatible, or when any label is
            at least ``vocab_size``. Gather occurs before masking, so this also
            applies to a positive out-of-range ``ignore_index``.

    Example:
        >>> logits = torch.randn(32, 1000)
        >>> labels = torch.randint(0, 1000, (32,))
        >>> loss = label_smoothing_cross_entropy(logits, labels, smoothing=0.1)
    """
    if smoothing < 0.0 or smoothing >= 1.0:
        raise ValueError(f"Label smoothing must be in [0, 1), got {smoothing}")

    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

    vocab_size = logits.size(-1)

    log_probs = F.log_softmax(logits, dim=-1)

    mask = labels != ignore_index
    nll_loss = -log_probs.gather(
        dim=-1, index=labels.clamp(min=0).unsqueeze(-1)
    ).squeeze()

    smooth_loss = -log_probs.sum(dim=-1)

    if smoothing > 0:
        loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss / vocab_size
    else:
        loss = nll_loss

    if mask.any():
        return loss[mask].mean()
    return loss.mean()
