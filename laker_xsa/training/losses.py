"""
Loss functions for training LAKER-XSA models.

This module provides loss functions commonly used for training
Transformer language models.
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
    """
    Cross-entropy loss with label smoothing.

    Label smoothing replaces the one-hot target distribution with:

    .. math::

        p(y|x) = (1 - \\epsilon) \\cdot \\text{one_hot}(y) + \\epsilon / V

    where :math:`\\epsilon` is the smoothing factor and :math:`V` is
    the vocabulary size.

    This prevents the model from becoming overconfident and can improve
    generalization.

    Args:
        logits: Model output logits, shape ``(N, vocab_size)`` or
            ``(batch, seq_len, vocab_size)``.
        labels: Target labels, shape ``(N,)`` or ``(batch, seq_len)``.
        smoothing: Label smoothing factor in [0, 1).
        ignore_index: Index to ignore in loss computation.

    Returns:
        Scalar loss tensor.

    Example:
        >>> logits = torch.randn(32, 1000)
        >>> labels = torch.randint(0, 1000, (32,))
        >>> loss = label_smoothing_cross_entropy(logits, labels, smoothing=0.1)
    """
    if smoothing < 0.0 or smoothing >= 1.0:
        raise ValueError(f"Label smoothing must be in [0, 1), got {smoothing}")

    # Flatten if needed
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

    vocab_size = logits.size(-1)

    # Log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # NLL loss (negative log likelihood of correct class)
    mask = labels != ignore_index
    nll_loss = -log_probs.gather(
        dim=-1, index=labels.clamp(min=0).unsqueeze(-1)
    ).squeeze()

    # Smooth loss (negative mean log probability)
    smooth_loss = -log_probs.sum(dim=-1)

    # Combine
    if smoothing > 0:
        loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss / vocab_size
    else:
        loss = nll_loss

    # Apply mask and average
    if mask.any():
        return loss[mask].mean()
    return loss.mean()
