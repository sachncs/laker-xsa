from __future__ import annotations

"""
Training utilities for LAKER-XSA models.

This package provides training loops, loss functions, and utilities
for training Transformer models with XSA and LAKER attention.
"""

from laker_xsa.training.trainer import Trainer, TrainingConfig
from laker_xsa.training.losses import label_smoothing_cross_entropy

__all__ = [
    "Trainer",
    "TrainingConfig",
    "label_smoothing_cross_entropy",
]
