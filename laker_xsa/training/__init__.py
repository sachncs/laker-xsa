"""Training configuration, step orchestration, and label-smoothed loss.

``Trainer`` constructs AdamW when no optimizer is supplied and can step an
injected scheduler. Callers provide dataloaders and orchestrate epochs, logging,
evaluation cadence, and checkpointing.
"""

from __future__ import annotations

from laker_xsa.training.trainer import Trainer, TrainingConfig
from laker_xsa.training.losses import label_smoothing_cross_entropy

__all__ = [
    "Trainer",
    "TrainingConfig",
    "label_smoothing_cross_entropy",
]
