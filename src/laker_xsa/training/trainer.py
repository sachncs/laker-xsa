"""
Training loop and configuration for LAKER-XSA models.

This module provides a simple but complete training framework for
training Transformer models with XSA and LAKER attention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """
    Configuration for training.

    Attributes:
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_steps: Number of warmup steps for learning rate schedule.
        max_grad_norm: Maximum gradient norm for clipping.
        label_smoothing: Label smoothing factor for cross-entropy.
        log_interval: Steps between logging.
        eval_interval: Steps between evaluation.
        seed: Random seed for reproducibility.
    """

    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    log_interval: int = 100
    eval_interval: int = 1000
    seed: int = 42


class Trainer:
    """
    Trainer for LAKER-XSA Transformer models.

    Handles training loop, learning rate scheduling, gradient clipping,
    and logging.

    Attributes:
        model: Model to train.
        config: Training configuration.
        device: Device to train on.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.

    Example:
        >>> trainer = Trainer(model, training_config, device)
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         trainer.train_step(batch)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        optimizer: Optional[AdamW] = None,
        scheduler: Optional[LRScheduler] = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Model to train.
            config: Training configuration.
            device: Device for training.
            optimizer: Optional optimizer. If None, creates AdamW.
            scheduler: Optional scheduler. If None, uses cosine decay.
        """
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = optimizer or AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = scheduler
        self.step_count = 0

        # Set seed
        torch.manual_seed(config.seed)

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Tuple of (input_ids, labels).

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        logits = self.model(input_ids)

        # Compute loss with label smoothing
        loss = self._compute_loss(logits, labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        self.step_count += 1

        return {"loss": loss.item()}

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with label smoothing.

        Args:
            logits: Model output, shape ``(batch, seq_len, vocab_size)``.
            labels: Target labels, shape ``(batch, seq_len)``.

        Returns:
            Scalar loss tensor.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Label smoothing
        if self.config.label_smoothing > 0:
            log_probs = torch.log_softmax(logits_flat, dim=-1)
            nll_loss = -log_probs.gather(dim=-1, index=labels_flat.unsqueeze(-1)).squeeze()
            smooth_loss = -log_probs.sum(dim=-1)
            loss = (1 - self.config.label_smoothing) * nll_loss + \
                   self.config.label_smoothing * smooth_loss
        else:
            loss = nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-1)

        return loss.mean()

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(input_ids)
                loss = self._compute_loss(logits, labels)

                total_loss += loss.item()
                num_batches += 1

        return {"eval_loss": total_loss / num_batches}

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Dictionary with epoch metrics.
        """
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch in dataloader:
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            num_batches += 1

        elapsed = time.time() - start_time
        avg_loss = total_loss / num_batches

        return {
            "epoch_loss": avg_loss,
            "elapsed": elapsed,
            "steps": self.step_count,
        }
