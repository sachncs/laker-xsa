"""Training loop for LAKER-XSA Transformer models.

This module provides :class:`Trainer`, a small collection of training and
evaluation steps around a model, default AdamW optimizer, optional gradient
clipping, and optional scheduler. It does not own dataloaders or orchestrate
epochs, logging, checkpointing, or evaluation cadence; callers invoke
``train_epoch`` and ``evaluate`` themselves.

The companion :class:`TrainingConfig` dataclass exposes several
knobs. Not all of them are consumed directly by :class:`Trainer`;
some (notably ``num_epochs``, ``warmup_steps``, ``log_interval``
and ``eval_interval``) are stored on the config for callers to read.
The bundled training CLI only reads ``num_epochs`` (and forwards
``learning_rate``, ``warmup_steps``, ``seed``); it does not consult
``log_interval`` or ``eval_interval``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from laker_xsa.training.losses import label_smoothing_cross_entropy


@dataclass
class TrainingConfig:
    """Knobs for a training run.

    The dataclass mixes fields that :class:`Trainer` consumes
    internally with fields that are stored for the *caller* (e.g.
    the CLI) to read. This split is deliberate: it lets a single
    config object describe a full training run without forcing the
    trainer to take on scheduling and logging responsibilities it
    was not designed for.

    Fields consumed by :class:`Trainer`:

    * :attr:`learning_rate` - learning rate used to construct the default
      AdamW optimizer.
    * :attr:`weight_decay` - weight decay used by the default
      AdamW optimizer.
    * :attr:`max_grad_norm` - maximum gradient ``L2`` norm used by
      :func:`torch.nn.utils.clip_grad_norm_` on every step.
    * :attr:`label_smoothing` - smoothing factor passed to
      :func:`laker_xsa.training.losses.label_smoothing_cross_entropy`
      on every step.
    * :attr:`seed` - used by the trainer to seed the PyTorch global
      RNG at construction. Note that this does **not** seed the
      Python or NumPy RNGs; for full reproducibility use
      :func:`laker_xsa.utils.seed.set_seed` instead.

    Fields stored for the caller:

    * :attr:`num_epochs` - the outer-loop epoch count. The trainer
      does not iterate epochs; the caller is expected to do so.
    * :attr:`warmup_steps` - metadata for caller-defined scheduling. The
      trainer does not create a scheduler.
    * :attr:`log_interval` / :attr:`eval_interval` - logging and
      evaluation cadences. The trainer does not log or trigger
      evaluation on its own, and the bundled CLI does not read them
      either; they are provided for callers that implement their own
      logging/evaluation cadence.

    Attributes:
        num_epochs: Number of epochs the caller intends to run.
        learning_rate: Learning rate for the default AdamW optimizer.
        weight_decay: Weight decay for the default AdamW optimizer.
        warmup_steps: Warmup length for caller-driven LR schedules.
        max_grad_norm: Gradient clipping ``L2`` norm ceiling.
        label_smoothing: Smoothing factor for the cross-entropy
            loss.
        log_interval: Steps between caller-driven log lines.
        eval_interval: Steps between caller-driven evaluation
            passes.
        seed: Seed for ``torch.manual_seed`` at trainer
            construction. Defaults to ``42`` for convenience but
            callers needing broader RNG coverage can use
            :func:`laker_xsa.utils.seed.set_seed`.
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
    """Minimal training loop for LAKER-XSA Transformer models.

    The trainer wraps a model and an optimizer and exposes three
    operations:

    * :meth:`train_step` - one forward / backward / optimizer step
      over a single batch. This is the workhorse of the training
      loop and includes gradient clipping and an optional scheduler
      step.
    * :meth:`train_epoch` - convenience wrapper that iterates a
      dataloader and aggregates the per-step losses.
    * :meth:`evaluate` - a single pass over a validation
      dataloader, returning the average loss.

    The trainer does **not** implement learning-rate scheduling
    on its own: if no scheduler is supplied, the learning rate is
    constant for the entire run. Callers that want warmup or decay
    should construct an ``LRScheduler`` (or a warmup-then-cosine
    composition) and pass it to ``__init__``.

    The trainer also does **not** implement logging or
    checkpointing. ``TrainingConfig.log_interval`` and
    ``TrainingConfig.eval_interval`` are stored on the config but
    not consulted by the trainer; the outer loop is expected to
    read them and decide when to print or call :meth:`evaluate`.

    Determinism:

    * The trainer seeds PyTorch's global RNG during construction, after the
      supplied model has already been initialized. Seed before model creation
      when initialization reproducibility is required.
    * The trainer does not configure deterministic algorithms or cuDNN.
      :func:`laker_xsa.utils.seed.set_seed` covers more global generators and
      cuDNN settings but still cannot guarantee reproducibility for every
      operation and environment.

    Training, evaluation, optimizer, scheduler, and counter state are mutable;
    calls on one trainer instance must be serialized by the caller.

    Attributes:
        model: The model being trained. Moved to ``device`` only if
            the caller does so explicitly; the trainer assumes the
            model is already on the right device.
        config: The :class:`TrainingConfig` instance.
        device: The :class:`torch.device` the trainer moves batch
            tensors to before the forward pass.
        optimizer: The optimizer used to update parameters. The
            default is AdamW constructed from the config; an
            externally-built optimizer can be injected.
        scheduler: Optional learning-rate scheduler. ``None`` when no
            scheduler is configured.
        step_count: Cumulative number of optimizer steps performed
            by the trainer. Useful for caller-driven schedulers and
            log lines. Starts at ``0`` and is incremented at the end
            of every :meth:`train_step`.

    Example:
        >>> trainer = Trainer(model, training_config, device)
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         metrics = trainer.train_step(batch)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        optimizer: Optional[AdamW] = None,
        scheduler: Optional[LRScheduler] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The model to train. The trainer does not call
                ``model.to(device)``; the caller is responsible for
                moving the model to ``device`` before constructing
                the trainer.
            config: Training configuration. See :class:`TrainingConfig`
                for the per-field contract.
            device: Device used for the batch tensors during the
                forward pass. The model itself is not moved.
            optimizer: Optional optimizer. When ``None`` (the
                default), an AdamW optimizer is built from
                ``config.learning_rate`` and ``config.weight_decay``.
                Although the parameter is annotated ``Optional[AdamW]``,
                any optimizer exposing the standard
                ``zero_grad``/``step`` interface works at runtime, including
                optimizers with parameter groups or fused implementations.
            scheduler: Optional learning-rate scheduler. When
                ``None`` (the default), no scheduler is used and
                the learning rate stays constant for the entire
                run. The trainer calls ``scheduler.step()`` once
                per :meth:`train_step` when a scheduler is
                provided. The trainer does **not** construct a
                warmup or cosine schedule on its own; callers
                that want one should build it here.

        Side Effects:
            Seeds PyTorch's global RNG with ``config.seed`` and, when no
            optimizer is supplied, constructs an AdamW optimizer over
            ``model.parameters()``.
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

        torch.manual_seed(config.seed)

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, float]:
        """Perform a single training step.

        Runs the forward pass, computes the loss, clears the
        gradients, runs the backward pass, clips the gradients to
        ``config.max_grad_norm``, applies the optimizer, and steps
        the scheduler (if any). :attr:`step_count` is incremented
        at the end of the call.

        Args:
            batch: A 2-tuple ``(input_ids, labels)`` as produced by
                a typical language-modeling dataloader. ``input_ids``
                is expected to be shape ``(batch, seq_len)`` of
                integer token IDs; ``labels`` has the same shape.
                Both are moved to :attr:`device` before the forward
                pass.

        Returns:
            A single-key dictionary ``{"loss": loss.item()}``
            containing the float value of the loss for this step.
            Callers are expected to aggregate these values across
            a training epoch. The keys are deliberately stable so
            downstream logging code can rely on them.

        Side Effects:
            Sets training mode, clears gradients through the optimizer,
            backpropagates, clips gradients, updates parameters/optimizer state,
            optionally advances the scheduler, and increments ``step_count``.

        Raises:
            RuntimeError: Propagated from model execution, loss computation,
                autograd, gradient clipping, optimizer, or scheduler operations.
        """
        self.model.train()
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(input_ids)

        loss = self.compute_loss(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.step_count += 1

        return {"loss": loss.item()}

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the label-smoothed cross-entropy loss.

        Thin wrapper over
        :func:`laker_xsa.training.losses.label_smoothing_cross_entropy`
        that supplies the smoothing factor from
        :attr:`TrainingConfig.label_smoothing` and pins
        ``ignore_index`` to ``-1`` (the convention used by the rest
        of the codebase).

        Args:
            logits: Model output shaped either ``(N, vocab_size)`` or
                ``(batch, seq_len, vocab_size)``.
            labels: Matching labels shaped ``(N,)`` or ``(batch, seq_len)``.
                Values equal to ``-1`` are omitted unless every label is
                ignored, in which case the underlying helper returns its
                unfiltered mean.

        Returns:
            Scalar (``ndim == 0``) loss tensor.
        """
        return label_smoothing_cross_entropy(
            logits,
            labels,
            smoothing=self.config.label_smoothing,
            ignore_index=-1,
        )

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate the model on a validation dataloader.

        Iterates ``dataloader`` under ``torch.no_grad()``, computes
        the loss for every batch, and returns the average. The
        model is set to evaluation mode for the duration of the
        call and restored implicitly by the next :meth:`train_step`
        (which calls ``self.model.train()``).

        Args:
            dataloader: A :class:`torch.utils.data.DataLoader`
                yielding ``(input_ids, labels)`` batches with the
                same conventions as :meth:`train_step`.

        Returns:
            ``{"eval_loss": average_loss}``, where each batch contributes one
            equally weighted scalar regardless of batch size.

        Raises:
            ZeroDivisionError: If ``dataloader`` yields no batches, the
                final ``total_loss / num_batches`` divides by zero.

        Side Effects:
            * Sets the model to evaluation mode
              (``self.model.eval()``). The model is left in eval mode on
              return; it is not restored to its prior mode until a later
              :meth:`train_step` calls ``self.model.train()``.
            * Disables gradient computation for the duration of the
              pass.
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
                loss = self.compute_loss(logits, labels)

                total_loss += loss.item()
                num_batches += 1

        return {"eval_loss": total_loss / num_batches}

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one pass over ``dataloader``.

        Iterates the dataloader, calling :meth:`train_step` on each
        batch, and aggregates the per-step losses. The wall-clock
        duration of the pass is measured and included in the
        returned metrics.

        Args:
            dataloader: A :class:`torch.utils.data.DataLoader`
                yielding ``(input_ids, labels)`` batches with the
                same conventions as :meth:`train_step`.

        Returns:
            A dictionary with the following keys:

            * ``"epoch_loss"`` - mean of the per-step ``loss``
              values over the epoch.
            * ``"elapsed"`` - wall-clock seconds for the epoch,
              measured with :func:`time.time` around the dataloader
              loop. Includes model forward/backward time but not
              any caller-side work outside the loop.
            * ``"steps"`` - the trainer's cumulative
              :attr:`step_count` after the epoch.

        Raises:
            ZeroDivisionError: If ``dataloader`` yields no batches, the
                final ``total_loss / num_batches`` divides by zero.

        Side Effects:
            * Increments :attr:`step_count` by the number of
              batches in ``dataloader``.
            * Sets the model to training mode (the
              :meth:`train_step` calls do this implicitly on
              every batch); the model is left in training mode on
              return.
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
