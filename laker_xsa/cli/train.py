#!/usr/bin/env python3
"""Training script for LAKER-XSA models.

Constructs a Transformer with a configurable attention back-end,
generates a fully synthetic (input, label) pair from
``torch.randint``, and runs a short AdamW training loop to exercise the
end-to-end pipeline.

Usage:
    python -m laker_xsa.cli.train
    python -m laker_xsa.cli.train --attention-type fused_v2 --num-epochs 2

Important methodological notes:

* The training data is **dummy**: both inputs and labels are drawn
  independently from ``torch.randint(0, vocab_size, (num_samples, seq_len))``.
  There is no structural relationship between input and target. A sufficiently
  large model could memorize a finite sampled dataset, but metrics here do not
  measure generalization or a meaningful task.

* ``--attention-type`` accepts ``"standard"``, ``"xsa"``, ``"kernel"``,
  ``"fused"``, and ``"fused_v2"`` (the v2
  :class:`~laker_xsa.attention.laker.LakerAttention`). Note that only
  ``"fused"`` sets ``XSA_LAKER_Config(use_fused=True)``; ``"fused_v2"``
  is selected at the
  :class:`~laker_xsa.model.full_model.XSALAKERTransformer` block level
  rather than the legacy ``use_fused`` flag.

* argparse behaviour: :func:`argparse.ArgumentParser.parse_args` is the
  source of every flag; calling it can raise :class:`SystemExit` on
  invalid flags (including an invalid ``--attention-type``), missing
  required arguments, or ``--help``. All flags here are optional and
  have safe defaults.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.model.full_model import XSALAKERTransformer
from laker_xsa.training.trainer import Trainer, TrainingConfig


def create_dummy_data(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic ``(input_ids, labels)`` pair for smoke training.

    Both tensors are drawn independently from
    ``torch.randint(0, vocab_size, (num_samples, seq_len))``; there is no
    causal or semantic relationship between them.

    Args:
        num_samples: Number of synthetic sequences to generate.
        seq_len: Sequence length of each example.
        vocab_size: Vocabulary size; samples are drawn from
            ``[0, vocab_size)``.

    Returns:
        ``(input_ids, labels)`` with each tensor having shape
        ``(num_samples, seq_len)`` and dtype ``torch.int64`` (the default
        of :func:`torch.randint`).

    Side Effects:
        Allocates two CPU tensors and advances PyTorch's global CPU RNG state.
        :class:`Trainer` later moves batches to its configured device.

    Raises:
        RuntimeError: If a requested dimension is negative or ``vocab_size`` is
            not positive.

    Complexity:
        ``O(num_samples * seq_len)``.

    Note:
        Labels are independent of inputs. A finite dataset can still be
        memorized, so this helper is suitable only for pipeline smoke tests.
    """
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))
    return input_ids, labels


def main() -> None:
    """CLI entry point for the training script.

    Workflow:

        1. Parse CLI flags (see below).
        2. Seed ``torch`` with ``args.seed``.
        3. Select device: ``cuda`` when ``args.cuda`` is set and a CUDA
           device is visible, otherwise ``cpu``.
        4. Build :class:`~laker_xsa.config.XSA_LAKER_Config`. ``use_fused`` is
           true only when ``args.attention_type == "fused"``;
           ``"fused_v2"`` selects :class:`~laker_xsa.attention.laker.LakerAttention`
           through the Transformer block's ``attention_type`` argument.
        5. Build :class:`~laker_xsa.model.full_model.XSALAKERTransformer`
           with the chosen attention type, ``d_ff = 4 * d_model``,
           ``dropout=0.1``, ``max_seq_len=args.seq_len``.
        6. Generate dummy train and val splits
           (val size = ``num_samples // 10``).
        7. Construct :class:`~laker_xsa.training.trainer.Trainer` and
           run ``args.num_epochs`` epochs, printing per-epoch train
           loss, val loss, and elapsed seconds.

    Flags:
        ``--d-model`` (int, default 256) — model dimension.
        ``--num-heads`` (int, default 4) — number of heads.
        ``--num-layers`` (int, default 4) — number of Transformer blocks.
        ``--vocab-size`` (int, default 1000) — vocabulary size used
            both by the embedding and by the dummy-data sampler.
        ``--num-epochs`` (int, default 5).
        ``--batch-size`` (int, default 8).
        ``--seq-len`` (int, default 64) — fixed sequence length used by
            both the position-embedding ceiling and the dummy data.
        ``--num-samples`` (int, default 1000) — number of training
            samples; validation samples default to ``num_samples // 10``.
        ``--attention-type`` (str, default ``"fused"``; choices
            ``{"standard", "xsa", "kernel", "fused", "fused_v2"}``) —
            which attention back-end each block uses.
        ``--seed`` (int, default 42) — passed to ``torch.manual_seed``
            and to :class:`~laker_xsa.training.trainer.TrainingConfig`.
        ``--cuda`` (flag) — request CUDA; falls back to CPU if
            unavailable.

    Side Effects:
        Sets the global torch seed. Prints progress text and per-epoch
        metrics to :data:`sys.stdout` via :func:`print`. Allocates the
        full Transformer plus optimiser state on the selected device.

    Limitations:
        Training data has independent random labels. ``warmup_steps=100`` is
        stored on ``TrainingConfig`` but no scheduler is supplied, so it has no
        effect. ``num_samples < 10`` creates an empty validation loader and
        causes :meth:`Trainer.evaluate` to divide by zero.

    Raises:
        SystemExit: From :func:`argparse.ArgumentParser.parse_args` on
            invalid flags (including an unrecognised
            ``--attention-type``), missing arguments, or ``--help``.
    """
    parser = argparse.ArgumentParser(description="Train LAKER-XSA model")
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model dimension",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of Transformer layers",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--attention-type",
        type=str,
        default="fused",
        choices=["standard", "xsa", "kernel", "fused", "fused_v2"],
        help="Attention type to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available",
    )

    args = parser.parse_args()

    # Set seed for reproducibility of the dummy data and parameter init.
    torch.manual_seed(args.seed)

    # Device: prefer CUDA only when both requested *and* available.
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ``use_fused`` is true only for the deprecated ``"fused"`` choice;
    # ``"fused_v2"`` is selected directly by the Transformer blocks.
    config = XSA_LAKER_Config(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_iterations=10,
        preconditioner_rank=args.d_model // 16,
        kernel_type="rbf",
        xsa_mode="subtract_projection",
        use_fused=args.attention_type == "fused",
    )

    # Model.
    model = XSALAKERTransformer(
        config,
        num_layers=args.num_layers,
        d_ff=args.d_model * 4,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        dropout=0.1,
        attention_type=args.attention_type,
    )
    model = model.to(device)

    # Count parameters (printed as a sanity check).
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} / {total_params:,}")

    # Dummy data — labels are independent of inputs by construction, so
    # no learning signal is present.
    train_input, train_labels = create_dummy_data(
        args.num_samples, args.seq_len, args.vocab_size
    )
    train_dataset = TensorDataset(train_input, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Validation split is one-tenth the training size.
    val_input, val_labels = create_dummy_data(
        args.num_samples // 10, args.seq_len, args.vocab_size
    )
    val_dataset = TensorDataset(val_input, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # ``warmup_steps`` is metadata here because no scheduler is supplied.
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=1e-3,
        warmup_steps=100,
        seed=args.seed,
    )

    trainer = Trainer(model, training_config, device)

    # Training loop: per-epoch print of train loss, val loss, elapsed.
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Attention type: {args.attention_type}")
    print(f"Train samples: {args.num_samples}, Val samples: {args.num_samples // 10}")

    for epoch in range(args.num_epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_metrics['epoch_loss']:.4f} - "
            f"Val Loss: {val_metrics['eval_loss']:.4f} - "
            f"Time: {train_metrics['elapsed']:.1f}s"
        )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
