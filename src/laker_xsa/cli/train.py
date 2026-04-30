#!/usr/bin/env python3
"""
Training script for LAKER-XSA models.

Usage:
    python -m laker_xsa.cli.train --config config.json

Or with defaults:
    python -m laker_xsa.cli.train
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
    """
    Create dummy training data for testing.

    Args:
        num_samples: Number of samples.
        seq_len: Sequence length.
        vocab_size: Vocabulary size.

    Returns:
        Tuple of (input_ids, labels), each shape (num_samples, seq_len).
    """
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))
    return input_ids, labels


def main() -> None:
    """Main training entry point."""
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
        choices=["standard", "xsa", "kernel", "fused"],
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

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    config = XSA_LAKER_Config(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_iterations=10,
        preconditioner_rank=args.d_model // 16,
        kernel_type="rbf",
        xsa_mode="subtract_projection",
        use_fused=args.attention_type == "fused",
    )

    # Model
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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} / {total_params:,}")

    # Data
    train_input, train_labels = create_dummy_data(
        args.num_samples, args.seq_len, args.vocab_size
    )
    train_dataset = TensorDataset(train_input, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_input, val_labels = create_dummy_data(
        args.num_samples // 10, args.seq_len, args.vocab_size
    )
    val_dataset = TensorDataset(val_input, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Training config
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=1e-3,
        warmup_steps=100,
        seed=args.seed,
    )

    # Trainer
    trainer = Trainer(model, training_config, device)

    # Training loop
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
