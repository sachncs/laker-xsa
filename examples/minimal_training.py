#!/usr/bin/env python3
"""Minimal training smoke test: build a small
:class:`XSALAKERTransformer` with ``attention_type="fused"``
(deprecated v1 — use ``"fused_v2"`` for the current
:class:`LakerAttention`), train on a synthetic reversal task, and
print predictions.

AdamW without explicit weight decay, cross-entropy over flattened
logits, L2-norm gradient clipping at 1.0, and ``d_ff=256`` are
hard-coded.

Usage:
    python -m examples.minimal_training
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.model.full_model import XSALAKERTransformer


def create_copy_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build copy tasks: target equals input.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``.
    """
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    target_ids = input_ids.clone()
    return input_ids, target_ids


def create_reversal_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build reversal tasks: target is the input reversed.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``.
    """
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    target_ids = input_ids.flip(dims=[1])
    return input_ids, target_ids


def main() -> None:
    """Run the minimal training smoke test end-to-end."""
    print("=" * 60)
    print("LAKER-XSA Minimal Training Example")
    print("=" * 60)

    # Configuration
    config = XSA_LAKER_Config(
        d_model=128,
        num_heads=4,
        num_iterations=10,
        preconditioner_rank=8,
        dropout=0.1,
    )

    # Hyperparameters
    vocab_size = 100
    seq_len = 32
    num_train = 500
    num_val = 100
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-3

    print(f"\nConfiguration:")
    print(f"  Task: Sequence reversal")
    print(f"  Vocabulary: {vocab_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Train samples: {num_train}")
    print(f"  Val samples: {num_val}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Create model
    model = XSALAKERTransformer(
        config,
        num_layers=4,
        d_ff=256,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout=config.dropout,
        attention_type="fused",
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Create data
    train_input, train_target = create_reversal_task(num_train, seq_len, vocab_size)
    val_input, val_target = create_reversal_task(num_val, seq_len, vocab_size)

    train_dataset = TensorDataset(train_input, train_target)
    val_dataset = TensorDataset(val_input, val_target)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            input_ids, target_ids = batch
            optimizer.zero_grad()

            logits = model(input_ids)

            # Reshape for loss
            loss = criterion(
                logits.view(-1, vocab_size),
                target_ids.view(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, target_ids = batch
                logits = model(input_ids)

                loss = criterion(
                    logits.view(-1, vocab_size),
                    target_ids.view(-1),
                )

                val_loss += loss.item()
                val_batches += 1

        avg_train = train_loss / train_batches
        avg_val = val_loss / val_batches
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train: {avg_train:.4f} - "
            f"Val: {avg_val:.4f} - "
            f"Time: {elapsed:.1f}s"
        )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Test on a few examples
    model.eval()
    print("\nSample predictions (first 5 tokens):")
    with torch.no_grad():
        test_input = val_input[:4]
        test_target = val_target[:4]
        predictions = model(test_input)
        preds = predictions.argmax(dim=-1)

        for i in range(4):
            input_seq = test_input[i, :10].tolist()
            target_seq = test_target[i, :10].tolist()
            pred_seq = preds[i, :10].tolist()

            print(f"  Input:    {input_seq}")
            print(f"  Target:   {target_seq}")
            print(f"  Predicted: {pred_seq}")
            print()


if __name__ == "__main__":
    main()
