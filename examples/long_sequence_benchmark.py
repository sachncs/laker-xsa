#!/usr/bin/env python3
"""Long-sequence scaling benchmark: train a fresh model per attention
type and per sequence length, then report test loss, accuracy, speed,
and memory. The deprecated v1 path is selected via
``attention_type="fused"``; ``"fused_v2"`` selects the current
:class:`LakerAttention` instead.

Usage:
    python -m examples.long_sequence_benchmark --task retrieval --max-seq-len 512
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
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
    """Build reversal tasks: target is the input reversed along the time axis.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``.
    """
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    target_ids = input_ids.flip(dims=[1])
    return input_ids, target_ids


def create_retrieval_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build long-context retrieval examples with random distractor positions.

    Layout (per sample):

    * Position 0 carries the ``query_marker`` (token ``vocab_size - 1``).
    * The target sits at a random offset in ``[seq_len // 4, seq_len - 1)``;
      every other position is a random distractor.

    Targets emit the target value at the query position and zero elsewhere.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``
        and dtype ``torch.long``.
    """
    input_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)
    target_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)

    query_marker = vocab_size - 1

    for i in range(num_samples):
        # Query-position information is implicit in the marker.
        target_offset = torch.randint(seq_len // 4, seq_len - 1, (1,)).item()
        target_value = torch.randint(1, vocab_size // 2, (1,)).item()

        # Fill with random distractors across all positions.
        input_ids[i] = torch.randint(1, vocab_size - 1, (seq_len,))

        # Query marker at position 0.
        input_ids[i, 0] = query_marker

        # Place the target value at the computed offset.
        input_ids[i, target_offset] = target_value

        # Target emits the value at the query position; everything else is 0.
        target_ids[i, 0] = target_value
        target_ids[i, 1:] = 0

    return input_ids, target_ids


def create_first_last_match_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build first-last matching tasks with a binary answer at the final slot.

    For each sample, the target is ``1`` iff the value at position 0
    appears anywhere in the middle positions; ``0`` otherwise.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``
        and dtype ``torch.long``; ``target_ids`` is zero except for the
        final slot.
    """
    input_ids = torch.randint(1, vocab_size - 1, (num_samples, seq_len))
    target_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)

    for i in range(num_samples):
        first_token = input_ids[i, 0]
        # Middle positions only — the final position holds the answer.
        middle = input_ids[i, 1:-1]
        has_match = (middle == first_token).any().item()

        # Target: binary at the last position.
        target_ids[i, -1] = 1 if has_match else 0

    return input_ids, target_ids


def create_model(
    d_model: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int,
    max_seq_len: int,
    attention_type: str,
    dropout: float = 0.1,
) -> XSALAKERTransformer:
    """Construct an :class:`XSALAKERTransformer` with the requested attention.

    Uses ``XSA_LAKER_Config`` with ``kernel_type="rbf"``,
    ``xsa_mode="subtract_projection"``, ``num_iterations=10``,
    ``preconditioner_rank=d_model // 16``, and ``d_ff = 4 * d_model``.

    The benchmark loop only ever passes ``"standard"`` and ``"fused"``.
    """
    config = XSA_LAKER_Config(
        d_model=d_model,
        num_heads=num_heads,
        num_iterations=10,
        preconditioner_rank=d_model // 16,
        kernel_type="rbf",
        xsa_mode="subtract_projection",
        dropout=dropout,
    )

    model = XSALAKERTransformer(
        config,
        num_layers=num_layers,
        d_ff=d_model * 4,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout=dropout,
        attention_type=attention_type,
    )
    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Train ``model`` with AdamW (lr, weight_decay 0.01) and grad-norm clip 1.0.

    Per-token loss uses ``reduction="none"`` and is averaged to a scalar
    per batch.

    Returns:
        ``{"train_losses": [...], "val_losses": [...]}`` lists of per-epoch
        means.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss(reduction="none")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            # Mean over flattened positions — equivalent to default CE.
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss = loss.mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        model.eval()
        epoch_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, target_ids = batch
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss = loss.mean()

                epoch_val_loss += loss.item()
                num_batches += 1

        train_losses.append(epoch_train_loss / len(train_loader))
        val_losses.append(epoch_val_loss / len(val_loader))

    return {"train_losses": train_losses, "val_losses": val_losses}


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Return ``(avg_loss, accuracy)`` across the whole test loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch in test_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)

            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.mean().item()

            preds = logits.argmax(dim=-1)
            correct += (preds == target_ids).sum().item()
            total += target_ids.numel()

    return total_loss / len(test_loader), correct / total


def measure_inference_speed(
    model: XSALAKERTransformer,
    seq_len: int,
    device: torch.device,
    num_runs: int = 20,
) -> Dict[str, float]:
    """Measure forward-pass throughput for long sequence lengths.

    Methodology: batch size ``2``; three warm-up forwards precede timing;
    no :func:`torch.cuda.synchronize` gating.

    Returns:
        ``{"ms_per_sample": float, "samples_per_second": float}``.
    """
    model.eval()
    batch_size = 2  # Smaller batch for long sequences
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_ids)

    # Timing window
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_ids)
    elapsed = time.perf_counter() - start

    ms_per_batch = (elapsed / num_runs) * 1000 / batch_size
    return {
        "ms_per_sample": ms_per_batch,
        "samples_per_second": 1000 / ms_per_batch if ms_per_batch > 0 else float("inf"),
    }


def measure_memory(
    model: nn.Module,
    seq_len: int,
    device: torch.device,
) -> Dict[str, float]:
    """Estimate peak memory during one forward pass (batch size 2).

    On CUDA: :func:`torch.cuda.max_memory_allocated` in MiB.
    On CPU: ``3 * param_mb`` heuristic — coarse sanity check only.

    Returns:
        ``{"peak_memory_mb": float}``.
    """
    model.eval()
    batch_size = 2
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(input_ids)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        # CPU heuristic: parameter footprint times a small multiplier.
        param_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
        peak_mb = param_mb * 3

    return {"peak_memory_mb": peak_mb}


TASKS = {
    "copy": create_copy_task,
    "reversal": create_reversal_task,
    "retrieval": create_retrieval_task,
    "first_last": create_first_last_match_task,
}

SEQ_LENS = [128, 256, 512]  # Can add 1024 if memory allows


def run_scaling_benchmark(
    task_name: str,
    max_seq_len: int = 512,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    vocab_size: int = 100,
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train both attention types on each probed sequence length.

    Iterates over ``[s for s in SEQ_LENS if s <= max_seq_len]``.

    Raises:
        ValueError: If ``task_name`` is not in :data:`TASKS`.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    task_fn = TASKS.get(task_name)
    if task_fn is None:
        raise ValueError(f"Unknown task: {task_name}")

    # Filter SEQ_LENS to those at or below the user-selected cap.
    seq_lens = [s for s in SEQ_LENS if s <= max_seq_len]

    results: Dict[str, Any] = {
        "task": task_name,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "num_epochs": num_epochs,
        "sequence_lengths": {},
    }

    for seq_len in seq_lens:
        print(f"\n{'=' * 60}")
        print(f"Testing sequence length: {seq_len}")
        print("=" * 60)

        # Dataset sizing shrinks with longer sequences.
        num_train = max(500, 2000 - seq_len * 2)
        num_val = 200
        num_test = 200
        batch_size = max(4, 32 - seq_len // 32)

        train_input, train_target = task_fn(num_train, seq_len, vocab_size)
        val_input, val_target = task_fn(num_val, seq_len, vocab_size)
        test_input, test_target = task_fn(num_test, seq_len, vocab_size)

        train_dataset = TensorDataset(train_input, train_target)
        val_dataset = TensorDataset(val_input, val_target)
        test_dataset = TensorDataset(test_input, test_target)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        seq_results: Dict[str, Any] = {"attention_types": {}}

        for attention_type in ["standard", "fused"]:
            print(f"\nTraining {attention_type} attention...")

            model = create_model(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                vocab_size=vocab_size,
                max_seq_len=seq_len,
                attention_type=attention_type,
            )

            total_params = sum(p.numel() for p in model.parameters())

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
            )

            test_loss, test_accuracy = evaluate_model(model, test_loader, device)
            speed = measure_inference_speed(model, seq_len, device)
            memory = measure_memory(model, seq_len, device)

            seq_results["attention_types"][attention_type] = {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "final_train_loss": history["train_losses"][-1],
                "speed_metrics": speed,
                "memory_metrics": memory,
                "total_params": total_params,
            }

            print(
                f"  {attention_type.upper()}: Loss={test_loss:.4f}, "
                f"Acc={test_accuracy:.4f}, Speed={speed['samples_per_second']:.1f}/s"
            )

        # Compute comparison when both runs completed.
        std_acc = seq_results["attention_types"]["standard"]["test_accuracy"]
        fused_acc = seq_results["attention_types"]["fused"]["test_accuracy"]
        std_speed = seq_results["attention_types"]["standard"]["speed_metrics"][
            "samples_per_second"
        ]
        fused_speed = seq_results["attention_types"]["fused"]["speed_metrics"][
            "samples_per_second"
        ]

        seq_results["comparison"] = {
            "accuracy_difference": fused_acc - std_acc,
            "speed_ratio": std_speed / max(fused_speed, 0.01),
        }

        results["sequence_lengths"][seq_len] = seq_results

        print(
            f"\n  Comparison @ {seq_len}: Acc diff={fused_acc - std_acc:+.4f}, "
            f"Slowdown={std_speed/fused_speed:.2f}x"
        )

    return results


def main() -> None:
    """Parse CLI args, run the scaling benchmark, and optionally write JSON."""
    parser = argparse.ArgumentParser(description="Long sequence scaling benchmark")
    parser.add_argument(
        "--task",
        type=str,
        default="retrieval",
        choices=["copy", "reversal", "retrieval", "first_last"],
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=512, help="Maximum sequence length to test"
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("Long Sequence Scaling Benchmark")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(
        f"Model: d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}"
    )

    results = run_scaling_benchmark(
        task_name=args.task,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
