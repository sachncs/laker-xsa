#!/usr/bin/env python3
"""Hard benchmark: train both attention types on synthetic high-signal
tasks (``retrieval``, ``multihop``, ``noisy_copy``, ``binding``) chosen
to be difficult enough that neither model achieves 100% accuracy.

The deprecated v1 path is selected via ``attention_type="fused"``
(:class:`FusedXSALAKERAttention`); ``"fused_v2"`` selects the current
:class:`LakerAttention` instead.

Usage:
    python -m examples.hard_benchmark --task retrieval --seq-len 64
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

# Hard task definitions


def create_retrieval_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build long-context retrieval examples with random distractor positions.

    Layout (per sample):

    * Position 0 holds the ``query_marker`` (token ``vocab_size - 1``).
    * Position ``target_pos`` (chosen from ``[seq_len // 2, seq_len - 1)``)
      carries the target value; every other non-query position is a random
      distractor token from ``[1, vocab_size - 2)``.

    Targets emit the target value at the query position and a copy of the
    input elsewhere — this is CE-supervised on the entire sequence; the
    query-position accuracy is the metric of interest.

    Args:
        num_samples: Number of independently sampled examples.
        seq_len: Total sequence length including the query marker.
        vocab_size: Vocabulary size; reserves ``vocab_size - 1`` as the
            query marker.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``
        and dtype ``torch.long``.
    """
    input_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)
    target_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)

    # Token reservations: 0=padding, vocab-1=query marker, vocab-2=target marker.
    query_marker = vocab_size - 1
    target_marker = vocab_size - 2

    for i in range(num_samples):
        # Random target position in the second half.
        target_pos = torch.randint(seq_len // 2, seq_len - 1, (1,)).item()

        # Query marker at position 0.
        input_ids[i, 0] = query_marker

        # Fill all other positions with random distractors.
        distractors = torch.randint(1, vocab_size - 2, (seq_len - 1,))

        # Place target value at the chosen position.
        target_value = torch.randint(1, vocab_size // 2, (1,)).item()
        distractors[target_pos - 1] = target_value

        input_ids[i, 1:] = distractors

        # Target: target_value at the query position; copy rest for CE training.
        target_ids[i, 0] = target_value
        target_ids[i, 1:] = input_ids[i, 1:]

    return input_ids, target_ids


def create_multihop_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build chain-of-pointers examples that require multi-hop reasoning.

    A chain of length 3–5 selects random positions ``chain_positions``.
    Each chain position ``j`` (except the last) holds a pointer
    ``max_value + chain_positions[j + 1]``; the last chain position holds
    the final value. Position 0 is the query marker pointing at the
    first chain position.

    Args:
        num_samples: Number of independently sampled examples.
        seq_len: Total sequence length.
        vocab_size: Vocabulary size; the upper half
            ``[vocab_size // 2, vocab_size)`` is reserved for pointers.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``
        and dtype ``torch.long``.
    """
    input_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)
    target_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)

    # Reserve the upper half of the vocab for pointer markers.
    max_value = vocab_size // 2

    for i in range(num_samples):
        # Random chain length of 3-5, clipped to fit the sequence.
        chain_length = torch.randint(3, min(6, seq_len // 2)).item()
        chain_positions = torch.randperm(seq_len - 1)[:chain_length].sort().values + 1

        # Random values for each chain node (last node carries the "answer").
        values = torch.randint(1, max_value, (chain_length + 1,))

        for j, pos in enumerate(chain_positions):
            if j < chain_length - 1:
                # Intermediate nodes hold a pointer to the next chain node.
                input_ids[i, pos] = max_value + chain_positions[j + 1]
            else:
                # The terminal node stores the final value.
                input_ids[i, pos] = values[j]

        # Position 0 is the query marker pointing at the first chain node.
        input_ids[i, 0] = max_value + chain_positions[0]

        # Target: the final chain value at the query position; copy elsewhere.
        target_ids[i, 0] = values[-1]
        target_ids[i, 1:] = input_ids[i, 1:]

    return input_ids, target_ids


def create_noisy_copy_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    noise_ratio: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build copy tasks where a fraction of positions are masked with noise.

    The targets remain the original (pre-corruption) token at every
    position, so the model must reconstruct each token from context when
    its own position has been overwritten with the noise marker
    ``vocab_size - 1``.

    Args:
        num_samples: Number of independently sampled examples.
        seq_len: Sequence length.
        vocab_size: Vocabulary size; ``vocab_size - 1`` is the noise marker.
        noise_ratio: Fraction of positions to overwrite; defaults to ``0.3``.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``
        and dtype ``torch.long``.
    """
    input_ids = torch.randint(1, vocab_size - 1, (num_samples, seq_len))
    target_ids = input_ids.clone()

    noise_token = vocab_size - 1

    for i in range(num_samples):
        # Random subset of positions to corrupt with the noise token.
        num_noisy = int(seq_len * noise_ratio)
        noisy_positions = torch.randperm(seq_len)[:num_noisy]

        # Overwrite the chosen positions with the noise marker.
        input_ids[i, noisy_positions] = noise_token

        # Targets remain the original (pre-noise) values at those positions.

    return input_ids, target_ids


def create_binding_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    num_bindings: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build key-value binding examples followed by queries.

    Layout (per sample):

    * Positions ``[0, 2 * num_bindings)`` hold interleaved ``(key, value)``
      pairs.
    * Remaining positions are filled with ``(query_marker, query_key)``
      pairs whose target is the value associated with that key.

    Args:
        num_samples: Number of independently sampled examples.
        seq_len: Total sequence length. Must accommodate both the
            ``num_bindings`` pairs and at least one query.
        vocab_size: Vocabulary size.
        num_bindings: Number of (key, value) pairs at the top of the
            sequence. Defaults to 4.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``
        and dtype ``torch.long``.

    Notes:
        With the default ``num_bindings=4``, key IDs ``[2, 6)`` and value IDs
        ``[10, 14)`` are disjoint. Larger counts can overlap, and callers must
        choose a vocabulary large enough for every generated ID.
    """
    input_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)
    target_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)

    # Reserve tokens: 0=pad, 1=query_marker, vocab_size-1=value_marker
    query_marker = 1
    value_marker = vocab_size - 1

    keys = list(range(2, 2 + num_bindings))  # Key tokens
    values = list(range(10, 10 + num_bindings))  # Value tokens

    for i in range(num_samples):
        pos = 0
        bindings = {}

        # Lay down the key-value pairs first.
        for k, v in zip(keys, values):
            if pos + 1 < seq_len - 2:
                input_ids[i, pos] = k
                input_ids[i, pos + 1] = v
                bindings[k] = v
                pos += 2

        # Fill remaining slots with queries.
        while pos + 1 < seq_len:
            query_key = keys[torch.randint(0, len(keys), (1,)).item()]
            input_ids[i, pos] = query_marker
            input_ids[i, pos + 1] = query_key

            # Target is the looked-up value at the query-key position.
            target_ids[i, pos + 1] = bindings[query_key]
            target_ids[i, :pos] = input_ids[i, :pos]
            pos += 2

    return input_ids, target_ids


# Model and training


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
    ``xsa_mode="subtract_projection"``, ``num_iterations=10`` (v1 path
    knobs), ``preconditioner_rank=d_model // 16``, and ``d_ff = 4 * d_model``.

    Args:
        d_model: Embedding / hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of stacked Transformer blocks.
        vocab_size: Token vocabulary size.
        max_seq_len: Positional-embedding length.
        attention_type: Forwarded to :class:`XSALAKERTransformer`;
            see that class for the accepted literals.
        dropout: Dropout probability.

    Returns:
        A newly initialized ``XSALAKERTransformer`` on CPU.
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

    Returns:
        ``{"train_losses": [...], "val_losses": [...]}`` lists of per-epoch
        mean cross-entropy losses.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
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

            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
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

                epoch_val_loss += loss.item()
                num_batches += 1

        train_losses.append(epoch_train_loss / len(train_loader))
        val_losses.append(epoch_val_loss / len(val_loader))

    return {"train_losses": train_losses, "val_losses": val_losses}


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    query_positions: torch.Tensor = None,
) -> Tuple[float, float, float]:
    """Evaluate ``model`` and return loss, overall accuracy, query-only accuracy.

    Query-position scoring uses ``query_positions`` sliced by
    ``batch_idx * test_loader.batch_size``; ``None`` skips query-only
    scoring (which then defaults to ``0``).

    Returns:
        ``(avg_loss, accuracy, query_accuracy)``.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    query_correct = 0
    query_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)

            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)

            # Overall accuracy across every position.
            correct += (preds == target_ids).sum().item()
            total += target_ids.numel()

            # Query-only accuracy if positions were supplied.
            if query_positions is not None:
                batch_query_positions = query_positions[
                    batch_idx
                    * test_loader.batch_size : (batch_idx + 1)
                    * test_loader.batch_size
                ]
                for b in range(input_ids.size(0)):
                    for qpos in batch_query_positions[b]:
                        if qpos >= 0:
                            query_total += 1
                            if preds[b, qpos] == target_ids[b, qpos]:
                                query_correct += 1

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    query_accuracy = query_correct / max(query_total, 1)

    return avg_loss, accuracy, query_accuracy


def measure_conditioning(
    model: XSALAKERTransformer,
    seq_len: int,
    device: torch.device,
) -> Dict[str, float]:
    """Compute the trace-vs-diagonal conditioning proxy on the first block.

    Reads ``blocks[0].attention.w_q`` / ``w_k`` / ``kernel_fn`` when the
    attention module exposes them (i.e. ``attention_type="fused"``); returns
    ``{"condition_estimate": inf}`` otherwise.

    Returns:
        ``{"trace_norm": float, "diag_sum": float, "condition_estimate": float}``
        on success; ``{"condition_estimate": float("inf")}`` if the
        attention has no ``kernel_fn`` attribute.

    Notes:
        This is a *trace proxy*, not the true condition number
        ``sigma_max / sigma_min``. For rigorous conditioning analysis use
        :func:`laker_xsa.benchmarks.conditioning.compute_conditioning_metrics`.
    """
    model.eval()
    config = model.config

    x = torch.randn(1, seq_len, config.d_model, device=device)

    with torch.no_grad():
        block = model.blocks[0]
        if hasattr(block.attention, "kernel_fn"):
            q = block.attention.w_q(x)
            k = block.attention.w_k(x)
            q = q.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
            k = k.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)

            kernel = block.attention.kernel_fn(q, k)
            kernel_flat = kernel[0, 0]

            trace = kernel_flat.abs().sum()
            diag_sum = torch.diagonal(kernel_flat).abs().sum()
            condition_estimate = trace / (diag_sum + 1e-6)

            return {
                "trace_norm": trace.item(),
                "diag_sum": diag_sum.item(),
                "condition_estimate": condition_estimate.item(),
            }

    return {"condition_estimate": float("inf")}


def measure_inference_speed(
    model: XSALAKERTransformer,
    seq_len: int,
    device: torch.device,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Measure forward-pass throughput under ``eval()`` mode.

    Methodology: batch size fixed at ``4``; five warm-up forwards precede
    the timed window; timing uses :func:`time.perf_counter` only (no
    :func:`torch.cuda.synchronize` is called).

    Args:
        model: ``XSALAKERTransformer`` already on ``device``.
        seq_len: Length of the synthetic random input.
        device: Target :class:`torch.device`.
        num_runs: Number of timed forward passes.

    Returns:
        ``{"ms_per_sample": float, "samples_per_second": float}``.
    """
    model.eval()
    batch_size = 4
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids)

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


# Main benchmark


TASKS = {
    "retrieval": create_retrieval_task,
    "multihop": create_multihop_task,
    "noisy_copy": create_noisy_copy_task,
    "binding": create_binding_task,
}

ATTENTION_TYPES = ["standard", "fused"]


def run_benchmark(
    task_name: str,
    seq_len: int,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    vocab_size: int = 100,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train both attention types on ``task_name`` and report a comparison.

    Iterates over ``ATTENTION_TYPES = ["standard", "fused"]`` with the same
    data splits and seed. Dataset sizes are hard-coded (2000 / 400 / 400
    train / val / test, batch size 16).

    Raises:
        ValueError: If ``task_name`` is not in :data:`TASKS`.
    """
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    num_train = 2000
    num_val = 400
    num_test = 400
    batch_size = 16

    task_fn = TASKS.get(task_name)
    if task_fn is None:
        raise ValueError(f"Unknown task: {task_name}")

    print(f"\nCreating {task_name} task dataset...")
    train_input, train_target = task_fn(num_train, seq_len, vocab_size)
    val_input, val_target = task_fn(num_val, seq_len, vocab_size)
    test_input, test_target = task_fn(num_test, seq_len, vocab_size)

    train_dataset = TensorDataset(train_input, train_target)
    val_dataset = TensorDataset(val_input, val_target)
    test_dataset = TensorDataset(test_input, test_target)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    results: Dict[str, Any] = {
        "task": task_name,
        "seq_len": seq_len,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "num_epochs": num_epochs,
        "attention_types": {},
    }

    for attention_type in ATTENTION_TYPES:
        print(f"\n{'=' * 60}")
        print(f"Training {attention_type} attention model...")
        print("=" * 60)

        model = create_model(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            attention_type=attention_type,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )

        test_loss, test_accuracy, query_accuracy = evaluate_model(
            model, test_loader, device
        )

        conditioning = measure_conditioning(model, seq_len, device)
        speed_metrics = measure_inference_speed(model, seq_len, device)

        results["attention_types"][attention_type] = {
            "final_train_loss": history["train_losses"][-1],
            "final_val_loss": history["val_losses"][-1],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "query_accuracy": query_accuracy,
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "conditioning": conditioning,
            "speed_metrics": speed_metrics,
            "total_params": total_params,
        }

        print(f"\n{attention_type.upper()} Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Query Accuracy: {query_accuracy:.4f}")
        print(
            f"  Condition Estimate: {conditioning.get('condition_estimate', 'N/A'):.4f}"
        )
        print(
            f"  Inference Speed: {speed_metrics['samples_per_second']:.1f} samples/sec"
        )

    # Compute comparison metrics when both runs completed.
    if (
        "standard" in results["attention_types"]
        and "fused" in results["attention_types"]
    ):
        std_acc = results["attention_types"]["standard"]["query_accuracy"]
        fused_acc = results["attention_types"]["fused"]["query_accuracy"]

        std_loss = results["attention_types"]["standard"]["test_loss"]
        fused_loss = results["attention_types"]["fused"]["test_loss"]

        std_speed = results["attention_types"]["standard"]["speed_metrics"][
            "samples_per_second"
        ]
        fused_speed = results["attention_types"]["fused"]["speed_metrics"][
            "samples_per_second"
        ]

        results["comparison"] = {
            "query_accuracy_improvement": fused_acc - std_acc,
            "query_accuracy_improvement_pct": (
                (fused_acc - std_acc) / max(std_acc, 0.01)
            )
            * 100,
            "loss_reduction": std_loss - fused_loss,
            "loss_reduction_pct": ((std_loss - fused_loss) / max(std_loss, 0.01)) * 100,
            "speed_slowdown": std_speed / max(fused_speed, 0.01),
        }

        print(f"\n{'=' * 60}")
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(
            f"Query Accuracy Improvement: {results['comparison']['query_accuracy_improvement']:.4f} "
            f"({results['comparison']['query_accuracy_improvement_pct']:.1f}%)"
        )
        print(
            f"Loss Reduction: {results['comparison']['loss_reduction']:.4f} "
            f"({results['comparison']['loss_reduction_pct']:.1f}%)"
        )
        print(f"Speed Slowdown: {results['comparison']['speed_slowdown']:.2f}x")

    return results


def main() -> None:
    """Parse CLI args and run the hard benchmark; optionally write JSON."""
    parser = argparse.ArgumentParser(
        description="Hard benchmark: XSA+LAKER vs Standard Transformer"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="binding",
        choices=["retrieval", "multihop", "noisy_copy", "binding"],
        help="Task to evaluate on",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
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
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LAKER-XSA Hard Benchmark")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Sequence Length: {args.seq_len}")
    print(
        f"Model: d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}"
    )
    print(f"Training: {args.epochs} epochs, lr={args.lr}")

    results = run_benchmark(
        task_name=args.task,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )

    if args.output:
        output_results = results.copy()
        for attn_type in output_results["attention_types"]:
            output_results["attention_types"][attn_type]["train_losses"] = []
            output_results["attention_types"][attn_type]["val_losses"] = []

        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
