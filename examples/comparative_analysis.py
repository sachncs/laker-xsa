#!/usr/bin/env python3
"""Comparative analysis: standard vs. fused (v1) attention on tasks
that stress long-range dependency modelling, context-only aggregation,
and numerical conditioning.

Set ``attention_type="fused_v2"`` for the current
:class:`LakerAttention`.

Usage:
    python -m examples.comparative_analysis --task copy --seq-len 64
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
from laker_xsa.attention.standard_attention import StandardMultiHeadAttention
from laker_xsa.attention.kernel_attention import FusedXSALAKERAttention

# Task definitions


def create_copy_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a copy task where the target equals the input.

    Args:
        num_samples: Number of independently sampled sequences.
        seq_len: Length of each sequence.
        vocab_size: Vocabulary size.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``
        and dtype ``torch.long``.
    """
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    target_ids = input_ids.clone()
    return input_ids, target_ids


def create_reversal_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a reversal task where the target is the reversed input.

    Args:
        num_samples: Number of independently sampled sequences.
        seq_len: Length of each sequence.
        vocab_size: Vocabulary size.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``
        and dtype ``torch.long``; ``target_ids`` is ``input_ids`` reversed
        along the time axis.
    """
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    target_ids = input_ids.flip(dims=[1])
    return input_ids, target_ids


def create_induction_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build an induction-head task that predicts the next repeat of each token.

    For position ``i`` the target is the next occurrence of
    ``input_ids[:, i]`` in ``input_ids[:, i+1:]``; if no such occurrence
    exists the input token itself is copied. The final position has no
    future context, so its target echoes the input token.

    Args:
        num_samples: Number of independently sampled sequences.
        seq_len: Length of each sequence.
        vocab_size: Vocabulary size.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``
        and dtype ``torch.long``.

    Notes:
        ``O(num_samples * seq_len ** 2)`` Python work — fine for the small
        tasks used here, not for large-scale experiments.
    """
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    target_ids = torch.zeros_like(input_ids)

    for i in range(seq_len - 1):
        for batch in range(num_samples):
            # Find next occurrence of current token.
            current_token = input_ids[batch, i]
            next_pos = torch.where(input_ids[batch, i + 1 :] == current_token)[0]
            if len(next_pos) > 0:
                target_ids[batch, i] = input_ids[batch, i + 1 + next_pos[0]]
            else:
                # Default to self when no future occurrence exists.
                target_ids[batch, i] = input_ids[batch, i]

    # Last position has no future context, so echo the input token.
    target_ids[:, -1] = input_ids[:, -1]

    return input_ids, target_ids


def create_addition_task(
    num_samples: int,
    seq_len: int,
    vocab_size: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build an addition task where the model must sum a numeric prefix.

    Layout:

    * Positions ``[0, seq_len - 2)``: integer digits sampled uniformly from
      ``[0, 10)``.
    * Position ``seq_len - 2``: separator token ``vocab_size - 1``.
    * Position ``seq_len - 1``: reserved output slot (zero on the input
      side, sum modulo ``vocab_size`` on the target side).

    Args:
        num_samples: Number of independently sampled sequences.
        seq_len: Total sequence length; must be at least 3.
        vocab_size: Vocabulary size. Defaults to 100. The final digit
            target is reduced modulo ``vocab_size``.

    Returns:
        ``(input_ids, target_ids)`` of shape ``(num_samples, seq_len)``
        and dtype ``torch.long``.
    """
    # Generate random digits 0-9 in the payload slots.
    numbers = torch.randint(0, 10, (num_samples, seq_len - 2))

    # Separator marker at position ``seq_len - 2``; final slot held at zero.
    sep_token = vocab_size - 1
    input_ids = torch.cat(
        [
            numbers,
            torch.full((num_samples, 1), sep_token),
            torch.zeros((num_samples, 1), dtype=torch.long),
        ],
        dim=1,
    )

    # Target: sum modulo vocab_size placed at the final position.
    target_ids = torch.zeros_like(input_ids)
    sums = numbers.sum(dim=1)
    target_ids[:, seq_len - 1] = sums % vocab_size

    return input_ids, target_ids


# Model training


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
    knob), ``preconditioner_rank=d_model // 16``, and ``d_ff = 4 * d_model``.

    Args:
        d_model: Embedding / hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of stacked Transformer blocks.
        vocab_size: Token vocabulary size.
        max_seq_len: Maximum sequence length for the positional embedding.
        attention_type: Forwarded to :class:`XSALAKERTransformer`.
        dropout: Dropout probability.
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
    """Train ``model`` with AdamW (weight_decay=0.01) + L2-norm grad clip 1.0.

    Returns:
        ``{"train_losses": [...], "val_losses": [...]}`` lists of per-epoch
        mean cross-entropy values.

    Notes:
        Loss is :class:`nn.CrossEntropyLoss` on flattened
        ``(batch * seq_len, vocab)`` logits against flattened targets; no
        label smoothing or PAD-mask handling is applied.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
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

        # Validation phase
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
) -> Tuple[float, float]:
    """Return ``(avg_loss, accuracy)`` across the whole test loader.

    Accuracy is computed across every position (no PAD-mask awareness).
    The model is placed in ``eval()`` mode for the call and left there.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)

            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == target_ids).sum().item()
            total += target_ids.numel()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return avg_loss, accuracy


# Analysis functions


def measure_conditioning(
    model: XSALAKERTransformer,
    seq_len: int,
    device: torch.device,
) -> Dict[str, float]:
    """Trace-vs-diagonal conditioning proxy on the first block's kernel.

    Reads ``blocks[0].attention.w_q`` / ``w_k`` / ``kernel_fn`` (present
    only when ``attention_type`` exposes them — i.e. ``"fused"``); returns
    ``{"condition_estimate": inf}`` otherwise.

    Returns:
        ``{"trace_norm": float, "diag_sum": float, "condition_estimate": float}``
        on success; ``{"condition_estimate": float("inf")}`` if the
        attention has no ``kernel_fn`` attribute.

    Notes:
        This is a *trace proxy*, not the true ``sigma_max / sigma_min``
        condition number. For rigorous analysis use
        :func:`laker_xsa.benchmarks.conditioning.compute_conditioning_metrics`.
    """
    model.eval()
    config = model.config

    # Random probe input: shape (1, seq_len, d_model) on the target device.
    x = torch.randn(1, seq_len, config.d_model, device=device)

    # Pull the kernel from the first layer's attention module (v1 path).
    with torch.no_grad():
        block = model.blocks[0]
        if hasattr(block.attention, "kernel_fn"):
            q = block.attention.w_q(x)
            k = block.attention.w_k(x)
            q = q.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
            k = k.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)

            kernel = block.attention.kernel_fn(q, k)

            # Reduce to a single head for the proxy.
            kernel_flat = kernel[0, 0]
            trace = kernel_flat.abs().sum()
            diag_sum = torch.diagonal(kernel_flat).abs().sum()

            # Off-diagonal-heavy kernels yield larger trace/diag ratios.
            condition_estimate = trace / (diag_sum + 1e-6)

            return {
                "trace_norm": trace.item(),
                "diag_sum": diag_sum.item(),
                "condition_estimate": condition_estimate.item(),
            }

    return {"condition_estimate": float("inf")}


def measure_gradient_norm(
    model: XSALAKERTransformer,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Forward + ``logits.sum().backward()``; report gradient magnitudes.

    A fresh AdamW (lr 1e-3) is constructed; parameters are not stepped.
    Per-parameter grads default to ``0.0`` if the parameter lacked a
    gradient (e.g. unused embedding rows).

    Returns:
        ``{"total_gradient_norm", "embedding_grad", "output_proj_grad"}``.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # Accumulate per-parameter L2 norms and a global L2.
    grad_norms = {}
    total_norm = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm().item()
            grad_norms[name] = param_norm
            total_norm += param_norm**2

    total_norm = total_norm**0.5

    return {
        "total_gradient_norm": total_norm,
        "embedding_grad": grad_norms.get("token_embedding.weight", 0.0),
        "output_proj_grad": grad_norms.get("output_proj.weight", 0.0),
    }


def measure_inference_speed(
    model: XSALAKERTransformer,
    seq_len: int,
    device: torch.device,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Measure forward-pass throughput under ``eval()`` mode.

    Methodology: batch size ``4``; five warm-up forwards precede
    :func:`time.perf_counter`-gated timing; CUDA devices additionally
    call :func:`torch.cuda.synchronize` around the timed window.

    Returns:
        ``{"ms_per_sample": float, "samples_per_second": float}``.
    """
    model.eval()
    batch_size = 4
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids)

    # Timed window with optional CUDA sync.
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_ids)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    ms_per_batch = (elapsed / num_runs) * 1000 / batch_size

    return {
        "ms_per_sample": ms_per_batch,
        "samples_per_second": 1000 / ms_per_batch if ms_per_batch > 0 else float("inf"),
    }


# Main analysis


TASKS = {
    "copy": create_copy_task,
    "reversal": create_reversal_task,
    "induction": create_induction_task,
    "addition": create_addition_task,
}

ATTENTION_TYPES = ["standard", "fused"]


def run_comparative_analysis(
    task_name: str,
    seq_len: int,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    vocab_size: int = 100,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train both attention types on ``task_name`` and report a comparison.

    Dataset sizes are hard-coded (1000 / 200 / 200 train / val / test,
    batch size 16).

    Raises:
        ValueError: If ``task_name`` is not in :data:`TASKS`.
    """
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create dataset
    num_train = 1000
    num_val = 200
    num_test = 200
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

        # Create model
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

        # Evaluate
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)

        # Measure conditioning
        conditioning = measure_conditioning(model, seq_len, device)

        # Measure gradient norms
        grad_metrics = measure_gradient_norm(model, test_input.to(device), device)

        # Measure inference speed
        speed_metrics = measure_inference_speed(model, seq_len, device)

        results["attention_types"][attention_type] = {
            "final_train_loss": history["train_losses"][-1],
            "final_val_loss": history["val_losses"][-1],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "conditioning": conditioning,
            "gradient_metrics": grad_metrics,
            "speed_metrics": speed_metrics,
            "total_params": total_params,
        }

        print(f"\n{attention_type.upper()} Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(
            f"  Condition Estimate: {conditioning.get('condition_estimate', 'N/A'):.4f}"
        )
        print(f"  Total Gradient Norm: {grad_metrics['total_gradient_norm']:.4f}")
        print(
            f"  Inference Speed: {speed_metrics['samples_per_second']:.1f} samples/sec"
        )

    # Compute improvement metrics when both runs completed.
    if (
        "standard" in results["attention_types"]
        and "fused" in results["attention_types"]
    ):
        std_acc = results["attention_types"]["standard"]["test_accuracy"]
        fused_acc = results["attention_types"]["fused"]["test_accuracy"]

        std_loss = results["attention_types"]["standard"]["test_loss"]
        fused_loss = results["attention_types"]["fused"]["test_loss"]

        std_speed = results["attention_types"]["standard"]["speed_metrics"][
            "samples_per_second"
        ]
        fused_speed = results["attention_types"]["fused"]["speed_metrics"][
            "samples_per_second"
        ]

        results["comparison"] = {
            "accuracy_improvement": fused_acc - std_acc,
            "accuracy_improvement_pct": ((fused_acc - std_acc) / max(std_acc, 0.01))
            * 100,
            "loss_reduction": std_loss - fused_loss,
            "loss_reduction_pct": ((std_loss - fused_loss) / max(std_loss, 0.01)) * 100,
            "speed_slowdown": std_speed / max(fused_speed, 0.01),
        }

        print(f"\n{'=' * 60}")
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(
            f"Accuracy Improvement: {results['comparison']['accuracy_improvement']:.4f} "
            f"({results['comparison']['accuracy_improvement_pct']:.1f}%)"
        )
        print(
            f"Loss Reduction: {results['comparison']['loss_reduction']:.4f} "
            f"({results['comparison']['loss_reduction_pct']:.1f}%)"
        )
        print(f"Speed Slowdown: {results['comparison']['speed_slowdown']:.2f}x")

    return results


def main() -> None:
    """Parse CLI args, run the analysis, and optionally write JSON output."""
    parser = argparse.ArgumentParser(
        description="Comparative analysis: XSA+LAKER vs Standard Transformer"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="reversal",
        choices=["copy", "reversal", "induction", "addition"],
        help="Task to evaluate on",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
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
        default=20,
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
    print("LAKER-XSA Comparative Analysis")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Sequence Length: {args.seq_len}")
    print(
        f"Model: d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}"
    )
    print(f"Training: {args.epochs} epochs, lr={args.lr}")

    results = run_comparative_analysis(
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
        # Strip the per-epoch loss lists to keep the JSON output compact.
        output_results = results.copy()
        for attn_type in output_results["attention_types"]:
            output_results["attention_types"][attn_type]["train_losses"] = []
            output_results["attention_types"][attn_type]["val_losses"] = []

        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
