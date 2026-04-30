"""
Long-context scaling benchmark for LAKER-XSA.

This benchmark evaluates how different attention mechanisms scale
with increasing sequence length.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.standard_attention import StandardMultiHeadAttention
from laker_xsa.attention.xsa_attention import ExclusiveSelfAttention
from laker_xsa.attention.kernel_attention import (
    KernelAttentionRegression,
    FusedXSALAKERAttention,
)


def create_long_context_task(
    seq_len: int,
    d_model: int,
    batch_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a proxy task for long-context evaluation.

    The task is to predict the last token based on all previous tokens.
    This tests the model's ability to aggregate information across the
    entire sequence.

    Args:
        seq_len: Total sequence length.
        d_model: Embedding dimension (used as vocab size).
        batch_size: Batch size.

    Returns:
        Tuple of (input_ids, target), where target is the last token.
    """
    # Random sequence
    input_ids = torch.randint(0, d_model, (batch_size, seq_len))

    # Target: predict a function of all tokens (e.g., sum mod vocab_size)
    target = input_ids.sum(dim=1) % d_model

    return input_ids, target


def evaluate_attention_module(
    attn_module: nn.Module,
    config: XSA_LAKER_Config,
    seq_len: int,
    num_trials: int = 3,
) -> Dict[str, float]:
    """
    Evaluate an attention module on the long-context task.

    Args:
        attn_module: Attention module to evaluate.
        config: Configuration object.
        seq_len: Sequence length.
        num_trials: Number of evaluation trials.

    Returns:
        Dictionary with accuracy and loss metrics.
    """
    attn_module.eval()
    device = next(attn_module.parameters()).device

    accuracies = []
    losses = []

    for _ in range(num_trials):
        # Create task
        input_ids, target = create_long_context_task(
            seq_len, config.d_model, batch_size=4
        )
        input_ids = input_ids.to(device)
        target = target.to(device)

        # Create simple readout
        x = torch.randn(input_ids.shape[0], seq_len, config.d_model, device=device)

        with torch.no_grad():
            # Get attention output
            out = attn_module(x)

            # Pool over sequence (use last token)
            pooled = out[:, -1, :]  # (batch, d_model)

            # Simple linear readout (random projection)
            # In a real model, this would be trained
            readout = nn.Linear(config.d_model, config.d_model, device=device)
            logits = readout(pooled)

            # Compute loss and accuracy
            loss = nn.functional.cross_entropy(logits, target)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == target).float().mean().item()

            accuracies.append(accuracy)
            losses.append(loss.item())

    return {
        "accuracy": sum(accuracies) / len(accuracies),
        "loss": sum(losses) / len(losses),
        "accuracy_std": torch.tensor(accuracies).std().item(),
    }


def long_context_benchmark(
    d_model: int = 256,
    num_heads: int = 4,
    seq_lens: List[int] = None,
    num_trials: int = 3,
) -> Dict[str, Any]:
    """
    Run long-context scaling benchmark.

    Evaluates all attention variants across multiple sequence lengths.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        seq_lens: List of sequence lengths to test.
        num_trials: Number of trials per configuration.

    Returns:
        Dictionary with benchmark results.
    """
    if seq_lens is None:
        seq_lens = [64, 128, 256, 512, 1024]

    config = XSA_LAKER_Config(
        d_model=d_model,
        num_heads=num_heads,
        num_iterations=10,
        preconditioner_rank=d_model // 16,
    )

    results: Dict[str, Any] = {
        "config": {
            "d_model": d_model,
            "num_heads": num_heads,
            "seq_lens": seq_lens,
        },
        "attention_types": ["standard", "xsa", "kernel", "fused"],
        "results": {},
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seq_len in seq_lens:
        print(f"\nEvaluating seq_len={seq_len}...")
        results["results"][seq_len] = {}

        # Standard
        attn_std = StandardMultiHeadAttention(config).to(device)
        results["results"][seq_len]["standard"] = evaluate_attention_module(
            attn_std, config, seq_len, num_trials
        )
        print(f"  Standard: acc={results['results'][seq_len]['standard']['accuracy']:.3f}")

        # XSA
        attn_xsa = ExclusiveSelfAttention(config).to(device)
        results["results"][seq_len]["xsa"] = evaluate_attention_module(
            attn_xsa, config, seq_len, num_trials
        )
        print(f"  XSA: acc={results['results'][seq_len]['xsa']['accuracy']:.3f}")

        # Kernel
        attn_kernel = KernelAttentionRegression(config).to(device)
        results["results"][seq_len]["kernel"] = evaluate_attention_module(
            attn_kernel, config, seq_len, num_trials
        )
        print(f"  Kernel: acc={results['results'][seq_len]['kernel']['accuracy']:.3f}")

        # Fused
        attn_fused = FusedXSALAKERAttention(config).to(device)
        results["results"][seq_len]["fused"] = evaluate_attention_module(
            attn_fused, config, seq_len, num_trials
        )
        print(f"  Fused: acc={results['results'][seq_len]['fused']['accuracy']:.3f}")

    return results
