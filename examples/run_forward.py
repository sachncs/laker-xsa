#!/usr/bin/env python3
"""
Simple forward pass example with LAKER-XSA.

This script demonstrates basic usage of the fused XSA + LAKER attention.

Usage:
    python -m examples.run_forward
"""

from __future__ import annotations

import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.standard_attention import StandardMultiHeadAttention
from laker_xsa.attention.kernel_attention import FusedXSALAKERAttention
from laker_xsa.model.full_model import XSALAKERTransformer


def main() -> None:
    """Run forward pass examples."""
    print("=" * 60)
    print("LAKER-XSA Forward Pass Example")
    print("=" * 60)

    # Configuration
    config = XSA_LAKER_Config(
        d_model=256,
        num_heads=4,
        num_iterations=10,
        preconditioner_rank=16,
        kernel_type="rbf",
        xsa_mode="subtract_projection",
    )

    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_iterations: {config.num_iterations}")
    print(f"  preconditioner_rank: {config.preconditioner_rank}")
    print(f"  kernel_type: {config.kernel_type}")

    # Create sample input
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, config.d_model)

    print(f"\nInput shape: {x.shape}")

    # Standard attention
    print("\n--- Standard Multi-Head Attention ---")
    attn_std = StandardMultiHeadAttention(config)
    with torch.no_grad():
        out_std = attn_std(x)
    print(f"Output shape: {out_std.shape}")
    print(f"Output norm: {out_std.norm().item():.4f}")

    # Fused XSA + LAKER
    print("\n--- Fused XSA + LAKER Attention ---")
    attn_fused = FusedXSALAKERAttention(config)
    with torch.no_grad():
        out_fused = attn_fused(x)
    print(f"Output shape: {out_fused.shape}")
    print(f"Output norm: {out_fused.norm().item():.4f}")

    # Difference
    diff = (out_fused - out_std).abs()
    print(f"\nDifference (fused vs standard):")
    print(f"  Mean abs diff: {diff.mean().item():.6f}")
    print(f"  Max abs diff: {diff.max().item():.6f}")

    # Full model
    print("\n--- Full Transformer Model ---")
    vocab_size = 1000
    model = XSALAKERTransformer(
        config,
        num_layers=4,
        vocab_size=vocab_size,
        max_seq_len=128,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input IDs shape: {input_ids.shape}")

    with torch.no_grad():
        logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits norm: {logits.norm().item():.4f}")

    print("\n" + "=" * 60)
    print("Forward pass complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
