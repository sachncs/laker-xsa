#!/usr/bin/env python3
"""
Evaluation script for LAKER-XSA models.

Usage:
    python -m laker_xsa.cli.evaluate --checkpoint path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse

import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.model.full_model import XSALAKERTransformer


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate LAKER-XSA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Reconstruct model from config
    config = XSA_LAKER_Config(**checkpoint["config"])
    model = XSALAKERTransformer(
        config,
        num_layers=checkpoint.get("num_layers", 6),
        vocab_size=checkpoint.get("vocab_size"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  d_model: {config.d_model}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_layers: {checkpoint.get('num_layers', 6)}")

    # Run inference on random input
    print(f"\nRunning inference on random input...")
    with torch.no_grad():
        x = torch.randint(
            0, checkpoint.get("vocab_size", 1000),
            (args.batch_size, args.seq_len),
            device=device,
        )
        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output mean: {output.mean().item():.4f}")
        print(f"  Output std: {output.std().item():.4f}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
