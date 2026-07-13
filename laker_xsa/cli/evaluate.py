#!/usr/bin/env python3
"""Evaluation script for LAKER-XSA models.

Loads a checkpoint and runs one inference pass on random token IDs. This is a
checkpoint/model/device smoke test, not dataset evaluation.

Usage:
    python -m laker_xsa.cli.evaluate --checkpoint path/to/checkpoint.pt

Important methodological notes:

* This is a **smoke test**, not a real evaluation. The token IDs are
  drawn uniformly at random and the printed output statistics are
  shape / mean / std only. There is no ground-truth labels dataset, no
  metric against a held-out split, and no comparison to a baseline.

* The checkpoint is loaded with ``torch.load(..., weights_only=True)``,
  which restricts deserialisation to tensors and a small allow-list of
  safe container types and refuses arbitrary pickled Python objects.
  This means the script does **not** rely on trusting arbitrary pickle
  payloads; a checkpoint carrying disallowed objects raises rather than
  executing them.

* Unknown config keys are dropped before constructing
  :class:`~laker_xsa.config.XSA_LAKER_Config`. This prevents extra keys alone
  from causing a constructor error; it does not guarantee architectural or
  state-dictionary compatibility across versions.

* ``num_layers`` defaults to ``6``. ``vocab_size`` has no usable fallback for
  this token-ID smoke path: an explicit ``None`` fails in ``torch.randint``;
  a missing key uses ``1000`` for sampling but builds an embedding-free model,
  which rejects or misinterprets the resulting 2-D token tensor.

* argparse behaviour: :func:`argparse.ArgumentParser.parse_args` is the
  source of every flag; calling it can raise :class:`SystemExit` on
  invalid input or ``--help``. The ``--checkpoint`` flag is required.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

import torch

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.model.full_model import XSALAKERTransformer


def main() -> None:
    """CLI entry point for checkpoint evaluation.

    Workflow:

        1. Parse args (``--checkpoint`` required, ``--seq-len`` /
           ``--batch-size`` / ``--cuda`` optional).
        2. Select device: ``cuda`` when ``args.cuda`` is set and a CUDA
           device is visible, otherwise ``cpu``.
        3. Load the checkpoint with ``torch.load(weights_only=True)``.
        4. Reconstruct ``XSA_LAKER_Config`` from the checkpoint's
           ``"config"`` sub-dict, filtering to known dataclass fields
           so newer config keys don't break older checkpoints.
        5. Build :class:`~laker_xsa.model.full_model.XSALAKERTransformer`
           using ``num_layers`` and ``vocab_size`` from the checkpoint
           (with safe defaults), load ``model_state_dict``, move to
           device, and call ``.eval()``.
        6. Run a single forward pass on random token IDs and print the
           input / output shapes and the first two moments of the
           output.

    Side Effects:
        Writes progress and result text to :data:`sys.stdout` via
        :func:`print`. Reads ``args.checkpoint`` from the local
        filesystem.

    Assumptions:
        The checkpoint contains configuration and state compatible with the
        current model, including a positive non-``None`` ``vocab_size`` for
        random token sampling.

    Raises:
        SystemExit: From :func:`argparse.ArgumentParser.parse_args` on
            invalid flags or ``--help``.
        FileNotFoundError: If ``args.checkpoint`` does not exist.
        KeyError: If required checkpoint keys are missing.
        TypeError: If checkpoint containers/config values have incompatible
            types or ``vocab_size`` is explicitly ``None`` during sampling.
        IndexError: If generated sequence positions exceed the reconstructed
            positional embedding table.
        RuntimeError: From checkpoint loading, state loading, tensor creation,
            device transfer, or model execution.
    """
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

    # ``args.cuda`` is a *prerequisite* for choosing CUDA, not a
    # guarantee: only use CUDA when both the flag is set and the runtime
    # actually exposes a CUDA device.
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint with ``weights_only=True`` to restrict
    # deserialisation to tensors and safe containers; arbitrary pickle
    # payloads in the checkpoint will raise.
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Drop config keys unknown to this package version. This does not establish
    # compatibility of the remaining architecture or state dictionary.
    config_dict = checkpoint["config"]
    known_fields = {f.name for f in fields(XSA_LAKER_Config)}
    filtered_config = {k: v for k, v in config_dict.items() if k in known_fields}
    config = XSA_LAKER_Config(**filtered_config)
    # ``num_layers`` (default 6) and ``vocab_size`` (default None) are
    # optional in older checkpoints.
    model = XSALAKERTransformer(
        config,
        num_layers=checkpoint.get("num_layers", 6),
        vocab_size=checkpoint.get("vocab_size"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")
    print(f"  d_model: {config.d_model}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_layers: {checkpoint.get('num_layers', 6)}")

    # Smoke-test inference on a uniformly-random token batch. The output
    # statistics (mean / std) are printed for inspection only; they are
    # not checked, so NaN/Inf would simply be printed rather than
    # detected or rejected.
    print("\nRunning inference on random input...")
    with torch.no_grad():
        x = torch.randint(
            0,
            checkpoint.get("vocab_size", 1000),
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
