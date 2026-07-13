"""Long-context scaling benchmark for LAKER-XSA.

This benchmark evaluates how different attention variants scale as the
sequence length grows. It instantiates one module from each of four
implementation paths and measures a training-free accuracy/loss signal
(a stress/smoke metric; runtime and memory are measured separately).

Important methodological notes:

* The ``kernel`` and ``fused`` arms instantiate deprecated v1
  ``KernelAttentionRegression`` and ``FusedXSALAKERAttention`` rather than
  :class:`~laker_xsa.attention.laker.LakerAttention`. Results from those arms
  do not characterize the v2 implementation.

* The reported accuracy is a stress/smoke metric: every trial uses randomly
  initialized embeddings, attention weights, and a fresh untrained readout.
  It is not a measure of learned long-context quality.

* Memory/runtime are measured separately by
  :mod:`laker_xsa.benchmarks.runtime`; this module only gathers the
  accuracy/loss summary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from laker_xsa.attention import (
    ExclusiveSelfAttention,
    StandardMultiHeadAttention,
)
from laker_xsa.attention._legacy import (
    FusedXSALAKERAttention,
    KernelAttentionRegression,
)
from laker_xsa.config import XSA_LAKER_Config

logger = logging.getLogger(__name__)


def create_long_context_task(
    seq_len: int,
    d_model: int,
    batch_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a synthetic long-context proxy task.

    The task is to predict a deterministic function of all ``input_ids``
    in the sequence. Specifically, the target is
    ``(sum of token IDs along axis 1) mod d_model``, which tests the
    ability to aggregate information across the full sequence.

    Args:
        seq_len: Total sequence length (number of tokens).
        d_model: Embedding dimension. Treated as the vocabulary size for
            the synthetic classification task (``d_model`` classes).
        batch_size: Number of independent sequences in the batch.

    Returns:
        ``(input_ids, target)`` where:

        * ``input_ids`` has shape ``(batch_size, seq_len)``, dtype ``int64``.
        * ``target`` has shape ``(batch_size,)``, dtype ``int64``, with
          values in ``[0, d_model)``.

    Side Effects:
        Advances PyTorch's global CPU RNG while creating token IDs.

    Complexity:
        ``O(batch_size * seq_len)``.
    """
    # Random integer token IDs in [0, d_model).
    input_ids = torch.randint(0, d_model, (batch_size, seq_len))

    # Target: deterministic aggregate of all tokens (sum mod vocab_size).
    target = input_ids.sum(dim=1) % d_model

    return input_ids, target


def evaluate_attention_module(
    attn_module: nn.Module,
    config: XSA_LAKER_Config,
    seq_len: int,
    num_trials: int = 3,
) -> Dict[str, float]:
    """Run an untrained single-module forward pass and report toy metrics.

    The module is put in ``eval()`` mode (no dropout, no parameter update)
    and fed randomly initialised embeddings; a *fresh, untrained* linear
    readout is used to convert the last token's representation into class
    logits.

    Args:
        attn_module: Any attention module with a
            ``forward(x: (batch, seq_len, d_model)) -> (batch, seq_len, d_model)``
            signature.
        config: LAKER-XSA configuration; only ``config.d_model`` is read.
        seq_len: Sequence length for the synthetic input.
        num_trials: Number of random trials per module. Zero causes division by
            zero during aggregation; one produces ``NaN`` for the unbiased
            accuracy standard deviation.

    Returns:
        A dictionary with three scalar entries:

        * ``accuracy`` — fraction of correctly classified samples, averaged
          over trials. **This is a stress/smoke metric only** because the
          readout is fresh and untrained in every trial.
        * ``loss`` — cross-entropy of the untrained readout, averaged.
        * ``accuracy_std`` — standard deviation of per-trial accuracies.

    Raises:
        StopIteration: If ``attn_module`` has no parameters from which to infer
            a device.
        ZeroDivisionError: If ``num_trials`` is zero.
        RuntimeError: Propagated from module execution or metric computation.

    Side Effects:
        Calls ``attn_module.eval()`` and leaves it in evaluation mode. Forward
        passes run under ``torch.no_grad()``. Random inputs and readout
        initialization advance the relevant CPU/device PyTorch RNG states.

    Assumptions:
        The attention module's parameters live on a single device; the
        device of the synthetic input is set to ``next(attn_module.parameters()).device``.
    """
    # Eval mode disables dropout and any training-only behaviour, and
    # makes the run deterministic w.r.t. parameter updates.
    attn_module.eval()
    device = next(attn_module.parameters()).device

    accuracies = []
    losses = []

    for _ in range(num_trials):
        # Synthetic integer token stream. ``config.d_model`` is reused as
        # the (arbitrary) vocabulary size for this stress task.
        input_ids, target = create_long_context_task(
            seq_len, config.d_model, batch_size=4
        )
        input_ids = input_ids.to(device)
        target = target.to(device)

        # Random continuous embeddings — uniform across batch/seq/feature.
        # Using ``randn`` (not ``randint``) so the module sees a tensor it
        # can project through Q/K/V.
        x = torch.randn(input_ids.shape[0], seq_len, config.d_model, device=device)

        with torch.no_grad():
            # Get attention output.
            out = attn_module(x)

            # Pool over sequence (use last token), shape (batch, d_model).
            pooled = out[:, -1, :]

            # Simple linear readout (random projection). Constructed fresh
            # in every trial so it is **untrained**, which is why the
            # reported accuracy is a stress/smoke metric rather than a
            # quality signal.
            readout = nn.Linear(config.d_model, config.d_model, device=device)
            logits = readout(pooled)

            # Loss and accuracy against the synthetic ``sum-mod-d_model``
            # target from ``create_long_context_task``.
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
    seq_lens: Optional[List[int]] = None,
    num_trials: int = 3,
) -> Dict[str, Any]:
    """Run the full long-context scaling benchmark.

    For each sequence length in ``seq_lens`` the benchmark instantiates
    and evaluates four modules:

        * ``"standard"`` — :class:`~laker_xsa.attention.StandardMultiHeadAttention`
          (baseline dense softmax attention).
        * ``"xsa"`` — :class:`~laker_xsa.attention.ExclusiveSelfAttention`
          (Exclusive Self Attention, no kernel regression).
        * ``"kernel"`` — :class:`~laker_xsa.attention._legacy.KernelAttentionRegression`
          (deprecated v1 kernel regression with fixed Richardson iterations).
        * ``"fused"`` — :class:`~laker_xsa.attention._legacy.FusedXSALAKERAttention`
          (deprecated v1 fused XSA + LAKER path).

    The v2 :class:`~laker_xsa.attention.laker.LakerAttention` is not included.

    Args:
        d_model: Model (embedding) dimension used for every arm.
        num_heads: Number of attention heads; ``d_model`` must be
            divisible by ``num_heads``.
        seq_lens: Sequence lengths to evaluate. Defaults to
            ``[64, 128, 256, 512, 1024]`` if not provided.
        num_trials: Number of random trials per arm and sequence length. Zero
            propagates the aggregation failure documented by
            :func:`evaluate_attention_module`.

    Returns:
        A nested dictionary:

        * ``"config"`` — echo of the benchmark configuration.
        * ``"attention_types"`` — list of arm names actually evaluated.
        * ``"results"`` — ``{seq_len: {arm_name: evaluate_attention_module(...))}``.

    Side Effects:
        Instantiates four modules per sequence length on CUDA when available,
        otherwise CPU, advancing PyTorch RNG state and emitting deprecation
        warnings for v1 modules. Emits progress through the module logger;
        visibility and destination depend on the application's logging
        configuration. Each evaluated module is left in evaluation mode.

    Limitations:
        The reported accuracy is a **stress/smoke metric only** (see
        module-level docstring). Runtime and memory are not measured here;
        use :func:`laker_xsa.benchmarks.runtime.runtime_profile` for
        those.

    Complexity:
        Executes four dense-attention-style modules and one ``d_model``-class
        readout per trial. All attention arms materialize a quadratic
        ``(seq_len, seq_len)`` score or kernel matrix; legacy iterative arms
        additionally repeat dense kernel products. Autograd is disabled.
    """
    if seq_lens is None:
        seq_lens = [64, 128, 256, 512, 1024]

    # The benchmark hard-codes ``num_iterations=10`` and a preconditioner
    # rank of ``d_model // 16``; these are not exposed as CLI knobs.
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
        logger.info("Evaluating seq_len=%d", seq_len)
        results["results"][seq_len] = {}

        # Standard dense softmax attention (baseline reference).
        attn_std = StandardMultiHeadAttention(config).to(device)
        results["results"][seq_len]["standard"] = evaluate_attention_module(
            attn_std, config, seq_len, num_trials
        )
        logger.info(
            "  Standard: acc=%.3f", results["results"][seq_len]["standard"]["accuracy"]
        )

        # Exclusive Self Attention without kernel regression.
        attn_xsa = ExclusiveSelfAttention(config).to(device)
        results["results"][seq_len]["xsa"] = evaluate_attention_module(
            attn_xsa, config, seq_len, num_trials
        )
        logger.info("  XSA: acc=%.3f", results["results"][seq_len]["xsa"]["accuracy"])

        # Deprecated v1 kernel regression; this is not LakerAttention v2.
        attn_kernel = KernelAttentionRegression(config).to(device)
        results["results"][seq_len]["kernel"] = evaluate_attention_module(
            attn_kernel, config, seq_len, num_trials
        )
        logger.info(
            "  Kernel: acc=%.3f", results["results"][seq_len]["kernel"]["accuracy"]
        )

        # Deprecated v1 fused implementation; this is not LakerAttention v2.
        attn_fused = FusedXSALAKERAttention(config).to(device)
        results["results"][seq_len]["fused"] = evaluate_attention_module(
            attn_fused, config, seq_len, num_trials
        )
        logger.info(
            "  Fused: acc=%.3f", results["results"][seq_len]["fused"]["accuracy"]
        )

    return results
