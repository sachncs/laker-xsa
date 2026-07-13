"""Global random-number-generator seeding and state snapshots.

The helpers cover Python, NumPy, PyTorch CPU, and visible CUDA generators.
``set_seed`` also changes cuDNN determinism settings when CUDA is available.
Reproducibility still depends on the hardware, software versions, algorithms,
operation order, and any random generators not managed here.
"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    This is a one-shot convenience for the common "set everything
    to the same seed" workflow. It does not save or restore any
    previous state - use :func:`get_rng_states` /
    :func:`set_rng_states` for that.

    When CUDA is available, the function additionally enables
    deterministic cuDNN kernels and disables cuDNN's
    autotuner:

    .. code-block:: python

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    These settings reduce nondeterminism in cuDNN operations but do not by
    themselves guarantee bit-exact execution. Deterministic kernels can be
    slower than autotuned alternatives. Callers should evaluate this trade-off
    for their workload.

    Side Effects:
        * Seeds Python's :mod:`random` module.
        * Seeds NumPy's global RNG.
        * Seeds PyTorch's CPU RNG.
        * Seeds every visible CUDA device's RNG (when CUDA is
          available).
        * On CUDA, sets ``torch.backends.cudnn.deterministic =
          True`` and ``torch.backends.cudnn.benchmark = False``.

    Args:
        seed: Integer accepted by the underlying RNG APIs. Repeating a seed
            resets those global generators; identical downstream values still
            depend on using the same operation order and environment.

    Example:
        >>> set_seed(42)
        >>> x = torch.randn(10)  # Reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_states() -> Dict[str, Any]:
    """Snapshot the current RNG state for all supported libraries.

    Captures Python, NumPy, and PyTorch CPU state, plus all visible CUDA RNG
    states when CUDA is available. A caller can store the pickleable dictionary
    with a checkpoint and restore it before resuming the same sequence of random
    operations.

    Keys:

    * ``"python"`` - output of :func:`random.getstate`.
    * ``"numpy"`` - output of :func:`numpy.random.get_state`.
    * ``"torch"`` - output of :func:`torch.get_rng_state`.
    * ``"torch_cuda"`` - output of
      :func:`torch.cuda.get_rng_state_all` (only present when
      CUDA is available).

    Returns:
        Dictionary containing the opaque state objects returned by each
        library. ``"torch_cuda"`` is omitted when CUDA is unavailable.

    Example:
        >>> state = get_rng_states()
        >>> # ... some operations ...
        >>> set_rng_states(state)  # Restore state
    """
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()

    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restore RNG states from a snapshot produced by :func:`get_rng_states`.

    Applies the snapshot in library order: Python first, NumPy
    second, PyTorch CPU third, and PyTorch CUDA last (if the
    snapshot contains a CUDA state and CUDA is currently
    available). The CUDA state is applied with
    :func:`torch.cuda.set_rng_state_all`, which fans out to
    every visible device.

    The function does not verify that the snapshot matches the
    current device layout: a snapshot taken on 4 GPUs will be
    applied verbatim to whatever GPUs are visible at the time
    of the call. On a different topology this can still fail to
    reproduce a run exactly; the snapshot/restore contract
    assumes the same hardware and driver stack.

    Args:
        states: Snapshot dictionary produced by
            :func:`get_rng_states`. Must contain ``"python"``,
            ``"numpy"``, and ``"torch"``. The ``"torch_cuda"``
            key is optional and only used when CUDA is
            available.

    Raises:
        KeyError: If any of the required keys (``"python"``,
            ``"numpy"``, ``"torch"``) is missing from
            ``states``.

    Example:
        >>> state = get_rng_states()
        >>> # ... some operations ...
        >>> set_rng_states(state)  # Back to original state
    """
    if "python" not in states:
        raise KeyError("Missing 'python' key in states")
    if "numpy" not in states:
        raise KeyError("Missing 'numpy' key in states")
    if "torch" not in states:
        raise KeyError("Missing 'torch' key in states")

    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])

    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
