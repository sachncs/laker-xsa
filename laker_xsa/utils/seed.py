"""
Random seed utilities for reproducible training.

This module provides functions for setting random seeds and
capturing/restoring RNG states.
"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - Python random module
    - NumPy RNG
    - PyTorch CPU RNG
    - PyTorch GPU RNG (if CUDA available)

    Also enables deterministic algorithms in cuDNN (may reduce performance).

    Args:
        seed: Integer seed value.

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

        # Enable deterministic algorithms (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_states() -> Dict[str, Any]:
    """
    Capture current RNG states for reproducibility.

    Returns:
        Dictionary containing states for:
        - 'python': Python random state
        - 'numpy': NumPy RNG state
        - 'torch': PyTorch CPU RNG state
        - 'torch_cuda': PyTorch GPU RNG state (if CUDA available)

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
    """
    Restore RNG states from captured snapshot.

    Args:
        states: Dictionary from ``get_rng_states()``.

    Raises:
        KeyError: If required keys are missing from states dict.

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
