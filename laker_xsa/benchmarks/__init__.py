"""Benchmarking utilities for LAKER-XSA.

This package provides tools for profiling and evaluating LAKER-XSA
attention variants:

* :func:`long_context_benchmark`  — long-context scaling benchmark.
* :func:`compute_conditioning_metrics` — kernel-conditioning analysis.
* :func:`runtime_profile` / :func:`profile_iterations` — runtime and
  per-iteration convergence profiling.

Several benchmark paths still instantiate deprecated v1 attention classes.
Their results do not characterize :class:`~laker_xsa.attention.laker.LakerAttention`
v2. ``profile_iterations`` is available from :mod:`laker_xsa.benchmarks.runtime`
but is not re-exported here.
"""

from __future__ import annotations

from laker_xsa.benchmarks.conditioning import compute_conditioning_metrics
from laker_xsa.benchmarks.long_context import long_context_benchmark
from laker_xsa.benchmarks.runtime import runtime_profile

__all__ = [
    "long_context_benchmark",
    "compute_conditioning_metrics",
    "runtime_profile",
]
