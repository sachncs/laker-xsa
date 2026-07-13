"""Test suite for the LAKER-XSA package.

This package collects the pytest modules that verify shape, gradient,
numerical stability, configuration, CLI, and benchmark behaviour for the
public API.

The modules that import deprecated v1 symbols (``KernelAttentionRegression``,
``FusedXSALAKERAttention``, ``LearnedPreconditioner``, ``KernelFunction``)
declare ``pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")``
so the deprecation warnings stay out of test logs.
"""
