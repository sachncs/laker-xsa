# Frequently Asked Questions

## General

### What is LAKER-XSA?

LAKER-XSA is a production-grade implementation of two attention mechanisms for Transformer models:

1. **Exclusive Self Attention (XSA)**: Removes self-aligned components from attention output, forcing each token to aggregate only from other tokens.
2. **LAKER Kernel Attention**: Treats attention as kernel ridge regression with a learned preconditioner, solved via Preconditioned Conjugate Gradient.

The library fuses these two mechanisms into a single attention module (`LakerAttention`) that addresses both self-bias and spectral collapse in standard attention.

### Is this a research prototype or production-ready?

It is production-quality code with full type hints, comprehensive tests (269 tests, 88% coverage), CI passing (pylint, mypy, pytest), and a clean modular API. However, it has computational overhead (8-10x slower than standard attention) and is best suited for research or settings where the accuracy benefits justify the cost.

### What papers does this implement?

- **XSA**: [arXiv:2603.09078](https://arxiv.org/abs/2603.09078) - Exclusive Self Attention
- **LAKER**: [arXiv:2604.25138](https://arxiv.org/html/2604.25138v1) - Learned Preconditioning for Attention Kernel Regression

The v2 fusion (`LakerAttention`) is a novel unpublished combination of both.

## Installation

### What Python versions are supported?

Python 3.9, 3.10, 3.11, and 3.12.

### Do I need a GPU?

No. LAKER-XSA runs on CPU, but a CUDA-capable GPU is recommended for reasonable performance on sequences longer than a few hundred tokens.

### How do I install development dependencies?

```bash
pip install -e ".[dev]"
```

This installs pytest, pytest-cov, pylint, black, and mypy.

## Usage

### Which attention type should I use?

For most use cases, use `fused_v2` (`LakerAttention`). It combines both XSA and LAKER for the strongest theoretical grounding. Use `standard` for baseline comparisons.

### How do I configure the attention?

Use the `XSA_LAKER_Config` dataclass:

```python
from laker_xsa import XSA_LAKER_Config

config = XSA_LAKER_Config(
    d_model=512,           # Embedding dimension
    num_heads=8,           # Number of attention heads
    dropout=0.1,           # Dropout rate
    lambda_init=3.0,       # Regularization for kernel system
    kernel_type="exp_attention",  # Kernel function
    xsa_mode="subtract_projection",  # XSA strategy
    preconditioner_type="fast",  # Preconditioner mode
    pcg_max_iterations=20,  # Max solver iterations
    pcg_tolerance=1e-2,    # Solver convergence tolerance
)
```

See the [API Reference](../README.md#api-reference) for all options.

### Why is it slower than standard attention?

LAKER-XSA computes a full kernel matrix O(n^2) and solves an iterative system per attention call. The trade-off is theoretically stronger attention at the cost of ~8-10x compute overhead. See [Limitations](limitations.md) for details.

### Can I use it with Hugging Face models?

Not directly. LAKER-XSA provides its own Transformer implementation (`XSALAKERTransformer`). To use it with Hugging Face, you would need to replace the attention layers in an existing model.

## Troubleshooting

### Training diverges or produces NaN

- Increase `lambda_init` (e.g., from 3.0 to 10.0)
- Reduce learning rate
- Use `preconditioner_type="fast"` instead of `"cccp"` for sequences > 1024
- Enable gradient clipping in your training loop

### Out of memory on long sequences

- Reduce sequence length (practical limit ~2048 tokens)
- Reduce `preconditioner_rank` (default 32)
- Reduce `pcg_max_iterations`
- Use a smaller `d_model`

### Tests fail after installation

Ensure you have the dev dependencies installed:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup, coding standards, and pull request process.
