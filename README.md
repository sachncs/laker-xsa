# LAKER-XSA: Fused Exclusive Self Attention and LAKER Kernel Attention

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-grade implementation of fused **Exclusive Self Attention (XSA)** and **LAKER-style Kernel Attention Regression** for Transformer models.

## Papers

- **XSA**: [arXiv:2603.09078](https://arxiv.org/abs/2603.09078) - Exclusive Self Attention
- **LAKER**: [arXiv:2604.25138v1](https://arxiv.org/html/2604.25138v1) - Learned Preconditioning for Attention Kernel Regression

## Overview

LAKER-XSA fuses two attention mechanisms to improve Transformer models:

1. **Exclusive Self Attention (XSA)**: Removes self-aligned components from attention output, forcing each token to aggregate only from OTHER tokens in the sequence.

2. **LAKER Kernel Attention**: Treats attention as kernel ridge regression with a learned preconditioner (CCCP-based Tyler's M-estimator), solved via Preconditioned Conjugate Gradient (PCG).

### Key Features

- **Mathematically faithful**: Implements equations directly from papers
- **Production-ready**: Full type hints, Google-style docstrings, CI-passing (pylint, mypy, pytest)
- **Well-tested**: 269 tests, 88% coverage, zero deprecation warnings
- **Benchmarked**: Scaling, conditioning, and runtime benchmarks included
- **Dual API**: Class-based Modules for training, stateless `functional` API for inference

## Installation

```bash
git clone https://github.com/your-org/laker-xsa.git
cd laker-xsa

# Install with core dependencies
pip install -e .

# With development, benchmark, and training dependencies
pip install -e ".[dev,bench,train]"
```

## Quick Start

```python
import torch
from laker_xsa import XSA_LAKER_Config, LakerAttention
from laker_xsa.model.full_model import XSALAKERTransformer

# Configuration
config = XSA_LAKER_Config(
    d_model=512,
    num_heads=8,
    dropout=0.1,
    xsa_mode="subtract_projection",
)

# Single attention layer (v2 — flagship)
attn = LakerAttention(config)
x = torch.randn(2, 128, 512)  # (batch, seq_len, d_model)
out = attn(x)  # (2, 128, 512)

# Full Transformer model
model = XSALAKERTransformer(
    config,
    num_layers=6,
    vocab_size=32000,
    max_seq_len=512,
    attention_type="fused_v2",
)

input_ids = torch.randint(0, 32000, (2, 128))
logits = model(input_ids)  # (2, 128, 32000)
```

## Usage

### CLI Training

```bash
python -m laker_xsa.cli.train \
    --d-model 256 \
    --num-heads 4 \
    --num-layers 4 \
    --num-epochs 10 \
    --batch-size 8 \
    --attention-type fused_v2
```

### CLI Benchmarking

```bash
python -m laker_xsa.cli.benchmark \
    --d-model 512 \
    --num-heads 8 \
    --num-runs 50 \
    --output results.json
```

### Evaluate Checkpoint

```bash
python -m laker_xsa.cli.evaluate --checkpoint path/to/checkpoint.pt
```

## Repository Structure

```
laker-xsa/
├── laker_xsa/                # Main package
│   ├── config.py             # Configuration dataclass
│   ├── attention/            # Attention implementations
│   │   ├── core.py           # Base class, QKV projection, reshape utils
│   │   ├── standard.py       # Standard scaled dot-product attention
│   │   ├── xsa.py            # Exclusive Self Attention
│   │   ├── laker.py          # Fused XSA + LAKER (v2, flagship)
│   │   ├── kernels.py        # AttentionKernel module
│   │   ├── functional.py     # Stateless compute_kernel_matrix
│   │   └── _legacy.py        # Deprecated v1 (KernelAttentionRegression, etc.)
│   ├── solver/               # Iterative solvers
│   │   ├── laker_preconditioner.py  # CCCP / fast / diagonal preconditioner
│   │   ├── conjugate_gradient.py    # PCG + Richardson solvers
│   │   ├── functional.py     # Stateless apply_kernel_operator
│   │   └── preconditioner.py # Deprecated v1 preconditioner
│   ├── model/                # Transformer models
│   │   ├── transformer_block.py
│   │   └── full_model.py
│   ├── training/             # Training utilities
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── benchmarks/           # Benchmark suites
│   ├── cli/                  # CLI entry points
│   └── utils/                # Tensor ops, seed, stability
├── tests/                    # 269 tests, 88% coverage
├── examples/                 # Example scripts
├── docs/                     # Architecture, math, design docs
└── .github/workflows/        # CI (pylint, mypy, pytest)
```

## API Reference

### Configuration

```python
from laker_xsa import XSA_LAKER_Config

config = XSA_LAKER_Config(
    d_model=512,                # Embedding dimension
    num_heads=8,                # Number of attention heads
    head_dim=None,              # Per-head dim (default: d_model // num_heads)
    dropout=0.0,                # Dropout rate
    eps=1e-6,                   # Numerical stability epsilon
    lambda_init=3.0,            # Initial regularization for kernel system
    kernel_type="exp_attention",# 'exp_attention', 'rbf', 'linear', or 'cosine'
    xsa_mode="subtract_projection",  # 'subtract_projection', 'zero_diagonal', 'mask'
    preconditioner_type="fast", # 'cccp', 'fast', 'diagonal', 'none'
    preconditioner_rank=32,     # Low-rank dimension for fast preconditioner
    pcg_max_iterations=20,      # Maximum PCG iterations
    pcg_tolerance=1e-2,         # Relative residual tolerance
    kernel_temperature=1.0,     # Temperature for exp kernel
)
```

### Attention Modules

| Module | Location | Description |
|--------|----------|-------------|
| `LakerAttention` | `laker_xsa.attention.laker` | **Flagship v2** — Fused XSA + LAKER with PCG |
| `LakerAttentionLayer` | `laker_xsa.attention.laker` | Multi-layer wrapper with per-layer config |
| `AttentionKernel` | `laker_xsa.attention.kernels` | Exp kernel K = exp(cosine(Q,K) / T) |
| `StandardMultiHeadAttention` | `laker_xsa.attention.standard` | Baseline scaled dot-product attention |
| `ExclusiveSelfAttention` | `laker_xsa.attention.xsa` | XSA-only with 3 exclusion strategies |
| `FusedXSALAKERAttention` | `laker_xsa.attention._legacy` | **Deprecated v1** — use LakerAttention |
| `KernelAttentionRegression` | `laker_xsa.attention._legacy` | **Deprecated v1** — use LakerAttention |

### Solver Modules

| Module | Description |
|--------|-------------|
| `LakerPreconditioner` | CCCP, fast gradient, diagonal, or none modes |
| `pcg_solve` | Preconditioned Conjugate Gradient with adaptive early stopping |
| `richardson_solve` | Preconditioned Richardson iteration (baseline) |

### Functional API (Stateless)

| Function | Module |
|----------|--------|
| `compute_kernel_matrix` | `laker_xsa.attention.functional` |
| `apply_kernel_operator` | `laker_xsa.solver.functional` |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=laker_xsa --cov-report=html

# Run specific categories
pytest tests/test_attention.py -v
pytest tests/test_laker_v2.py -v
pytest tests/test_gradients.py -v
```

## Benchmarks

### Runtime Comparison (typical)

| Attention Type | Forward (ms) | Backward (ms) | Relative |
|----------------|--------------|---------------|----------|
| Standard | 0.2 | 0.4 | 1.0x |
| XSA | 0.3 | 0.5 | 1.5x |
| Kernel (v1) | 1.5 | 2.5 | 6.0x |
| Fused (v1) | 1.8 | 3.0 | 8.0x |

*Results vary by hardware and sequence length*

### Conditioning Improvement

The CCCP learned preconditioner reduces kernel condition numbers by 10-1000x, enabling dramatically faster PCG convergence versus unpreconditioned solvers.

## Design Principles

1. **Mathematical fidelity**: Implement equations directly from papers
2. **Clarity over cleverness**: Prefer readable code over optimization
3. **Explicit over implicit**: Document all assumptions and approximations
4. **Testable**: Comprehensive test coverage with zero warnings
5. **Reproducible**: Configurable random seeds, deterministic options

## Limitations

- **O(n²) complexity**: Limited to ~2048 tokens without modifications
- **Runtime overhead**: 8-10x slower than standard attention
- **CCCP mode**: O(n³) per iteration; use `fast` mode for n > 1024
- **Hyperparameter sensitivity**: Requires tuning for optimal results

See [`docs/limitations.md`](docs/limitations.md) for details.

## Citation

If you use this implementation in your research:

```bibtex
@software{laker-xsa,
  title = {LAKER-XSA: Fused Exclusive Self Attention and LAKER Kernel Attention},
  author = {LAKER-XSA Contributors},
  year = {2026},
  url = {https://github.com/your-org/laker-xsa},
}

@article{xsa_paper,
  title = {Exclusive Self Attention},
  author = {XSA Authors},
  journal = {arXiv preprint arXiv:2603.09078},
  year = {2026},
}

@article{laker_paper,
  title = {Learned Preconditioning for Attention Kernel Regression},
  author = {LAKER Authors},
  journal = {arXiv preprint arXiv:2604.25138},
  year = {2026},
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Run linting: `pylint laker_xsa/ --rcfile=pyproject.toml`
6. Run type checking: `mypy laker_xsa/ --ignore-missing-imports`
7. Submit a pull request
