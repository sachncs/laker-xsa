# LAKER-XSA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/sachn-cs/laker-xsa/actions/workflows/ci.yml/badge.svg)](https://github.com/sachn-cs/laker-xsa/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)

Fused Exclusive Self Attention and LAKER Kernel Attention for Transformer models.

Production-grade implementation of two complementary attention mechanisms that address fundamental failure modes of standard attention: **self-bias** (tokens copying themselves) and **spectral collapse** (eigenvalue decay).

## Features

- **Exclusive Self Attention (XSA)** — Removes self-aligned components, forcing context-only aggregation ([arXiv:2603.09078](https://arxiv.org/abs/2603.09078))
- **LAKER Kernel Attention** — Kernel ridge regression with CCCP-based learned preconditioning ([arXiv:2604.25138](https://arxiv.org/html/2604.25138v1))
- **Fused v2 (`LakerAttention`)** — Novel combination of XSA + LAKER in a single module
- **Dual API** — Class-based modules for training, stateless functional API for inference
- **CLI tools** — Train, benchmark, and evaluate from the command line
- **Well-tested** — 269 tests, 88% coverage, zero deprecation warnings
- **Type-safe** — Full type hints, passes mypy and pylint

## Installation

```bash
git clone https://github.com/sachn-cs/laker-xsa.git
cd laker-xsa
pip install -e .

# With development, benchmark, and training dependencies
pip install -e ".[dev,bench,train]"
```

## Quick Start

```python
import torch
from laker_xsa import XSA_LAKER_Config, LakerAttention
from laker_xsa.model.full_model import XSALAKERTransformer

config = XSA_LAKER_Config(d_model=512, num_heads=8, dropout=0.1)

# Single attention layer
attn = LakerAttention(config)
x = torch.randn(2, 128, 512)
out = attn(x)  # (2, 128, 512)

# Full Transformer model
model = XSALAKERTransformer(
    config, num_layers=6, vocab_size=32000,
    max_seq_len=512, attention_type="fused_v2",
)
logits = model(torch.randint(0, 32000, (2, 128)))
```

## Usage

### CLI Training

```bash
python -m laker_xsa.cli.train \
    --d-model 256 --num-heads 4 --num-layers 4 \
    --num-epochs 10 --batch-size 8 --attention-type fused_v2
```

### CLI Benchmarking

```bash
python -m laker_xsa.cli.benchmark \
    --d-model 512 --num-heads 8 --num-runs 50 --output results.json
```

### Evaluate Checkpoint

```bash
python -m laker_xsa.cli.evaluate --checkpoint path/to/checkpoint.pt
```

## Configuration

LAKER-XSA uses a single configuration dataclass:

```python
from laker_xsa import XSA_LAKER_Config

config = XSA_LAKER_Config(
    d_model=512,                      # Embedding dimension
    num_heads=8,                      # Number of attention heads
    head_dim=None,                    # Per-head dim (default: d_model // num_heads)
    dropout=0.0,                      # Dropout rate
    eps=1e-6,                         # Numerical stability epsilon
    lambda_init=3.0,                  # Regularization for kernel system
    kernel_type="exp_attention",      # 'exp_attention', 'rbf', 'linear', 'cosine'
    xsa_mode="subtract_projection",   # 'subtract_projection', 'zero_diagonal', 'mask'
    preconditioner_type="fast",       # 'cccp', 'fast', 'diagonal', 'none'
    preconditioner_rank=32,           # Low-rank dimension for fast preconditioner
    pcg_max_iterations=20,            # Maximum PCG iterations
    pcg_tolerance=1e-2,               # Relative residual tolerance
    kernel_temperature=1.0,           # Temperature for exp kernel
)
```

No environment variables are required. All configuration is passed through the `XSA_LAKER_Config` dataclass.

## Project Structure

```
laker-xsa/
├── laker_xsa/                # Main package
│   ├── config.py             # Configuration dataclass
│   ├── attention/            # Attention implementations
│   │   ├── core.py           # Base class, QKV projection
│   │   ├── standard.py       # Standard scaled dot-product
│   │   ├── xsa.py            # Exclusive Self Attention
│   │   ├── laker.py          # Fused XSA + LAKER (v2, flagship)
│   │   ├── kernels.py        # AttentionKernel module
│   │   ├── functional.py     # Stateless compute_kernel_matrix
│   │   └── _legacy.py        # Deprecated v1 classes
│   ├── solver/               # Iterative solvers
│   │   ├── laker_preconditioner.py  # CCCP/fast/diagonal preconditioner
│   │   ├── conjugate_gradient.py    # PCG + Richardson solvers
│   │   └── functional.py     # Stateless apply_kernel_operator
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
└── .github/workflows/        # CI pipeline
```

## Development

```bash
# Setup development environment
bash setup.sh

# Or manually:
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,bench,train]"
```

### Commands

| Command | Description |
|---------|-------------|
| `pytest tests/ -v` | Run all tests |
| `pytest tests/ --cov=laker_xsa` | Run tests with coverage |
| `pylint laker_xsa/ --rcfile=pyproject.toml` | Lint |
| `mypy laker_xsa/ --ignore-missing-imports` | Type check |
| `black laker_xsa/ tests/` | Format code |
| `python -m build` | Build distribution |

## API Reference

### Attention Modules

| Module | Description |
|--------|-------------|
| `LakerAttention` | **Flagship v2** — Fused XSA + LAKER with PCG solver |
| `LakerAttentionLayer` | Multi-layer wrapper with per-layer config |
| `AttentionKernel` | Exp kernel K = exp(cosine(Q,K) / T) |
| `StandardMultiHeadAttention` | Baseline scaled dot-product attention |
| `ExclusiveSelfAttention` | XSA-only with 3 exclusion strategies |

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

## Benchmarks

| Attention Type | Forward (ms) | Backward (ms) | Relative |
|----------------|--------------|---------------|----------|
| Standard | 0.2 | 0.4 | 1.0x |
| XSA | 0.3 | 0.5 | 1.5x |
| Kernel (v1) | 1.5 | 2.5 | 6.0x |
| Fused (v1) | 1.8 | 3.0 | 8.0x |

*Results vary by hardware and sequence length. See [RESULTS.md](RESULTS.md) for full details.*

## Roadmap

- [ ] Sparse kernel implementation for long sequences
- [ ] Custom CUDA kernels for fused operations
- [ ] Adaptive iteration count based on residual
- [ ] Mixed precision (AMP) support
- [ ] Hugging Face integration
- [ ] FlashAttention-style kernel fusion

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Security

To report security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## Citation

```bibtex
@software{laker-xsa,
  title = {LAKER-XSA: Fused Exclusive Self Attention and LAKER Kernel Attention},
  author = {LAKER-XSA Contributors},
  year = {2026},
  url = {https://github.com/sachn-cs/laker-xsa},
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

[MIT](LICENSE)
