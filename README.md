<p align="center">
  <h1 align="center">LAKER-XSA</h1>
  <p align="center">Fused Exclusive Self Attention and LAKER Kernel Attention for Transformer models.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/laker-xsa/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/laker-xsa/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/laker-xsa/"><img src="https://img.shields.io/pypi/v/laker-xsa" alt="PyPI"></a>
    <a href="https://github.com/sachncs/laker-xsa/stargazers"><img src="https://img.shields.io/github/stars/sachncs/laker-xsa" alt="Stars"></a>
    <a href="https://github.com/sachncs/laker-xsa/blob/master/pyproject.toml"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch"></a>
  </p>
</p>

**LAKER-XSA** is a production-grade Python library that implements two complementary
attention mechanisms for Transformer models, addressing fundamental failure modes
of standard scaled dot-product attention: **self-bias** (tokens copying themselves)
and **spectral collapse** (eigenvalue decay).

It ships **Exclusive Self Attention (XSA)**, which removes self-aligned components
to force context-only aggregation, and **LAKER Kernel Attention**, a kernel
ridge-regression formulation with CCCP-based learned preconditioning. The
flagship **LakerAttention (v2)** fuses both into a single module solved with a
Preconditioned Conjugate Gradient (PCG) iteration.

---

## Features

- **Exclusive Self Attention (XSA)** — Removes self-aligned components, forcing
  context-only aggregation ([arXiv:2603.09078](https://arxiv.org/abs/2603.09078)).
- **LAKER Kernel Attention** — Kernel ridge regression with CCCP-based learned
  preconditioning ([arXiv:2604.25138](https://arxiv.org/html/2604.25138v1)).
- **Fused v2 (`LakerAttention`)** — Novel combination of XSA + LAKER in a
  single module, solved by Preconditioned Conjugate Gradient.
- **Dual API** — Class-based modules for training, stateless functional API
  for inference (`compute_kernel_matrix`, `apply_kernel_operator`).
- **CLI Tools** — `laker-xsa-train`, `laker-xsa-benchmark`, `laker-xsa-evaluate`
  for training, profiling, and checkpoint evaluation.
- **Well-tested** — 269 tests, 88% coverage, zero deprecation warnings.
- **Type-safe** — Full type hints, passes mypy and pylint at 10.00/10.

---

## Installation

### From PyPI

```bash
pip install laker-xsa
```

### From source

```bash
git clone https://github.com/sachncs/laker-xsa.git
cd laker-xsa
pip install -e .
```

### With dev, benchmark, and training dependencies

```bash
pip install -e ".[dev,bench,train]"
```

---

## Quick Start

### CLI

```bash
# Train a model
python -m laker_xsa.cli.train \
    --d-model 256 --num-heads 4 --num-layers 4 \
    --num-epochs 10 --batch-size 8 --attention-type fused_v2

# Benchmark attention variants
python -m laker_xsa.cli.benchmark \
    --d-model 512 --num-heads 8 --num-runs 50 --output results.json

# Evaluate a checkpoint
python -m laker_xsa.cli.evaluate --checkpoint path/to/checkpoint.pt
```

### Python API

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

---

## Configuration

LAKER-XSA uses a single `XSA_LAKER_Config` dataclass — no environment variables
required.

### Kernel Type

| Value           | Definition                                                |
|-----------------|-----------------------------------------------------------|
| `exp_attention` | `K = exp(cosine(Q, K) / T)` with L2-normalized Q/K (default) |
| `rbf`           | Gaussian RBF on pairwise Q/K distances                    |
| `linear`        | Linear kernel `K = Q K^T`                                 |
| `cosine`        | Cosine similarity without exponential scaling             |

### XSA Mode

| Value                  | Definition                                                |
|------------------------|-----------------------------------------------------------|
| `subtract_projection`  | Subtract each token's self-projection from output (default) |
| `zero_diagonal`        | Zero the diagonal of the kernel matrix                    |
| `mask`                 | Apply an additive -inf mask on the diagonal               |

### Preconditioner

| Value      | Definition                                                  |
|------------|-------------------------------------------------------------|
| `cccp`     | CCCP fixed-point iteration with Tyler's M-estimator        |
| `fast`     | Gradient-based low-rank + diagonal (default)                |
| `diagonal` | Jacobi-style diagonal preconditioner                        |
| `none`     | Identity preconditioner                                     |

---

## API

| Symbol                                  | Type     | Description                                       |
|-----------------------------------------|----------|---------------------------------------------------|
| `XSA_LAKER_Config`                      | dataclass | Hyperparameters for every attention variant      |
| `LakerAttention`                        | class    | Fused XSA + LAKER (v2) attention module           |
| `XSALAKERTransformer`                   | class    | Full Transformer model with fused attention       |
| `compute_kernel_matrix`                 | function | Stateless kernel matrix construction              |
| `apply_kernel_operator`                 | function | Stateless kernel operator application            |
| `AttentionKernel`                       | class    | Base attention module with shared QKV projection  |
| `StandardAttention`                     | class    | Standard scaled dot-product attention baseline    |
| `XSA`                                   | class    | Exclusive Self Attention (no self-bias)           |
| `LakerAttentionKernel`                  | class    | LAKER kernel attention (v1)                       |

---

## Examples

```bash
# Train a small fused model on synthetic data
laker-xsa-train --d-model 256 --num-heads 4 --num-layers 4 \
    --num-epochs 10 --batch-size 8 --attention-type fused_v2

# Run a benchmark sweep across attention variants
laker-xsa-benchmark --d-model 512 --num-heads 8 --num-runs 50 --output results.json

# Evaluate a saved checkpoint and emit metrics
laker-xsa-evaluate --checkpoint artifacts/last.pt
```

See [`examples/`](examples/) for end-to-end scripts covering each attention
variant and the full Transformer pipeline.

---

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

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev,bench,train]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=laker_xsa

# Lint
pylint laker_xsa/ --rcfile=pyproject.toml

# Type check
mypy laker_xsa/ --ignore-missing-imports

# Format code
black laker_xsa/ tests/

# Build distribution
python -m build
```

### Code Style

- **Line length**: 88 (black default)
- **Quotes**: double (`"`)
- **Formatter**: [black](https://github.com/psf/black) — `black laker_xsa/ tests/`
- **Type hints**: required on all public signatures; passes mypy
- **Linter**: [pylint](https://pylint.pycqa.org/) at 10.00/10 with project config
- **Docstrings**: Google-style throughout
- **Naming**: no semi-private (`_foo`) names — all identifiers are public

### Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add sparse kernel implementation
fix: correct softplus import for non-callable lint
docs: restructure README to reference template
refactor: extract yaml_escape to static method
test: add round-trip serialization tests
chore: update pyproject config
```

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=laker_xsa
```

---

## Build

```bash
python -m build
```

---

## Release

See [docs/release.md](docs/release.md) — version is bumped in `pyproject.toml`,
changelog updated in `CHANGELOG.md`, tagged `vX.Y.Z`, and the PyPI publishing
workflow publishes the distribution to TestPyPI then PyPI.

---

## Tech Stack

| Category       | Technology                                          |
|----------------|-----------------------------------------------------|
| Language       | Python 3.9+                                         |
| Deep Learning  | [PyTorch](https://pytorch.org/) 2.0+                |
| Numerical      | [NumPy](https://numpy.org/) 1.20+                   |
| Testing        | [pytest](https://docs.pytest.org/) + pytest-cov     |
| Lint           | [pylint](https://pylint.pycqa.org/) 2.17+           |
| Format         | [black](https://github.com/psf/black) 23+           |
| Type Check     | [mypy](https://mypy-lang.org/) 1.0+                  |
| Benchmarks     | [matplotlib](https://matplotlib.org/), pandas       |
| Training       | [tqdm](https://tqdm.github.io/)                     |

---

## Roadmap

- Sparse kernel implementation for long sequences
- Custom CUDA kernels for fused operations
- Adaptive iteration count based on residual
- Mixed precision (AMP) support
- Hugging Face integration
- FlashAttention-style kernel fusion

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Pull request process
- Coding standards
- Test expectations

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Security

To report security vulnerabilities, please see [SECURITY.md](SECURITY.md).

---

## Citation

```bibtex
@software{laker-xsa,
  title = {LAKER-XSA: Fused Exclusive Self Attention and LAKER Kernel Attention},
  author = {sachin},
  year = {2026},
  url = {https://github.com/sachncs/laker-xsa},
}

@article{xsa_paper,
  title = {Exclusive Self Attention},
  author = {Sachin},
  journal = {arXiv preprint arXiv:2603.09078},
  year = {2026},
}

@article{laker_paper,
  title = {Learned Preconditioning for Attention Kernel Regression},
  author = {Sachin},
  journal = {arXiv preprint arXiv:2604.25138},
  year = {2026},
}
```

## License

[MIT](LICENSE) © 2026 sachin

Contact: [sachncs@gmail.com](mailto:sachncs@gmail.com)
