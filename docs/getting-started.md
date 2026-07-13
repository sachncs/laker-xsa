# Getting Started

This guide walks you through installing LAKER-XSA and running your first experiment.

## Prerequisites

- **Python** 3.9 or later
- **PyTorch** 2.0 or later
- A CUDA-capable GPU (recommended, not required)

## Installation

### From source (recommended)

```bash
git clone https://github.com/sachncs/laker-xsa.git
cd laker-xsa
pip install -e .
```

### With optional dependencies

```bash
# Development tools (testing, linting, formatting)
pip install -e ".[dev]"

# Benchmarking tools (matplotlib, pandas)
pip install -e ".[bench]"

# Training tools (tqdm progress bars)
pip install -e ".[train]"

# Everything
pip install -e ".[dev,bench,train]"
```

## Verify Installation

```bash
# Run the test suite
pytest tests/ -v --tb=short
```

## Your First Experiment

### 1. Single Attention Layer

```python
import torch
from laker_xsa import XSA_LAKER_Config, LakerAttention

config = XSA_LAKER_Config(
    d_model=512,
    num_heads=8,
    dropout=0.1,
    xsa_mode="subtract_projection",
)

attn = LakerAttention(config)
x = torch.randn(2, 128, 512)  # (batch, seq_len, d_model)
out = attn(x)
print(out.shape)  # torch.Size([2, 128, 512])
```

### 2. Full Transformer Model

```python
from laker_xsa.model.full_model import XSALAKERTransformer

model = XSALAKERTransformer(
    config,
    num_layers=6,
    vocab_size=32000,
    max_seq_len=512,
    attention_type="fused_v2",
)

input_ids = torch.randint(0, 32000, (2, 128))
logits = model(input_ids)
print(logits.shape)  # torch.Size([2, 128, 32000])
```

### 3. CLI Training

```bash
python -m laker_xsa.cli.train \
    --d-model 256 \
    --num-heads 4 \
    --num-layers 4 \
    --num-epochs 10 \
    --batch-size 8 \
    --attention-type fused_v2
```

### 4. CLI Benchmarking

```bash
python -m laker_xsa.cli.benchmark \
    --d-model 512 \
    --num-heads 8 \
    --num-runs 50 \
    --output results.json
```

## Choosing an Attention Type

| Type | Module | Use Case |
|------|--------|----------|
| `standard` | `StandardMultiHeadAttention` | Baseline, fast inference |
| `xsa` | `ExclusiveSelfAttention` | Research on XSA mechanism |
| `fused_v2` | `LakerAttention` | Full XSA + LAKER fusion (recommended) |

## Next Steps

- Read the [Architecture documentation](architecture.md) for design details
- See the [API Reference](../README.md#api-reference) for configuration options
- Review [Limitations](limitations.md) before production use
- Check out the [examples/](../examples/) directory for runnable scripts
