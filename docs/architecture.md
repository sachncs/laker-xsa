# Architecture Documentation

## Overview

LAKER-XSA fuses two attention mechanisms:

1. **Exclusive Self Attention (XSA)** from arXiv:2603.09078
2. **LAKER-style Kernel Attention** from arXiv:2604.25138v1

This document explains the architectural decisions and mathematical foundations.

## Module Structure

```
laker_xsa/
├── config.py              # Configuration dataclass
├── attention/
│   ├── standard_attention.py   # Baseline MHA
│   ├── xsa_attention.py        # XSA implementation
│   └── kernel_attention.py     # LAKER and fused implementations
├── solver/
│   ├── preconditioner.py       # Learned preconditioner
│   └── conjugate_gradient.py   # CG solver (alternative)
├── model/
│   ├── transformer_block.py    # Transformer block
│   └── full_model.py           # Full Transformer
├── training/
│   ├── trainer.py              # Training loop
│   └── losses.py               # Loss functions
├── benchmarks/
│   ├── long_context.py         # Scaling analysis
│   ├── conditioning.py         # Condition number analysis
│   └── runtime.py              # Profiling utilities
└── utils/
    ├── tensor_ops.py           # Tensor utilities
    ├── stability.py            # Numerical stability
    └── seed.py                 # Reproducibility
```

## Attention Flow

### Standard Attention

```
Input -> [Q, K, V projections] -> Scaled Dot-Product -> Softmax -> Output
```

### XSA Attention

```
Input -> [Q, K, V projections] -> Scaled Dot-Product -> Softmax -> 
         Output -> Subtract projection onto own value -> Output
```

### LAKER Kernel Attention

```
Input -> [Q, K, V projections] -> Kernel Matrix K -> 
         Solve (K + λI)α = V with preconditioned iteration -> 
         Output = K @ α
```

### Fused XSA + LAKER

```
Input -> [Q, K, V projections] -> Kernel Matrix K -> 
         Zero diagonal (XSA) -> Solve (K + λI)α = V -> 
         Output = K @ α -> Optional projection subtraction
```

## Key Design Decisions

### 1. Pre-Norm Architecture

We use pre-normalization (LayerNorm before attention/MLP) rather than post-normalization because:
- More stable gradients during training
- Better empirical performance for deep models
- Standard in modern Transformer implementations

### 2. Richardson Iteration over CG

We use preconditioned Richardson iteration instead of Conjugate Gradient because:
- Simpler to make fully differentiable
- Inner products in CG vary with sequence length
- With good preconditioning, convergence is adequate
- More stable gradients in deep networks

### 3. Low-Rank + Diagonal Preconditioner

The preconditioner uses:
```
P = diag(d) + U @ U^T
```

This balances:
- **Expressivity**: Low-rank captures cross-token correlations
- **Efficiency**: O(n · rank) parameters, not O(n²)
- **Stability**: Diagonal ensures positive definiteness

### 4. Position-Generated Low-Rank Factors

The low-rank factor U is generated from learned position embeddings:
- Allows variable sequence lengths
- Shares parameters across positions
- Enables extrapolation to longer sequences

## Mathematical Formulations

### XSA Projection Removal

Given attention output y_i and value v_i for token i:

```
y_i^XSA = y_i - (y_i · v_i) / (v_i · v_i + ε) · v_i
```

This ensures y_i^XSA is orthogonal to v_i.

### RBF Kernel

```
K_ij = exp(-||q_i - k_j||² / (2σ²))
```

Computed efficiently as:
```
||q - k||² = ||q||² + ||k||² - 2 · q · k
```

### Preconditioned Richardson Iteration

```
α_{t+1} = α_t + P · (V - (K + λI) · α_t)
```

Where P is the learned preconditioner.

## Approximations and Underspecified Details

### From XSA Paper (arXiv:2603.09078)

The paper describes projection removal but does not specify:
- **Learnable scale**: We add a learnable scale parameter for the projection subtraction
- **Mode variants**: We implement three modes (subtract_projection, zero_diagonal, mask)

### From LAKER Paper (arXiv:2604.25138v1)

The paper describes kernel attention with preconditioning but does not fully specify:
- **Preconditioner parameterization**: We use low-rank + diagonal
- **Position dependence**: We generate low-rank factors from position embeddings
- **Iteration count**: We use 10 as default, configurable

## Extending the Architecture

### For Longer Sequences

1. Reduce `num_iterations` for early layers
2. Use block-sparse kernel computation
3. Implement Nyström approximation in `KernelFunction`

### For Different Tasks

1. Adjust `d_model` and `num_heads` for capacity
2. Tune `preconditioner_rank` for speed/accuracy tradeoff
3. Try different `kernel_type` (rbf, linear, cosine)

### For Production Use

1. Add gradient checkpointing for memory efficiency
2. Implement FlashAttention-style kernel fusion
3. Add mixed precision (AMP) support
