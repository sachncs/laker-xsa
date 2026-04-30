# Design Decisions

## Overview

This document details all design decisions made during implementation, including:
- Assumptions not specified in the papers
- Rationale for implementation choices
- What was preserved exactly vs. approximated

## Paper Fidelity

### XSA (arXiv:2603.09078)

**Exactly preserved:**
- Core projection removal formula: y_i^XSA = y_i - proj_{v_i}(y_i)
- Multi-head attention structure
- Orthogonality objective

**Underspecified / Our additions:**
1. **Learnable scale parameter**: The paper doesn't specify whether the projection removal should be scaled. We add a learnable `xsa_scale` parameter.

2. **Alternative exclusion modes**: The paper focuses on projection removal. We also implement:
   - `zero_diagonal`: Zero attention score diagonal
   - `mask`: Explicit binary masking

3. **When to apply exclusion**: The paper doesn't specify if exclusion should be applied before or after the output projection. We apply it before (to the attention output).

### LAKER (arXiv:2604.25138v1)

**Exactly preserved:**
- Kernel regression formulation
- Preconditioned iterative solving concept
- Learnable preconditioner approach

**Underspecified / Our additions:**
1. **Preconditioner parameterization**: The paper doesn't fully specify the preconditioner structure. We use:
   ```
   P = diag(d) + U @ U^T
   ```
   This is a common choice balancing expressivity and efficiency.

2. **Position dependence**: We generate low-rank factors from learned position embeddings to handle variable sequence lengths.

3. **Iteration count**: Not specified. We use 10 as default.

4. **Kernel type**: Paper mentions kernel attention but doesn't mandate a specific kernel. We support RBF, linear, and cosine.

## Architecture Decisions

### 1. Pre-Norm vs Post-Norm

**Decision:** Pre-norm (LayerNorm before attention/MLP)

**Rationale:**
- More stable gradients (prevents amplification through residual)
- Better empirical performance for deep models
- Standard in modern implementations (e.g., Llama, PaLM)

**Trade-off:** Slightly different training dynamics than original Transformer

### 2. Richardson vs Conjugate Gradient

**Decision:** Preconditioned Richardson iteration

**Rationale:**
- Simpler to implement differentiably
- CG requires inner products that vary with sequence length
- With good preconditioning, convergence is adequate
- More stable gradients in deep networks

**Trade-off:** May require more iterations than CG

### 3. Low-Rank + Diagonal Preconditioner

**Decision:** P = diag(d) + U @ U^T

**Rationale:**
- Diagonal captures per-token scaling
- Low-rank captures cross-token correlations
- O(nr) parameters instead of O(n²)
- Ensures positive definiteness

**Trade-off:** May not capture all correlations in K

### 4. Position-Generated Low-Rank Factors

**Decision:** Generate U from learned position embeddings

**Rationale:**
- Handles variable sequence lengths
- Shares parameters across positions
- Enables some extrapolation

**Trade-off:** Fixed max positions (2048 default)

### 5. RBF Kernel Default

**Decision:** Use RBF kernel as default

**Rationale:**
- Positive definite by construction
- Smooth, translation invariant
- Well-studied theoretically

**Trade-off:** Requires bandwidth tuning (learned)

## Numerical Stability Decisions

### 1. Epsilon Placement

We add epsilon in multiple locations:
- Projection denominator: `v_norm_sq + eps`
- Kernel diagonal: implicit through regularization
- Regularization: `softplus(lambda) + eps`

**Rationale:** Prevent division by zero and ensure positivity

### 2. Clipping

We clip α values to [-1e6, 1e6] during iteration.

**Rationale:** Prevent numerical overflow in long iterations

### 3. Softplus for Positivity

We use softplus for lambda and preconditioner diagonal:

```python
lambda_eff = softplus(lambda_param) + eps
diag_precond = softplus(kernel_diag) * scale + reg
```

**Rationale:** Ensures positivity while allowing gradient flow

## API Design Decisions

### 1. Configuration Dataclass

**Decision:** Single config object for all components

**Rationale:**
- Consistent hyperparameters across modules
- Easy serialization
- Clear documentation of options

### 2. Modular Architecture

**Decision:** Separate modules for each attention type

**Rationale:**
- Easy ablation studies
- Clear comparison between variants
- Reusable components

### 3. Type Hints Throughout

**Decision:** Full type annotations on all public APIs

**Rationale:**
- Better IDE support
- Clearer documentation
- Catch errors early with mypy

## Testing Decisions

### 1. Shape Tests

**Decision:** Extensive shape verification tests

**Rationale:** Catch broadcasting errors and dimension mismatches

### 2. Gradient Tests

**Decision:** Verify gradients flow through all components

**Rationale:** Ensure differentiability for training

### 3. Numerical Stability Tests

**Decision:** Test with extreme values and long sequences

**Rationale:** Catch NaN/Inf issues before they affect training

## Known Limitations

### 1. Sequence Length

Current implementation is O(n²) in sequence length due to full kernel matrix.

**Mitigation:** Future work should implement:
- Block-sparse kernels
- Nyström approximation
- Linear attention variants

### 2. CUDA Optimization

No custom CUDA kernels; relies on PyTorch operations.

**Mitigation:** Future work could implement:
- Fused kernel + attention kernel
- Custom CG solver kernel

### 3. Batched Variable Length

No native support for variable-length sequences in batch.

**Mitigation:** Use padding mask; inefficient for highly variable lengths

## Future Directions

### Near-term
1. Add gradient checkpointing for memory efficiency
2. Implement FlashAttention-style fusion
3. Add mixed precision (AMP) support

### Long-term
1. Sparse kernel implementations
2. Kernel approximation methods
3. Integration with popular model libraries
