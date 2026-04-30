# Limitations

## Known Limitations

This document lists known limitations of the LAKER-XSA implementation.

## Computational Complexity

### O(n²) Memory and Compute

**Issue:** The kernel matrix K has shape (batch, num_heads, seq_len, seq_len), requiring O(n²) memory and compute.

**Impact:**
- Limited to ~2048 tokens on typical GPUs
- Slow for long sequences

**Mitigation:**
- Use fewer iterations for early layers
- Implement block-sparse attention
- Use kernel approximation (Nyström, linear attention)

### Iterative Solve Overhead

**Issue:** Richardson iteration adds T× overhead where T is iteration count.

**Impact:**
- 8-10× slower than standard attention in benchmarks
- May not be suitable for latency-critical applications

**Mitigation:**
- Reduce iterations after convergence analysis
- Use warm starting from previous layer's α
- Implement early stopping based on residual

## Numerical Stability

### Ill-Conditioned Kernels

**Issue:** RBF kernel can be ill-conditioned for certain data distributions.

**Symptoms:**
- Slow solver convergence
- Occasional NaN in gradients

**Mitigation:**
- Increase regularization λ
- Use linear or cosine kernel instead
- Increase preconditioner rank

### Long Sequence Instability

**Issue:** Very long sequences (>1024) can cause numerical issues.

**Symptoms:**
- Increased condition number
- Solver divergence

**Mitigation:**
- Use more iterations
- Increase ε values
- Implement kernel normalization

## Model Capacity

### Fixed Position Embeddings

**Issue:** Preconditioner uses fixed max position embeddings (2048 default).

**Impact:** Cannot handle sequences longer than max without modification.

**Mitigation:**
- Increase max_positions parameter
- Implement extrapolation (e.g., RoPE-style)

### Limited Preconditioner Expressivity

**Issue:** Low-rank + diagonal may not capture all correlations.

**Impact:** Suboptimal preconditioning for some inputs.

**Mitigation:**
- Increase preconditioner_rank
- Add higher-rank corrections

## Training Considerations

### Hyperparameter Sensitivity

**Issue:** Performance sensitive to:
- num_iterations
- preconditioner_rank
- lambda_init
- kernel bandwidth

**Mitigation:**
- Use provided defaults as starting point
- Run ablation studies for your use case
- Monitor solver convergence during training

### Gradient Flow

**Issue:** Many iterations can cause gradient vanishing/exploding.

**Symptoms:**
- Unstable training
- Poor convergence

**Mitigation:**
- Limit iterations (10 is usually sufficient)
- Use gradient clipping
- Monitor gradient norms

## Benchmark Caveats

### Synthetic Data

**Issue:** Benchmarks use random tensors, not real data.

**Impact:** May not reflect real-world performance.

**Mitigation:** Run benchmarks on your actual data distribution.

### Small Model Scale

**Issue:** Default benchmarks use small models (d_model=128-512).

**Impact:** Overhead may be different at production scale.

**Mitigation:** Scale up benchmarks to your target model size.

### Single GPU

**Issue:** Benchmarks run on single GPU.

**Impact:** Multi-GPU scaling not characterized.

**Mitigation:** Test distributed training for your setup.

## Paper Limitations

These limitations inherit from the underlying papers:

### XSA (arXiv:2603.09078)

- Assumes projection removal is sufficient for exclusion
- Doesn't address interaction with other attention modifications
- Limited analysis of gradient flow properties

### LAKER (arXiv:2604.25138v1)

- Assumes kernel formulation is appropriate for all attention tasks
- Preconditioner design not fully specified
- Limited analysis of generalization to unseen sequence lengths

## Underspecified Details

The following details are not specified in the papers:

1. **Exact preconditioner architecture**: We chose low-rank + diagonal
2. **Iteration count**: We use 10 as default
3. **Kernel bandwidth initialization**: We learn it from 1.0
4. **When to apply XSA**: We apply before output projection
5. **Interaction between XSA and kernel**: We zero kernel diagonal AND apply projection removal

See `design_decisions.md` for full details.

## Recommended Usage

### When to Use LAKER-XSA

- Research on attention mechanisms
- Tasks requiring strong context integration
- Settings where numerical stability is critical
- When you can afford the compute overhead

### When NOT to Use LAKER-XSA

- Production latency-critical applications
- Very long sequences (>2048 tokens)
- Resource-constrained environments
- When standard attention works well enough

## Future Work

Priority improvements:

1. **Sparse kernel implementation** for long sequences
2. **Custom CUDA kernels** for fused operations
3. **Better preconditioner** with more expressivity
4. **Adaptive iteration** based on residual
5. **Mixed precision** support for memory efficiency
