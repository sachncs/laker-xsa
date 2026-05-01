# LAKER+XSA Implementation Audit: Gaps vs Published Papers

**Date:** 2026-04-30
**Sources:** XSA (arXiv:2603.09078), LAKER (arXiv:2604.25138)

---

## Executive Summary

The current implementation has **critical gaps** in the LAKER component — the preconditioner learning mechanism is completely different from the published paper. The XSA component is mathematically correct but tested in a regime where the paper itself says gains are minimal. The synergy between the two is novel and unexplored.

---

## 1. LAKER Implementation: CRITICAL GAPS

### 1.1 Wrong System Formulation

| Aspect | LAKER Paper | Current Code | Status |
|--------|------------|-------------|--------|
| Kernel | `G_{ij} = exp(⟨e_i, e_j⟩)` (single embedding) | `K_{ij} = kernel(q_i, k_j)` (separate Q/K projections) | **WRONG** |
| Targets | `y ∈ ℝ^n` (scalar RSS values) | `V ∈ ℝ^{n×d_head}` (multi-head value vectors) | **WRONG** |
| Kernel types | Attention kernel only (inner product + exp) | RBF, linear, cosine (choice of 3) | **WRONG** |
| Problem | `(G + λI)α = y` (scalar regression) | `(K + λI)α = V` (vector regression per head) | **WRONG** |

The LAKER paper addresses **spectrum cartography** — reconstructing scalar radio signal strength from spatial measurements. The code applies it to **multi-head self-attention** in Transformers. These are fundamentally different problems.

### 1.2 Completely Wrong Preconditioner

| Aspect | LAKER Paper | Current Code |
|--------|------------|-------------|
| Learning method | CCCP (convex-concave procedure) on Tyler's M-estimator | None — just parameterized directly |
| Data source | Angular samples `ū_k = u_k/||u_k||` from `u_k ~ N(0, (λI+G)²)` | Kernel diagonal only |
| Parameterization | `P = Σ^{-1/2}` where `Σ ∈ ℝ^{n×n}` is full matrix | `P = diag(d) + UU^T` (low-rank + diagonal) |
| What it captures | Full inverse spectral structure of `(λI+G)` | Position-based correlations (no spectral info) |

The paper's preconditioner captures the **inverse spectral structure** of the kernel system via angular sampling and Tyler's M-estimator. The code's position-embedding-based low-rank approach captures nothing about the kernel's eigenvalue spectrum.

### 1.3 Missing CCCP Algorithm

The paper uses this iterative procedure to learn Σ (Algorithm 1, lines 5-13):

```
1. Sample z_k ~ N(0, I_n) for k = 1,...,N_r
2. u_k ← (λI+G)·z_k,  ū_k ← u_k / ||u_k||₂
3. Initialize Σ₀ ← I
4. CCCP iteration:
   Σ_{t+1} = (1/(1+γ/n)) · [(n/N_r)·Σ_k (ū_k ū_k^⊤)/(ū_k^⊤ Σ_t^{-1} ū_k + ε) + γI]
5. Shrinkage: Σ̃ ← (1-ρ)·Σ_{t+1} + ρI
6. Normalize: Σ ← Σ̃ / (tr(Σ̃)/n)
7. Set P = Σ^{-1/2}
```

**None of this exists in the code.** The code's `LearnedPreconditioner` is a completely different mechanism.

### 1.4 Missing Safeguards

| Feature | In Paper | In Code |
|---------|----------|---------|
| ε-safeguard in denominator | ✓ | ✗ |
| Adaptive shrinkage ρ | ✓ (increases when λ_min too small) | ✗ |
| Trace normalization | ✓ (ensures tr(Σ) = n) | ✗ |
| Fixed-point convergence check | ✓ (CCCP termination) | ✗ |

### 1.5 Solver: Richardson vs PCG

| Aspect | LAKER Paper | Current Code |
|--------|------------|-------------|
| Solver | Preconditioned Conjugate Gradient (PCG) | Richardson iteration |
| Convergence | Quadratic (CG property) | Linear |
| Iterations needed | ~30 (at n=2000) | Fixed 10 (no convergence check) |
| Convergence monitoring | Relative residual `||r||/||y||` | None (fixed iterations) |
| Adaptive termination | Yes (ε_tol) | No |

The code has a `conjugate_gradient.py` that implements standard CG, but it is explicitly **not used** by the fused attention — the docstring says Richardson is "preferred." The LAKER paper shows PCG is essential for the 22× speedup.

### 1.6 Missing Matrix-Free Operations

The paper (Remark 3): "The method only requires matrix–vector products... does not require forming or storing the full kernel matrix G."

The code: Builds the **full** `(batch, num_heads, seq_len, seq_len)` kernel matrix explicitly. At n=2000, this is a 2000×2000 matrix per head per batch element.

---

## 2. XSA Implementation: CORRECT BUT UNDEREXPLORED

### 2.1 What's Correct

- `_subtract_projection()` — Mathematically matches the paper's orthogonal projection removal
- `_zero_diagonal_attention()` — Correctly zeros self-attention pre-softmax
- Multi-head reshape and output projection — Standard and correct
- Learnable `xsa_scale` parameter — Reasonable extension

### 2.2 What's Underexplored

| Gap | Paper Finding | Current Testing |
|-----|--------------|-----------------|
| **Sequence length** | Gains grow with seq len (tested 512-16384) | Only tested ≤256 tokens |
| **Model scale** | Gains grow with model size (0.7B→2.7B) | Only tested tiny models (d_model=64-128) |
| **Real language** | Tested on FineWeb-100BT, 8 downstream tasks | Only synthetic tasks |
| **Attention sinks** | XSA maintains margin even with attention sinks (Fig 6) | Not tested |
| **Learning rate robustness** | Constant margin across 4 LRs (Fig 4) | Not tested |
| **XSA mode comparison** | Paper uses subtract_projection implicitly | 3 modes exist, never systematically compared |

**Key finding from the paper**: Gains are minimal at small scale and short sequences. The paper's 0.7B model only gets +0.26 pp improvement. The current benchmarks use models ~1000× smaller — the paper's own results predict negligible gains in this regime.

### 2.3 Exploration Opportunities

1. **Long-context benchmarks** (512-2048 tokens) where XSA claims larger gains
2. **Larger models** (at least d_model=512-1024) to see if gains emerge with scale
3. **Real NLP data** (WikiText-2, C4) instead of synthetic copy/reversal tasks
4. **Mode ablation**: Compare `subtract_projection` vs `zero_diagonal` vs `mask` systematically
5. **Layer-wise analysis**: XSA paper shows value correlation increases with layer depth — test if XSA is more beneficial in deeper layers
6. **Attention sink interaction**: Test with/without learned attention sinks
7. **Per-task analysis**: BoolQ and LAMBADA showed largest gains (+3.88, +1.86) — test on QA and language modeling specifically

---

## 3. LAKER+XSA Synergy: UNEXPLORED

### 3.1 Fundamental Mismatch

The two papers address completely different problems:

| | XSA (2603.09078) | LAKER (2604.25138) |
|---|---|---|
| Domain | LLM self-attention | Spectrum cartography |
| Output | Next-token logits | Scalar radio field |
| Kernel | Softmax(QK^T/√d) | exp(EE^T) |
| Optimization | Standard gradient descent | PCG with learned preconditioner |
| Scale | Up to 2.7B params, 2048 ctx | Up to n=2000 |

The "fusion" of these two methods is a **novel, unpublished idea**. There is no paper studying how XSA-style self-exclusion interacts with kernel regression attention. This is both an opportunity (novel contribution) and a challenge (no guidance on how to combine them correctly).

### 3.2 Unresolved Design Questions

1. **Should LAKER's kernel replace softmax attention entirely** or augment it?
2. **How does XSA's zero-diagonal** interact with LAKER's regularization λI (which also modifies the diagonal)?
3. **Should the preconditioner be learned per-layer** (as LAKER paper) or shared across layers?
4. **Can the CCCP procedure be differentiated through** for end-to-end training?
5. **Is the Richardson→PCG switch necessary** for Transformer training (vs one-shot solve in LAKER)?
6. **Does the "attention kernel" G=exp(EE^T)** need separate Q/K projections like standard attention?

### 3.3 Optimization Opportunities

1. **Learned preconditioner warm-start**: CCCP could be run once and the learned Σ reused, avoiding per-forward-pass CCCP
2. **Layer-specific preconditioners**: Different layers have different attention patterns
3. **Adaptive iteration count**: Use residual norm to decide when to stop iterating
4. **Kernel function ablation**: Compare RBF vs attention-kernel (exp(QK^T)) for the Transformer use case

---

## 4. Prioritized Fix List

### Critical (LAKER is broken)

1. **Replace preconditioner with CCCP-based learning** — implement Algorithm 1 lines 5-13 from the LAKER paper
2. **Switch from Richardson to PCG** — the conjugate_gradient.py exists, just needs to be wired in with the learned preconditioner
3. **Add convergence-based termination** instead of fixed iterations
4. **Add safeguards**: ε-stabilization, adaptive shrinkage ρ, trace normalization

### High (Bridging LAKER to Transformers)

5. **Reconcile kernel formulations** — the LAKER paper uses `G=exp(EE^T)` from a single embedding; the Transformer needs Q/K projections. This requires design work
6. **Handle multi-head value vectors** — LAKER solves scalar systems; Transformer values are `d_head`-dimensional
7. **Make matrix-free** — use matrix-vector products instead of building the full kernel matrix

### Medium (XSA exploration)

8. **Run long-context benchmarks** (512-2048 tokens)
9. **Test at larger model scales** where paper predicts gains
10. **Compare XSA modes systematically**

### Low (Documentation)

11. **Document the LAKER→Transformer adaptation** — since this fusion is novel, the design decisions need to be recorded
12. **Add convergence monitoring** to benchmark suite

---

## 5. Summary Table

| Component | Correctness | Completeness | Test Coverage | Notes |
|-----------|------------|-------------|---------------|-------|
| XSA core (`_subtract_projection`) | ✓ Correct | ✓ Complete | ✓ Tested | Matches paper |
| XSA core (`_zero_diagonal`) | ✓ Correct | ✓ Complete | ✓ Tested | Matches paper |
| LAKER kernel construction | ✗ Wrong | ✗ Missing | ✗ Not tested vs paper | Uses different kernel entirely |
| LAKER preconditioner | ✗ Wrong | ✗ Missing | ✗ Not tested vs paper | CCCP completely absent |
| LAKER solver | ⚠️ Suboptimal | ⚠️ Partial | ⚠️ PCG exists but unused | Should use PCG not Richardson |
| LAKER safeguards | ✗ Missing | ✗ Missing | ✗ N/A | ε, ρ, trace norm all absent |
| XSA+LAKER fusion | ⚠️ Novel | ✗ Unexplored | ✗ N/A | No published guidance |
| Full Transformer | ✓ Correct | ✓ Complete | ⚠️ Only synthetic tasks | Architecture is sound |
