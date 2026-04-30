# Mathematical Foundations

## 1. Exclusive Self Attention (XSA)

### Standard Self-Attention

Given queries Q, keys K, and values V:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

For token i, the output is:

$$y_i = \sum_j \text{softmax}(Q_i \cdot K_j / \sqrt{d}) \cdot V_j$$

This includes self-attention when j = i.

### XSA Projection Removal

XSA removes the component of y_i aligned with v_i:

$$y_i^{\text{XSA}} = y_i - \text{proj}_{v_i}(y_i)$$

Where the projection is:

$$\text{proj}_{v_i}(y_i) = \frac{y_i \cdot v_i}{v_i \cdot v_i + \epsilon} v_i$$

**Derivation:**

The projection of vector a onto vector b is:

$$\text{proj}_b(a) = \frac{a \cdot b}{||b||^2} b$$

We add ε for numerical stability. After subtraction:

$$y_i^{\text{XSA}} \cdot v_i = \left(y_i - \frac{y_i \cdot v_i}{v_i \cdot v_i} v_i\right) \cdot v_i = 0$$

Thus y_i^XSA is orthogonal to v_i.

### Alternative XSA Formulations

**Zero-diagonal attention:**
Set attention score diagonal to -∞ before softmax:

$$A_{ii} = -\infty \implies \text{softmax}(A)_{ii} = 0$$

**Mask-based exclusion:**
Use explicit binary mask M where M_{ii} = 0.

## 2. Kernel Attention Regression

### From Attention to Kernel Regression

Standard attention:

$$\text{output} = \text{softmax}(QK^T/\sqrt{d}) V$$

Kernel regression formulation:

$$(K + \lambda I)\alpha = V$$
$$\text{output} = K\alpha$$

Where K is a positive definite kernel matrix.

### Kernel Functions

**RBF (Gaussian) Kernel:**

$$k(x, y) = \exp\left(-\frac{||x - y||^2}{2\sigma^2}\right)$$

Properties:
- Positive definite
- Translation invariant
- Parameterized by bandwidth σ

**Linear Kernel:**

$$k(x, y) = x \cdot y + 1$$

Properties:
- Simple, fast computation
- May not capture complex relationships

**Cosine Kernel:**

$$k(x, y) = \frac{x \cdot y}{||x|| \cdot ||y||} + 1$$

Properties:
- Scale invariant
- Range [0, 2]

### Kernel Ridge Regression

The formulation (K + λI)α = V is kernel ridge regression with:
- Kernel matrix K
- Regularization λ
- Target values V

The closed-form solution is:

$$\alpha = (K + \lambda I)^{-1} V$$

For large sequences, we solve iteratively.

## 3. Preconditioned Iterative Solving

### Richardson Iteration

Basic Richardson iteration for Ax = b:

$$x_{t+1} = x_t + (b - Ax_t)$$

Converges when spectral radius ρ(I - A) < 1.

### Preconditioned Richardson

With preconditioner P ≈ A⁻¹:

$$x_{t+1} = x_t + P(b - Ax_t)$$

Converges faster when P better approximates A⁻¹.

### Our Preconditioner

We use:

$$P = \text{diag}(d) + UU^T$$

Where:
- d is learned per-head diagonal scaling
- U is low-rank factor (n × r)

**Application:**

$$Pr = d \odot r + U(U^T r)$$

Cost: O(nr) instead of O(n²) for full P.

### Convergence Analysis

For system (K + λI)α = V:

- Without preconditioning: convergence depends on condition number κ(K + λI)
- With preconditioning: effective condition number κ(P(K + λI))

The learned preconditioner adapts to the data distribution.

## 4. Conditioning Analysis

### Condition Number

For matrix A, the condition number is:

$$\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

High condition number → ill-conditioned → slow iterative convergence.

### Effect of Regularization

Adding λI to K:

$$\kappa(K + \lambda I) < \kappa(K)$$

Because eigenvalues shift: σ_i(K + λI) = σ_i(K) + λ

### Effect of Preconditioning

Ideal preconditioner P = (K + λI)⁻¹ gives:

$$\kappa(P(K + \lambda I)) = \kappa(I) = 1$$

Our learned P approximates this ideal.

## 5. Gradient Flow

### Differentiability of Iterative Solve

The unrolled Richardson iteration is differentiable:

$$\frac{\partial \alpha_T}{\partial \theta} = \sum_{t=0}^{T-1} \frac{\partial \alpha_{t+1}}{\partial \alpha_t} \cdots \frac{\partial \alpha_1}{\partial \theta}$$

Where θ includes kernel parameters and preconditioner weights.

### Gradient Stability

Potential issues:
- Vanishing gradients for many iterations
- Exploding gradients for ill-conditioned K

Mitigations:
- Limit iterations (default: 10)
- Clip α values during iteration
- Use learned preconditioner for better conditioning

## 6. Computational Complexity

| Operation | Standard | Fused XSA+LAKER |
|-----------|----------|-----------------|
| Q, K, V projection | O(n·d²) | O(n·d²) |
| Kernel/Scores | O(n²·d) | O(n²·d) |
| Solve | O(1) (direct) | O(T·n²·d) |
| Preconditioner | - | O(T·n·r·d) |
| **Total** | **O(n·d² + n²·d)** | **O(n·d² + T·n²·d)** |

Where:
- n = sequence length
- d = model dimension
- r = preconditioner rank
- T = number of iterations
