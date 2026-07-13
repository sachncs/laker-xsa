"""Configuration values shared by LAKER-XSA attention and solver modules.

The :class:`XSA_LAKER_Config` dataclass groups shape, kernel, exclusion,
preconditioner, and iterative-solver settings. Some retained compatibility
fields are consumed only by deprecated v1 paths, while ``use_fused`` and
``seed`` are currently metadata. Training uses the separate
:class:`laker_xsa.training.trainer.TrainingConfig`.

:meth:`XSA_LAKER_Config.__post_init__` derives ``head_dim`` and performs the
limited validation documented on that method; it does not validate every
cross-field relationship or numeric range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class XSA_LAKER_Config:
    """Hyperparameters for XSA + LAKER attention.

    The dataclass is the primary source of attention/solver
    hyperparameters for the attention and preconditioner modules.
    Construction runs :meth:`__post_init__`, which fills in the default
    ``head_dim`` and performs a limited set of validation checks (the
    categorical fields, the ``d_model`` / ``num_heads`` divisibility, and
    a few numeric ranges). It does **not** validate every field: for
    example ``d_model`` / ``num_heads`` positivity is not checked, and an
    explicitly supplied ``head_dim`` is stored as-is without checking it
    against ``d_model // num_heads``.

    The configuration separates three concerns:

    1. *Attention shape* (``d_model``, ``num_heads``, ``head_dim``,
       ``dropout``).
    2. *Self-exclusion and kernel formulation* (``xsa_mode``,
       ``kernel_type``, ``kernel_temperature``, ``kernel_symmetric``,
       ``kernel_normalize_qk``).
    3. *Linear-system solve and preconditioning* (``lambda_init``,
       ``preconditioner_type`` and its parameters, ``pcg_max_iterations``,
       ``pcg_tolerance``, ``num_iterations``, ``precond_update_frequency``,
       ``clip_abs``).

    Attributes:
        d_model: Token embedding dimension used by every projection in
            the attention block. Expected to be a positive integer
            divisible by ``num_heads`` (positivity itself is not
            validated).
        num_heads: Number of attention heads. ``d_model`` must be
            divisible by ``num_heads``.
        head_dim: Per-head dimension. When ``None`` (the default),
            ``__post_init__`` sets it to ``d_model // num_heads``. An
            explicitly supplied value is used verbatim and is not checked
            for consistency with ``d_model // num_heads``.
        dropout: Attention dropout probability. Must lie in ``[0.0, 1.0]``.
        eps: Numerical-stability epsilon. Must be strictly positive.
            Used in several denominators/regularizers (e.g. kernel score
            stabilization, RMS/projection normalization, and the positive
            ``lambda`` floor), but not literally in every division across
            the codebase.
        lambda_init: Initial value of the isotropic ridge regularizer
            added to the kernel matrix in the LAKER solve. Must be
            non-negative. It initializes a single scalar
            ``nn.Parameter`` per attention module (one value shared
            across tokens and heads, reparameterized via softplus), not a
            per-token value.
        kernel_type: Kernel selector used only by deprecated v1
            ``KernelFunction`` paths. ``"rbf"`` and the legacy
            ``"exp_attention"`` alias both select the RBF implementation;
            ``"linear"`` and ``"cosine"`` select their named kernels. The v2
            :class:`~laker_xsa.attention.laker.LakerAttention` always builds
            :class:`~laker_xsa.attention.kernels.AttentionKernel` and ignores
            this field.
        xsa_mode: Strategy used to remove the self-aligned component.
            It is consumed primarily by
            :class:`~laker_xsa.attention.ExclusiveSelfAttention`:

            * ``"subtract_projection"`` (default) subtracts each token's
              self-projection from the output; requires no masking.
            * ``"zero_diagonal"`` masks/zeroes the self-similarity on the
              diagonal.
            * ``"mask"`` excludes each token from attending to itself via
              an explicit self-mask.

            The v2 :class:`~laker_xsa.attention.laker.LakerAttention` only
            branches on ``"subtract_projection"`` (to apply output
            projection removal); it zeroes the kernel diagonal solely when
            no external mask is passed, independent of this field.

        use_fused: Compatibility metadata. No model or attention module reads
            this field. The bundled training CLI sets it only for
            ``attention_type == "fused"``, so it does not reliably encode the
            selected attention implementation.
        seed: Optional metadata retained on the configuration. Attention and
            preconditioner modules do not read it; callers must seed the random
            number generators explicitly when reproducibility is required.

        preconditioner_type: Which preconditioner to use for the
            PCG-based LAKER solve. One of:

            * ``"cccp"`` - angular-sampling and fixed-point preconditioner;
              its cost depends on ``cccp_num_directions`` and
              ``cccp_max_iterations``.
            * ``"fast"`` - learned low-rank-plus-diagonal preconditioner
              (default), with rank controlled by ``preconditioner_rank``.
            * ``"diagonal"`` - Jacobi-style diagonal preconditioner.
            * ``"none"`` - skip preconditioning; PCG relies purely on
              its iteration budget.

        preconditioner_rank: Low-rank dimension used by the
            ``"fast"`` preconditioner (default ``32``). ``None`` or
            ``0`` disables the low-rank factor entirely, leaving a
            diagonal-only preconditioner; no ``d_model``-proportional
            default is substituted. The CLI passes ``d_model // 16``
            explicitly.
        cccp_num_directions: Number of random angular directions ``N_r``
            sampled by the CCCP preconditioner per solve.
        cccp_max_iterations: Maximum number of CCCP fixed-point
            iterations.
        cccp_gamma: Nuclear norm regularization weight used by the CCCP
            update.
        cccp_shrinkage_rho: Initial isotropic shrinkage strength
            applied within CCCP.
        cccp_shrinkage_eps: Safeguard added to angular-sample norms,
            quadratic-form denominators, and trace normalization in the CCCP
            implementation. Its sign is not validated here.

        pcg_max_iterations: Hard upper bound on the number of PCG
            iterations performed for a single forward pass through the
            LAKER solve. PCG may stop earlier once
            ``pcg_tolerance`` is satisfied.
        pcg_tolerance: Relative-residual threshold used by PCG's early-stop
            check. A smaller value requests a stricter residual, but the solver
            may exhaust its iteration budget or return non-finite values without
            reporting convergence.
        num_iterations: Legacy alias kept for backwards compatibility
            with code written against an earlier configuration schema.
            The LAKER solve itself reads ``pcg_max_iterations``; the
            legacy ``KernelAttentionRegression`` paths in
            ``attention._legacy`` still consume this field directly.
            New code should prefer ``pcg_max_iterations``.

        precond_update_frequency: Cache refresh cadence for ``"cccp"`` and
            ``"fast"`` preconditioners. ``1`` recomputes every forward pass;
            values greater than ``1`` reuse the cached payload between refresh
            steps; non-positive values populate the cache once and then keep
            reusing it. The diagonal and identity modes do not use this cache.

        kernel_temperature: Divisor inside the v2 exponential kernel,
            ``exp(sim / T)``. For fixed similarities, a larger positive value
            compresses score differences and a smaller value amplifies them.
            The effective temperature is clamped by ``AttentionKernel``.
        kernel_symmetric: If ``True``, the kernel matrix is explicitly
            symmetrized (``0.5 * (K + K^T)``) before being used. This
            removes asymmetry but does **not** by itself guarantee
            positive (semi-)definiteness of the kernel.
        kernel_normalize_qk: If ``True``, ``Q`` and ``K`` are
            L2-normalized along the feature axis before kernel
            evaluation. This pairs naturally with ``"cosine"`` and
            ``"exp_attention"`` kernels.
        clip_abs: Absolute value clamp bound. It is consumed only by the
            legacy Richardson solves in
            :mod:`laker_xsa.attention._legacy` (which clamp the iterate to
            ``[-clip_abs, clip_abs]``). The v2 PCG path does not read this
            field; it uses a hardcoded ``stable_clip`` bound instead. The
            default (``1e6``) is intentionally large.
    """

    d_model: int
    num_heads: int
    head_dim: Optional[int] = None
    dropout: float = 0.0
    eps: float = 1e-6
    lambda_init: float = 3.0
    kernel_type: Literal["exp_attention", "rbf", "linear", "cosine"] = "exp_attention"
    xsa_mode: Literal["subtract_projection", "zero_diagonal", "mask"] = (
        "subtract_projection"
    )
    use_fused: bool = True
    seed: Optional[int] = None

    preconditioner_type: Literal["cccp", "fast", "diagonal", "none"] = "fast"
    preconditioner_rank: Optional[int] = 32
    cccp_num_directions: int = 64
    cccp_max_iterations: int = 20
    cccp_gamma: float = 0.1
    cccp_shrinkage_rho: float = 0.01
    cccp_shrinkage_eps: float = 1e-8

    pcg_max_iterations: int = 20
    pcg_tolerance: float = 1e-2
    num_iterations: int = 10

    precond_update_frequency: int = 1

    kernel_temperature: float = 1.0
    kernel_symmetric: bool = False
    kernel_normalize_qk: bool = True

    clip_abs: float = 1e6

    def __post_init__(self) -> None:
        """Validate and finalize the configuration.

        The post-init hook performs three classes of work:

        1. **Default filling.** If ``head_dim`` is ``None``, it is set
           to ``d_model // num_heads``.
        2. **Divisibility check.** ``d_model`` must be divisible by
           ``num_heads`` so that the heads can evenly split the
           embedding dimension.
        3. **Value validation.** The categorical fields
           (``kernel_type``, ``xsa_mode``, ``preconditioner_type``) are
           checked against their allowed values, and the numerical
           fields are range-checked (``dropout`` in ``[0, 1]``,
           ``eps > 0``, ``lambda_init >= 0``,
           ``pcg_max_iterations >= 1``, ``num_iterations >= 1``).

        This is not exhaustive: positivity of ``d_model`` / ``num_heads``
        is not checked, an explicitly supplied ``head_dim`` is not
        validated against ``d_model // num_heads``, and the many other
        numeric fields (e.g. the CCCP and ``pcg_tolerance`` values) are
        not range-checked here.

        Raises:
            ValueError: If a validated categorical or numerical field is
                outside its accepted range, or if ``d_model`` is not divisible
                by a nonzero ``num_heads``.
            ZeroDivisionError: If ``num_heads`` is zero while deriving
                ``head_dim`` or checking divisibility.
        """
        if self.head_dim is None:
            self.head_dim = self.d_model // self.num_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        valid_kernels = ("exp_attention", "rbf", "linear", "cosine")
        if self.kernel_type not in valid_kernels:
            raise ValueError(
                f"kernel_type must be one of {valid_kernels}, "
                f"got '{self.kernel_type}'"
            )

        valid_xsa = ("subtract_projection", "zero_diagonal", "mask")
        if self.xsa_mode not in valid_xsa:
            raise ValueError(
                f"xsa_mode must be one of {valid_xsa}, got '{self.xsa_mode}'"
            )

        valid_precond = ("cccp", "fast", "diagonal", "none")
        if self.preconditioner_type not in valid_precond:
            raise ValueError(
                f"preconditioner_type must be one of {valid_precond}, "
                f"got '{self.preconditioner_type}'"
            )

        if self.pcg_max_iterations < 1:
            raise ValueError("pcg_max_iterations must be >= 1")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError("dropout must be in [0,1]")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.lambda_init < 0:
            raise ValueError("lambda_init must be non-negative")

    @property
    def effective_pcg_iters(self) -> int:
        """Effective PCG iteration cap used by the LAKER solve.

        Returns:
            The current value of :attr:`pcg_max_iterations`. This is a
            direct accessor; it does **not** fall back to
            :attr:`num_iterations`. The legacy field is preserved for
            older call sites that consume it directly, but the
            modern LAKER solve path always uses ``pcg_max_iterations``.
        """
        return self.pcg_max_iterations
