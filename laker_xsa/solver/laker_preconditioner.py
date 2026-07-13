"""LAKER preconditioner via CCCP.

The CCCP-based preconditioner implements the angular-sampling pipeline already
cited in this repository under the LAKER reference
(``arXiv:2604.25138``). A user-facing preconditioner
:class:`LakerPreconditioner` provides ``"cccp"``, ``"fast"``, and
``"diagonal"`` modes; ``"none"`` is treated as identity by
:func:`pcg_solve` and :func:`richardson_solve`.

Notes on cache lifetime and ``state_dict``:

- :attr:`LakerPreconditioner.cached_preconditioner` is a plain Python attribute
  and is therefore not part of ``state_dict`` and is not moved by
  ``module.to(device)``. After ``load_state_dict`` or a device transfer, the
  cache is left in its prior state, including possibly a stale payload from a
  different device.
- :attr:`LakerPreconditioner.step_counter` is a registered buffer and so
  persists across ``load_state_dict`` calls.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, cast

import torch
from torch import nn
from torch.linalg import eigh

_SOFTPLUS = nn.Softplus()

PreconditionerData = Union[
    torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], None
]


class LakerPreconditioner(nn.Module):
    """Learned preconditioner for LAKER attention.

    Modes have different numerical properties. The ``"fast"`` form is
    positive-definite for finite values when its diagonal is strictly positive;
    the normal config path enforces ``eps > 0``. The CCCP form clamps
    eigenvalues before constructing ``P``. Neither property makes the separate
    system matrix ``K + lambda I`` satisfy PCG's SPD requirement, and the code
    does not validate that requirement.

    - ``'cccp'``: angular sampling, shrinkage-stabilized fixed-point updates,
      trace normalization, and ``P = Sigma^{-1/2}`` via eigendecomposition.
      The dominant cost is an :math:`O(n^3)` batched inverse per
      fixed-point update. Code paths here call :func:`torch.linalg.inv` and
      :func:`torch.linalg.eigh` unconditionally.
    - ``'fast'``: diagonal ``d`` plus low-rank factor ``U`` such that
      ``P = diag(d) + U U^T``. ``U`` may be signed: ``lr_base`` is
      unconstrained and ``lr_importance`` only enters as a
      non-negative ``softplus`` weighting that scales the per-rank
      components (a multiplicative magnitude, not a sign choice).
      ``U U^T`` is positive-semidefinite for any real ``U`` regardless
      of the sign of its entries; the softplus weighting does not need
      to be sign-aware. Applying ``P`` to a residual of feature width
      ``d`` costs :math:`O(n r d)` per batch/head.
    - ``'diagonal'``: returns only ``diag(d)``; no low-rank term.
    - ``'none'``: falls through :meth:`compute_preconditioner` and returns
      ``None``; callers such as :func:`pcg_solve` treat ``None`` as the
      identity preconditioner. The mode string is accepted so the
      configuration literal in :class:`XSA_LAKER_Config` can include it.

    The module keeps a forward-pass cache (:attr:`cached_preconditioner`)
    and a registered buffer (:attr:`step_counter`). The cache is a plain
    Python attribute and is not part of ``state_dict``; see the module-level
    docstring.

    The cache and counter mutate during ``forward``. Concurrent forwards on the
    same instance are therefore not thread-safe without external
    synchronization.

    Attributes:
        num_heads: Number of attention heads the preconditioner is
            shared over. Set at construction; not updated by forward.
        mode: Preconditioner mode, one of ``'cccp'``, ``'fast'``,
            ``'diagonal'`` or ``'none'``. ``'none'`` causes
            :meth:`compute_preconditioner` to return ``None``, which the
            downstream solvers treat as the identity preconditioner.
        rank: Low-rank dimension used by ``'fast'`` mode. ``None`` or
            ``0`` disables the low-rank factor. Has no effect in
            ``'cccp'`` or ``'diagonal'`` modes.
        gamma: Nuclear-norm-like regularization weight used inside the
            CCCP iteration to bias :math:`\\Sigma` toward :math:`I`.
            Larger values yield a more isotropic preconditioner.
         rho: Isotropic coefficient in
             :math:`\\tilde\\Sigma = (1 - \\rho) F + \\rho I`. Values are not
             constrained to ``[0, 1]``.

        eps_safeguard: Scalar added to denominators. Its sign is not
            validated; zero or negative values can produce non-finite or
            inverted safeguards.
        N_r: Number of random angular-sample directions generated per
            CCCP preconditioner build. Only used in ``'cccp'`` mode.
        max_cccp_iters: Maximum CCCP fixed-point iterations per
            preconditioner build. Only used in ``'cccp'`` mode.
        max_seq_len: ``'fast'`` mode's per-head basis buffer has a fixed
            ``num_heads x max_seq_len x rank`` size. Slicing
            ``lr_base[:, :seq_len, :]`` with ``seq_len > max_seq_len`` does
            not itself raise an ``IndexError`` (PyTorch returns the available
            prefix); the shape mismatch only surfaces later in the
            ``matmul`` / ``einsum`` of the downstream solve.
        eps: Lower bound applied to eigenvalues before
            :math:`Sigma^{-1/2}`. Its sign is not validated.

    Note:
        ``state_dict`` only stores :attr:`step_counter` and the learnable
        parameters (``diag_scale``, ``lr_base``, ``lr_importance`` depending
        on mode). The Python attribute :attr:`cached_preconditioner` is not
        part of ``state_dict``, and :meth:`load_state_dict` does not clear it.
    """

    def __init__(
        self,
        num_heads: int,
        mode: str = "fast",
        rank: Optional[int] = 32,
        gamma: float = 0.1,
        rho: float = 0.01,
        eps_safeguard: float = 1e-8,
        n_random_directions: int = 64,
        max_cccp_iters: int = 20,
        max_seq_len: int = 2048,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the LAKER preconditioner.

        Args:
            num_heads: Number of attention heads.
            mode: One of ``'cccp'``, ``'fast'``, ``'diagonal'`` or
                ``'none'``. ``'none'`` and any other unrecognized mode
                cause :meth:`compute_preconditioner` to return ``None``;
                the solvers interpret ``None`` as the identity
                preconditioner.
            rank: Low-rank dimension for ``'fast'`` mode. ``None`` or
                ``0`` disables the low-rank factor.
            gamma: CCCP nuclear-norm regularization weight.
            rho: CCCP isotropic shrinkage strength.
            eps_safeguard: Scalar added to denominators. Its sign is not
                validated.
            n_random_directions: Number of CCCP angular samples
                :math:`N_r`.
            max_cccp_iters: Number passed to ``range``. A non-positive
                value skips the fixed-point loop without an explicit
                check.
            max_seq_len: ``'fast'`` basis buffer size.
            eps: Lower bound on clamped eigenvalues.

        Side Effects:
            Allocates parameters and buffers. ``"fast"`` mode initializes
            ``lr_base`` randomly and advances PyTorch's global RNG state.
        """
        super().__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.rank = rank
        self.gamma = gamma
        self.rho = rho
        self.eps_safeguard = eps_safeguard
        self.N_r = n_random_directions
        self.max_cccp_iters = max_cccp_iters
        self.max_seq_len = max_seq_len
        self.eps = eps

        if mode == "fast" and rank is not None and rank > 0:
            self.diag_scale = nn.Parameter(torch.ones(1, num_heads, 1))
            self.lr_base = nn.Parameter(
                torch.randn(num_heads, max_seq_len, rank) * 0.01
            )
            self.lr_importance = nn.Parameter(torch.zeros(num_heads, rank))
        elif mode == "diagonal":
            self.diag_scale = nn.Parameter(torch.ones(1, num_heads, 1))
            self.register_buffer("lr_base", torch.empty(0, 0, 0))
            self.register_buffer("lr_importance", torch.empty(0, 0))
        else:
            self.register_buffer("diag_scale", torch.ones(1, 1, 1))
            self.register_buffer("lr_base", torch.empty(0, 0, 0))
            self.register_buffer("lr_importance", torch.empty(0, 0))

        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))
        self.cached_preconditioner: Optional[
            Union[
                Tuple[torch.Tensor],
                Tuple[torch.Tensor, Optional[torch.Tensor]],
            ]
        ] = None

    def generate_angular_samples(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
    ) -> PreconditionerData:
        """Generate angular samples :math:`\\bar u_k` from the kernel system.

        For each :math:`k = 1, \\dots, N_r`:

        .. math::

            z_k \\sim \\mathcal{N}(0, I),\\quad
            u_k = (\\lambda I + K)\\, z_k,\\quad
            \\bar u_k = u_k / \\|u_k\\|_2

        The vectors :math:`\\bar u_k` are normalized along the ``n`` dimension.
        The implementation does not validate that the resulting norms or
        samples are finite.

        Args:
            kernel: Kernel matrix with shape
                ``(batch, num_heads, n, n)``.
            lambda_reg: Regularization coefficient. Broadcastable to
                ``(batch, num_heads, 1, 1)`` so it scales the identity
                term in the operator application.

        Returns:
            torch.Tensor: Stacked angular samples with shape
            ``(N_r, batch, num_heads, n)``. The leading dim is the
            sample index; the trailing ``n`` corresponds to the
            sequence dimension.

        Raises:
            RuntimeError: Propagated from :func:`torch.matmul` or
                :func:`torch.linalg.vector_norm` for incompatible
                shapes, dtypes, or devices.

        Notes:
            - Each direction ``z`` is drawn independently with the
              same ``device`` and ``dtype`` as ``kernel`` so that the
              operator application ``(lambda*I + K) @ z`` stays on
              device and in dtype.
            - The normalize-by-``u_norm`` step adds
              ``eps_safeguard`` to avoid division by zero if ``u_k``
              is identically zero (which has probability zero for a
              continuous Gaussian but is included as a numerical
              safety).
            - The unit vectors are returned without the trailing
              column-vector axis (a trailing singleton inserted before
              the ``matmul`` so that ``z`` is treated as a column). The
              attention ``head_dim`` axis is unrelated; callers that
              need ``u`` as a column-vector should ``unsqueeze(-1)``
              the result.
            - ``lambda_reg`` is added component-wise to ``Kz``; its
              broadcast shape is whatever PyTorch accepts, not a single
              checked shape.
        """
        batch, num_heads, n, _ = kernel.shape
        device = kernel.device
        dtype = kernel.dtype

        ubar_list = []

        for _ in range(self.N_r):
            z = torch.randn(batch, num_heads, n, 1, device=device, dtype=dtype)

            Kz = torch.matmul(kernel, z)
            u = Kz + lambda_reg * z

            u_norm = torch.sqrt(torch.sum(u * u, dim=-2, keepdim=True))
            ubar = u / (u_norm + self.eps_safeguard)

            ubar_list.append(ubar.squeeze(-1))

        return torch.stack(ubar_list, dim=0)

    def cccp_iteration(
        self,
        ubar_samples: torch.Tensor,
        Sigma: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """Single CCCP fixed-point step for Tyler's M-estimator.

        Implements the shrinkage-stabilized update (Eq. 35-37 of the
        LAKER paper):

        .. math::

            F_\\gamma &= \\frac{1}{1 + \\gamma/n}\\left[
                \\frac{n}{N_r} \\sum_{k=1}^{N_r}
                \\frac{\\bar u_k \\bar u_k^T}
                     {\\bar u_k^T \\Sigma^{-1} \\bar u_k + \\epsilon}
                + \\gamma I
            \\right] \\\\
            \\tilde\\Sigma &= (1 - \\rho)\\, F_\\gamma + \\rho\\, I \\\\
            \\Sigma_{\\text{new}} &= \\tilde\\Sigma /
                \\mathrm{tr}(\\tilde\\Sigma)/n

        The ``1 / (1 + gamma/n)`` rescaling and the ``+ gamma I`` term
        implement the nuclear-norm-like regularization that biases the
        shape matrix toward ``I``; the ``+ rho I`` shrinkage stabilizes
        the iteration against noise from a finite number of samples;
        and the trace normalization fixes the scale of the estimator
        (Tyler's M-estimator is itself scale-invariant).

        Args:
            ubar_samples: Angular samples with shape
                ``(N_r, batch, num_heads, n)``. ``N_r`` must match
                ``self.N_r`` (it is read from this tensor's shape, not
                enforced).
            Sigma: Current matrix with shape
                ``(batch, num_heads, n, n)``. The update preserves symmetry for
                finite symmetric input. Positive-definiteness depends on the
                supplied samples and unconstrained ``gamma``, ``rho``, and
                safeguard values.
            n: Sequence length, equal to ``Sigma.shape[-1]``. Passed in
                explicitly so the function can build identity matrices
                of the correct size.

        Returns:
            torch.Tensor: Updated :math:`\\Sigma_{\\text{new}}` with
            shape ``(batch, num_heads, n, n)``.

        Notes:
            - The method computes :func:`torch.linalg.inv`, which is
              :math:`O(n^3)` per call and is invoked once per fixed-point
              iteration.
            - The outer products ``u u^T`` and the scalar denominators
              are computed with batched ``einsum`` and broadcasting so
              the per-iteration cost scales as
              :math:`O(N_r \\cdot \\text{batch} \\cdot
              \\text{num\\_heads} \\cdot n^2)`.
            - ``eps_safeguard`` is added to the per-sample denominator
              ``\\bar u_k^T \\Sigma^{-1} \\bar u_k``. It reduces division-by-zero
              risk when positive, but its sign and finite value are not checked.
            - The identity matrices are constructed once with
              ``unsqueeze(0).unsqueeze(0)`` so that broadcasting
              handles the per-(batch, head) expansion.
            - The trace normalization divides by ``tr(Sigma_tilde) /
              n`` rather than by ``tr(Sigma_tilde)`` to preserve the
              average-eigenvalue constraint ``tr(Sigma) / n = 1`` that
              the estimator targets.
        """
        batch, num_heads = Sigma.shape[:2]
        device = Sigma.device
        dtype = Sigma.dtype

        gamma = self.gamma
        N_r = ubar_samples.shape[0]

        Sigma_inv = torch.linalg.inv(Sigma)  # pylint: disable=not-callable

        denom_sum = torch.zeros(batch, num_heads, n, n, device=device, dtype=dtype)

        for k in range(N_r):
            u = ubar_samples[k]

            Su = torch.matmul(Sigma_inv, u.unsqueeze(-1)).squeeze(-1)
            # Batched reduction of u^T Sigma^{-1} u over the n axis.
            denom = (u * Su).sum(dim=-1)

            outer = torch.einsum("...i,...j->...ij", u, u)

            # Broadcasting: per-(batch, head) denom lifted to (..., 1, 1)
            # to divide the per-(batch, head) outer product.
            denom_sum = denom_sum + outer / (
                denom.unsqueeze(-1).unsqueeze(-1) + self.eps_safeguard
            )

        scale = n / N_r
        # Broadcasting: per-(n, n) identity is unsqueezed to (1, 1, n, n)
        # so the (gamma * I) term expands over (batch, num_heads).
        F_gamma = scale * denom_sum + gamma * torch.eye(
            n, device=device, dtype=dtype
        ).unsqueeze(0).unsqueeze(0)
        F_gamma = F_gamma / (1.0 + gamma / n)

        rho = self.rho
        # Shrinkage: bias Sigma toward I for numerical stability of the
        # subsequent eigendecomposition; (1 - rho) F + rho I.
        Sigma_tilde = (1.0 - rho) * F_gamma + rho * torch.eye(
            n, device=device, dtype=dtype
        ).unsqueeze(0).unsqueeze(0)

        trace = torch.diagonal(Sigma_tilde, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        # Trace normalization: divide by tr/n so tr(Sigma) = n exactly;
        # preserves the average-eigenvalue constraint of Tyler's estimator.
        Sigma_new = Sigma_tilde * n / (trace.unsqueeze(-1) + self.eps_safeguard)

        return Sigma_new

    def cccp_preconditioner(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
    ) -> torch.Tensor:
        """Full CCCP pipeline: angular samples :math:`\\to` CCCP :math:`\\to` :math:`P = \\Sigma^{-1/2}`.

        Performs the following:

        1. Generate ``N_r`` angular samples from the kernel system.
        2. Initialize :math:`\\Sigma = I` (broadcast over batch and
           head).
        3. Run :meth:`cccp_iteration` exactly ``max_cccp_iters`` times. The
           update builds :math:`\\Sigma` from outer products,
           :math:`\\gamma I` and :math:`\\rho I`, and scalar normalization.
           These operations preserve symmetry for finite values; the method
           does not check fixed-point convergence.
        4. Eigendecompose the symmetric :math:`\\Sigma`, clamp the
           eigenvalues below ``eps``, and assemble
           :math:`P = V\\, \\mathrm{diag}(\\lambda^{-1/2})\\, V^T`.

        Args:
            kernel: Kernel matrix with shape
                ``(batch, num_heads, n, n)``. Used only inside
                :meth:`generate_angular_samples` to build
                :math:`(K + \\lambda I)\\, z`; the kernel itself is
                not passed to :func:`torch.linalg.eigh` and need not
                be symmetric.
            lambda_reg: Regularization coefficient, broadcastable to
                ``(batch, num_heads, 1, 1)``.

        Returns:
            Matrix with shape ``(batch, num_heads, n, n)`` assembled as
            :math:`P = V\\, \\mathrm{diag}(\\lambda^{-1/2})\\, V^T` after clamping
            eigenvalues with ``min=self.eps``. Positive ``eps`` and finite
            decomposition outputs produce positive clamped eigenvalues; direct
            callers can configure values that violate those conditions.


        Notes:
            - The eigendecomposition uses :func:`torch.linalg.eigh`
              on the symmetric :math:`\\Sigma` constructed by the
              CCCP iteration, not on the (possibly nonsymmetric)
              ``kernel``. Each additive term in
              :meth:`cccp_iteration` is symmetric, so :math:`\\Sigma`
              is symmetric for any real input.
            - Eigenvalues are clamped to ``[eps, +inf)`` to dampen
              small-magnitude noise. NaNs in the eigenvalues are not
              separately handled and would propagate to ``P``.
            - The full iteration unconditionally iterates
              ``self.max_cccp_iters`` times with no internal convergence
              check and no tolerance on the change in ``Sigma``.
            - The dominant per-iteration cost is
              :func:`torch.linalg.inv` of :math:`\\Sigma`, which is
              :math:`O(n^3)` per (batch, head); the outer-product
              accumulation over ``N_r`` samples contributes
              :math:`O(N_r \\cdot \\text{batch} \\cdot \\text{num\\_heads}
              \\cdot n^2)`.
        """
        batch, num_heads, n, _ = kernel.shape
        device = kernel.device
        dtype = kernel.dtype

        ubar_samples = cast(
            torch.Tensor, self.generate_angular_samples(kernel, lambda_reg)
        )

        eye = torch.eye(n, device=device, dtype=dtype)
        Sigma = eye.unsqueeze(0).unsqueeze(0).expand(batch, num_heads, -1, -1).clone()

        for _ in range(self.max_cccp_iters):
            Sigma_new = self.cccp_iteration(ubar_samples, Sigma, n)
            Sigma = Sigma_new

        eigenvalues, eigenvectors = eigh(Sigma)  # pylint: disable=not-callable

        eigenvalues = torch.clamp(eigenvalues, min=self.eps)

        inv_sqrt_eig = eigenvalues.pow(-0.5)
        P = eigenvectors @ (inv_sqrt_eig.unsqueeze(-1) * eigenvectors.transpose(-2, -1))

        return cast(torch.Tensor, P)

    def fast_preconditioner(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fast gradient-based preconditioner (low-rank + diagonal).

        Returns ``(diag, lr_factor)`` for the ``"fast"`` form
        ``P = diag(d) + U U^T``. The implementation computes

            d = softplus(kernel_diag + lambda_reg) * |diag_scale| + eps
            U = lr_base[:, :seq_len, :] * softplus(lr_importance)

        and returns ``(d, U)`` (with ``U`` broadcast over the batch
        dimension). ``lr_base`` itself is unconstrained; only the
        per-rank ``lr_importance`` is wrapped in ``softplus`` to scale
        the rank components by a non-negative magnitude.

        Args:
            kernel: Kernel matrix with shape
                ``(batch, num_heads, n, n)``. Only ``kernel.diagonal``
                is read.
            lambda_reg: Regularization coefficient. The implementation
                squeezes the trailing two dims (``(-1, -1)``) so this
                should broadcast to a per-(batch, head) scalar.
            seq_len: Actual sequence length used to slice the first
                ``seq_len`` rows of the low-rank basis buffer
                ``lr_base[:, :seq_len, :]``. The method does not
                check ``seq_len`` against ``max_seq_len``; passing
                ``seq_len > max_seq_len`` does not itself raise, but
                the slice returns a prefix of length ``max_seq_len``
                and the downstream matmul / apply step fails with a
                shape mismatch.

        Returns:
            Tuple ``(diag_precond, lr_factor)``:

            - ``diag_precond`` has shape ``(batch, num_heads, n)`` and equals
              ``softplus(kernel_diag + lambda_reg) * abs(diag_scale) + eps``.
              With finite values and the normal config path's ``eps > 0``, it is
              strictly positive; direct construction does not validate ``eps``.
            - ``lr_factor`` has shape
              ``(batch, num_heads, min(seq_len, max_seq_len), rank)`` when the
              low-rank component is active; ``None`` otherwise. Overlength
              factors fail only when later combined with a longer residual.

        Notes:
            - The diagonal formula is exactly
              ``softplus(kernel_diag + lambda_reg) * abs(diag_scale) + eps``.
              The config-path guarantee of a strictly positive diagonal
              holds because :class:`laker_xsa.config.XSA_LAKER_Config`
              constrains ``eps > 0`` (see
              :meth:`XSA_LAKER_Config.__post_init__`). Direct callers
              who construct :class:`LakerPreconditioner` with a
              non-positive ``eps`` bypass that check.
            - ``lr_factor`` is ``unsqueeze(0).expand(batch, ...)``,
              sharing storage with the per-head basis.
            - The full ``P`` is not formed; applying it to a residual
              of feature width ``d`` costs
              :math:`O(n r d)` per batch/head via the factored
              ``diag(d) r + U (U^T r)`` form.
        """
        batch = kernel.shape[0]

        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        # Diagonal formula: softplus(kernel_diag + lambda) * |diag_scale| + eps.
        # softplus is strictly positive for finite input, |diag_scale| is
        # non-negative, and the user-supplied eps may be any sign. The
        # config path guarantees eps > 0, which is the only knob that can
        # make this expression non-positive for direct callers.
        diag = _SOFTPLUS(kernel_diag + lambda_reg.squeeze(-1).squeeze(-1))
        # |diag_scale| prevents the learnable scalar from flipping the
        # sign of the softplus output; the trailing + eps is a floor whose
        # sign is not validated.
        diag = diag * self.diag_scale.abs() + self.eps

        lr_factor = None
        if self.rank is not None and self.rank > 0 and self.lr_base.numel() > 0:
            lr_pos = self.lr_base[:, :seq_len, :]

            # softplus on lr_importance yields non-negative per-rank
            # weights. U U^T is positive-semidefinite for any real U, so
            # the non-negativity of importance only controls the
            # per-rank magnitude of the U U^T term, not its sign or PSD
            # status.
            importance = _SOFTPLUS(self.lr_importance)

            # Broadcasting: importance has shape (num_heads, rank) and is
            # unsqueezed to (num_heads, 1, rank) to scale each position.
            lr_scaled = lr_pos * importance.unsqueeze(1)

            # Broadcasting: share the same (num_heads, seq_len, rank)
            # basis across the batch dimension.
            lr_factor = lr_scaled.unsqueeze(0).expand(batch, -1, -1, -1)

        return diag, lr_factor

    def diagonal_preconditioner(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
    ) -> PreconditionerData:
        """Compute a Jacobi-style diagonal preconditioner.

        Returns :math:`d` of shape ``(batch, num_heads, n)`` computed
        as the same formula used by :meth:`fast_preconditioner` for
        the diagonal entry:

            d = softplus(kernel_diag + lambda_reg) * |diag_scale| + eps

        No caching is performed for this mode; the diagonal is rebuilt
        on every call.

        Args:
            kernel: Kernel matrix with shape
                ``(batch, num_heads, n, n)``. Only the diagonal is read.
            lambda_reg: Regularization coefficient, broadcastable to
                a per-(batch, head) scalar after squeezing the trailing
                two dims.

        Returns:
            ``d`` of shape ``(batch, num_heads, n)``. For finite inputs
            the only term that can be non-positive is the
            user-supplied ``eps``; the config-path guarantee of a
            strictly positive diagonal comes from
            :class:`laker_xsa.config.XSA_LAKER_Config` constraining
            ``eps > 0``. This does not by itself make the surrounding
            system SPD.
        """
        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag = _SOFTPLUS(kernel_diag + lambda_reg.squeeze(-1).squeeze(-1))
        return cast(PreconditionerData, diag * self.diag_scale.abs() + self.eps)

    def compute_preconditioner(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
        seq_len: int,
        force_update: bool = False,
        update_frequency: int = 1,
    ) -> PreconditionerData:
        """Build (or fetch cached) preconditioner payload.

        The cache is updated if any of the following holds:

        - ``force_update`` is ``True``.
        - The cache is empty (e.g. on the first call; ``load_state_dict``
          does **not** clear this plain-Python cache).
        - ``update_frequency > 0`` and ``step_counter[0] % update_frequency
          == 0``.

        When ``update_frequency <= 0`` (including the documented
        ``0``), ``should_update`` is permanently ``False`` and the cache
        is only populated on the first call; every subsequent call
        reuses the cached payload.

        The step counter is **not** incremented here; it is incremented
        by :meth:`forward`. Despite that, this method is **not** side
        effect free: it mutates :attr:`cached_preconditioner` for
        ``"cccp"`` and ``"fast"`` modes whenever the cache is stale
        or empty. Treat it as an idempotent-from-the-caller's-perspective
        call only with respect to :attr:`step_counter`.

        Args:
            kernel: Kernel matrix with shape
                ``(batch, num_heads, n, n)``.
            lambda_reg: Regularization coefficient.
            seq_len: Actual sequence length.
            force_update: If ``True``, always rebuild and overwrite the
                cache. This is the documented way to invalidate the
                cache after a ``load_state_dict`` call or a
                ``module.to(device)`` move, neither of which clears the
                Python attribute automatically.
            update_frequency: Recompute cadence. ``1`` (default) means
                rebuild on every call; ``N > 1`` means rebuild when
                ``step_counter % N == 0``; ``0`` (or any non-positive
                value) leaves the cache frozen after the first
                computation and reuses it forever after.

        Returns:
            For ``"cccp"``, the cached preconditioner matrix ``P`` (a
            single tensor) — the implementation stores it internally as
            a 1-tuple but returns the unwrapped tensor. For ``"fast"``,
            the cached ``(diag, lr_factor)`` tuple. For ``"diagonal"``,
            a fresh ``diag`` tensor is computed without touching the
            cache. For ``"none"`` and any other unrecognized mode,
            ``None``.

        Raises:
            RuntimeError: Propagated from the selected construction for
                incompatible shapes, dtypes, devices, singular inverses, or
                failed decompositions. An overlength fast factor is returned
                truncated here and fails only when later applied to a longer
                residual.

        Side Effects:
            Mutates :attr:`cached_preconditioner` for ``"cccp"`` and
            ``"fast"`` modes whenever the cache is stale or empty.

        Notes:
            - The cache is a plain Python attribute
              (:attr:`cached_preconditioner`) and is not part of
              ``state_dict``. ``load_state_dict`` and
              ``module.to(device)`` do not clear or move it; a stale
              payload from a previous device may be observed
              otherwise.
            - When ``update_frequency <= 0``, the ``should_update``
              boolean is always ``False`` and the cache is reused on
              every call. The first call still stores a freshly built
              payload.
        """
        should_update = force_update or (
            update_frequency > 0 and (int(self.step_counter[0]) % update_frequency == 0)
        )

        # Cache cadence: rebuild when should_update is True (forced or
        # step_counter % update_frequency == 0 with update_frequency > 0)
        # or when the cache is still None. With update_frequency <= 0,
        # should_update is permanently False and the cache is populated
        # only on the first call.

        if self.mode == "cccp":
            if should_update or self.cached_preconditioner is None:
                P = self.cccp_preconditioner(kernel, lambda_reg)
                self.cached_preconditioner = (P,)
            return cast(Tuple[torch.Tensor], self.cached_preconditioner)[0]

        if self.mode == "fast":
            if should_update or self.cached_preconditioner is None:
                diag, lr = self.fast_preconditioner(kernel, lambda_reg, seq_len)
                self.cached_preconditioner = (diag, lr)
            return cast(
                Tuple[torch.Tensor, Optional[torch.Tensor]],
                self.cached_preconditioner,
            )

        if self.mode == "diagonal":
            return self.diagonal_preconditioner(kernel, lambda_reg)

        return None

    def apply_preconditioner(
        self,
        residual: torch.Tensor,
        precond_data: PreconditionerData,
    ) -> torch.Tensor:
        """Apply the preconditioner: compute :math:`P\\, r`.

        Args:
            residual: Residual tensor with shape
                ``(batch, num_heads, n, head_dim)``.
            precond_data: Payload from
                :meth:`compute_preconditioner`. ``None`` is treated as
                the identity preconditioner.

        Returns:
            torch.Tensor: Preconditioned residual, same shape and
            dtype as ``residual``.

        Raises:
            RuntimeError: Propagated from :func:`torch.matmul` or
                elementwise operations for incompatible shapes,
                dtypes, or devices. The method does not validate
                ``precond_data`` against ``self.mode`` beyond
                dispatching on ``self.mode``.

        Notes:
            - ``'fast'`` mode applies
              :math:`r \\mapsto \\mathrm{diag}(d)\\, r + U\\,(U^T\\,r)`,
              which evaluates :math:`(UU^T) r` as
              :math:`U (U^T r)` to avoid forming the :math:`n \\times n`
              matrix explicitly.
            - ``'diagonal'`` mode is purely element-wise
              (``residual * diag.unsqueeze(-1)``).
            - ``'cccp'`` mode applies a full ``matmul(P, residual)``;
              the per-call cost is :math:`O(n^2 d)` per (batch, head)
              for the matmul alone.
            - If ``precond_data`` is ``None``, ``residual`` is returned
              unchanged, so the caller can transparently use this
              function whether or not preconditioning is active.
            - If ``self.mode`` is not recognized, ``residual`` is
              returned unchanged.
        """
        if precond_data is None:
            return residual

        if self.mode == "cccp":
            P = cast(torch.Tensor, precond_data)
            return torch.matmul(P, residual)

        if self.mode == "fast":
            diag, lr_factor = cast(
                Tuple[torch.Tensor, Optional[torch.Tensor]], precond_data
            )
            # Broadcasting: diag has shape (batch, num_heads, n) and is
            # unsqueezed to (..., 1) so it multiplies each head_dim slot.
            out = residual * diag.unsqueeze(-1)
            if lr_factor is not None:
                UT_r = torch.matmul(lr_factor.transpose(-2, -1), residual)
                out = out + torch.matmul(lr_factor, UT_r)
            return cast(torch.Tensor, out)

        if self.mode == "diagonal":
            return cast(
                torch.Tensor, residual * cast(torch.Tensor, precond_data).unsqueeze(-1)
            )

        return residual

    def forward(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
        seq_len: int,
        force_update: bool = False,
        update_frequency: int = 1,
    ) -> PreconditionerData:
        """Build / refresh the preconditioner and bump the step counter.

        This is the primary :class:`torch.nn.Module` entry point. It
        delegates to :meth:`compute_preconditioner` and then increments
        :attr:`step_counter`, so subsequent calls observe the cadence
        encoded in ``update_frequency``.

        Args:
            kernel: Kernel matrix with shape
                ``(batch, num_heads, n, n)``.
            lambda_reg: Regularization coefficient.
            seq_len: Actual sequence length.
            force_update: Bypass the cache and always rebuild.
            update_frequency: Recompute cadence (see
                :meth:`compute_preconditioner`).

        Returns:
            PreconditionerData: Same payload contract as
            :meth:`compute_preconditioner`.

        Raises:
            RuntimeError: Propagated from :meth:`compute_preconditioner`
                (see that method's ``Raises`` section).

        Side Effects:
            Increments :attr:`step_counter` by one in place. The
            underlying :meth:`compute_preconditioner` may mutate
            :attr:`cached_preconditioner` for ``"cccp"`` and
            ``"fast"`` modes.
        """
        precond_data = self.compute_preconditioner(
            kernel, lambda_reg, seq_len, force_update, update_frequency
        )
        # step_counter drives the cache cadence in compute_preconditioner.
        self.step_counter.add_(1)
        return precond_data

    step_counter: torch.Tensor
    diag_scale: torch.Tensor
    lr_base: torch.Tensor
    lr_importance: torch.Tensor
