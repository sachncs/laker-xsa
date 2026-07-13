"""Fused XSA + LAKER attention.

Combines attention behavior associated with the two references already cited by
this repository:

* XSA (``arXiv:2603.09078``), implemented here through kernel-diagonal removal
  and optional output projection subtraction.
* LAKER (``arXiv:2604.25138``), implemented here as attention-kernel regression
  with a configurable preconditioner and iterative solve.

Pipeline (per head; see :meth:`LakerAttention.compute_attention`):

1. Project the input to ``Q``, ``K``, and ``V`` in the base class.
2. Compute ``K_{ij} = exp(sim(q_i, k_j) / temperature)``.
3. Multiply by a supplied mask. If no mask is supplied, zero the kernel
   diagonal regardless of ``xsa_mode``.
4. Compute the positive regularization
   ``lambda = softplus(raw_lambda) + eps``. Positivity alone does not make
   ``K + lambda * I`` symmetric or positive-definite.
5. Build or reuse the configured preconditioner.
6. Apply :func:`~laker_xsa.solver.conjugate_gradient.pcg_solve` to
   ``(K + lambda*I) @ alpha = V``. PCG requires an SPD operator and suitable
   preconditioner, but this path does not check those conditions.
7. Clamp and RMS-normalize ``alpha``.
8. If ``xsa_mode == "subtract_projection"``, subtract a scaled, regularized
   projection of ``alpha`` onto each corresponding value vector.
9. Merge heads and apply the output projection in the base class.

Backward compatibility:
    The aliases ``FusedXSALAKERAttentionV2`` and ``XSALAKERAttentionV2`` are
    re-exported from :mod:`laker_xsa.attention.fused_attention_v2`;
    ``LakerAttention`` replaces the deprecated v1 ``FusedXSALAKERAttention``
    in :mod:`laker_xsa.attention._legacy`.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, cast

import torch
from torch import nn
from torch.nn.functional import softplus

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.attention.core import (
    BaseMultiHeadAttention,
    stable_clip,
)
from laker_xsa.attention.kernels import AttentionKernel
from laker_xsa.solver.laker_preconditioner import LakerPreconditioner
from laker_xsa.solver.conjugate_gradient import pcg_solve

logger = logging.getLogger(__name__)


class LakerAttention(BaseMultiHeadAttention):
    """Fused XSA + LAKER attention (v2).

    Combines kernel-regression inverse mixing with diagonal removal and optional
    output projection subtraction. The implementation uses PCG without
    validating that its matrix and preconditioner satisfy the SPD assumptions
    required by that algorithm. Its preconditioner mutates a cache and step
    counter during ``forward``, so concurrent calls on one instance require
    external synchronization.

    Attributes:
        kernel_fn: :class:`~laker_xsa.attention.kernels.AttentionKernel`
            producing the exponential kernel matrix with score clamping. Some
            dtypes can still overflow during exponentiation.
        preconditioner: :class:`~laker_xsa.solver.laker_preconditioner.LakerPreconditioner`
            used by the inner PCG solve; its mode follows
            ``config.preconditioner_type`` (``"cccp"``, ``"fast"``,
            ``"diagonal"`` or ``"none"``).
        raw_lambda: ``nn.Parameter`` backing :attr:`lambda_reg`; softplus plus
            ``config.eps`` keeps the effective regularisation positive but
            does not establish symmetry or positive-definiteness of the
            regularized kernel.
        xsa_scale: ``nn.Parameter`` for the projection-removal strength when
            ``xsa_mode == "subtract_projection"``; for other modes it is a
            non-trainable buffer of ones (state-dict compatible).

    Input:  ``(batch, seq_len, d_model)``
    Output: ``(batch, seq_len, d_model)``
    """

    def __init__(self, config: XSA_LAKER_Config) -> None:
        """Initialise the fused XSA + LAKER attention module.

        Builds the :class:`AttentionKernel`, the
        :class:`LakerPreconditioner`, the learnable regulariser, and
        the projection-removal scale from the supplied configuration,
        then calls :meth:`init_weights` to initialise the
        Q/K/V/output projection weights.

        Args:
            config: :class:`laker_xsa.config.XSA_LAKER_Config` supplying shape,
                kernel, preconditioner, regularization, and XSA settings used by
                this implementation. Compatibility fields such as
                ``kernel_type``, ``use_fused``, ``seed``, and ``clip_abs`` are
                not consumed here.

        Side Effects:
            Allocates parameters and buffers, initializes projection weights
            from PyTorch's global RNG, and creates the preconditioner cache and
            step counter. ``xsa_scale`` is trainable only for
            ``"subtract_projection"``; otherwise it is an unused buffer.
        """
        super().__init__(config)

        self.kernel_fn = AttentionKernel(
            head_dim=self.head_dim,
            temperature=config.kernel_temperature,
            symmetric=config.kernel_symmetric,
            learnable_temperature=True,
            normalize_qk=config.kernel_normalize_qk,
            eps=config.eps,
        )

        self.preconditioner = LakerPreconditioner(
            num_heads=config.num_heads,
            mode=config.preconditioner_type,
            rank=config.preconditioner_rank,
            gamma=config.cccp_gamma,
            rho=config.cccp_shrinkage_rho,
            eps_safeguard=config.cccp_shrinkage_eps,
            n_random_directions=config.cccp_num_directions,
            max_cccp_iters=config.cccp_max_iterations,
            eps=config.eps,
        )

        self.raw_lambda = nn.Parameter(torch.tensor(config.lambda_init))

        if config.xsa_mode == "subtract_projection":
            self.xsa_scale = nn.Parameter(torch.ones(1))
        else:
            # ``xsa_scale`` is unused outside projection mode. A buffer keeps
            # it in module state and moves it with the module.
            self.register_buffer("xsa_scale", torch.ones(1))

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise the Q/K/V/output projections with a Gaussian.

        Each projection weight is sampled with mean ``0`` and standard
        deviation ``0.02 / sqrt(2)``.

        Side Effects:
            Mutates the ``weight`` tensors of
            ``self.qkv_proj.w_q``, ``self.qkv_proj.w_k``,
            ``self.qkv_proj.w_v`` and ``self.w_o`` in place via
            :func:`torch.nn.init.normal_`.
        """
        std = 0.02 / math.sqrt(2.0)
        for proj in [self.qkv_proj.w_q, self.qkv_proj.w_k, self.qkv_proj.w_v, self.w_o]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)

    @property
    def lambda_reg(self) -> torch.Tensor:
        """Return ``softplus(raw_lambda) + config.eps``.

        This parameterization makes the scalar positive. It does not
        guarantee that ``K + lambda * I`` is SPD when ``K`` is
        nonsymmetric or indefinite.

        Returns:
            Scalar 0-d :class:`torch.Tensor` of dtype matching
            ``raw_lambda``, equal to ``softplus(raw_lambda) + config.eps``.
        """
        # pylint: disable-next=not-callable
        return softplus(self.raw_lambda) + self.config.eps

    def zero_diagonal(self, kernel: torch.Tensor) -> torch.Tensor:
        """Zero the kernel diagonal — XSA diagonal removal.

        Args:
            kernel: Kernel matrix of shape
                ``(batch, num_heads, seq_len, seq_len)``.

        Returns:
            New tensor with ``kernel[..., i, i] == 0`` for every ``i``; the
            off-diagonal entries are unchanged. The operation does not modify
            ``kernel`` in place.
        """
        _, _, n, _ = kernel.shape
        diag_mask = torch.eye(n, device=kernel.device, dtype=kernel.dtype)
        diag_mask = diag_mask.view(1, 1, n, n)
        return kernel * (1.0 - diag_mask)

    def clean_self_projection(
        self, output: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Subtract a regularized self-value projection from each output.

        Implements (per token ``i``)

            y_i^xsa = y_i - scale * (y_i · v_i) / (v_i · v_i + eps) * v_i

        with ``scale = self.xsa_scale``. Because ``eps`` is added to the
        denominator and ``scale`` is a learnable parameter, the result
        is a regularized approximation of the orthogonal projection
        subtraction: it coincides with the exact subtraction only in
        the limiting case ``eps == 0`` and ``scale == 1``. For any
        other setting a residual component along ``v_i`` is expected
        to remain.

        Args:
            output: Tensor of shape
                ``(batch, num_heads, seq_len, head_dim)`` (typically the PCG
                solution ``alpha`` or a weighted sum of values).
            values: ``v`` of shape
                ``(batch, num_heads, seq_len, head_dim)`` — the value vectors
                to project against.

        Returns:
            Tensor with the scaled regularized projection subtracted. The
            self-aligned component need not be eliminated exactly, and
            cross-token contributions aligned with the same value direction
            are also affected.
        """
        dot = (output * values).sum(dim=-1, keepdim=True)
        v_norm_sq = (values * values).sum(dim=-1, keepdim=True) + self.config.eps
        return output - self.xsa_scale * (dot / v_norm_sq) * values

    def rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by the per-batch, per-head root mean square.

        The RMS is computed across sequence and feature dimensions. Adding
        ``config.eps`` inside the square root avoids division by zero.

        Args:
            x: Tensor of shape
                ``(batch, num_heads, seq_len, head_dim)``.

        Returns:
            Normalised tensor with the same shape and dtype as ``x``.
        """
        rms = torch.sqrt((x * x).mean(dim=(-2, -1), keepdim=True) + self.config.eps)
        return x / rms

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute fused XSA + LAKER attention per head.

        The pipeline is documented at module level. If the PCG call
        raises ``RuntimeError`` (e.g. an upstream ``torch.linalg`` op
        fails on a degenerate matrix), this method logs a warning and
        retries the same regularized system with
        :func:`torch.linalg.solve` on the dense kernel
        ``K + lambda * I``. That dense solve can itself raise
        ``RuntimeError``; the exception is not caught here and will
        propagate to the caller. PCG can additionally return a
        non-finite or inaccurate tensor without raising, in which case
        this fallback is not used; the method does not detect NaN/Inf
        in the PCG result.

        Args:
            q: Queries ``(batch, num_heads, seq_len, head_dim)`` produced by
                the base-class QKV projections.
            k: Keys of the same shape.
            v: Values of the same shape.
            mask: Optional 3- or 4-D mask broadcastable to ``(batch, num_heads,
                seq_len, seq_len)``; ``None`` for bidirectional attention.

        Returns:
            Per-head inverse-mixing output ``alpha`` of shape
            ``(batch, num_heads, seq_len, head_dim)`` after XSA output
            cleaning (when ``xsa_mode == "subtract_projection"``) and RMS
            normalisation. The base class will then merge heads and apply
            ``w_o``.

        Raises:
            RuntimeError: Propagated by kernel construction, masking,
                preconditioner construction, PCG setup, or the dense fallback
                for incompatible inputs or failed linear algebra. A
                ``RuntimeError`` raised inside PCG is intercepted once; a
                fallback error propagates.

        Side Effects:
            Calls the underlying
            :class:`~laker_xsa.solver.laker_preconditioner.LakerPreconditioner`,
            which may update its :attr:`cached_preconditioner` and
            increments its :attr:`step_counter` (see
            :meth:`LakerPreconditioner.forward`). On the PCG-raises
            fallback, a single warning is logged with the current
            sequence length and ``lambda`` value.

        Numerical notes:
            A typical causal mask retains the diagonal and zeros entries above
            it, producing a nonsymmetric lower-triangular kernel. This method
            skips its own diagonal removal whenever any mask is supplied; it
            neither infers mask semantics nor checks symmetry or
            positive-definiteness. With no mask, it always zeros the diagonal,
            independently of ``xsa_mode``. PCG requires SPD inputs, so callers
            must ensure the resulting operator and preconditioner meet PCG's
            assumptions.
        """
        _, _, seq_len, _ = q.shape

        # Attention kernel K = exp(sim(q_i, k_j) / temperature).
        kernel = self.kernel_fn(q, k)

        # Multiply by the supplied mask. Boolean masks act as keep/drop masks;
        # non-boolean values scale kernel entries rather than merely selecting
        # them. Mask semantics are otherwise not inferred.
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            kernel = kernel * mask.to(dtype=kernel.dtype)

        # The implementation removes the diagonal only when no mask is passed.
        # Typical causal masks retain their diagonal, so masked kernels are not
        # generally diagonal-free and may be nonsymmetric.
        if mask is None:
            kernel = self.zero_diagonal(kernel)

        # Positive scalar regularization; this alone does not make a
        # nonsymmetric or indefinite kernel SPD.
        lam = self.lambda_reg.view(1, 1, 1, 1)

        # (Possibly cached) preconditioner for (K + lambda*I).
        precond_data = self.preconditioner(
            kernel,
            lam,
            seq_len,
            force_update=False,
            update_frequency=self.config.precond_update_frequency,
        )

        # PCG assumes an SPD operator and suitable preconditioner; neither is
        # validated here. Only a raised RuntimeError triggers the dense solve.
        try:
            alpha = pcg_solve(
                kernel=kernel,
                b=v,
                lambda_reg=lam,
                precond_data=precond_data,
                apply_preconditioner=self.preconditioner.apply_preconditioner,
                max_iterations=self.config.effective_pcg_iters,
                tolerance=self.config.pcg_tolerance,
                min_iterations=3,
            )
        except RuntimeError:
            logger.warning(
                "PCG solve failed; falling back to direct solve. "
                "seq_len=%d, lambda=%.4f",
                seq_len,
                self.lambda_reg.item(),
            )
            eye = torch.eye(seq_len, device=kernel.device, dtype=kernel.dtype)
            eye = eye.view(1, 1, seq_len, seq_len)
            kernel_reg = kernel + lam * eye
            alpha = torch.linalg.solve(kernel_reg, v)  # pylint: disable=not-callable

        # Bound the returned values, then normalize their per-head RMS.
        alpha = stable_clip(alpha)
        alpha = self.rms_normalize(alpha)

        # XSA output cleaning (only meaningful for subtract_projection mode).
        if self.config.xsa_mode == "subtract_projection":
            alpha = self.clean_self_projection(alpha, v)

        return cast(torch.Tensor, alpha)


class LakerAttentionLayer(nn.Module):
    """Thin ``nn.Module`` wrapper around :class:`LakerAttention`.

    Each wrapper constructs an independent attention module. ``layer_idx`` and
    ``share_preconditioner_across_layers`` are stored as metadata but do not
    affect construction or forwarding; in particular, setting the sharing flag
    does not share a preconditioner.

    Attributes:
        layer_idx: Zero-based layer index in the host Transformer stack.
        share_preconditioner: Stored metadata with no current effect.
        attention: The underlying :class:`LakerAttention` module.
    """

    def __init__(
        self,
        config: XSA_LAKER_Config,
        layer_idx: int = 0,
        share_preconditioner_across_layers: bool = False,
    ) -> None:
        """Initialise the wrapper around :class:`LakerAttention`.

        Args:
            config: :class:`laker_xsa.config.XSA_LAKER_Config` forwarded
                to the underlying :class:`LakerAttention`.
            layer_idx: Stored zero-based layer metadata; unused by ``forward``.
            share_preconditioner_across_layers: Stored as
                ``share_preconditioner`` but otherwise ignored. Every wrapper
                still constructs a separate :class:`LakerAttention`.

        Side Effects:
            Constructs the underlying :class:`LakerAttention` and
            registers it as the :attr:`attention` submodule. No
            preconditioner sharing is performed at this time.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.share_preconditioner = share_preconditioner_across_layers
        self.attention = LakerAttention(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the wrapped :class:`LakerAttention`.

        Args:
            x: Token embeddings ``(batch, seq_len, d_model)``.
            mask: Optional attention mask; semantics are documented on
                :meth:`LakerAttention.compute_attention`.

        Returns:
            Tensor ``(batch, seq_len, d_model)``.
        """
        return cast(torch.Tensor, self.attention(x, mask))

    attention: LakerAttention
