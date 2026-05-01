"""
LAKER Learned Preconditioner via CCCP (Convex-Concave Procedure).

Implements Algorithm 1 from arXiv:2604.25138 for learning a data-dependent
preconditioner P that approximates (lambda*I + K)^{-1}.

The preconditioner captures the inverse spectral structure of the attention
kernel system through:
1. Angular sampling: z ~ N(0,I), u = (lambda*I + K) * z, ubar = u/||u||
2. Tyler's M-estimator on angular data via regularized CCCP
3. Shrinkage stabilization and trace normalization
4. Extraction: P = Sigma^{-1/2}

Two variants are provided:
- Full CCCP: High-quality, O(n^3) per iteration, best for n <= 1024
- Fast: Gradient-based low-rank + diagonal, O(n*r^2), best for n > 1024
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LakerPreconditioner(nn.Module):
    """
    Learned preconditioner for the kernel attention system.

    Learns P such that P @ (lambda*I + K) approx I, dramatically reducing
    the condition number of the kernel system and accelerating PCG convergence.

    Supports three modes:
    - 'cccp': Full CCCP with Tyler's M-estimator and shrinkage
    - 'fast': Gradient-based diagonal + low-rank, differentiable
    - 'diagonal': Jacobi-style diagonal preconditioner only

    Attributes:
        num_heads: Number of attention heads.
        mode: Preconditioner mode.
        rank: Low-rank dimension (fast mode only).
        gamma: Nuclear norm regularization (CCCP only).
        rho: Shrinkage strength (CCCP only).
        eps_safeguard: Denominator safeguard epsilon.
        N_r: Number of random directions (CCCP only).
        max_cccp_iters: Maximum CCCP iterations.
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

        # For 'fast' mode: learnable diagonal + low-rank factors
        if mode == "fast" and rank is not None and rank > 0:
            self.diag_scale = nn.Parameter(
                torch.ones(1, num_heads, 1)
            )
            self.lr_base = nn.Parameter(
                torch.randn(num_heads, max_seq_len, rank) * 0.01
            )
            self.lr_importance = nn.Parameter(
                torch.zeros(num_heads, rank)
            )
        elif mode == "diagonal":
            self.diag_scale = nn.Parameter(
                torch.ones(1, num_heads, 1)
            )
            self.register_buffer("lr_base", torch.empty(0, 0, 0))
            self.register_buffer("lr_importance", torch.empty(0, 0))
        else:
            self.register_buffer("diag_scale", torch.ones(1, 1, 1))
            self.register_buffer("lr_base", torch.empty(0, 0, 0))
            self.register_buffer("lr_importance", torch.empty(0, 0))

        # Forward pass counter for periodic updates
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))
        self.cached_preconditioner: Optional[Tuple[torch.Tensor, ...]] = None

    def generate_angular_samples(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate angular samples ubar_k from the kernel system.

        For each k:
            z_k ~ N(0, I)
            u_k = (lambda*I + K) @ z_k
            ubar_k = u_k / ||u_k||

        The unit vectors ubar_k follow an angular distribution determined
        by the spectral structure of (lambda*I + K)^2.

        Args:
            kernel: (batch, num_heads, n, n) kernel matrix K.
            lambda_reg: Scalar or (batch, num_heads, 1, 1) regularization.

        Returns:
            ubar_samples: (N_r, batch, num_heads, n) unit vectors.
        """
        batch, num_heads, n, _ = kernel.shape
        device = kernel.device
        dtype = kernel.dtype

        ubar_list = []

        for _ in range(self.N_r):
            # Sample random direction
            z = torch.randn(batch, num_heads, n, 1, device=device, dtype=dtype)

            # u = (lambda*I + K) @ z
            Kz = torch.matmul(kernel, z)
            u = Kz + lambda_reg * z

            # Normalize to unit vector
            u_norm = torch.linalg.vector_norm(u, dim=-2, keepdim=True)
            ubar = u / (u_norm + self.eps_safeguard)

            ubar_list.append(ubar.squeeze(-1))

        return torch.stack(ubar_list, dim=0)

    def cccp_iteration(
        self,
        ubar_samples: torch.Tensor,
        Sigma: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """
        Single CCCP iteration for Tyler's M-estimator.

        Updates Sigma using the shrinkage-stabilized CCCP step (Eq. 35-37):

            F_gamma = 1/(1+gamma/n) * [
                (n/N_r) * sum_k (ubar_k @ ubar_k^T) / (ubar_k^T @ Sigma^{-1} @ ubar_k + eps)
                + gamma * I
            ]
            Sigma_tilde = (1-rho) * F_gamma + rho * I
            Sigma_new = Sigma_tilde / (tr(Sigma_tilde) / n)

        Args:
            ubar_samples: (N_r, batch, num_heads, n) unit vectors.
            Sigma: (batch, num_heads, n, n) current estimate.
            n: Sequence length.

        Returns:
            Updated Sigma (batch, num_heads, n, n).
        """
        batch, num_heads = Sigma.shape[:2]
        device = Sigma.device
        dtype = Sigma.dtype

        gamma = self.gamma
        N_r = ubar_samples.shape[0]

        # Compute Sigma^{-1} via batched inverse
        # For n up to ~1024 this is manageable
        Sigma_inv = torch.linalg.inv(Sigma)

        # Denominator term: ubar_k^T @ Sigma^{-1} @ ubar_k
        denom_sum = torch.zeros(batch, num_heads, n, n, device=device, dtype=dtype)

        for k in range(N_r):
            u = ubar_samples[k]

            # denom = u^T @ Sigma^{-1} @ u (batch, num_heads)
            Su = torch.matmul(Sigma_inv, u.unsqueeze(-1)).squeeze(-1)
            denom = (u * Su).sum(dim=-1)

            # outer product u @ u^T
            outer = torch.einsum("...i,...j->...ij", u, u)

            denom_sum = denom_sum + outer / (denom.unsqueeze(-1).unsqueeze(-1) + self.eps_safeguard)

        # F_gamma
        scale = n / N_r
        F_gamma = scale * denom_sum + gamma * torch.eye(n, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        F_gamma = F_gamma / (1.0 + gamma / n)

        # Shrinkage
        rho = self.rho
        Sigma_tilde = (1.0 - rho) * F_gamma + rho * torch.eye(n, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

        # Trace normalization
        trace = torch.diagonal(Sigma_tilde, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        Sigma_new = Sigma_tilde * n / (trace.unsqueeze(-1) + self.eps_safeguard)

        return Sigma_new

    def cccp_preconditioner(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full CCCP pipeline: samples -> CCCP -> P = Sigma^{-1/2}.

        Returns preconditioner matrix P of shape (batch, num_heads, n, n).
        """
        batch, num_heads, n, _ = kernel.shape
        device = kernel.device
        dtype = kernel.dtype

        # Generate angular samples
        ubar_samples = self.generate_angular_samples(kernel, lambda_reg)

        # Initialize Sigma = I
        Sigma = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(batch, num_heads, -1, -1).clone()

        # CCCP fixed-point iterations
        for _ in range(self.max_cccp_iters):
            Sigma_new = self.cccp_iteration(ubar_samples, Sigma, n)
            Sigma = Sigma_new

        # Extract P = Sigma^{-1/2} via eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)

        # Clamp eigenvalues for numerical stability
        eigenvalues = torch.clamp(eigenvalues, min=self.eps)

        # P = V @ diag(lambda^{-1/2}) @ V^T
        inv_sqrt_eig = eigenvalues.pow(-0.5)
        P = eigenvectors @ (inv_sqrt_eig.unsqueeze(-1) * eigenvectors.transpose(-2, -1))

        return P

    def fast_preconditioner(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fast gradient-based preconditioner.

        Returns (diag, lr_factor) where P = diag(d) + U @ U^T.
        This is fully differentiable and efficient for long sequences.

        Args:
            kernel: (batch, num_heads, n, n).
            lambda_reg: Regularization.
            seq_len: Actual sequence length.

        Returns:
            (diag_precond, lr_factor) for apply_precondition.
        """
        batch = kernel.shape[0]

        # Diagonal part: use kernel diagonal + lambda for scaling
        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        # softplus ensures positivity, important for preconditioner quality
        diag = F.softplus(kernel_diag + lambda_reg.squeeze(-1).squeeze(-1))
        diag = diag * self.diag_scale.abs() + self.eps

        # Low-rank factor
        lr_factor = None
        if self.rank is not None and self.rank > 0 and self.lr_base.numel() > 0:
            # Get position-dependent basis: (num_heads, seq_len, rank)
            lr_pos = self.lr_base[:, :seq_len, :]

            # Apply learned importance per rank component
            importance = F.softplus(self.lr_importance)

            # Scale basis by importance
            lr_scaled = lr_pos * importance.unsqueeze(1)

            # Expand batch: (batch, num_heads, seq_len, rank)
            lr_factor = lr_scaled.unsqueeze(0).expand(batch, -1, -1, -1)

        return diag, lr_factor

    def diagonal_preconditioner(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
    ) -> torch.Tensor:
        """Jacobi-style diagonal preconditioner."""
        kernel_diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        diag = F.softplus(kernel_diag + lambda_reg.squeeze(-1).squeeze(-1))
        return diag * self.diag_scale.abs() + self.eps

    def compute_preconditioner(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
        seq_len: int,
        force_update: bool = False,
        update_frequency: int = 1,
    ) -> torch.Tensor:
        """
        Main entry point. Returns preconditioner application data.

        Args:
            kernel: (batch, num_heads, n, n).
            lambda_reg: Scalar tensor.
            seq_len: Actual sequence length.
            force_update: Always recompute.
            update_frequency: Update every N calls.

        Returns:
            For 'cccp': P matrix (batch, num_heads, n, n).
            For 'fast': (diag, lr_factor) tuple.
            For 'diagonal': diag tensor (batch, num_heads, n).
        """
        should_update = force_update or (
            update_frequency > 0
            and (self.step_counter.item() % update_frequency == 0)
        )

        if self.mode == "cccp":
            if should_update or self.cached_preconditioner is None:
                P = self.cccp_preconditioner(kernel, lambda_reg)
                self.cached_preconditioner = (P,)
            return self.cached_preconditioner[0]

        elif self.mode == "fast":
            if should_update or self.cached_preconditioner is None:
                diag, lr = self.fast_preconditioner(kernel, lambda_reg, seq_len)
                self.cached_preconditioner = (diag, lr)
            return self.cached_preconditioner

        elif self.mode == "diagonal":
            return self.diagonal_preconditioner(kernel, lambda_reg)

        else:
            return None

    def apply_preconditioner(
        self,
        residual: torch.Tensor,
        precond_data,
    ) -> torch.Tensor:
        """
        Apply preconditioner: P @ residual.

        Args:
            residual: (batch, num_heads, n, head_dim).
            precond_data: From compute_preconditioner.

        Returns:
            Preconditioned residual, same shape.
        """
        if precond_data is None:
            return residual

        if self.mode == "cccp":
            # P is (batch, num_heads, n, n), apply via matmul
            P = precond_data
            return torch.matmul(P, residual)

        elif self.mode == "fast":
            diag, lr_factor = precond_data
            # Diagonal: element-wise
            out = residual * diag.unsqueeze(-1)
            # Low-rank: U @ (U^T @ r)
            if lr_factor is not None:
                UT_r = torch.matmul(lr_factor.transpose(-2, -1), residual)
                out = out + torch.matmul(lr_factor, UT_r)
            return out

        elif self.mode == "diagonal":
            return residual * precond_data.unsqueeze(-1)

        return residual

    def forward(
        self,
        kernel: torch.Tensor,
        lambda_reg: torch.Tensor,
        seq_len: int,
        force_update: bool = False,
        update_frequency: int = 1,
    ) -> torch.Tensor:
        """Compute and return preconditioner application data.

        This is the primary nn.Module entry point. Increments internal step
        counter for periodic update cadence.
        """
        precond_data = self.compute_preconditioner(
            kernel, lambda_reg, seq_len, force_update, update_frequency
        )
        self.step_counter.add_(1)
        return precond_data
