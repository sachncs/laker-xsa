from __future__ import annotations

"""
Solver module for iterative linear system solving.

Provides:
1. LAKER learned preconditioner (CCCP-based, fast gradient-based, diagonal)
2. Preconditioned Conjugate Gradient (PCG) solver
3. Richardson iteration solver (baseline)
"""

from laker_xsa.solver.preconditioner import LearnedPreconditioner
from laker_xsa.solver.laker_preconditioner import LakerPreconditioner
from laker_xsa.solver.conjugate_gradient import pcg_solve, richardson_solve

__all__ = [
    "LearnedPreconditioner",
    "LakerPreconditioner",
    "pcg_solve",
    "richardson_solve",
]
