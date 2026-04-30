"""
Solver module for iterative linear system solving.

This package provides iterative solvers for the kernel attention system,
including preconditioned Richardson iteration and Conjugate Gradient.
"""

from laker_xsa.solver.preconditioner import LearnedPreconditioner
from laker_xsa.solver.conjugate_gradient import conjugate_gradient_solve

__all__ = [
    "LearnedPreconditioner",
    "conjugate_gradient_solve",
]
