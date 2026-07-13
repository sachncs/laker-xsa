"""Linear-system operators, iterative routines, and preconditioners.

The attention path uses these modules with the regularized operator

.. math::

    A(\\alpha) = K\\alpha + \\lambda\\alpha.

The LAKER reference already cited by this repository is
``arXiv:2604.25138``. The executable API here includes:

* :func:`pcg_solve`, a residual-monitored PCG-style recurrence. Classical PCG
  requires an SPD operator and compatible preconditioner; the code does not
  enforce those assumptions or report convergence status.
* :func:`richardson_solve`, a fixed-budget preconditioned Richardson iteration.
* :class:`LakerPreconditioner`, with ``"cccp"``, ``"fast"``, ``"diagonal"``,
  and ``"none"`` behavior.
* :class:`LearnedPreconditioner`, the position-based legacy implementation.

The functions return their final iterates even if the requested residual
threshold was not reached or non-finite values arose.
"""

from __future__ import annotations

from laker_xsa.solver.preconditioner import LearnedPreconditioner
from laker_xsa.solver.laker_preconditioner import LakerPreconditioner
from laker_xsa.solver.conjugate_gradient import pcg_solve, richardson_solve

__all__ = [
    "LearnedPreconditioner",
    "LakerPreconditioner",
    "pcg_solve",
    "richardson_solve",
]
