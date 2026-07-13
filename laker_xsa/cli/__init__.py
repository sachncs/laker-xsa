"""Command-line interface for LAKER-XSA.

This package exposes the user-facing entry points:

* :mod:`laker_xsa.cli.train`     — :func:`main` for training entry.
* :mod:`laker_xsa.cli.evaluate`  — :func:`main` for checkpoint evaluation.
* :mod:`laker_xsa.cli.benchmark` — :func:`main` for runtime benchmarks.

Each ``main`` function is the target of a ``python -m laker_xsa.cli.<name>``
invocation and uses ``argparse`` to parse CLI options (calling
``argparse.ArgumentParser.parse_args`` which calls :class:`SystemExit` on
errors and on ``--help``).
"""

from __future__ import annotations
