"""
PyDynSys: Python Dynamical Systems Library

A library for working with dynamical systems in Euclidean space, providing:
- Autonomous and non-autonomous system support
- Symbolic to numerical system conversion
- Trajectory computation and composition
- Phase space constraints and time horizons
"""

__version__ = "0.1.0"

# Expose submodules
from . import core, vis


__all__ = [
    'core',
    'vis',
]
