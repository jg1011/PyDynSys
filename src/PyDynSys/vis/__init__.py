"""Visualization utilities for PyDynSys."""

# Expose submodule
from . import euclidean

# Convenience re-exports from euclidean for top-level imports
from .euclidean import plot_phase_space

__all__ = [
    'plot_phase_space',
    'euclidean',
]
