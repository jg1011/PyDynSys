"""Core module for PyFlow dynamical systems library."""

# Type exports from utility modules (direct imports)
from .types import (
    # Vector field types
    AutonomousVectorField,
    NonAutonomousVectorField,
    VectorField,
    # Symbolic types
    SymbolicODE,
    SystemParameters,
    # Solution types
    TrajectoryCacheKey,
    SciPyIvpSolution,
    TrajectorySegmentMergePolicy,
)

# System builder from utility module
from .sym_utils import SymbolicSystemBuilder, SymbolicToVectorFieldResult

# Expose euclidean as a submodule
from . import euclidean

__all__ = [
    # System builder
    'SymbolicSystemBuilder',
    # Vector field types
    'AutonomousVectorField',
    'NonAutonomousVectorField',
    'VectorField',
    # Symbolic types
    'SymbolicODE',
    'SystemParameters',
    # Solution types
    'TrajectoryCacheKey',
    'TrajectorySegmentMergePolicy',
    # Result types
    'SymbolicToVectorFieldResult',
    # Submodules
    'euclidean',
]
