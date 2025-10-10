"""Core module for PyFlow dynamical systems library."""

# Type exports
from .types import (
    # Vector field types
    AutonomousVectorField,
    NonAutonomousVectorField,
    VectorField,
    # Symbolic types
    SymbolicODE,
    SystemParameters,
    # Solution types
    IvpParams,
    SciPyIvpSolution,
    # Phase space types
    PhaseSpace,
    TimeHorizon,
    # Result types
    SymbolicToVectorFieldResult,
)

# System builder
from .sym_utils import SymbolicSystemBuilder

# Dynamical systems
from .euclidean_sys import (
    EuclideanDS,
    AutonomousEuclideanDS,
    NonAutonomousEuclideanDS,
)

__all__ = [
    # Dynamical system classes
    'EuclideanDS',
    'AutonomousEuclideanDS', 
    'NonAutonomousEuclideanDS',
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
    'IvpParams',
    'SciPyIvpSolution',
    # Phase space types
    'PhaseSpace',
    'TimeHorizon',
    # Result types
    'SymbolicToVectorFieldResult',
]