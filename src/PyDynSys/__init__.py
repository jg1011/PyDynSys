"""
PyDynSys: Python Dynamical Systems Library

A library for working with dynamical systems in Euclidean space, providing:
- Autonomous and non-autonomous system support
- Symbolic to numerical system conversion
- Trajectory computation and composition
- Phase space constraints and time horizons
"""

__version__ = "0.1.0"

# Re-export core functionality for convenience
from .core import (
    # Dynamical system classes
    EuclideanDS,
    AutonomousEuclideanDS,
    NonAutonomousEuclideanDS,
    EuclideanTrajectorySegment,
    EuclideanTrajectory,
    # System builder
    SymbolicSystemBuilder,
    SymbolicToVectorFieldResult,
    # Vector field types
    AutonomousVectorField,
    NonAutonomousVectorField,
    VectorField,
    # Symbolic types
    SymbolicODE,
    SystemParameters,
    # Solution types
    TrajectoryCacheKey,
    TrajectorySegmentMergePolicy,
    # Phase space types
    PhaseSpace,
    TimeHorizon,
)

__all__ = [
    # Dynamical system classes
    'EuclideanDS',
    'AutonomousEuclideanDS',
    'NonAutonomousEuclideanDS',
    'EuclideanTrajectorySegment',
    'EuclideanTrajectory',
    # System builder
    'SymbolicSystemBuilder',
    'SymbolicToVectorFieldResult',
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
    # Phase space types
    'PhaseSpace',
    'TimeHorizon',
]

