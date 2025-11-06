"""
Euclidean dynamical systems module.

This module provides the core architecture for Euclidean dynamical systems:
- Autonomous systems: dx/dt = F(x)
- Non-autonomous systems: dx/dt = F(x, t)

Architecture:
- Public API: AutDynSys and NonAutDynSys (concrete implementations of base ABCS)
- Internal ABCs: _AutDynSys and _NonAutDynSys (for subclassing, in base submodule)
- Property mixins: LinearMixin, etc. (in properties submodule)
- Support classes: Trajectory, TimeHorizon, PhaseSpace, VectorField, etc. (in support submodule)
"""

from .base.autonomous import AutDynSys
from .base.non_autonomous import NonAutDynSys
from .support.trajectory import TrajectorySegment, Trajectory
from .support.time_horizon import TimeHorizon
from .support.phase_space import PhaseSpace
from .support.vector_field import AutVectorField, NonAutVectorField, vector_field_factory

# Re-export property mixins
from .properties import LinearMixin

__all__ = [
    # Public API (concrete implementations)
    'AutDynSys',
    'NonAutDynSys',
    # Trajectory classes
    'TrajectorySegment',
    'Trajectory',
    # Domain classes
    'TimeHorizon',
    'PhaseSpace',
    # Vector field classes
    'AutVectorField',
    'NonAutVectorField',
    # Vector field factory
    'vector_field_factory',
    # Property mixins
    'LinearMixin',
]