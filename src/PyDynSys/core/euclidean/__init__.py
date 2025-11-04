"""Euclidean dynamical systems module."""

from .base import EuclideanDS  # Or inline definition here
from .autonomous import AutonomousEuclideanDS
from .non_autonomous import NonAutonomousEuclideanDS
from .trajectory import EuclideanTrajectorySegment, EuclideanTrajectory
from .time_horizon import TimeHorizon
from .phase_space import PhaseSpace

__all__ = [
    'EuclideanDS',
    'AutonomousEuclideanDS',
    'NonAutonomousEuclideanDS',
    'EuclideanTrajectorySegment',
    'EuclideanTrajectory',
    'TimeHorizon',
    'PhaseSpace',
]