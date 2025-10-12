"""Euclidean dynamical systems module."""

from .base import EuclideanDS  # Or inline definition here
from .autonomous import AutonomousEuclideanDS
from .non_autonomous import NonAutonomousEuclideanDS
from .trajectory import EuclideanTrajectorySegment, EuclideanTrajectory

__all__ = [
    'EuclideanDS',
    'AutonomousEuclideanDS',
    'NonAutonomousEuclideanDS',
    'EuclideanTrajectorySegment',
    'EuclideanTrajectory',
]