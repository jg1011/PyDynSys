"""Euclidean dynamical systems module."""

from .base import EuclideanDS  # Or inline definition here
from .autonomous import AutonomousEuclideanDS
from .non_autonomous import NonAutonomousEuclideanDS

__all__ = [
    'EuclideanDS',
    'AutonomousEuclideanDS',
    'NonAutonomousEuclideanDS',
]