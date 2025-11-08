"""
This module provides a robust, extensible caching system for storing and
retrieving computationally expensive trajectory objects.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from cachetools import LRUCache

if TYPE_CHECKING:
    from .trajectory import Trajectory


@dataclass(frozen=True)
class TrajectoryCacheKey:
    """
    A raw, high-precision key representing a unique trajectory request.

    This class acts as a Data Transfer Object (DTO) to gather all parameters
    that define a trajectory before they are normalized for hashing.
    """
    system_signature: tuple
    initial_state: NDArray[np.float64]
    t_span: tuple
    t_eval: NDArray[np.float64]
    t0: float | None  # For non-autonomous systems
    solver_options: Dict[str, Any]


class KeyNormalizer:
    """
    Normalizes a TrajectoryCacheKey into a stable, hashable tuple.

    This class is the core of the cache's robustness. It handles the
    unreliable nature of floating-point numbers and the instability of
    dictionary ordering to produce a reliable key for the cache's internal
    dictionary.
    """
    def __init__(self, **kwargs):
        """
        Initializes the KeyNormalizer.

        Args:
            **kwargs: Configuration for normalization. Currently supports:
                - precision (int): The number of decimal places to round
                  floating-point numbers to. Defaults to 8.
        """
        self.precision = kwargs.get('precision', 8)

    def normalize(self, raw_key: TrajectoryCacheKey) -> tuple:
        """
        Converts a raw TrajectoryCacheKey into a normalized, hashable tuple.
        """
        # 1. Normalize numerical arrays by rounding and converting to tuples
        norm_x0 = tuple(np.round(raw_key.initial_state, self.precision))
        norm_t_eval = tuple(np.round(raw_key.t_eval, self.precision))

        # 2. Normalize floating-point numbers
        norm_t_span = tuple(round(t, self.precision) for t in raw_key.t_span)
        norm_t0 = round(raw_key.t0, self.precision) if raw_key.t0 is not None else None

        # 3. Normalize the solver_options dictionary by sorting its items
        #    This ensures that the order of kwargs does not affect the key.
        norm_solver_opts = tuple(sorted(raw_key.solver_options.items()))

        # 4. Combine all components into a final, stable tuple.
        #    The system_signature is already a tuple and can be used directly.
        return (
            raw_key.system_signature,
            norm_x0,
            norm_t_span,
            norm_t_eval,
            norm_t0,
            norm_solver_opts,
        )


class TrajectoryCache:
    """
    A Least Recently Used (LRU) cache for storing and retrieving trajectories.

    This class encapsulates all caching logic, including key normalization,
    storage, and retrieval, using a memory-bounded LRU scheme to prevent

    uncontrolled memory growth.
    """
    def __init__(self, size: int = 128, **normalizer_kwargs):
        """
        Initializes the TrajectoryCache.

        Args:
            size: The maximum number of trajectories to store in the cache.
            **normalizer_kwargs: Configuration options passed to the
                                 KeyNormalizer (e.g., `precision`).
        """
        self._lru = LRUCache(maxsize=size)
        self._normalizer = KeyNormalizer(**normalizer_kwargs)

    def get(self, raw_key: TrajectoryCacheKey) -> Trajectory | None:
        """
        Retrieves a trajectory from the cache using a raw key.
        """
        normalized_key = self._normalizer.normalize(raw_key)
        return self._lru.get(normalized_key)

    def insert(self, raw_key: TrajectoryCacheKey, trajectory: Trajectory):
        """
        Inserts a new trajectory into the cache using a raw key.
        """
        normalized_key = self._normalizer.normalize(raw_key)
        self._lru[normalized_key] = trajectory

    def clear(self):
        """Clears all items from the cache."""
        self._lru.clear()

    def info(self) -> str:
        """Returns a string with information about the cache's state."""
        return (
            f"Cache Info: Size={self._lru.currsize}/{self._lru.maxsize}, "
            f"Hits={self._lru.hits}, Misses={self._lru.misses}"
        )
