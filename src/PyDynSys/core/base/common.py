"""
Shared infrastructure mixin for Euclidean dynamical systems.

Provides common functionality used by both AutDynSys and NonAutDynSys, including:
- Cache management
- Validation methods
- Common utilities
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, TYPE_CHECKING, Optional
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..types import TrajectoryCacheKey, TrajectoryCacheQuery
    from ..support.phase_space import PhaseSpace
    from ..support.trajectory import Trajectory

class _DynSys:
    """
    Mixin providing shared infrastructure for Euclidean dynamical systems; inherited by AutDynSys & NonAutDynSys.
    
    Fields (must be set by subclasses in their __init__):
        dimension: int
        phase_space: PhaseSpace
        _solutions_cache: Dict[TrajectoryCacheKey, 'Trajectory']
        
    Public Methods: 
        - empty_cache()
        - delete_cache_items()
        - replace_cache_items()
    
    NOTE: This mixin does NOT provide an __init__ method. Inheritors must setthese attributes themselves. 
    This design avoids multiple inheritance issues with __init__ & allows inheritors to control their 
    constructor signature.
    """
    
    # Type annotations for attributes that subclasses must set
    dimension: int
    phase_space: PhaseSpace
    _solutions_cache: Dict[TrajectoryCacheKey, 'Trajectory']
    
    
    ### --- Validation Methods --- ###
    
    
    def _validate_state(self, state: NDArray[np.float64]) -> None:
        """
        Validate state vector dimension and phase space membership.
        
        Args:
            state: State vector to validate
            
        Raises:
            ValueError: If state.shape != (dimension,) or state ∉ X
        """
        if state.shape != (self.dimension,):
            raise ValueError(
                f"State vector has incorrect dimension: "
                f"expected ({self.dimension},), got {state.shape}"
            )
        
        # Check phase space membership
        if not self.phase_space.contains_point(state):
            raise ValueError(
                f"State {state} is not in phase space X. "
                f"Phase space constraints violated."
            )
    
    
    def _validate_time_span(self, t_span: Tuple[float, float]) -> None:
        """
        Base validation: check non-zero length interval.
        
        Args:
            t_span: Time span tuple (t_start, t_end)
            
        Raises:
            ValueError: If t_span[0] >= t_span[1]
        """
        if t_span[0] >= t_span[1]:
            raise ValueError(
                f"Time span must have non-zero length, got {t_span}"
            )
    
    
    def _validate_t_eval(
        self, 
        t_eval: NDArray[np.float64], 
        t_span: Tuple[float, float]
    ) -> None:
        """
        Base validation: check t_eval within t_span bounds.
        
        Args:
            t_eval: Evaluation time points
            t_span: Time span tuple (t_start, t_end)
            
        Raises:
            ValueError: If any t_eval point is outside [t_span[0], t_span[1]]
        """
        t_start, t_end = t_span
        t_min = np.min(t_eval)
        t_max = np.max(t_eval)
        
        if t_min < t_start or t_max > t_end:
            raise ValueError(
                f"Evaluation points t_eval must be within time span [{t_start}, {t_end}], "
                f"got range [{t_min}, {t_max}]"
            )
    
    
    def _validate_time_step(self, dt: float) -> None:
        """
        Base validation: check positive time step.
        
        Args:
            dt: Time step
            
        Raises:
            ValueError: If dt <= 0
        """
        if dt <= 0:
            raise ValueError(
                f"Time step must be positive, got {dt}"
            )
    
    
    ### --- Cache Management --- ###
    
    
    def empty_cache(self) -> None:
        """
        Clear all cached trajectories.
        
        Removes all entries from the solution cache.
        """
        self._solutions_cache.clear()
    
    
    def delete_cache_items(
        self,
        initial_state: Optional[NDArray[np.float64]] = None,
        initial_time: Optional[float] = None,
        t_eval: Optional[NDArray[np.float64]] = None
    ) -> None:
        """
        Clear cached trajectories matching query.
        
        None parameters act as wildcards (match any value).
        This enables flexible cache management:
        - Clear all trajectories from a specific initial state
        - Clear all trajectories starting at a specific time
        - Clear a specific trajectory
        
        Args:
            initial_state: If provided, only clear trajectories from this state
            initial_time: If provided, only clear trajectories starting at this time
            t_eval: If provided, only clear trajectories with this evaluation array
            
        Examples:
            # Clear all trajectories from x0
            sys.clear_trajectory_cache(initial_state=x0)
            
            # Clear specific trajectory
            sys.clear_trajectory_cache(
                initial_state=x0,
                initial_time=0.0,
                t_eval=t_eval
            )
        """
        # Build query
        query = TrajectoryCacheQuery(
            initial_conditions=tuple(initial_state) if initial_state is not None else None,
            initial_time=initial_time,
            t_eval_tuple=tuple(t_eval) if t_eval is not None else None
        )
        
        # Find matching keys
        keys_to_remove = [
            key for key in self._solutions_cache.keys()
            if query.matches(key)
        ]
        
        # Remove matches
        for key in keys_to_remove:
            del self._solutions_cache[key]
    
    
    def replace_trajectory_cache(
        self,
        initial_state: NDArray[np.float64],
        initial_time: float,
        t_eval: NDArray[np.float64],
        trajectory: 'Trajectory'
    ) -> None:
        """
        Explicitly replace cached trajectory.
        
        Creates new cache entry or overwrites existing one.
        Useful when recomputing with different solver method.
        
        Args:
            initial_state: Initial condition
            initial_time: Initial time
            t_eval: Evaluation time points
            trajectory: Trajectory to cache
        """
        key = TrajectoryCacheKey(
            initial_conditions=tuple(initial_state),
            initial_time=initial_time,
            t_eval_tuple=tuple(t_eval)
        )
        self._solutions_cache[key] = trajectory

