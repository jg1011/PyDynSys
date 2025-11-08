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
    from ..support.phase_space import PhaseSpace
    from ..support.trajectory import Trajectory
    from ..support.cache import TrajectoryCache

class _DynSys:
    """
    Mixin providing shared infrastructure for Euclidean dynamical systems; inherited by AutDynSys & NonAutDynSys.
    
    Fields (must be set by subclasses in their __init__):
        dimension: int
        phase_space: PhaseSpace
        _trajectory_cache: TrajectoryCache | None
        
    Public Methods: 
        - empty_cache()
    
    NOTE: This mixin does NOT provide an __init__ method. Inheritors must set these attributes themselves. 
    This design avoids multiple inheritance issues with __init__ & allows inheritors to control their 
    constructor signature.
    """
    
    # Type annotations for attributes that subclasses must set
    dimension: int
    phase_space: PhaseSpace
    _trajectory_cache: TrajectoryCache | None
    
    
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
        if self._trajectory_cache:
            self._trajectory_cache.clear()

