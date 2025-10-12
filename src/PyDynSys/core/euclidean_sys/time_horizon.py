from dataclasses import dataclass
from typing import Optional, Tuple, Callable

@dataclass
class RealLineTimeHorizon:
    """
    Time horizon T subset of R for non-autonomous systems.
    
    Supports:
    - Unbounded: T = R (default)
    - Interval: T = [a, b]
    - Predicate: T = {t : constraint(t) = True}
    
    Fields:
        bounds (Tuple[float, float] | None): (t_min, t_max) or None for ℝ
        constraint (Callable | None): Custom time constraint predicate
    """
    bounds: Optional[Tuple[float, float]] = None
    constraint: Optional[Callable[[float], bool]] = None
    
    def contains(self, t: float) -> bool:
        """
        Check if t is in T.
        
        Args:
            t: Time point to test
            
        Returns:
            bool: True if t in T, False otherwise
        """
        if self.constraint is not None:
            return self.constraint(t)
        elif self.bounds is not None:
            return self.bounds[0] <= t <= self.bounds[1]
        else:
            return True  # T = R
    
    @classmethod
    def real_line(cls) -> 'RealLineTimeHorizon':
        """
        Factory: T = R (entire real line).
        
        This is the DEFAULT time horizon for non-autonomous systems.
        
        Returns:
            TimeHorizon representing R
        """
        return cls()
    
    @classmethod
    def interval(cls, t_min: float, t_max: float) -> 'RealLineTimeHorizon':
        """
        Factory: T = [t_min, t_max] (bounded interval).
        
        Args:
            t_min: Lower bound
            t_max: Upper bound
            
        Returns:
            TimeHorizon with interval constraints
            
        Raises:
            ValueError: If t_min >= t_max
        """
        if t_min >= t_max:
            raise ValueError(f"Interval time horizon must have t_min < t_max, got ({t_min}, {t_max})")
        return cls(bounds=(t_min, t_max))