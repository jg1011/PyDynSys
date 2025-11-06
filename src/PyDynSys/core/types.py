"""Type definitions for PyDynSys dynamical systems."""

from typing import Union, List, Callable, Dict, Tuple, Any, Literal, Optional, Set
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import sympy as syp


### Vector Field Types ###

# NOTE: We use explicit Callable types instead of type aliases to avoid type bloat.
# For callable-only vector fields, use:
#   - Autonomous: Callable[[NDArray[np.float64]], NDArray[np.float64]]
#   - Non-autonomous: Callable[[NDArray[np.float64], float], NDArray[np.float64]]
# For dual representation (symbolic + callable), use the VectorField class from euclidean.vector_field

VectorFieldCallable = Union[
    Callable[[NDArray[np.float64]], NDArray[np.float64]],  # Autonomous: F(x) -> dx/dt
    Callable[[NDArray[np.float64], float], NDArray[np.float64]]  # Non-autonomous: F(x, t) -> dx/dt
]
"""
Union type for callable vector field functions (internal use only).

Used internally in SymbolicToVectorFieldResult to represent the compiled
callable result from SymbolicSystemBuilder.

NOTE: The primary vector field representation is the VectorField class
(in euclidean.vector_field), which provides dual symbolic/callable forms.
"""


### Symbolic Types ###


SymbolicODE = Union[List[syp.Expr], syp.Expr, str, List[str]]
"""
Union type for symbolic ODE representations.
Accepts: single expression, list of expressions, or string forms.
Expected format: d(x_i)/dt - F_i(x, t) = 0 (we drop the "= 0")
"""

SystemParameters = Union[Dict[str, float], Dict[syp.Symbol, float]]
"""
Parameter substitution dictionary for symbolic systems.
Keys: parameter names (str) or SymPy symbols
Values: numerical parameter values
NOTE: All keys in a single dict must be same type (all str or all Symbol)
"""


### IVP Solution Types ###


@dataclass(frozen=True)
class TrajectoryCacheKey:
    """
    Immutable cache key for trajectory solutions.
    
    DESIGN RATIONALE - Why This Key Structure:
    ------------------------------------------
    Caches by initial conditions and evaluation domain, NOT by method.
    This allows retrieval regardless of solver used. Note: multi-method
    trajectories (segments solved with different methods) are valid.
    
    Why method is NOT in the cache key:
    - Removed to support multi-method trajectories (e.g., bidirectional integration
      where backward uses RK45 and forward uses DOP853)
    - Same IVP solved with different methods should give similar results (modulo
      numerical error), so caching without method is reasonable
    - Simplifies cache logic and enables trajectory composition
    
    Why initial_time IS in the cache key:
    - For autonomous systems (dx/dt = F(x)): initial_time doesn't affect trajectory
      SHAPE in phase space, only the parameterization. However, we store it for
      consistency and potential future time-shifted cache lookups.
    - For non-autonomous systems (dx/dt = F(x,t)): initial_time is CRITICAL!
      The same initial state x_0 at different times evolves completely differently
      due to time-dependent forcing. Example:
        - System: dx/dt = -x + sin(t)
        - x(0) = 1.0 at t_0=0: affected by sin(0)=0
        - x(0) = 1.0 at t_0=π/2: affected by sin(π/2)=1
      These produce entirely different trajectories despite same x_0!
    
    For autonomous systems: initial_time is conventionally t_span[0]
    For non-autonomous systems: initial_time matters and varies
    
    Fields:
        initial_conditions: x(t_0) as hashable tuple
        initial_time: t_0 (critical for non-autonomous, stored for autonomous too)
        t_eval_tuple: Evaluation time points as hashable tuple
        
    TODO: Future enhancements (smart caching):
        - Smart cache lookup: merge cached [0,2] + [1,5] to get requested [1,3]
          (use EuclideanTrajectory.from_segments to compose cached segments)
        - Cache slicing: slice cached [1,3] to get requested [1,2]
          (extract subset of evaluation points from cached trajectory)
    """
    # Frozen dataclass ensures immutability → hashable → usable as dict key
    initial_conditions: Tuple[float, ...]
    initial_time: float
    t_eval_tuple: Tuple[float, ...]


@dataclass
class TrajectoryCacheQuery:
    """
    Mutable query for cache operations (clear, replace).
    
    Supports partial matching: None fields match any value (wildcard).
    
    This enables flexible cache management:
    - Clear all trajectories from a specific initial state
    - Clear all trajectories starting at a specific time
    - Clear a specific trajectory
    
    Examples:
        # Clear all trajectories from x0
        query = TrajectoryCacheQuery(initial_conditions=tuple(x0))
        
        # Clear specific trajectory
        query = TrajectoryCacheQuery(
            initial_conditions=tuple(x0),
            initial_time=0.0,
            t_eval_tuple=tuple(t_eval)
        )
    """
    initial_conditions: Optional[Tuple[float, ...]] = None
    initial_time: Optional[float] = None
    t_eval_tuple: Optional[Tuple[float, ...]] = None
    
    def matches(self, key: TrajectoryCacheKey) -> bool:
        """
        Check if this query matches a cache key.
        
        None fields act as wildcards (match any value).
        
        Args:
            key: Cache key to test against
            
        Returns:
            True if query matches key (all non-None fields match)
        """
        if self.initial_conditions is not None:
            if self.initial_conditions != key.initial_conditions:
                return False
        
        if self.initial_time is not None:
            # Use tolerance for float comparison
            if abs(self.initial_time - key.initial_time) > 1e-10:
                return False
        
        if self.t_eval_tuple is not None:
            if self.t_eval_tuple != key.t_eval_tuple:
                return False
        
        return True
    
    def to_key(self) -> TrajectoryCacheKey:
        """
        Convert to immutable key (requires all fields set).
        
        Returns:
            TrajectoryCacheKey with all fields set
            
        Raises:
            ValueError: If any field is None
        """
        if self.initial_conditions is None:
            raise ValueError("initial_conditions required for key conversion")
        if self.initial_time is None:
            raise ValueError("initial_time required for key conversion")
        if self.t_eval_tuple is None:
            raise ValueError("t_eval_tuple required for key conversion")
        
        return TrajectoryCacheKey(
            initial_conditions=self.initial_conditions,
            initial_time=self.initial_time,
            t_eval_tuple=self.t_eval_tuple
        )


@dataclass
class SciPyIvpSolution:
    """
    Type-safe wrapper for scipy.integrate.solve_ivp results.
    
    Wraps scipy's OdeResult (not properly typed in scipy.integrate) with
    explicit property accessors for common attributes.
    
    Fields:
        raw_solution (Any): The scipy OdeResult object
    """
    raw_solution: Any  # scipy.integrate.OdeResult not exposed in type stubs
    
    @property
    def t(self) -> NDArray[np.float64]:
        """Time points where solution is evaluated - shape (n_points,)"""
        return self.raw_solution.t
    
    @property
    def y(self) -> NDArray[np.float64]:
        """State vectors at each time point - shape (n_dim, n_points)"""
        return self.raw_solution.y
    
    @property
    def sol(self) -> Any:
        """
        Continuous solution interpolant: callable f(t) -> x(t)
        
        NOTE: Most accurate within [min(t), max(t)], extrapolates beyond.
        Only available if dense_output=True was passed to solve_ivp.
        """
        return self.raw_solution.sol
    
    @property
    def success(self) -> bool:
        """Whether integration succeeded"""
        return self.raw_solution.success
    
    @property
    def message(self) -> str:
        """Human-readable description of termination reason"""
        return self.raw_solution.message


### Trajectory Types ###


TrajectorySegmentMergePolicy = Literal['average', 'left', 'right', 'stitch']
"""
Strategy for merging overlapping trajectory segments in EuclideanTrajectory.from_segments().

WHEN OVERLAPS OCCUR:
--------------------
Overlapping domains arise when the same physical trajectory is computed multiple
times over intersecting time intervals. Common scenarios:
  1. Re-solving for comparison: Solve [0,1.5] with RK45, then [0.5,2] with DOP853
  2. Patching numerical errors: Re-solve unstable region with tighter tolerances
  3. Composing from cache: Merging cached trajectories to avoid recomputation

MERGE POLICIES:
---------------
When two segments overlap on [a, b], we must decide:
  - Which y values to use at shared evaluation points?
  - Which interpolant to use for continuous evaluation in [a, b]?

Available policies:
- 'average' (DEFAULT): Average y values at shared evaluation points in overlap region.
                       Takes midpoint between competing numerical approximations.
                       Interpolant set to None (limitation of current implementation).
                       Use case: Equal trust in both segments, want best estimate.
                       
- 'left': Prioritize left segment's values and interpolant in overlap region.
          Use case: Left segment has higher accuracy (tighter tolerance, better method).
          Status: Not yet implemented (raises NotImplementedError).
          
- 'right': Prioritize right segment's values and interpolant in overlap region.
           Use case: Right segment has higher accuracy or is more recent computation.
           Status: Not yet implemented (raises NotImplementedError).
           
- 'stitch': Use left segment's interpolant until overlap midpoint, then right's.
            Creates continuous transition across overlap region.
            Use case: Both segments equally valid, want smooth transition.
            Status: Not yet implemented (raises NotImplementedError).

TANGENT DOMAINS (Special Case):
--------------------------------
When domains touch at exactly one point (e.g., [0,1] + [1,2]), the "overlap"
is just the boundary. The average policy automatically handles this by averaging
the single shared point, which is correct for tangent segments from bidirectional
integration where both segments share x(t_0) at the tangent point.

Note: Only 'average' is implemented in current version. Others raise NotImplementedError.
"""

