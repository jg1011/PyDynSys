"""Type definitions for PyFlow dynamical systems."""

from typing import Union, List, Callable, Dict, Tuple, Any, Literal
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import sympy as syp


### Vector Field Types ###


AutonomousVectorField = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""
Vector field for autonomous systems: F: R^n → R^n
Maps state vector x to derivative dx/dt = F(x)
"""

NonAutonomousVectorField = Callable[[NDArray[np.float64], float], NDArray[np.float64]]
"""
Vector field for non-autonomous systems: F: R^n × R → R^n
Maps (state x, time t) to derivative dx/dt = F(x, t)
"""

VectorField = Union[AutonomousVectorField, NonAutonomousVectorField]
"""
Union type for any vector field representation.
NOTE: For type safety, prefer specific types in implementations.
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
