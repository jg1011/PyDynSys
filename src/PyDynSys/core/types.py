"""Type definitions for PyFlow dynamical systems."""

from typing import Union, List, Callable, Dict, Tuple, Any
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import sympy as sp


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

SymbolicODE = Union[List[sp.Expr], sp.Expr, str, List[str]]
"""
Union type for symbolic ODE representations.
Accepts: single expression, list of expressions, or string forms.
Expected format: d(x_i)/dt - F_i(x, t) = 0 (we drop the "= 0")
"""

SystemParameters = Union[Dict[str, float], Dict[sp.Symbol, float]]
"""
Parameter substitution dictionary for symbolic systems.
Keys: parameter names (str) or SymPy symbols
Values: numerical parameter values
NOTE: All keys in a single dict must be same type (all str or all Symbol)
"""

### IVP Solution Types ###

@dataclass(frozen=True)
class IvpParams:
    """
    Immutable cache key for IVP solutions.
    
    Valid for both autonomous and non-autonomous systems. For autonomous
    systems, cache may contain redundant entries (time-shifted duplicates),
    but this simplifies implementation vs. normalizing to t=0.
    
    Fields:
        initial_conditions: x(t_0) as hashable tuple
        t_span: Integration interval (t_start, t_end)
        t_eval_tuple: Evaluation time points as hashable tuple
        method: ODE solver method (e.g. 'RK45', 'LSODA')
    """
    initial_conditions: Tuple[float, ...]
    t_span: Tuple[float, float]
    t_eval_tuple: Tuple[float, ...]
    method: str

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


### Phase Space and Time Horizon Types ###

from typing import Optional
from sympy.sets import ProductSet


@dataclass
class PhaseSpace:
    """
    Phase space X ⊆ ℝⁿ with flexible symbolic/callable representation.
    
    Supports three usage patterns:
    1. Symbolic only: Provides symbolic set, constraint auto-compiled (general)
    2. Callable only: Provides constraint directly (fast, no symbolic ops)
    3. Both: Provides both for optimal performance (fast + symbolic ops)
    
    Symbolic representation enables:
    - Rigorous mathematical operations (intersections, closures, etc.)
    - Future features (equilibria on manifolds, symbolic constraints)
    - Automatic bound extraction for optimization
    
    Callable representation provides O(1) membership testing for numerical work.
    
    Fields:
        dimension (int): Phase space dimension n
        symbolic_set (sp.Set | None): Optional SymPy set representation
        constraint (Callable | None): Optional callable for fast membership testing
        
    At least one of symbolic_set or constraint must be provided.
    """
    dimension: int
    symbolic_set: Optional[sp.Set] = None
    constraint: Optional[Callable[[NDArray[np.float64]], bool]] = None
    
    def __post_init__(self):
        """Validate and set up constraint if needed."""
        # Validate at least one representation provided
        if self.symbolic_set is None and self.constraint is None:
            raise ValueError(
                "PhaseSpace requires at least one representation: "
                "symbolic_set or constraint must be provided"
            )
        
        # Auto-compile constraint from symbolic if only symbolic provided
        if self.constraint is None and self.symbolic_set is not None:
            object.__setattr__(self, 'constraint', self._compile_constraint())
    
    def _compile_constraint(self) -> Callable[[NDArray[np.float64]], bool]:
        """
        Compile symbolic set to callable for fast numerical membership testing.
        
        Only called when symbolic_set is provided but constraint is not.
        
        Strategy:
        - For ℝⁿ (unbounded): return lambda x: True (no constraint)
        - For ProductSets of Intervals: extract bounds, compile to numpy checks
        - For general sets: use sympy's contains (slow, but general)
        
        Raises:
            AssertionError: If symbolic_set is None (should never happen)
        """
        assert self.symbolic_set is not None, "Cannot compile constraint without symbolic_set"
        
        # Special case: ℝⁿ (ProductSet of Reals or Reals**n)
        if self._is_full_euclidean_space():
            return lambda x: True
        
        # Special case: ProductSet of Intervals → box constraints
        if isinstance(self.symbolic_set, ProductSet):
            bounds = self._extract_box_bounds()
            if bounds is not None:
                def box_constraint(x: NDArray[np.float64]) -> bool:
                    return bool(np.all((x >= bounds[:, 0]) & (x <= bounds[:, 1])))
                return box_constraint
        
        # General case: use sympy (slow)
        # Prefer set.contains(Tuple(...)) over geometric Point for generic sets
        def symbolic_constraint(x: NDArray[np.float64]) -> bool:
            try:
                values = [sp.Float(float(v)) for v in x]
                elem = sp.Tuple(*values)
                contains_expr = self.symbolic_set.contains(elem)
                # SymPy returns a Boolean or a symbolic expression; coerce if possible
                return bool(contains_expr)
            except Exception:
                return False
        
        return symbolic_constraint
    
    def _is_full_euclidean_space(self) -> bool:
        """
        Check if symbolic set represents ℝⁿ.
        
        Returns False if symbolic_set is None.
        """
        if self.symbolic_set is None:
            return False
            
        if isinstance(self.symbolic_set, ProductSet):
            return all(s == sp.Reals for s in self.symbolic_set.args)
        # Also check for Reals**n notation
        if hasattr(self.symbolic_set, 'base') and hasattr(self.symbolic_set, 'exp'):
            return self.symbolic_set.base == sp.Reals and self.symbolic_set.exp == self.dimension
        return False
    
    def _extract_box_bounds(self) -> Optional[NDArray[np.float64]]:
        """
        Extract box bounds from ProductSet of Intervals.
        
        Returns:
            Array of shape (n, 2) with [[a1, b1], ..., [an, bn]], or None
            if not a product of intervals or if symbolic_set is None.
        """
        if self.symbolic_set is None or not isinstance(self.symbolic_set, ProductSet):
            return None
        
        bounds = []
        for component_set in self.symbolic_set.args:
            if isinstance(component_set, sp.Interval):
                a = float(component_set.start) if component_set.start.is_finite else -np.inf
                b = float(component_set.end) if component_set.end.is_finite else np.inf
                bounds.append([a, b])
            elif component_set == sp.Reals:
                bounds.append([-np.inf, np.inf])
            else:
                return None  # Not a simple interval
        
        return np.array(bounds, dtype=np.float64)
    
    def contains(self, x: NDArray[np.float64]) -> bool:
        """
        Check if x ∈ X using compiled constraint.
        
        Uses the constraint callable for fast membership testing.
        The constraint is guaranteed to exist after __post_init__.
        
        Args:
            x: Point in ℝⁿ to test
            
        Returns:
            bool: True if x ∈ X, False otherwise
        """
        assert self.constraint is not None, "Constraint should be set in __post_init__"
        return self.constraint(x)
    
    @classmethod
    def euclidean(cls, dimension: int) -> 'PhaseSpace':
        """
        Factory: X = ℝⁿ (full Euclidean space).
        
        This is the DEFAULT phase space for systems without constraints.
        Provides both symbolic representation and optimized constraint for
        best performance (no compilation overhead).
        
        Args:
            dimension: Phase space dimension n
            
        Returns:
            PhaseSpace representing ℝⁿ with optimal performance
        """
        symbolic = sp.Reals ** dimension
        # Provide constraint directly to avoid compilation overhead
        constraint = lambda x: True
        return cls(dimension=dimension, symbolic_set=symbolic, constraint=constraint)
    
    @classmethod
    def box(cls, bounds: NDArray[np.float64]) -> 'PhaseSpace':
        """
        Factory: X = [a₁, b₁] × ... × [aₙ, bₙ] (box constraints).
        
        Provides both symbolic representation and optimized numpy constraint
        for best performance.
        
        Args:
            bounds: Array of shape (n, 2) with [[a1, b1], ..., [an, bn]]
            
        Returns:
            PhaseSpace with box constraints and optimal performance
        """
        dimension = bounds.shape[0]
        intervals = [sp.Interval(bounds[i, 0], bounds[i, 1]) for i in range(dimension)]
        symbolic = ProductSet(*intervals)
        
        # Provide pre-compiled constraint for performance
        def box_constraint(x: NDArray[np.float64]) -> bool:
            return bool(np.all((x >= bounds[:, 0]) & (x <= bounds[:, 1])))
        
        return cls(dimension=dimension, symbolic_set=symbolic, constraint=box_constraint)
    
    @classmethod
    def from_symbolic(cls, symbolic_set: sp.Set, dimension: int) -> 'PhaseSpace':
        """
        Factory: X defined by arbitrary sympy Set.
        
        Constraint will be auto-compiled from symbolic representation.
        Use this when you need symbolic operations and can accept
        compilation overhead.
        
        Args:
            symbolic_set: SymPy set representation
            dimension: Phase space dimension (must match set dimension)
            
        Returns:
            PhaseSpace with custom symbolic set (constraint auto-compiled)
        """
        return cls(dimension=dimension, symbolic_set=symbolic_set)
    
    @classmethod
    def from_constraint(
        cls, 
        dimension: int, 
        constraint: Callable[[NDArray[np.float64]], bool]
    ) -> 'PhaseSpace':
        """
        Factory: X defined by callable constraint only (no symbolic).
        
        Use this for performance-critical applications where you don't need
        symbolic operations. The constraint is used directly without any
        compilation overhead.
        
        Args:
            dimension: Phase space dimension n
            constraint: Callable that returns True if x ∈ X, False otherwise
            
        Returns:
            PhaseSpace with fast constraint-only validation
            
        Example:
            >>> # Unit disk: {(x, y) : x² + y² < 1}
            >>> def unit_disk(x):
            ...     return x[0]**2 + x[1]**2 < 1.0
            >>> phase_space = PhaseSpace.from_constraint(2, unit_disk)
        """
        return cls(dimension=dimension, constraint=constraint)


@dataclass
class TimeHorizon:
    """
    Time horizon T ⊆ ℝ for non-autonomous systems.
    
    Supports:
    - Unbounded: T = ℝ (default)
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
        Check if t ∈ T.
        
        Args:
            t: Time point to test
            
        Returns:
            bool: True if t ∈ T, False otherwise
        """
        if self.constraint is not None:
            return self.constraint(t)
        elif self.bounds is not None:
            return self.bounds[0] <= t <= self.bounds[1]
        else:
            return True  # T = ℝ
    
    @classmethod
    def real_line(cls) -> 'TimeHorizon':
        """
        Factory: T = ℝ (entire real line).
        
        This is the DEFAULT time horizon for non-autonomous systems.
        
        Returns:
            TimeHorizon representing ℝ
        """
        return cls()
    
    @classmethod
    def interval(cls, t_min: float, t_max: float) -> 'TimeHorizon':
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
            raise ValueError(f"Time horizon must have t_min < t_max, got ({t_min}, {t_max})")
        return cls(bounds=(t_min, t_max))