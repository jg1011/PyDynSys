from typing import Optional, Callable
from sympy.sets import ProductSet
import sympy as syp
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class EuclideanPhaseSpace:
    """
    Phase space X subset of R^n with flexible symbolic/callable representation.
    
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
        symbolic_set (syp.Set | None): Optional SymPy set representation
        constraint (Callable | None): Optional callable for fast membership testing
        
    At least one of symbolic_set or constraint must be provided.
    """
    dimension: int
    symbolic_set: Optional[syp.Set] = None
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
        - For R^n (unbounded): return lambda x: True (no constraint)
        - For ProductSets of Intervals: extract bounds, compile to numpy checks
        - For general sets: use sympy's contains (slow, but general)
        
        Raises:
            AssertionError: If symbolic_set is None (should never happen)
        """
        assert self.symbolic_set is not None, "Cannot compile constraint without symbolic_set"
        
        # Special case: R^n (ProductSet of Reals or Reals**n)
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
                values = [syp.Float(float(v)) for v in x]
                elem = syp.Tuple(*values)
                contains_expr = self.symbolic_set.contains(elem)
                # SymPy returns a Boolean or a symbolic expression; coerce if possible
                return bool(contains_expr)
            except Exception:
                return False
        
        return symbolic_constraint
    
    def _is_full_euclidean_space(self) -> bool:
        """
        Check if symbolic set represents R^n.
        
        Returns False if symbolic_set is None.
        """
        if self.symbolic_set is None:
            return False
            
        if isinstance(self.symbolic_set, ProductSet):
            return all(s == syp.Reals for s in self.symbolic_set.args)
        # Also check for Reals**n notation
        if hasattr(self.symbolic_set, 'base') and hasattr(self.symbolic_set, 'exp'):
            return self.symbolic_set.base == syp.Reals and self.symbolic_set.exp == self.dimension
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
            if isinstance(component_set, syp.Interval):
                a = float(component_set.start) if component_set.start.is_finite else -np.inf
                b = float(component_set.end) if component_set.end.is_finite else np.inf
                bounds.append([a, b])
            elif component_set == syp.Reals:
                bounds.append([-np.inf, np.inf])
            else:
                return None  # Not a simple interval
        
        return np.array(bounds, dtype=np.float64)
    
    def contains(self, x: NDArray[np.float64]) -> bool:
        """
        Check if x in X using compiled constraint.
        
        Uses the constraint callable for fast membership testing.
        The constraint is guaranteed to exist after __post_init__.
        
        Args:
            x: Point in R^n to test
            
        Returns:
            bool: True if x in X, False otherwise
        """
        assert self.constraint is not None, "Constraint should be set in __post_init__"
        return self.constraint(x)
    
    @classmethod
    def euclidean(cls, dimension: int) -> 'EuclideanPhaseSpace':
        """
        Factory: X = R^n (full Euclidean space).
        
        This is the DEFAULT phase space for systems without constraints.
        Provides both symbolic representation and optimized constraint for
        best performance (no compilation overhead).
        
        Args:
            dimension (int): Phase space dimension n
        Returns:
            EuclideanPhaseSpace representing R^n with optimal performance
        """
        symbolic = syp.Reals ** dimension
        # Provide constraint directly to avoid compilation overhead
        constraint = lambda x: True
        return cls(dimension=dimension, symbolic_set=symbolic, constraint=constraint)
    
    @classmethod
    def box(cls, bounds: NDArray[np.float64]) -> 'EuclideanPhaseSpace':
        """
        Factory: X = [a_1, b_1] x ... x [a_n, b_n] (box constraints).
        
        Provides both symbolic representation and optimized numpy constraint
        for best performance.
        
        Args:
            bounds: Array of shape (n, 2) with [[a_1, b_1], ..., [a_n, b_n]]
            
        Returns:
            PhaseSpace with box constraints and optimal performance
        """
        dimension = bounds.shape[0]
        intervals = [syp.Interval(bounds[i, 0], bounds[i, 1]) for i in range(dimension)]
        symbolic = ProductSet(*intervals)
        
        # Provide pre-compiled constraint for performance
        def box_constraint(x: NDArray[np.float64]) -> bool:
            return bool(np.all((x >= bounds[:, 0]) & (x <= bounds[:, 1])))
        
        return cls(dimension=dimension, symbolic_set=symbolic, constraint=box_constraint)
    
    @classmethod
    def from_symbolic(cls, symbolic_set: syp.Set, dimension: int) -> 'EuclideanPhaseSpace':
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
    ) -> 'EuclideanPhaseSpace':
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
            >>> # Unit disk: {(x, y) : x^2 + y^2 < 1}
            >>> def unit_disk(x):
            ...     return x[0]**2 + x[1]**2 < 1.0
            >>> phase_space = PhaseSpace.from_constraint(2, unit_disk)
        """
        return cls(dimension=dimension, constraint=constraint)