"""
Euclidean Dynamical Systems abstract base class and concrete implementations.

Hierarchy:
    EuclideanDS (ABC) - Common infrastructure and factory
    ├── AutonomousEuclideanDS - Systems where dx/dt = F(x)
    └── NonAutonomousEuclideanDS - Systems where dx/dt = F(x, t)
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
import sympy as sp

from ..types import (
    IvpParams, 
    SciPyIvpSolution, 
    SystemParameters,
    SymbolicODE,
    PhaseSpace,
    TimeHorizon,
)


class EuclideanDS(ABC):
    """
    Abstract base class for Euclidean dynamical systems.
    
    Represents systems of first-order ODEs in R^n of form:
        dx/dt = F(x)       (autonomous)
        dx/dt = F(x, t)    (non-autonomous)
    
    Provides:
    - Shared infrastructure (dimension, solution caching, validation)
    - Abstract interface (subclasses must implement vector_field)
    - Factory method (from_symbolic auto-detects system type)
    - Common utilities (distance, volume, trajectory solving)
    
    NOTE: This class is abstract. Instantiate AutonomousEuclideanDS or
    NonAutonomousEuclideanDS directly, or use from_symbolic factory.
    """
    
    def __init__(self, dimension: int, phase_space: PhaseSpace):
        """
        Initialize Euclidean dynamical system.
        
        Args:
            dimension (int): Phase space dimension n (must be positive)
            phase_space (PhaseSpace): Phase space X ⊆ ℝⁿ
            
        Raises:
            ValueError: If dimension ≤ 0 or phase_space dimension mismatch
            
        NOTE: This constructor is NEVER called directly by users (abstract class).
        It's invoked via super().__init__() from subclasses.
        """
        # Validation
        if dimension <= 0:
            raise ValueError(f"Phase space dimension must be positive, got {dimension}")
        if phase_space.dimension != dimension:
            raise ValueError(
                f"Phase space dimension ({phase_space.dimension}) must match "
                f"system dimension ({dimension})"
            )
        
        self.dimension = dimension
        self.phase_space = phase_space
        self._solutions_cache: Dict[IvpParams, SciPyIvpSolution] = {}
    
    
    # ========================================================================
    # Factory Method - Auto-detects Autonomous vs Non-Autonomous
    # ========================================================================
    
    @classmethod
    def from_symbolic(
        cls,
        equations: SymbolicODE,
        variables: List[sp.Function],
        parameters: SystemParameters = None,
        phase_space: PhaseSpace = None,
        time_horizon: TimeHorizon = None
    ) -> 'EuclideanDS':
        """
        Factory method: construct system from symbolic equations.
        
        Auto-detects whether system is autonomous and returns appropriate
        subclass instance (AutonomousEuclideanDS or NonAutonomousEuclideanDS).
        
        Args:
            equations: Symbolic ODE in form [d(x_i)/dt - F_i(x, t), ...]
            variables: Dependent variables as SymPy Function objects
            parameters: Optional parameter substitution dict
            phase_space: Phase space X ⊆ ℝⁿ (defaults to X = ℝⁿ)
            time_horizon: Time horizon T ⊆ ℝ (only used for non-autonomous, defaults to T = ℝ)
            
        Returns:
            AutonomousEuclideanDS if ∂F/∂t ≡ 0, else NonAutonomousEuclideanDS
            
        Raises:
            ValueError: If system is not first-order
            
        Example:
            >>> t = sp.symbols('t')
            >>> x, y = sp.symbols('x y', cls=sp.Function)
            >>> x, y = x(t), y(t)
            >>> # Autonomous system
            >>> eqs = [sp.diff(x, t) - y, sp.diff(y, t) + x]
            >>> sys = EuclideanDS.from_symbolic(eqs, [x, y])
            >>> type(sys).__name__
            'AutonomousEuclideanDS'
        """
        from ..sym_utils import SymbolicSystemBuilder
        
        # Build vector field and detect system type
        result = SymbolicSystemBuilder.build_vector_field(
            equations, variables, parameters
        )
        
        # Create default phase space if not provided
        if phase_space is None:
            phase_space = PhaseSpace.euclidean(result.dimension)
        
        # Dispatch to appropriate subclass
        if result.is_autonomous:
            from .autonomous import AutonomousEuclideanDS
            return AutonomousEuclideanDS(
                dimension=result.dimension,
                vector_field=result.vector_field,
                phase_space=phase_space
            )
        else:
            from .non_autonomous import NonAutonomousEuclideanDS
            # Create default time horizon if not provided
            if time_horizon is None:
                time_horizon = TimeHorizon.real_line()
            return NonAutonomousEuclideanDS(
                dimension=result.dimension,
                vector_field=result.vector_field,
                phase_space=phase_space,
                time_horizon=time_horizon
            )
        
    
    # ========================================================================
    # Abstract Interface - Subclasses Must Implement
    # ========================================================================
    
    # NOTE: We do NOT define abstract vector_field() here because signature
    # differs between autonomous (x) and non-autonomous (x, t). Each subclass
    # defines its own signature.
    
    
    # ========================================================================
    # Validation
    # ========================================================================
    
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
        if not self.phase_space.contains(state):
            raise ValueError(
                f"State {state} is not in phase space X. "
                f"Phase space constraints violated."
            )
    
    def _validate_time_span(self, t_span: Tuple[float, float]) -> None:
        """
        Validate time span for integration.
        
        Allows:
        - Forward integration: t_start < t_end
        - Backward integration: t_start > t_end
        - Bidirectional: t_start interior to t_eval range (checked separately)
        
        Forbids:
        - Zero-length intervals: t_start == t_end
        
        Args:
            t_span: (t_start, t_end) interval
            
        Raises:
            ValueError: If t_start == t_end
        """
        t_start, t_end = t_span
        if t_start == t_end:
            raise ValueError(
                f"Time span must have nonzero length, got t_span = {t_span}"
            )
    
    def _validate_t_eval(
        self, 
        t_eval: NDArray[np.float64], 
        t_span: Tuple[float, float]
    ) -> None:
        """
        Validate evaluation time points.
        
        For unidirectional integration: t_eval must be within [min(t_span), max(t_span)]
        For bidirectional: t_span[0] is t_0, t_eval can span around it
        
        Args:
            t_eval: Time points for solution evaluation
            t_span: Integration interval
            
        Raises:
            ValueError: If t_eval points outside valid range for unidirectional integration
        """
        t_start, t_end = t_span
        t_min, t_max = min(t_start, t_end), max(t_start, t_end)
        
        # Check if bidirectional pattern: t_start is interior to t_eval
        t_eval_min, t_eval_max = np.min(t_eval), np.max(t_eval)
        is_bidirectional = (t_eval_min < t_start < t_eval_max)
        
        if not is_bidirectional:
            # Standard unidirectional: t_eval must be within t_span bounds
            if np.any(t_eval < t_min) or np.any(t_eval > t_max):
                raise ValueError(
                    f"All t_eval points must be in [{t_min}, {t_max}], "
                    f"got range [{t_eval_min}, {t_eval_max}]"
                )
    
    def _validate_time_step(self, dt: float) -> None:
        """
        Validate time step for single evolution.
        
        Args:
            dt: Time step
            
        Raises:
            ValueError: If dt <= 0
        """
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got dt = {dt}")
    
    
    # ========================================================================
    # Phase Space Geometry
    # ========================================================================
    
    def phase_space_distance(
        self, 
        x1: NDArray[np.float64], 
        x2: NDArray[np.float64]
    ) -> float:
        """
        Euclidean distance between two phase space points.
        
        Computes L2 norm: ||x1 - x2||_2 = sqrt(Σ(x1_i - x2_i)²)
        
        Args:
            x1, x2: Points in phase space R^n
            
        Returns:
            float: Euclidean distance
            
        Raises:
            ValueError: If x1 or x2 have incorrect dimension
        """
        self._validate_state(x1)
        self._validate_state(x2)
        return float(np.linalg.norm(x1 - x2))
    
    def phase_space_volume(self, points: NDArray[np.float64]) -> np.float64:
        """
        Estimate volume of point cloud in phase space.
        
        Uses ConvexHull if available, falls back to bounding box volume.
        Returns np.inf if phase space is unbounded and points span infinite extent.
        
        Args:
            points: Array of shape (n_points, dimension)
            
        Returns:
            np.float64: Estimated volume. Returns np.float64(np.inf) if phase 
                        space is unbounded and points span infinite extent.
                        Returns 0.0 if insufficient points for hull.
            
        Raises:
            ValueError: If points.shape[1] != dimension
        """
        if points.shape[1] != self.dimension:
            raise ValueError(
                f"Points must have dimension {self.dimension}, "
                f"got shape {points.shape}"
            )
        
        if points.shape[0] < self.dimension + 1:
            return np.float64(0.0)  # Insufficient points for non-degenerate hull
        
        # Check for infinite extent
        extents = np.max(points, axis=0) - np.min(points, axis=0)
        if np.any(np.isinf(extents)):
            return np.float64(np.inf)
        
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            return np.float64(hull.volume)
        except ImportError:
            # Fallback: bounding box volume
            return np.float64(np.prod(extents))
    
    
    # ========================================================================
    # Trajectory Computation - To Be Implemented in Subclasses
    # ========================================================================
    
    # These methods will be implemented in subclasses with type-specific
    # integrator signatures (autonomous vs non-autonomous)
    
    # NOTE: We could define abstract versions here, but they'd have identical
    # signatures across subclasses (only internal implementation differs).
    # Leaving as concrete implementations in subclasses.