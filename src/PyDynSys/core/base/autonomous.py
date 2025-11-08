"""
Autonomous Euclidean dynamical systems: dx/dt = F(x)

Implements dual ABC architecture:
- _AutonomousEuclideanDS (ABC): Abstract base class (internal)
- AutonomousEuclideanDS: Default implementation using scipy (public API)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, List, Optional, Union, Set, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
import sympy as syp

from .common import _DynSys
from ..properties.registry import _PropertyRegistry
from ..support.vector_field import AutVectorField
from ..support.trajectory import TrajectorySegment, Trajectory
from ..support.phase_space import PhaseSpace
from ..support.cache import TrajectoryCache, TrajectoryCacheKey
from ..types import (
    SciPyIvpSolution,
    SymbolicODE,
    SystemParameters
)
from ..utils.sym import SymbolicSystemBuilder


class _AutDynSys(ABC, _DynSys, _PropertyRegistry):
    """
    Abstract base class for autonomous Euclidean dynamical systems.
    
    Mathematical Definition:
    System: (X, F) where X ⊆ R^n is phase space, F: X → R^n is vector field
    Evolution: dx/dt = F(x)
    
    Properties:
    - Time-translation invariant: trajectory shape independent of initial time
    - Flow forms semigroup: φ_s(φ_t(x)) = φ_{s+t}(x)
    
    This ABC defines the contract that all autonomous systems must implement:
    - trajectory(): Solve IVP
    - vector_field: Get vector field
    
    NOTE: Factory methods are provided by the concrete implementation class,
    not by this ABC.
    
    NOTE: For examples of usage, see the concrete implementation class
    `AutDynSys` which provides factory methods and can be instantiated.
    """
    
    
    ### --- Abstract Methods --- ###
    
    
    @abstractmethod
    def _solve_trajectory(
        self,
        initial_state: NDArray[np.float64],
        t_span: Tuple[float, float],
        t_eval: NDArray[np.float64],
        **solver_kwargs: Any
    ) -> Trajectory:
        """
        Primitive operation for solving an IVP. Subclasses must implement this.
        
        This method should contain ONLY the solver-specific logic and should NOT
        implement any caching or validation.
        """
        pass
    
    @property
    @abstractmethod
    def vector_field(self) -> AutVectorField:
        """
        Vector field F: R^n → R^n.
        
        Abstract property - subclasses provide implementation.
        """
        pass

    @abstractmethod
    def get_system_signature(self) -> tuple:
        """
        Return a unique, hashable signature for the system's current state.

        This is used by the caching system to detect if the system's defining
        parameters have changed.
        """
        pass

    ### --- Public API --- ###

    def trajectory(
        self,
        initial_state: NDArray[np.float64],
        t_span: Tuple[float, float],
        t_eval: NDArray[np.float64],
        **solver_kwargs: Any
    ) -> Trajectory:
        """
        Solve initial value problem: dx/dt = F(x), x(t_0) = x_0.

        This method acts as a "template," handling the caching and validation
        scaffolding before delegating the core computation to the subclass's
        _solve_trajectory implementation.
        """
        if self._trajectory_cache is None:
            return self._solve_trajectory(initial_state, t_span, t_eval, **solver_kwargs)

        # Caching Template Logic
        raw_key = TrajectoryCacheKey(
            system_signature=self.get_system_signature(),
            initial_state=initial_state,
            t_span=t_span,
            t_eval=t_eval,
            t0=None,  # Not used for autonomous systems
            solver_options=solver_kwargs
        )

        cached_traj = self._trajectory_cache.get(raw_key)
        if cached_traj:
            return cached_traj
        
        new_traj = self._solve_trajectory(initial_state, t_span, t_eval, **solver_kwargs)
        self._trajectory_cache.insert(raw_key, new_traj)
        return new_traj


class AutDynSys(_AutDynSys):
    """
    Autonomous dynamical system implementation using scipy.integrate.solve_ivp.
    
    This is the primary public API for autonomous systems, and inherits from the _AutDynSys ABC.
    
    Factories: 
        - from_symbolic(equations, variables, parameters, phase_space) -> AutDynSys
    
    Inherited Abstract Methods: 
        - trajectory(initial_state, t_span, t_eval, method, dense_output) -> Trajectory
        - vector_field() -> AutVectorField 
        -> NOTE: These are implemented in this concrete class, not the _AutDynSys ABC.
    
    Public Methods: 
        - evolve(initial_state, t0, dt, method, validate) -> NDArray[np.float64]
            -> Numerically solve for x(t_0 + dt) from x(t_0) using the vector field with solve_ivp fn. 
            -> NOTE: This is a convenience method for rapid single-step integration without caching.
    
    Example:
        >>> sys = AutDynSys(
        ...     dimension=2,
        ...     vector_field=lambda x: np.array([x[1], -x[0]])
        ... )
        >>> traj = sys.trajectory(x0, t_span=(0, 10), t_eval=np.linspace(0, 10, 100))
        
    Future Work: 
        - Flow method (phi_t(x)) implementation that uses caching and interpolation, and if necessary trajectory 
        recomputation to compute phi_t(x) for fixed t, x. 
            -> The idea is if exists cached trajectory with IC x_0 approx x and t in t_eval range, 
            then phi_t(x) approx trajectory(t). If we have trajectories "either side" of x we can 
            interpolate. We can also compute a grid of trajectories for global phi_t(x) caching on 
            startup.  
    """
    
    
    ### --- Constructor --- ###
    
    
    def __init__(
        self, 
        dimension: int, 
        vector_field: Union[Callable[[NDArray[np.float64]], NDArray[np.float64]], AutVectorField],
        phase_space: PhaseSpace = None,
        cache_size: int | None = 128
    ):
        """
        Initialize autonomous system.
        
        Args:
            dimension (int): Phase space dimension n
            vector_field: Function F(x) -> dx/dt mapping R^n -> R^n, or AutonomousVectorField
            phase_space: Phase space X subset R^n. If None, uses vector_field's phase_space if available,
                        otherwise defaults to X = R^n
            cache_size: The number of recent trajectories to cache. If None or 0, caching is disabled.
            
        Raises:
            ValueError: If dimension <= 0 or phase_space dimension mismatch
        """
        # Handle vector field representation and domain delegation
        if isinstance(vector_field, AutVectorField):
            self._vector_field_repr = vector_field
            self._vector_field = vector_field.callable_field
            
            # Use vector field's domain if phase_space not provided
            if phase_space is None:
                if vector_field.phase_space is not None:
                    phase_space = vector_field.phase_space
                else:
                    phase_space = PhaseSpace.full(dimension)
            elif vector_field.phase_space is not None:
                # Validate consistency (optional - could be relaxed)
                if vector_field.phase_space != phase_space:
                    import warnings
                    warnings.warn(
                        f"Phase space mismatch: system phase_space != vector_field.phase_space. "
                        f"Using system's phase_space.",
                        UserWarning,
                        stacklevel=2
                    )
        else:
            self._vector_field = vector_field
            self._vector_field_repr = None
            # Enforce default: X = R^n
            if phase_space is None:
                phase_space = PhaseSpace.full(dimension)
        
        # Set instance attributes
        self.dimension = dimension
        self.phase_space = phase_space
        self._trajectory_cache = TrajectoryCache(size=cache_size) if cache_size and cache_size > 0 else None
        
        # Initialize property registry (initializes _properties dict)
        _PropertyRegistry.__init__(self)
    
    
    ### --- Factory Methods --- ###
    
    
    @classmethod
    def from_symbolic(
        cls,
        equations: SymbolicODE,
        variables: List[syp.Function],
        parameters: SystemParameters = None,
        phase_space: PhaseSpace = None
    ) -> 'AutDynSys':
        """
        Factory: Construct from symbolic equations.
        
        Process:
        1. Parse symbolic equations
        2. Build vector field (symbolic + callable)
        3. Detect properties (linear, hamiltonian, etc.)
        4. Return appropriate composed class
        
        Args:
            equations: Symbolic ODE in form [d(x_i)/dt - F_i(x), ...]
            variables: Dependent variables as SymPy Function objects
            parameters: Optional parameter substitution dict
            phase_space: Phase space X subset R^n (defaults to X = R^n)
            
        Returns:
            System instance (may be composed class with mixins if properties detected)
            
        Raises:
            ValueError: If system is not first-order or not autonomous
        """
        # Build vector field using SymbolicSystemBuilder
        result = SymbolicSystemBuilder.build_vector_field(
            equations, variables, parameters
        )
        
        # Ensure system is autonomous
        if not result.is_autonomous:
            raise ValueError(
                "from_symbolic() called on AutonomousEuclideanDS but system is non-autonomous. "
                "Use NonAutonomousEuclideanDS.from_symbolic() instead."
            )
        
        # result.vector_field is now an AutonomousVectorField
        vector_field = result.vector_field
        
        # Use vector field's phase_space if available and phase_space not provided
        if phase_space is None:
            if isinstance(vector_field, AutVectorField) and vector_field.phase_space is not None:
                phase_space = vector_field.phase_space
            else:
                phase_space = PhaseSpace.full(result.dimension)
        
        # Detect properties (for now, just linear - hamiltonian coming later)
        # Use callable_field for property detection
        callable_field = vector_field.callable_field if isinstance(vector_field, AutVectorField) else vector_field
        properties = _detect_properties(callable_field, result.dimension)
        
        # Build appropriate class based on properties
        return _build_with_properties(
            properties=properties,
            vector_field=vector_field,
            dimension=result.dimension,
            phase_space=phase_space
        )
    
    
    
    ### --- Abstract Method Implementations --- ###
    
    
    def get_system_signature(self) -> tuple:
        # This is a placeholder implementation. A more robust version would
        # inspect the vector field's bytecode or a user-provided params dict.
        # For symbolic fields, it should hash the equations.
        if self._vector_field_repr:
            return (str(self._vector_field_repr.symbolic_expr),)
        return (self._vector_field.__hash__(),)


    @property
    def vector_field(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """
        Vector field defining the dynamical system.
        """
        return self._vector_field
    
    
    def _solve_trajectory(
        self,
        initial_state: NDArray[np.float64],
        t_span: Tuple[float, float],
        t_eval: NDArray[np.float64],
        **solver_kwargs: Any
    ) -> Trajectory:
        """
        Solve initial value problem: dx/dt = F(x), x(t_0) = x_0 over interval I.
        
        Supports bidirectional integration: if t_span = (t_0, t_end) where 
        t_0 is interior to the t_eval range, performs:
        - Backward integration: t_0 → min(t_eval)
        - Forward integration: t_0 → max(t_eval)
        Then concatenates results.
        
        For autonomous systems, the initial time t_0 = t_span[0]. Time-translation
        invariance means the absolute value of t_0 doesn't affect the trajectory shape.
        
        Args:
            initial_state (NDArray[np.float64]): Initial condition x(t_0) ∈ R^n
            t_span (Tuple[float, float]): Integration bounds (t_start, t_end) where t_start = t_0
            t_eval (NDArray[np.float64]): Evaluation points (may span both sides of an interior t_0)
            method (str): ODE solver (default 'RK45' = 5th order Runge-Kutta w/ adaptive step size)
            dense_output (bool): If True, returns continuous interpolant sol(t) (default True)
            validate (bool): If True, validates input parameters (default True). \n
                - Set to False to skip validation for performance when inputs are guaranteed valid.
            
        Returns:
            Trajectory: Trajectory object with attributes:
                - t: Evaluation times (property, concatenated from segments)
                - y: States at each time (property, concatenated from segments)
                - segments: List of trajectory segments
                - interpolate(t): Method for continuous evaluation at time t
                - domains: List of domain intervals for each segment
                
        Raises:
            ValueError: If state dimension incorrect, t_span invalid, 
                       or t_eval points outside valid range (only if validate=True)
        """
        # Validation (optional for performance)
        validate = solver_kwargs.pop('validate', True)
        if validate:
            self._validate_state(initial_state)
            self._validate_time_span(t_span)
            self._validate_t_eval(t_eval, t_span)
        
        # ====================================================================
        # BIDIRECTIONAL INTEGRATION DETECTION
        # ====================================================================
        t_0 = t_span[0]  # Assume first element is initial time
        t_min, t_max = np.min(t_eval), np.max(t_eval)
        
        # Check if we need backward integration (t_0 above minimum)
        needs_backward = t_0 > t_min
        # Check if we need forward integration (t_0 below maximum)
        needs_forward = t_0 < t_max
        
        method = solver_kwargs.get('method', 'RK45')
        dense_output = solver_kwargs.get('dense_output', True)
        
        if needs_backward and needs_forward:
            # BIDIRECTIONAL CASE: t_0 is interior to [t_min, t_max]
            sol_backward_raw, sol_forward_raw = self._solve_bidirectional_raw(
                initial_state, t_0, t_eval, method, dense_output
            )
            
            # Wrap raw scipy solutions in segments
            seg_backward = TrajectorySegment.from_scipy_solution(
                SciPyIvpSolution(raw_solution=sol_backward_raw),
                method=method
            )
            seg_forward = TrajectorySegment.from_scipy_solution(
                SciPyIvpSolution(raw_solution=sol_forward_raw),
                method=method
            )
            
            # Merge via trajectory (handles tangent domain automatically)
            trajectory = Trajectory.from_segments([seg_backward, seg_forward])
        else:
            # UNIDIRECTIONAL CASE: Standard forward or backward integration
            solution_raw = solve_ivp(
                fun=lambda t, y: self.vector_field(y),  # Ignore time parameter (autonomous)
                t_span=t_span,
                y0=initial_state,
                t_eval=t_eval,
                method=method,
                dense_output=dense_output
            )
            
            # Wrap scipy solution in segment, then wrap segment in trajectory
            segment = TrajectorySegment.from_scipy_solution(
                SciPyIvpSolution(raw_solution=solution_raw),
                method=method
            )
            trajectory = Trajectory.from_segments([segment])
        
        # Return trajectory
        return trajectory
    
    
    ### --- Public Methods --- ###
    
    
    def evolve(
        self,
        initial_state: NDArray[np.float64],
        t0: float,
        dt: float,
        method: str = 'RK45',
        validate: bool = True
    ) -> NDArray[np.float64]:
        """
        Single time-step evolution: estimate x(t_0 + dt) from x(t_0).
        
        Convenience wrapper for rapid single-step integration without caching.
        For repeated evaluations or bidirectional integration, use trajectory().
        
        Args:
            initial_state: State at time t_0
            t0: Initial time
            dt: Time step (must be positive)
            method: ODE solver
            validate: If True, validates input parameters (default True).
                     Set to False to skip validation for performance when inputs are guaranteed valid.
            
        Returns:
            NDArray[np.float64]: Estimated state at t_0 + dt
            
        Raises:
            ValueError: If state dimension incorrect or dt ≤ 0 (only if validate=True)
        """
        # Validation (optional for performance)
        if validate:
            self._validate_state(initial_state)
            self._validate_time_step(dt)
        
        # Single-step integration without caching
        result = solve_ivp(
            fun=lambda t, y: self.vector_field(y),
            t_span=(t0, t0 + dt),
            y0=initial_state,
            t_eval=[t0 + dt],
            method=method,
            dense_output=False  # No interpolant needed
        )
        
        return result.y[:, 0]  # Extract final state vector
    
    
    ### --- Dunder Methods --- ### 
    
    def __repr__(self) -> str:
        return f"AutDynSys(dimension={self.dimension}, phase_space={self.phase_space}, vector_field={self.vector_field})"
    def __str__(self) -> str:
        return f"AutDynSys(dimension={self.dimension}, phase_space={self.phase_space}, vector_field={self.vector_field})"
    
    ### --- Private Methods --- ###
    
    
    def _solve_bidirectional_raw(
        self,
        initial_state: NDArray[np.float64],
        t_0: float,
        t_eval: NDArray[np.float64],
        method: str,
        dense_output: bool
    ) -> Tuple[Any, Any]:
        """
        Perform bidirectional integration: backward from t_0, forward from t_0.
        
        Returns raw scipy OdeResult objects from both integrations, without
        concatenation. Concatenation is handled by Trajectory.from_segments().
        
        Args:
            initial_state: x(t_0)
            t_0: Initial time (interior to t_eval range)
            t_eval: Full evaluation array
            method: Solver method
            dense_output: Whether to generate interpolant for each segment
            
        Returns:
            Tuple[Any, Any]: (sol_backward, sol_forward) raw scipy OdeResult objects
        """
        # Split t_eval into backward and forward portions around t_0
        t_eval_backward = t_eval[t_eval < t_0]   # Points strictly before t_0
        t_eval_forward = t_eval[t_eval >= t_0]   # Points at or after t_0
        
        # BACKWARD INTEGRATION: t_0 → min(t_eval)
        if len(t_eval_backward) > 0:
            # Append t_0 to backward evaluation points
            t_eval_backward_with_t0 = np.append(t_eval_backward, t_0)
            sol_backward = solve_ivp(
                fun=lambda t, y: self.vector_field(y),
                t_span=(t_0, t_eval_backward[0]),  # t_0 to minimum
                y0=initial_state,
                t_eval=t_eval_backward_with_t0[::-1],  # Reverse for increasing order
                method=method,
                dense_output=dense_output
            )
        else:
            sol_backward = None
        
        # FORWARD INTEGRATION: t_0 → max(t_eval)
        if len(t_eval_forward) > 0:
            # Check if t_0 is already the first point (within floating-point tolerance)
            if abs(t_eval_forward[0] - t_0) > 1e-10:
                # t_0 is NOT in t_eval_forward, prepend it explicitly
                t_eval_forward_with_t0 = np.concatenate([[t_0], t_eval_forward])
            else:
                # t_0 is already present, no modification needed
                t_eval_forward_with_t0 = t_eval_forward
            
            sol_forward = solve_ivp(
                fun=lambda t, y: self.vector_field(y),
                t_span=(t_0, t_eval_forward[-1]),  # t_0 to maximum
                y0=initial_state,
                t_eval=t_eval_forward_with_t0,
                method=method,
                dense_output=dense_output
            )
        else:
            sol_forward = None
        
        # Return raw solutions (let Trajectory handle merging)
        return sol_backward, sol_forward


### --- Helper Functions --- ###


def _detect_properties(
    vector_field: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dimension: int
) -> Set[str]:
    """
    Detect system properties from vector field.
    
    Currently detects:
    - linear: If F(x) = Ax for some matrix A
    
    Future: Will detect hamiltonian, planar, etc.
    
    Args:
        vector_field: Vector field function
        dimension: System dimension
        
    Returns:
        Set of detected property names
    """
    properties: Set[str] = set()
    
    # Detect linearity (simplified version - full implementation would be more robust)
    # For now, we'll detect linearity in the factory when building from symbolic
    # This is a placeholder for property detection logic
    
    return properties


def _build_with_properties(
    properties: Set[str],
    vector_field: Union[Callable[[NDArray[np.float64]], NDArray[np.float64]], AutVectorField],
    dimension: int,
    phase_space: PhaseSpace
) -> AutDynSys:
    """
    Build system class with detected properties.
    
    Args:
        properties: Set of detected property names
        vector_field: Vector field function
        dimension: System dimension
        phase_space: Phase space
        
    Returns:
        System instance (may be composed class if properties detected)
    """
    # For now, always return concrete base class
    # Property detection and composition will be enhanced later
    # when we add more mixins
    
    if 'linear' in properties:
        # Would return LinearAutonomousEuclideanDS
        # For now, just return concrete (linear detection not fully implemented)
        pass
    
    return AutDynSys(
        dimension=dimension,
        vector_field=vector_field,
        phase_space=phase_space
    )

