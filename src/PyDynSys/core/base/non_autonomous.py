"""
Non-autonomous Euclidean dynamical systems: dx/dt = F(x, t)

Implements dual ABC architecture:
- _NonAutDynSys (ABC): Abstract base class (internal)
- NonAutDynSys: Default implementation using scipy (public API)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, List, Optional, Union, Set, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
import sympy as syp

from .common import _DynSys
from ..properties.registry import _PropertyRegistry
from ..support.vector_field import NonAutVectorField
from ..support.trajectory import TrajectorySegment, Trajectory
from ..support.phase_space import PhaseSpace
from ..support.time_horizon import TimeHorizon
from ..types import (
    TrajectoryCacheKey, 
    SciPyIvpSolution,
    SymbolicODE,
    SystemParameters
)
from ..utils.sym import SymbolicSystemBuilder


class _NonAutDynSys(ABC, _DynSys, _PropertyRegistry):
    """
    Abstract base class for non-autonomous Euclidean dynamical systems.
    
    Mathematical Definition: \n
    System: (X, T, F) where X subset R^n is phase space, T subset R is time horizon and 
    F: X x T → R^n is vector field \n
    Evolution: dx/dt = F(x, t) over X x T
    
    Properties: \n
    - Flow phi_t(x, t_0) depends on initial time t_0 \n
    - Can be autonomized by augmenting state: y = (x, t), dy/dt = (F(x,t), 1)
    
    Use cases: \n
    - Periodically forced systems \n
    - Time-varying parameters
    
    This ABC defines the contract that all non-autonomous systems must implement: \n
    - trajectory(): Solve IVP with ICs (x_0, t_0)
    - vector_field: Get vector field F(x, t)
    - time_horizon: Get time horizon T
    
    NOTE: Factory methods are provided by the concrete implementation class,
    not by this ABC.
    """
    
    
    ### --- Abstract Methods --- ###
    
    
    @abstractmethod
    def trajectory(
        self,
        initial_state: NDArray[np.float64],
        t0: float,
        t_span: Tuple[float, float],
        t_eval: NDArray[np.float64],
        method: str = 'RK45',
        dense_output: bool = True
    ) -> Trajectory:
        """
        Solve initial value problem: dx/dt = F(x, t), x(t_0) = x_0.
        
        Abstract method - subclasses implement solver logic.
        
        Args:
            initial_state: Initial condition x0
            t0: Initial time (REQUIRED for non-autonomous systems)
            t_span: Integration bounds (t_start, t_end)
            t_eval: Evaluation time points
            method: Solver method
            dense_output: Whether to generate interpolant
            
        Returns:
            Trajectory: Trajectory object
        """
        pass
    
    
    @property
    @abstractmethod
    def vector_field(self) -> Callable[[NDArray[np.float64], float], NDArray[np.float64]]:
        """
        Vector field F: R^n × R → R^n.
        
        Abstract property - subclasses provide implementation.
        """
        pass
    
    
    @property
    @abstractmethod
    def time_horizon(self) -> TimeHorizon:
        """
        Time horizon T ⊆ R.
        
        Abstract property - subclasses provide implementation.
        """
        pass


class NonAutDynSys(_NonAutDynSys):
    """
    Non-autonomous Euclidean dynamical system implementation using scipy.solve_ivp.
    
    This is the primary public API for non-autonomous systems. 
    
    Provides standard trajectory solving with bidirectional integration support.
    
    Example:
        >>> sys = NonAutDynSys(
        ...     dimension=2,
        ...     vector_field=lambda x, t: np.array([x[1], -x[0] + np.sin(t)])
        ... )
        >>> traj = sys.trajectory(x0, t0=0.0, t_span=(-10, 10), t_eval=np.linspace(-10, 10, 1000))
    """
    
    
    ### --- Constructor --- ###
    
    
    def __init__(
        self, 
        dimension: int, 
        vector_field: Union[Callable[[NDArray[np.float64], float], NDArray[np.float64]], NonAutVectorField],
        phase_space: PhaseSpace = None,
        time_horizon: TimeHorizon = None
    ):
        """
        Initialize non-autonomous system.
        
        Args:
            dimension (int): Phase space dimension n
            vector_field: Function F(x, t): R^n × R → R^n, or NonAutVectorField
            phase_space: Phase space X subset R^n. If None, uses vector_field's phase_space if available,
                        otherwise defaults to X = R^n
            time_horizon: Time domain T subset R. If None, uses vector_field's time_horizon if available,
                        otherwise defaults to T = R
            
        Raises:
            ValueError: If dimension <= 0 or phase_space dimension mismatch
        """
        # Handle vector field representation and domain delegation
        if isinstance(vector_field, NonAutVectorField):
            self._vector_field_repr = vector_field
            self._vector_field = vector_field.callable_field
            
            # Use vector field's domain if not provided
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
            
            if time_horizon is None:
                if vector_field.time_horizon is not None:
                    time_horizon = vector_field.time_horizon
                else:
                    time_horizon = TimeHorizon.real_line()
            elif vector_field.time_horizon is not None:
                # Validate consistency (optional - could be relaxed)
                if vector_field.time_horizon != time_horizon:
                    import warnings
                    warnings.warn(
                        f"Time horizon mismatch: system time_horizon != vector_field.time_horizon. "
                        f"Using system's time_horizon.",
                        UserWarning,
                        stacklevel=2
                    )
        else:
            self._vector_field = vector_field
            self._vector_field_repr = None
            # Enforce defaults
            if phase_space is None:
                phase_space = PhaseSpace.full(dimension)
            if time_horizon is None:
                time_horizon = TimeHorizon.real_line()
        
        # Set instance attributes
        self.dimension = dimension
        self.phase_space = phase_space
        self._solutions_cache: dict = {}
        
        # Initialize property registry (initializes _properties dict)
        _PropertyRegistry.__init__(self)
        
        # Store time horizon
        self._time_horizon = time_horizon
    
    
    ### --- Factory Methods --- ###
    
    
    @classmethod
    def from_symbolic(
        cls,
        equations: SymbolicODE,
        variables: List[syp.Function],
        parameters: SystemParameters = None,
        phase_space: PhaseSpace = None,
        time_horizon: TimeHorizon = None
    ) -> 'NonAutDynSys':
        """
        Factory: Construct from symbolic equations.
        
        Process:
        1. Parse symbolic equations
        2. Build vector field (symbolic + callable)
        3. Detect properties (linear, etc.)
        4. Return appropriate composed class
        
        Args:
            equations: Symbolic ODE in form [d(x_i)/dt - F_i(x, t), ...]
            variables: Dependent variables as SymPy Function objects
            parameters: Optional parameter substitution dict
            phase_space: Phase space X subset R^n (defaults to X = R^n)
            time_horizon: Time horizon T subset R (defaults to T = R)
            
        Returns:
            System instance (may be composed class with mixins if properties detected)
            
        Raises:
            ValueError: If system is not first-order or is autonomous
        """
        # Build vector field using SymbolicSystemBuilder
        result = SymbolicSystemBuilder.build_vector_field(
            equations, variables, parameters
        )
        
        # Ensure system is non-autonomous
        if result.is_autonomous:
            raise ValueError(
                "from_symbolic() called on NonAutDynSys but system is autonomous. "
                "Use AutDynSys.from_symbolic() instead."
            )
        
        # result.vector_field is now a NonAutVectorField
        vector_field = result.vector_field
        
        # Use vector field's domains if available and not provided
        if phase_space is None:
            if isinstance(vector_field, NonAutVectorField) and vector_field.phase_space is not None:
                phase_space = vector_field.phase_space
            else:
                phase_space = PhaseSpace.full(result.dimension)
        if time_horizon is None:
            if isinstance(vector_field, NonAutVectorField) and vector_field.time_horizon is not None:
                time_horizon = vector_field.time_horizon
            else:
                time_horizon = TimeHorizon.real_line()
        
        # Detect properties (for now, just linear - hamiltonian coming later)
        # Use callable_field for property detection
        callable_field = vector_field.callable_field if isinstance(vector_field, NonAutVectorField) else vector_field
        properties = _detect_properties(callable_field, result.dimension)
        
        # Build appropriate class based on properties
        return _build_with_properties(
            properties=properties,
            vector_field=vector_field,
            dimension=result.dimension,
            phase_space=phase_space,
            time_horizon=time_horizon
        )
    
    
    @classmethod
    def from_vector_field(
        cls,
        vector_field: Union[Callable[[NDArray[np.float64], float], NDArray[np.float64]], NonAutVectorField],
        dimension: int,
        phase_space: PhaseSpace = None,
        time_horizon: TimeHorizon = None
    ) -> 'NonAutDynSys':
        """
        Factory: Direct construction from vector field.
        
        Args:
            vector_field: Vector field function or NonAutVectorField
            dimension: Phase space dimension n
            phase_space: Phase space X subset R^n (defaults to X = R^n or vector_field's phase_space)
            time_horizon: Time domain T subset R (defaults to T = R or vector_field's time_horizon)
            
        Returns:
            NonAutDynSys instance
        """
        return cls(
            dimension=dimension,
            vector_field=vector_field,
            phase_space=phase_space,
            time_horizon=time_horizon
        )
    
    
    ### --- Abstract Method Implementations --- ###
    
    
    @property
    def vector_field(self) -> Callable[[NDArray[np.float64], float], NDArray[np.float64]]:
        """
        Vector field defining the dynamical system.
        """
        return self._vector_field
    
    
    @property
    def time_horizon(self) -> TimeHorizon:
        """
        Time horizon T ⊆ R.
        """
        return self._time_horizon
    
    
    @property
    def symbolic_vector_field(self) -> Optional[List[syp.Expr]]:
        """
        Symbolic representation of vector field (if available).
        
        Returns:
            List of symbolic expressions [F_1, ..., F_n], or None if not available
        """
        if self._vector_field_repr:
            return self._vector_field_repr.symbolic_expr
        return None
    
    
    def trajectory(
        self,
        initial_state: NDArray[np.float64],
        t0: float,
        t_span: Tuple[float, float],
        t_eval: NDArray[np.float64],
        method: str = 'RK45',
        dense_output: bool = True,
        validate: bool = True
    ) -> Trajectory:
        """
        Solve initial value problem: dx/dt = F(x, t), x(t_0) = x_0.
        
        For non-autonomous systems, initial time t_0 matters and is required.
        If t_eval spans around t_0 (min(t_eval) < t_0 < max(t_eval)), performs
        bidirectional integration by integrating backward to min(t_eval) and
        forward to max(t_eval), then concatenating results.
        
        Args:
            initial_state (NDArray[np.float64]): Initial condition x(t_0) in Phase space X
            t0 (float): Initial time (REQUIRED for non-autonomous systems)
            t_span (Tuple[float, float]): Integration interval (t_start, t_end)
            t_eval (NDArray[np.float64]): Time points for explicit evaluation
            method (str): ODE solver (default 'RK45')
            dense_output (bool): If True, returns continuous interpolant (sol(t)) (default True)
            validate (bool): If True, validates the input parameters (default True)
            
        Returns:
            Trajectory: Trajectory object with attributes:
                - t: Evaluation times (property, concatenated from segments)
                - y: States at each time (property, concatenated from segments)
                - segments: List of trajectory segments
                - interpolate(t): Method for continuous evaluation at time t
                - domains: List of domain intervals for each segment
                
        Raises:
            ValueError: If state dimension incorrect, t_span invalid, 
                       or t_eval points outside valid range
        """
        # Validation (optional for performance)
        if validate:
            self._validate_state(initial_state)
            self._validate_time_span(t_span)
        
        t_min_eval = float(np.min(t_eval))
        t_max_eval = float(np.max(t_eval))
        spans_around_t0 = (t_min_eval < t0) and (t0 < t_max_eval)

        if spans_around_t0:
            # Bidirectional mode: ensure t0 and eval bounds are within time horizon
            if validate:
                if not self.time_horizon.contains_time(t0):
                    raise ValueError(
                        f"Initial time t0 = {t0} is not in time horizon T. "
                        f"Time horizon bounds: {self.time_horizon.bounds}"
                    )
                if not self.time_horizon.contains_time(t_min_eval) or not self.time_horizon.contains_time(t_max_eval):
                    raise ValueError(
                        f"Evaluation interval [{t_min_eval}, {t_max_eval}] not fully contained in time horizon T. "
                        f"Time horizon bounds: {self.time_horizon.bounds}"
                    )
        else:
            # Unidirectional mode: enforce standard rules and require t0 to coincide with t_span[0]
            if validate:
                if t0 != t_span[0]:
                    raise ValueError(
                        f"For unidirectional integration, t0 must equal t_span[0]. Got t0={t0}, t_span[0]={t_span[0]}"
                    )
                self._validate_t_eval(t_eval, t_span)
        
        # ====================================================================
        # CACHING STRATEGY 
        # ====================================================================
        # CRITICAL: for non-autonomous systems: initial_time t0 is part of the cache key!
        # 
        # Unlike autonomous systems where time-translation doesn't matter,
        # non-autonomous systems have F(x,t) depending explicitly on time.
        cache_key = TrajectoryCacheKey(
            initial_conditions=tuple(initial_state),
            initial_time=t0,  # Use actual t0 parameter 
            t_eval_tuple=tuple(t_eval)
        )
        
        # Cache hit: Return previously computed trajectory
        if cache_key in self._solutions_cache:
            return self._solutions_cache[cache_key]
        
        # Solve IVP
        if spans_around_t0:
            # BIDIRECTIONAL CASE: t0 is interior to [min(t_eval), max(t_eval)]
            sol_backward_raw, sol_forward_raw = self._solve_bidirectional_raw(
                initial_state=initial_state,
                t0=t0,
                t_eval=t_eval,
                method=method,
                dense_output=dense_output,
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
            
            # Merge via trajectory (handles tangent domain at t0 automatically)
            trajectory = Trajectory.from_segments([seg_backward, seg_forward])
        else:
            # Unidirectional case: standard scipy call
            solution_raw = solve_ivp(
                fun=lambda t, y: self.vector_field(y, t), 
                t_span=t_span,
                y0=initial_state,
                t_eval=t_eval,
                method=method,
                dense_output=dense_output
            )
            
            # Wrap in segment, then trajectory
            segment = TrajectorySegment.from_scipy_solution(
                SciPyIvpSolution(raw_solution=solution_raw),
                method=method
            )
            trajectory = Trajectory.from_segments([segment])
        
        # Cache and return
        self._solutions_cache[cache_key] = trajectory
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
        
        For non-autonomous systems, the initial time t_0 is CRUCIAL! 
        The same initial state at different times will evolve differently.
        
        Args:
            initial_state: State at time t_0
            t0: Initial time (ESSENTIAL for non-autonomous systems)
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
            fun=lambda t, y: self.vector_field(y, t),  # Use both time and state
            t_span=(t0, t0 + dt),
            y0=initial_state,
            t_eval=[t0 + dt],
            method=method,
            dense_output=False
        )
        
        return result.y[:, 0]  # Extract final state
    
    
    ### --- Private Methods --- ###
    
    
    def _validate_time_span(self, t_span: Tuple[float, float]) -> None:
        """
        Override base validation to also check time horizon membership.
        
        For non-autonomous systems, we must verify that both endpoints of the
        integration interval lie within the time horizon T subset of R.
        
        Args:
            t_span: (t_start, t_end) interval
            
        Raises:
            ValueError: If t_start == t_end or if either endpoint not in T
        """
        # First, perform base validation (non-zero length)
        super()._validate_time_span(t_span)
        
        # Then check time horizon membership
        t_start, t_end = t_span
        
        if not self.time_horizon.contains_time(t_start):
            raise ValueError(
                f"Initial time t_start = {t_start} is not in time horizon T. "
                f"Time horizon bounds: {self.time_horizon.bounds}"
            )
        
        if not self.time_horizon.contains_time(t_end):
            raise ValueError(
                f"Final time t_end = {t_end} is not in time horizon T. "
                f"Time horizon bounds: {self.time_horizon.bounds}"
            )
    
    
    def _solve_bidirectional_raw(
        self,
        initial_state: NDArray[np.float64],
        t0: float,
        t_eval: NDArray[np.float64],
        method: str,
        dense_output: bool,
    ) -> Tuple[Any, Any]:
        """
        Perform bidirectional integration: backward from t_0, forward from t_0.
        
        Returns raw scipy OdeResult objects from both integrations, without
        concatenation. Concatenation is handled by Trajectory.from_segments().
        
        Args:
            initial_state: x(t_0)
            t0: Initial time (interior to t_eval range)
            t_eval: Full evaluation array
            method: Solver method
            dense_output: Whether to generate interpolant for each segment
            
        Returns:
            Tuple[Any, Any]: (sol_backward, sol_forward) raw scipy OdeResult objects
        """
        # Split t_eval into backward and forward portions around t0
        t_eval_backward = t_eval[t_eval < t0]   # Points strictly before t0
        t_eval_forward = t_eval[t_eval >= t0]   # Points at or after t0

        # BACKWARD INTEGRATION: t0 → min(t_eval)
        if len(t_eval_backward) > 0:
            # Append t0 to backward evaluation points to ensure segment ends at t0
            t_eval_backward_with_t0 = np.append(t_eval_backward, t0)
            sol_backward = solve_ivp(
                fun=lambda t, y: self.vector_field(y, t),
                t_span=(t0, float(t_eval_backward[0])),
                y0=initial_state,
                t_eval=t_eval_backward_with_t0[::-1],
                method=method,
                dense_output=dense_output,
            )
        else:
            sol_backward = None

        # FORWARD INTEGRATION: t0 → max(t_eval)
        if len(t_eval_forward) > 0:
            # Check if t0 is already the first point (within floating-point tolerance)
            if abs(t_eval_forward[0] - t0) > 1e-10:
                # t0 is NOT in t_eval_forward, prepend it to ensure tangent domain
                t_eval_forward_with_t0 = np.concatenate([[t0], t_eval_forward])
            else:
                # t0 is already present, no modification needed
                t_eval_forward_with_t0 = t_eval_forward
            
            sol_forward = solve_ivp(
                fun=lambda t, y: self.vector_field(y, t),
                t_span=(t0, float(t_eval_forward[-1])),
                y0=initial_state,
                t_eval=t_eval_forward_with_t0,
                method=method,
                dense_output=dense_output,
            )
        else:
            sol_forward = None

        # Return raw solutions (let Trajectory handle merging)
        return sol_backward, sol_forward


### --- Helper Functions --- ###


def _detect_properties(
    vector_field: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
    dimension: int
) -> Set[str]:
    """
    Detect system properties from vector field.
    
    Currently detects:
    - linear: If F(x, t) = A(t)x + b(t) for some matrix A(t) and vector b(t)
    
    Future: Will detect other properties.
    
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
    vector_field: Union[Callable[[NDArray[np.float64], float], NDArray[np.float64]], NonAutVectorField],
    dimension: int,
    phase_space: PhaseSpace,
    time_horizon: TimeHorizon
) -> 'NonAutDynSys':  # Returns concrete implementation
    """
    Build system class with detected properties.
    
    Args:
        properties: Set of detected property names
        vector_field: Vector field function or NonAutVectorField
        dimension: System dimension
        phase_space: Phase space
        time_horizon: Time horizon
        
    Returns:
        System instance (may be composed class if properties detected)
    """
    # For now, always return concrete base class
    # Property detection and composition will be enhanced later
    # when we add more mixins
    
    if 'linear' in properties:
        # Would return LinearNonAutonomousEuclideanDS
        # For now, just return concrete (linear detection not fully implemented)
        pass
    
    return NonAutDynSys(
        dimension=dimension,
        vector_field=vector_field,
        phase_space=phase_space,
        time_horizon=time_horizon
    )

