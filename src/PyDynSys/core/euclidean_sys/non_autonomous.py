"""Non-autonomous Euclidean dynamical systems: dx/dt = F(x, t)"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .base import EuclideanDS
from ..types import (
    NonAutonomousVectorField, 
    IvpParams, 
    SciPyIvpSolution, 
    PhaseSpace,
    TimeHorizon
)


class NonAutonomousEuclideanDS(EuclideanDS):
    """
    Non-autonomous Euclidean dynamical system.
    
    Systems where dx/dt = F(x, t) with F: ℝⁿ × ℝ → ℝⁿ explicitly
    time-dependent. Equivalently, ∂F/∂t ≠ 0 for some (x, t).
    
    Mathematical Properties:
    - Flow φ_t(x, t_0) depends on initial time t_0
    - Can be autonomized by augmenting state: y = (x, t), dy/dt = (F(x,t), 1)
    - Useful for periodically forced systems, time-varying parameters
    
    Example:
        >>> # Driven harmonic oscillator: x'' + x = sin(t)
        >>> def driven_oscillator(x, t):
        ...     return np.array([x[1], -x[0] + np.sin(t)])
        >>> phase_space = PhaseSpace.euclidean(2)
        >>> time_horizon = TimeHorizon.real_line()
        >>> sys = NonAutonomousEuclideanDS(
        ...     dimension=2, 
        ...     vector_field=driven_oscillator,
        ...     phase_space=phase_space,
        ...     time_horizon=time_horizon
        ... )
    """
    
    def __init__(
        self, 
        dimension: int, 
        vector_field: NonAutonomousVectorField,
        phase_space: PhaseSpace = None,
        time_horizon: TimeHorizon = None
    ):
        """
        Initialize non-autonomous system.
        
        Args:
            dimension (int): Phase space dimension n
            vector_field: Function F(x, t) -> dx/dt mapping ℝⁿ × ℝ → ℝⁿ
            phase_space: Phase space X ⊆ ℝⁿ (defaults to X = ℝⁿ)
            time_horizon: Time domain T ⊆ ℝ (defaults to T = ℝ)
            
        Raises:
            ValueError: If dimension ≤ 0 or phase_space dimension mismatch
        """
        # Enforce defaults
        if phase_space is None:
            phase_space = PhaseSpace.euclidean(dimension)
        if time_horizon is None:
            time_horizon = TimeHorizon.real_line()
        
        super().__init__(dimension, phase_space)
        self.vector_field = vector_field
        self.time_horizon = time_horizon
    
    
    def _validate_time_span(self, t_span: Tuple[float, float]) -> None:
        """
        Override base validation to also check time horizon membership.
        
        For non-autonomous systems, we must verify that both endpoints of the
        integration interval lie within the time horizon T ⊆ ℝ.
        
        Args:
            t_span: (t_start, t_end) interval
            
        Raises:
            ValueError: If t_start == t_end or if either endpoint ∉ T
        """
        # First, perform base validation (non-zero length)
        super()._validate_time_span(t_span)
        
        # Then check time horizon membership
        t_start, t_end = t_span
        
        if not self.time_horizon.contains(t_start):
            raise ValueError(
                f"Initial time t_start = {t_start} is not in time horizon T. "
                f"Time horizon bounds: {self.time_horizon.bounds}"
            )
        
        if not self.time_horizon.contains(t_end):
            raise ValueError(
                f"Final time t_end = {t_end} is not in time horizon T. "
                f"Time horizon bounds: {self.time_horizon.bounds}"
            )
    
    
    def trajectory(
        self,
        initial_state: NDArray[np.float64],
        t_span: Tuple[float, float],
        t_eval: NDArray[np.float64],
        method: str = 'RK45',
        dense_output: bool = True
    ) -> SciPyIvpSolution:
        """
        Solve initial value problem: dx/dt = F(x, t), x(t_0) = x_0.
        
        For non-autonomous systems, initial time t_0 matters! The solution
        depends on both the initial state and the initial time.
        
        Args:
            initial_state: Initial condition x(t_0) ∈ ℝⁿ
            t_span: Integration interval (t_start, t_end)
            t_eval: Time points for explicit evaluation
            method: ODE solver (default 'RK45')
            dense_output: If True, returns continuous interpolant
            
        Returns:
            SciPyIvpSolution: Wrapped scipy solution with attributes:
                - t: Evaluation times (array)
                - y: States at each time (array of shape (n, len(t)))
                - sol: Continuous interpolant (if dense_output=True)
                - success: Whether integration succeeded
                - message: Termination reason
            
        Raises:
            ValueError: If state dimension incorrect, t_span invalid, 
                       or t_eval points outside valid range
            
        Implementation Notes:
            - Solutions cached by (initial_state, t_span, t_eval, method)
            - Integrator uses λ(t, x): self.vector_field(x, t) (uses both parameters)
            - Supports forward and backward integration via scipy
            - No existence/uniqueness checks performed (TODO: future)
        """
        # Validation
        self._validate_state(initial_state)
        self._validate_time_span(t_span)
        self._validate_t_eval(t_eval, t_span)
        
        # Check cache (includes method now)
        cache_key = IvpParams(
            initial_conditions=tuple(initial_state),
            t_span=t_span,
            t_eval_tuple=tuple(t_eval),
            method=method
        )
        
        if cache_key in self._solutions_cache:
            return self._solutions_cache[cache_key]
        
        # Solve IVP
        # NOTE: scipy signature is (t, y) and we use both
        solution_raw = solve_ivp(
            fun=lambda t, y: self.vector_field(y, t),  # Use both time and state
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method=method,
            dense_output=dense_output
        )
        
        # Wrap and cache
        wrapped_solution = SciPyIvpSolution(raw_solution=solution_raw)
        self._solutions_cache[cache_key] = wrapped_solution
        
        return wrapped_solution
    
    
    def evolve(
        self,
        initial_state: NDArray[np.float64],
        t0: float,
        dt: float,
        method: str = 'RK45'
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
            
        Returns:
            NDArray[np.float64]: Estimated state at t_0 + dt
            
        Raises:
            ValueError: If state dimension incorrect or dt ≤ 0
            
        Example:
            >>> # Driven oscillator evolves differently at different times
            >>> x0 = np.array([1.0, 0.0])
            >>> sys.evolve(x0, t0=0.0, dt=0.1)   # Different from
            >>> sys.evolve(x0, t0=10.0, dt=0.1)  # this!
        """
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