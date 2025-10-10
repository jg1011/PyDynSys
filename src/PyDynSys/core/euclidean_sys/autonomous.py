"""Autonomous Euclidean dynamical systems: dx/dt = F(x)"""

from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from types import SimpleNamespace

from .base import EuclideanDS
from ..types import AutonomousVectorField, IvpParams, SciPyIvpSolution, PhaseSpace


class AutonomousEuclideanDS(EuclideanDS):
    """
    Autonomous Euclidean dynamical system.
    
    Systems where dx/dt = F(x) with F: ℝⁿ → ℝⁿ independent of time.
    Equivalently, ∂F/∂t ≡ 0. 
    
    Mathematical Properties:
    - Flow φ_t(x) forms a semi-group: φ_s(φ_t(x)) = φ_{s+t}(x)
    - Time-translation invariant: shifting t_0 doesn't change trajectory shape
    - Equilibria x* satisfy F(x*) = 0
    
    Example:
        >>> # Harmonic oscillator: x'' + x = 0 → (x, y)' = (y, -x)
        >>> def harmonic_oscillator(x):
        ...     return np.array([x[1], -x[0]])
        >>> phase_space = PhaseSpace.euclidean(2)
        >>> sys = AutonomousEuclideanDS(
        ...     dimension=2, 
        ...     vector_field=harmonic_oscillator,
        ...     phase_space=phase_space
        ... )
    """
    
    def __init__(
        self, 
        dimension: int, 
        vector_field: AutonomousVectorField,
        phase_space: PhaseSpace = None
    ):
        """
        Initialize autonomous system.
        
        Args:
            dimension (int): Phase space dimension n
            vector_field: Function F(x) -> dx/dt mapping ℝⁿ → ℝⁿ
            phase_space: Phase space X ⊆ ℝⁿ (defaults to X = ℝⁿ)
            
        Raises:
            ValueError: If dimension ≤ 0 or phase_space dimension mismatch
        """
        # Enforce default: X = ℝⁿ with symbolic representation
        if phase_space is None:
            phase_space = PhaseSpace.euclidean(dimension)
        
        super().__init__(dimension, phase_space)
        self.vector_field = vector_field
        # NOTE: Could validate vector_field signature here via inspect
    
    
    def trajectory(
        self,
        initial_state: NDArray[np.float64],
        t_span: Tuple[float, float],
        t_eval: NDArray[np.float64],
        method: str = 'RK45',
        dense_output: bool = True
    ) -> SciPyIvpSolution:
        """
        Solve initial value problem: dx/dt = F(x), x(t_0) = x_0 over interval I.
        
        Supports bidirectional integration: if t_span = (t_0, t_end) where 
        t_0 is interior to the t_eval range, performs:
        - Backward integration: t_0 → min(t_eval)
        - Forward integration: t_0 → max(t_eval)
        Then concatenates results.
        
        Args:
            initial_state: Initial condition x(t_0) ∈ ℝⁿ
            t_span: Effective integration bounds (t_start, t_end)
            t_eval: Evaluation points (may span both sides of an interior t_0)
            method: ODE solver (default 'RK45' = 5th order Runge-Kutta)
            dense_output: If True, returns continuous interpolant sol(t)
            
        Returns:
            SciPyIvpSolution: Wrapped scipy solution with attributes:
                - t: Evaluation times (array)
                - y: States at each time (array of shape (n, len(t)))
                - sol: Continuous interpolant (if dense_output=True, None for bidirectional)
                - success: Whether integration succeeded
                - message: Termination reason
                
        Raises:
            ValueError: If state dimension incorrect, t_span invalid, 
                       or t_eval points outside valid range
            
        Implementation Notes:
            - Solutions cached by (initial_state, t_span, t_eval, method)
            - Integrator uses λ(t, x): self.vector_field(x) (ignores t)
            - Bidirectional: if t_0 interior to [min(t_eval), max(t_eval)]
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
        
        # Determine if bidirectional integration needed
        t_0 = t_span[0]  # Assume first element is initial time
        t_min, t_max = np.min(t_eval), np.max(t_eval)
        
        needs_backward = t_0 > t_min
        needs_forward = t_0 < t_max
        
        if needs_backward and needs_forward:
            # Bidirectional case: integrate backwards and forwards
            solution_raw = self._solve_bidirectional(
                initial_state, t_0, t_eval, method, dense_output
            )
        else:
            # Unidirectional case: standard scipy call
            solution_raw = solve_ivp(
                fun=lambda t, y: self.vector_field(y),  # Ignore time parameter
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
    
    def _solve_bidirectional(
        self,
        initial_state: NDArray[np.float64],
        t_0: float,
        t_eval: NDArray[np.float64],
        method: str,
        dense_output: bool
    ) -> Any:
        """
        Perform bidirectional integration: backward from t_0, forward from t_0.
        
        This handles the case where flow is desired on an open interval I around
        t_0, e.g., I = (-1, 1) with t_0 = 0.
        
        Args:
            initial_state: x(t_0)
            t_0: Initial time (interior to t_eval range)
            t_eval: Full evaluation array
            method: Solver method
            dense_output: Whether to generate interpolant (not supported for bidirectional)
            
        Returns:
            Concatenated solution mimicking scipy OdeResult structure
            
        NOTE: Dense output (continuous interpolant) not supported for bidirectional
        integration as it would require custom interpolation across t_0 discontinuity.
        """
        # Split t_eval into backward and forward
        t_eval_backward = t_eval[t_eval < t_0]
        t_eval_forward = t_eval[t_eval >= t_0]
        
        # Backward integration: t_0 → min(t_eval)
        if len(t_eval_backward) > 0:
            sol_backward = solve_ivp(
                fun=lambda t, y: self.vector_field(y),
                t_span=(t_0, t_eval_backward[0]),  # t_0 to minimum
                y0=initial_state,
                t_eval=t_eval_backward[::-1],  # Reverse for increasing order
                method=method,
                dense_output=False
            )
        else:
            sol_backward = None
        
        # Forward integration: t_0 → max(t_eval)
        if len(t_eval_forward) > 0:
            sol_forward = solve_ivp(
                fun=lambda t, y: self.vector_field(y),
                t_span=(t_0, t_eval_forward[-1]),  # t_0 to maximum
                y0=initial_state,
                t_eval=t_eval_forward,
                method=method,
                dense_output=False
            )
        else:
            sol_forward = None
        
        # Concatenate results
        if sol_backward and sol_forward:
            # Exclude duplicate t_0 point from forward (it's at index 0)
            t_combined = np.concatenate([
                sol_backward.t[::-1],  # Reverse backward times to ascending order
                sol_forward.t[1:]       # Skip t_0 duplicate
            ])
            y_combined = np.concatenate([
                sol_backward.y[:, ::-1],  # Reverse backward states
                sol_forward.y[:, 1:]       # Skip t_0 duplicate
            ], axis=1)
            success = sol_backward.success and sol_forward.success
            message = f"Bidirectional integration: backward {sol_backward.message}, forward {sol_forward.message}"
        elif sol_backward:
            t_combined = sol_backward.t[::-1]
            y_combined = sol_backward.y[:, ::-1]
            success = sol_backward.success
            message = f"Backward integration: {sol_backward.message}"
        else:  # sol_forward only
            t_combined = sol_forward.t
            y_combined = sol_forward.y
            success = sol_forward.success
            message = f"Forward integration: {sol_forward.message}"
        
        # Create combined result object (mimicking scipy OdeResult)
        combined_result = SimpleNamespace(
            t=t_combined,
            y=y_combined,
            sol=None,  # Dense output not supported for bidirectional
            success=success,
            message=message
        )
        
        return combined_result
    
    
    def evolve(
        self,
        initial_state: NDArray[np.float64],
        t0: float,
        dt: float,
        method: str = 'RK45'
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
            
        Returns:
            NDArray[np.float64]: Estimated state at t_0 + dt
            
        Raises:
            ValueError: If state dimension incorrect or dt ≤ 0
            
        NOTE: For backwards evolution or flow on intervals around t_0,
        use trajectory() instead which supports bidirectional integration.
        """
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