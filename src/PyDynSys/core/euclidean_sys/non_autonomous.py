"""Non-autonomous Euclidean dynamical systems: dx/dt = F(x, t)"""

from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from types import SimpleNamespace

from .base import EuclideanDS
from ..types import (
    NonAutonomousVectorField, 
    TrajectoryCacheKey, 
    SciPyIvpSolution, 
)
from .phase_space import EuclideanPhaseSpace
from .time_horizon import RealLineTimeHorizon
from .trajectory import EuclideanTrajectorySegment, EuclideanTrajectory


class NonAutonomousEuclideanDS(EuclideanDS):
    """
    Non-autonomous Euclidean dynamical system.
    
    Systems where dx/dt = F(x, t) with F: R^n x R → R^n explicitly
    time-dependent. Equivalently, PartialDerivative(F, t) nonzero at some (x, t).
    
    Mathematical Properties:
    - Flow phi_t(x, t_0) depends on initial time t_0
    - Can be autonomized by augmenting state: y = (x, t), dy/dt = (F(x,t), 1)
    - Useful for periodically forced systems, time-varying parameters
    
    NOTE: All non autonomous systems can be autonomized by augmenting state: y = (x, t), 
    dy/dt = (F(x,t), 1). However, this can alter topological properties and is not always 
    desirable, hence our support for non-autonomous systems.
    
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
        phase_space: EuclideanPhaseSpace = None,
        time_horizon: RealLineTimeHorizon = None
    ):
        """
        Initialize non-autonomous system.
        
        Args:
            dimension (int): Phase space dimension n
            vector_field: Function F(x, t): R^n x R → R^n, where F(x, t) = dx/dt
            phase_space (EuclideanPhaseSpace): Phase space X subset of R^n (defaults to X = R^n)
            time_horizon (RealLineTimeHorizon): Time domain T subset of R (defaults to T = R)
            
        Raises:
            ValueError: If dimension ≤ 0 or phase_space dimension mismatch
        """
        # Enforce defaults
        if phase_space is None:
            phase_space = EuclideanPhaseSpace.euclidean(dimension)
        if time_horizon is None:
            time_horizon = RealLineTimeHorizon.real_line()
        
        super().__init__(dimension, phase_space)
        self.vector_field = vector_field
        self.time_horizon = time_horizon
    
    
    def _validate_time_span(self, t_span: Tuple[float, float]) -> None:
        """
        Override base validation to also check time horizon membership.
        
        For non-autonomous systems, we must verify that both endpoints of the
        integration interval lie within the time horizon T subset of R.
        
        Args:
            t_span (Tuple[float, float]): (t_start, t_end) interval
            
        Raises:
            ValueError: If t_start == t_end or if either endpoint not in T
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
        t0: float,
        t_span: Tuple[float, float],
        t_eval: NDArray[np.float64],
        method: str = 'RK45',
        dense_output: bool = True
    ) -> EuclideanTrajectory:
        """
        Solve initial value problem: dx/dt = F(x, t), x(t_0) = x_0.
        
        For non-autonomous systems, initial time t_0 matters and is required.
        If t_eval spans around t_0 (min(t_eval) < t_0 < max(t_eval)), performs
        bidirectional integration by integrating backward to min(t_eval) and
        forward to max(t_eval), then concatenating results (sol=None in this mode).
        
        Args:
            initial_state (NDArray[np.float64]): Initial condition x(t_0) in Phase space X
            t_span (Tuple[float, float]): Integration interval (t_start, t_end)
            t_eval (NDArray[np.float64]): Time points for explicit evaluation
            method (str): ODE solver (default 'RK45')
            dense_output (bool): If True, returns continuous interpolant (sol(t))
                -> I am yet to see a use case where not true is needed.
            
        Returns:
            EuclideanTrajectory: Trajectory object with attributes:
                - t: Evaluation times (property, concatenated from segments)
                - y: States at each time (property, concatenated from segments)
                - segments: List of trajectory segments
                - interpolate(t): Method for continuous evaluation at time t
                - domains: List of domain intervals for each segment 
            
        Raises:
            ValueError: If state dimension incorrect, t_span invalid, 
                       or t_eval points outside valid range
            
        Implementation Notes:
            - Solutions cached by (initial_state, t_span, t_eval, method)
            - Integrator uses lambda (t, x): self.vector_field(x, t) (uses both parameters)
            - Supports forward and backward integration via scipy
            - No existence/uniqueness checks performed (TODO: future)
        """
        # Validation
        self._validate_state(initial_state)
        t_min_eval = float(np.min(t_eval))
        t_max_eval = float(np.max(t_eval))
        spans_around_t0 = (t_min_eval < t0) and (t0 < t_max_eval)

        if spans_around_t0:
            # Bidirectional mode: ensure t0 and eval bounds are within time horizon
            if not self.time_horizon.contains(t0):
                raise ValueError(
                    f"Initial time t0 = {t0} is not in time horizon T. "
                    f"Time horizon bounds: {self.time_horizon.bounds}"
                )
            if not self.time_horizon.contains(t_min_eval) or not self.time_horizon.contains(t_max_eval):
                raise ValueError(
                    f"Evaluation interval [{t_min_eval}, {t_max_eval}] not fully contained in time horizon T. "
                    f"Time horizon bounds: {self.time_horizon.bounds}"
                )
        else:
            # Unidirectional mode: enforce standard rules and require t0 to coincide with t_span[0]
            if t0 != t_span[0]:
                raise ValueError(
                    f"For unidirectional integration, t0 must equal t_span[0]. Got t0={t0}, t_span[0]={t_span[0]}"
                )
            self._validate_time_span(t_span)
            self._validate_t_eval(t_eval, t_span)
        
        # ====================================================================
        # CACHING STRATEGY (NON-AUTONOMOUS CRITICAL DIFFERENCE)
        # ====================================================================
        # Check cache (removed method field - trajectories can be multi-method)
        # 
        # CRITICAL for non-autonomous systems: initial_time t0 is part of the cache key!
        # 
        # Unlike autonomous systems where time-translation doesn't matter,
        # non-autonomous systems have F(x,t) depending explicitly on time.
        # 
        # Example demonstrating why t0 is critical:
        #   System: dx/dt = -x + sin(t)
        #   Same x_0 = 1.0 but different t0:
        #     - Solve from t0=0:   affected by sin(0)=0    → one trajectory
        #     - Solve from t0=π/2: affected by sin(π/2)=1 → DIFFERENT trajectory!
        # 
        # Therefore, (x_0, t0=0) and (x_0, t0=π/2) must be cached separately.
        # 
        # Cache key includes: (x_0, t_0, t_eval) but NOT method
        cache_key = TrajectoryCacheKey(
            initial_conditions=tuple(initial_state),
            initial_time=t0,  # Use actual t0 parameter (CRITICAL for non-autonomous!)
            t_eval_tuple=tuple(t_eval)
        )
        
        # Cache hit: Return previously computed trajectory
        if cache_key in self._solutions_cache:
            return self._solutions_cache[cache_key]
        
        # Solve IVP
        if spans_around_t0:
            # BIDIRECTIONAL CASE: t0 is interior to [min(t_eval), max(t_eval)]
            # 
            # For non-autonomous systems, this is particularly important because
            # the vector field F(x,t) changes with time. We need accurate solutions
            # both before AND after t0 to capture time-dependent behavior.
            #
            # Example: Periodically forced oscillator with F(x,t) = -x + sin(ωt)
            #   At t<0: forcing is negative (sin negative)
            #   At t>0: forcing is positive (sin positive)
            # Bidirectional integration captures this transition.
            
            # Bidirectional case: create two segments and let Trajectory handle tangent domain
            sol_backward_raw, sol_forward_raw = self._solve_bidirectional_raw(
                initial_state=initial_state,
                t0=t0,
                t_eval=t_eval,
                method=method,
                dense_output=dense_output,
            )
            
            # Wrap raw scipy solutions in segments
            # Each segment preserves its interpolant if dense_output=True
            seg_backward = EuclideanTrajectorySegment.from_scipy_solution(
                SciPyIvpSolution(raw_solution=sol_backward_raw),
                method=method
            )
            seg_forward = EuclideanTrajectorySegment.from_scipy_solution(
                SciPyIvpSolution(raw_solution=sol_forward_raw),
                method=method
            )
            
            # Merge via trajectory (handles tangent domain at t0 automatically)
            trajectory = EuclideanTrajectory.from_segments([seg_backward, seg_forward])
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
            segment = EuclideanTrajectorySegment.from_scipy_solution(
                SciPyIvpSolution(raw_solution=solution_raw),
                method=method
            )
            trajectory = EuclideanTrajectory.from_segments([segment])
        
        # Cache and return
        self._solutions_cache[cache_key] = trajectory
        return trajectory

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
        concatenation. Concatenation is handled by EuclideanTrajectory.from_segments().
        
        Args:
            initial_state: x(t_0)
            t0: Initial time (interior to t_eval range)
            t_eval: Full evaluation array
            method: Solver method
            dense_output: Whether to generate interpolant for each segment
            
        Returns:
            Tuple[Any, Any]: (sol_backward, sol_forward) raw scipy OdeResult objects
            
        NOTE: Each segment can have dense_output if requested. The trajectory
        will handle dispatching interpolation to the correct segment.
        """
        # ====================================================================
        # SPLIT EVALUATION ARRAY FOR BIDIRECTIONAL INTEGRATION
        # ====================================================================
        # Split t_eval into backward and forward portions around t0
        #
        # CRITICAL for non-autonomous systems: We must ensure BOTH segments include t0
        # to create tangent domains (no gap). This is even more important for non-autonomous
        # systems because interpolation across a gap would be undefined - we can't just
        # "fill in" missing values since the vector field F(x,t) depends on time.
        #
        # See autonomous.py::_solve_bidirectional_raw for detailed explanation of the
        # tangent domain issue and why we need to include t0 in both segments.
        
        t_eval_backward = t_eval[t_eval < t0]   # Points strictly before t0
        t_eval_forward = t_eval[t_eval >= t0]   # Points at or after t0

        # ====================================================================
        # BACKWARD INTEGRATION: t0 → min(t_eval)
        # ====================================================================
        # Include t0 in backward to ensure tangent domain with forward segment
        if len(t_eval_backward) > 0:
            # Append t0 to backward evaluation points to ensure segment ends at t0
            t_eval_backward_with_t0 = np.append(t_eval_backward, t0)
            sol_backward = solve_ivp(
                fun=lambda t, y: self.vector_field(y, t),
                t_span=(t0, float(t_eval_backward[0])),
                y0=initial_state,
                t_eval=t_eval_backward_with_t0[::-1],
                method=method,
                dense_output=dense_output,  # Allow dense output for segment
            )
        else:
            sol_backward = None

        # ====================================================================
        # FORWARD INTEGRATION: t0 → max(t_eval)
        # ====================================================================
        # Ensure t0 is at the start to guarantee tangent domain
        # (Same logic as autonomous case - see autonomous.py for detailed explanation)
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
                dense_output=dense_output,  # Allow dense output for segment
            )
        else:
            sol_forward = None

        # Return raw solutions (let Trajectory handle merging)
        return sol_backward, sol_forward
    
    
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