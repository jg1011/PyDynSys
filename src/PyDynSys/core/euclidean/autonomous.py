"""Autonomous Euclidean dynamical systems: dx/dt = F(x)"""

from typing import Tuple, Any, List
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from .base import EuclideanDS
from ..types import AutonomousVectorField, TrajectoryCacheKey, SciPyIvpSolution
from .phase_space import PhaseSpace
from .trajectory import EuclideanTrajectorySegment, EuclideanTrajectory


class AutonomousEuclideanDS(EuclideanDS):
    """
    Autonomous Euclidean dynamical system.
    
    Systems where dx/dt = F(x) with F: R^n -> R^n independent of time.
    Equivalently, Partial(df, dt) = 0. 
    
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
    
    
        ### --- Constructor --- ###
        
    
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
            vector_field: Function F(x) -> dx/dt mapping R^n -> R^n
            phase_space: Phase space X subset R^n (defaults to X = R^n)
            
        Raises:
            ValueError: If dimension <= 0 or phase_space dimension mismatch
        """
        # Enforce default: X = R^n with symbolic representation
        if phase_space is None:
            phase_space = PhaseSpace.euclidean(dimension)
        
        super().__init__(dimension, phase_space)
        self._vector_field = vector_field
        # NOTE: Could validate vector_field signature here via inspect
    
    
        ### --- EuclideanDS Method Impls --- ###
        
        
    @property
    def vector_field(self) -> AutonomousVectorField:
        """
        Vector field defining the dynamical system.
        """
        return self._vector_field
    
    
    def trajectory(
        self,
        initial_state: NDArray[np.float64],
        t_span: Tuple[float, float],
        t_eval: NDArray[np.float64],
        method: str = 'RK45',
        dense_output: bool = True
    ) -> EuclideanTrajectory:
        """
        Solve initial value problem: dx/dt = F(x), x(t_0) = x_0 over interval I.
        
        Supports bidirectional integration: if t_span = (t_0, t_end) where 
        t_0 is interior to the t_eval range, performs:
        - Backward integration: t_0 → min(t_eval)
        - Forward integration: t_0 → max(t_eval)
        Then concatenates results.
        
        Args:
            initial_state: Initial condition x(t_0) ∈ R^n
            t_span: Effective integration bounds (t_start, t_end)
            t_eval: Evaluation points (may span both sides of an interior t_0)
            method: ODE solver (default 'RK45' = 5th order Runge-Kutta w/ adaptive step size)
            dense_output: If True, returns continuous interpolant sol(t)
                -> NOTE: I am yet to see a use case where not true is useful, and it comes 
                at a minimal computational cost.
            
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
            - Integrator uses λ(t, x): self.vector_field(x) (ignores t)
            - Bidirectional: if t_0 interior to [min(t_eval), max(t_eval)]
            - No existence/uniqueness checks performed (TODO: future)
        """
        # Validation
        self._validate_state(initial_state)
        self._validate_time_span(t_span)
        self._validate_t_eval(t_eval, t_span)
        
        # ====================================================================
        # CACHING STRATEGY
        # ====================================================================
        # Check cache (removed method field - trajectories can be multi-method)
        # 
        # Cache key includes: (x_0, t_0, t_eval) but NOT method
        # This allows cache hits even with different methods, which is reasonable because:
        #   1. Different methods solve the same IVP (should give similar results)
        #   2. Multi-method trajectories are valid (bidirectional with different methods)
        #   3. Simpler cache logic (no need to track method combinations)
        #
        # For autonomous systems, t_0 doesn't affect trajectory shape (time-translation
        # invariant), but we store it for consistency with non-autonomous systems and
        # potential future optimizations (e.g., recognizing time-shifted duplicates).
        cache_key = TrajectoryCacheKey(
            initial_conditions=tuple(initial_state),
            initial_time=t_span[0],  # Store t_0 (typically 0 for autonomous, but can vary)
            t_eval_tuple=tuple(t_eval)
        )
        
        # Cache hit: Return previously computed trajectory (may have been computed with different method!)
        if cache_key in self._solutions_cache:
            return self._solutions_cache[cache_key]
        
        # ====================================================================
        # BIDIRECTIONAL INTEGRATION DETECTION
        # ====================================================================
        # Determine if bidirectional integration needed
        # 
        # Standard (unidirectional) integration: t_0 at boundary of t_eval range
        #   Example: t_0=0, t_eval=[0, 0.1, 0.2, ..., 10]
        #   Flow computed in ONE direction: forward from t_0 to max(t_eval)
        #
        # Bidirectional integration: t_0 INTERIOR to t_eval range
        #   Example: t_0=0, t_eval=[-5, -4, ..., -0.1, 0, 0.1, ..., 4, 5]
        #   Flow computed in TWO directions: 
        #     - Backward from t_0 to min(t_eval)
        #     - Forward from t_0 to max(t_eval)
        #   This is useful for studying orbits, limit cycles, and phase portraits
        #   where we want to see the trajectory evolve both forward and backward in time
        #
        # NOTE: scipy.solve_ivp does NOT support bidirectional integration natively,
        # so we manually split into two IVPs and merge the resulting segments.
        
        t_0 = t_span[0]  # Assume first element is initial time
        t_min, t_max = np.min(t_eval), np.max(t_eval)
        
        # Check if we need backward integration (t_0 above minimum)
        needs_backward = t_0 > t_min
        # Check if we need forward integration (t_0 below maximum)
        needs_forward = t_0 < t_max
        
        if needs_backward and needs_forward:
            # BIDIRECTIONAL CASE: t_0 is interior to [t_min, t_max]
            # Bidirectional case: integrate backwards and forwards
            # Create two segments and let Trajectory handle tangent domain at t_0
            # 
            # DESIGN DECISION: Why not manually concatenate like before?
            # ----------------------------------------------------------
            # Old approach (removed): Manually concatenate t and y arrays, return single
            #   SciPyIvpSolution with sol=None (no interpolant possible)
            # 
            # New approach: Create two segments, let EuclideanTrajectory.from_segments()
            #   handle merging. Benefits:
            #   1. Preserves BOTH interpolants (one for each segment)
            #   2. Allows seamless interpolation via trajectory.interpolate(t) dispatch
            #   3. Consistent with manual trajectory composition workflow
            #   4. Cleaner separation of concerns (merging logic in one place)
            sol_backward_raw, sol_forward_raw = self._solve_bidirectional_raw(
                initial_state, t_0, t_eval, method, dense_output
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
            
            # Merge via trajectory (handles tangent domain automatically)
            # The segments share t_0 → average merge policy merges the shared point
            # Result: Single trajectory with 2 segments, tangent at t_0
            trajectory = EuclideanTrajectory.from_segments([seg_backward, seg_forward])
        else:
            # UNIDIRECTIONAL CASE: Standard forward or backward integration
            # t_0 is at boundary of t_eval range (not interior)
            # 
            # Even though this is a single scipy solve_ivp call, we STILL wrap it
            # in EuclideanTrajectory (with one segment) for API consistency.
            # 
            # WHY? Uniform return type simplifies user code:
            #   result = sys.trajectory(...)  # Always get EuclideanTrajectory
            #   result.interpolate(t)          # Always works the same way
            #   result.segments                # Always accessible (just 1 segment here)
            # 
            # User doesn't need to check "is this bidirectional or unidirectional?"
            solution_raw = solve_ivp(
                fun=lambda t, y: self.vector_field(y),  # Ignore time parameter (autonomous)
                t_span=t_span,
                y0=initial_state,
                t_eval=t_eval,
                method=method,
                dense_output=dense_output
            )
            
            # Wrap scipy solution in segment, then wrap segment in trajectory
            # Result: EuclideanTrajectory with 1 segment
            segment = EuclideanTrajectorySegment.from_scipy_solution(
                SciPyIvpSolution(raw_solution=solution_raw),
                method=method
            )
            trajectory = EuclideanTrajectory.from_segments([segment])
        
        # Cache and return
        self._solutions_cache[cache_key] = trajectory
        return trajectory
    
    
        ### --- Public Methods --- ###
        
    
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
        concatenation. Concatenation is handled by EuclideanTrajectory.from_segments().
        
        This handles the case where flow is desired on an open interval I around
        t_0, e.g., I = (-1, 1) with t_0 = 0.
        
        Args:
            initial_state: x(t_0)
            t_0: Initial time (interior to t_eval range)
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
        # Split t_eval into backward and forward portions around t_0
        # 
        # Key insight: We need BOTH segments to include t_0 to ensure they are TANGENT
        # (i.e., they share the boundary point) with no gap in the trajectory domain.
        #
        # Without this fix, we had a gap:
        #   Backward: [..., -0.01, -0.005]  (ends NEAR t_0 but not AT t_0)
        #   Forward:  [0.005, 0.01, ...]    (starts NEAR t_0 but not AT t_0)
        #   Gap: (-0.005, 0.005) where interpolation fails!
        #
        # With this fix, segments are tangent:
        #   Backward: [..., -0.01, -0.005, 0.0]  (ends AT t_0)
        #   Forward:  [0.0, 0.005, 0.01, ...]    (starts AT t_0)
        #   No gap: domains touch at t_0
        
        t_eval_backward = t_eval[t_eval < t_0]   # Points strictly before t_0
        t_eval_forward = t_eval[t_eval >= t_0]   # Points at or after t_0
        
        # ====================================================================
        # BACKWARD INTEGRATION: t_0 → min(t_eval)
        # ====================================================================
        # Include t_0 in backward to ensure tangent domain with forward segment
        # This is CRITICAL for avoiding gaps in the trajectory domain
        if len(t_eval_backward) > 0:
            # Append t_0 to backward evaluation points
            # This ensures backward segment ends exactly at t_0
            t_eval_backward_with_t0 = np.append(t_eval_backward, t_0)
            sol_backward = solve_ivp(
                fun=lambda t, y: self.vector_field(y),
                t_span=(t_0, t_eval_backward[0]),  # t_0 to minimum
                y0=initial_state,
                t_eval=t_eval_backward_with_t0[::-1],  # Reverse for increasing order
                method=method,
                dense_output=dense_output  # Allow dense output for segment
            )
        else:
            sol_backward = None
        
        # ====================================================================
        # FORWARD INTEGRATION: t_0 → max(t_eval)
        # ====================================================================
        # Ensure t_0 is at the start to guarantee tangent domain
        # 
        # Edge case: When t_eval = linspace(a, b, n), t_0 may or may not be
        # exactly in the array due to floating-point spacing.
        #
        # Example problematic case:
        #   t_eval = linspace(-2, 2, 400) might give:
        #   [..., -0.005, 0.005, ...] where 0.0 is MISSING!
        #   
        # Even if we filter t_eval[t_eval >= t_0], we might get:
        #   t_eval_forward = [0.005, 0.01, ...] missing the exact t_0=0.0
        #
        # Solution: Explicitly prepend t_0 if not already present (within tolerance)
        if len(t_eval_forward) > 0:
            # Check if t_0 is already the first point (within floating-point tolerance)
            # Using 1e-10 tolerance to handle numerical precision issues
            if abs(t_eval_forward[0] - t_0) > 1e-10:
                # t_0 is NOT in t_eval_forward, prepend it explicitly
                # This ensures forward segment starts exactly at t_0
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
                dense_output=dense_output  # Allow dense output for segment
            )
        else:
            sol_forward = None
        
        # Return raw solutions (let Trajectory handle merging)
        return sol_backward, sol_forward