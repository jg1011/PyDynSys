"""Trajectory representation for Euclidean dynamical systems."""

from typing import Tuple, List, Optional, Callable, Dict, Any
import numpy as np
from numpy.typing import NDArray

from ..types import SciPyIvpSolution, TrajectorySegmentMergePolicy


class EuclideanTrajectorySegment: 
    """
    Represents a numerically computed segment of a trajectory on a monotone increasing evaluation space. 
    
    A segment corresponds to a single continuous solution from scipy.solve_ivp, representing
    the trajectory over a contiguous time interval with a single interpolant.
    
    Fields: 
        t (NDArray[np.float64]): Monotone increasing array of evaluation times, shape (len(t),)
        y (NDArray[np.float64]): Array of trajectory evaluations x(t), shape (n, len(t))
        domain (Tuple[float, float]): Time domain [t[0], t[-1]] where segment is defined
        interpolant (Optional[Callable]): Continuous interpolant x(t) on domain, or None
        method (str): ODE solver method used ('RK45', 'LSODA', etc.)
        meta (Dict[str, Any]): Metadata about numerical solution (success, message, etc.)
    
    Usage:
        Segments are created via from_scipy_solution() factory, not direct instantiation.
        Users primarily interact with EuclideanTrajectory, which aggregates segments.
    """
    
    def __init__(self):
        """
        Private constructor - use from_scipy_solution() factory instead.
        
        Direct instantiation is discouraged to enforce factory pattern and ensure
        proper initialization from scipy solve_ivp results.
        """
        pass
    
    @classmethod
    def from_scipy_solution(
        cls, 
        sol: SciPyIvpSolution, 
        method: str
    ) -> 'EuclideanTrajectorySegment':
        """
        Factory: Create segment from scipy solve_ivp solution.
        
        Handles backward integration (monotone decreasing t) by reversing arrays to
        enforce monotone increasing time convention for all segments.
        
        Args:
            sol (SciPyIvpSolution): Wrapped scipy OdeResult from solve_ivp
            method (str): ODE solver method used (e.g. 'RK45', 'LSODA')
        
        Returns:
            EuclideanTrajectorySegment: Segment with monotone increasing time
        
        Example:
            >>> from scipy.integrate import solve_ivp
            >>> result = solve_ivp(...)
            >>> wrapped = SciPyIvpSolution(raw_solution=result)
            >>> segment = EuclideanTrajectorySegment.from_scipy_solution(wrapped, 'RK45')
        """
        segment = cls()
        
        # Extract time and state arrays from scipy solution
        # t is shape (n_points,), y is shape (n_dim, n_points)
        t = sol.t
        y = sol.y
        
        # CRITICAL: Enforce monotone increasing time convention
        # Backward integration (t_span[0] > t_span[1]) produces t[0] > t[-1]
        # We reverse both arrays to maintain monotone increasing property
        # This is essential for:
        #   1. Consistent domain representation: domain = (t[0], t[-1]) always has t[0] < t[-1]
        #   2. Trajectory composition: merging segments requires consistent ordering
        #   3. Interpolation: Finding containing segment via binary search (future optimization)
        if len(t) > 1 and t[0] > t[-1]:
            # Backward integration detected (time decreases)
            # Reverse time array: [t_n, ..., t_1, t_0] → [t_0, t_1, ..., t_n]
            t = t[::-1]
            # Reverse state columns to match: [:, [n, ..., 1, 0]] → [:, [0, 1, ..., n]]
            y = y[:, ::-1]
        
        # Store arrays and extract domain endpoints
        segment.t = t
        segment.y = y
        # Domain is the time interval [a, b] where segment is defined
        # After reversing (if needed), t[0] is always the start, t[-1] is always the end
        segment.domain = (float(t[0]), float(t[-1]))
        
        # Store interpolant (callable or None)
        # If dense_output=True was used in solve_ivp, sol.sol is a callable f(t) → x(t)
        # This allows continuous evaluation between discrete points in t
        # If dense_output=False, sol.sol is None and interpolation will raise an error
        segment.interpolant = sol.sol
        
        # Store method and metadata for debugging/caching
        # Method string (e.g., 'RK45', 'LSODA') identifies the numerical integrator used
        # Metadata preserves scipy's success flag and termination message
        segment.method = method
        segment.meta = {
            'success': sol.success,
            'message': sol.message,
        }
        
        return segment
    
    def in_domain(self, t: float) -> bool:
        """
        Check if time t is within segment domain.
        
        Args:
            t (float): Time point to check
        
        Returns:
            bool: True if t ∈ [domain[0], domain[1]], False otherwise
        """
        return self.domain[0] <= t <= self.domain[1]
    
    def interpolate(self, t: float) -> NDArray[np.float64]:
        """
        Evaluate interpolant at time t.
        
        Uses scipy's dense output interpolant to compute x(t) continuously within
        the segment domain. Raises errors if t is outside domain or if no interpolant
        is available (dense_output=False in original solve_ivp call).
        
        Args:
            t (float): Time point for evaluation
        
        Returns:
            NDArray[np.float64]: State vector x(t) at time t, shape (n,)
        
        Raises:
            ValueError: If t outside segment domain or interpolant unavailable
        
        Example:
            >>> x_t = segment.interpolate(0.5)  # Evaluate at t=0.5
        """
        if not self.in_domain(t):
            raise ValueError(
                f"Time t={t} outside segment domain {self.domain}. "
                f"Interpolant only valid on [{self.domain[0]}, {self.domain[1]}]."
            )
        
        if self.interpolant is None:
            raise ValueError(
                "No interpolant available (dense_output was False). "
                "Set dense_output=True in trajectory() call to enable interpolation."
            )
        
        # Call scipy interpolant and return as 1D array
        result = self.interpolant(t)
        # Handle both scalar and array returns from interpolant
        if result.ndim == 2:
            return result[:, 0]  # Extract column vector as 1D
        return result
    
    
class EuclideanTrajectory: 
    """
    Represents a numerically computed trajectory on a subset of the real line ℝ.
    
    A trajectory is a composition of one or more segments with disjoint domains,
    providing seamless access to the complete solution across potentially
    non-contiguous time intervals.
    
    Key invariant: All segment domains are disjoint (validated in from_segments).
    
    Fields: 
        segments (List[EuclideanTrajectorySegment]): Trajectory segments in ascending domain order
        domains (List[Tuple[float, float]]): Domain intervals [t_i, t_{i+1}], guaranteed disjoint
        meta (Dict[str, Any]): Aggregate metadata from all segments
    
    Properties:
        t: Concatenated evaluation times from all segments
        y: Concatenated states from all segments
    
    Usage:
        Created via from_segments() factory. Primary user-facing class returned by
        system.trajectory() method. Provides seamless interpolation across segments.
    
    Example:
        >>> sys = AutonomousEuclideanDS(...)
        >>> traj = sys.trajectory(x0, t_span=(0, 10), t_eval=np.linspace(0, 10, 100))
        >>> x_at_5 = traj.interpolate(5.0)  # Seamlessly finds right segment
    """
    
    def __init__(self):
        """
        Private constructor - use from_segments() factory instead.
        """
        pass
    
    @classmethod
    def from_segments(
        cls,
        segments: List[EuclideanTrajectorySegment],
        merge_policy: TrajectorySegmentMergePolicy = 'average'  # Default: average overlapping values
    ) -> 'EuclideanTrajectory':
        """
        Factory: Create trajectory from list of segments, merging overlaps if needed.
        
        Sorts segments by domain start time, detects overlapping domains, and merges
        them according to specified policy. After merging, validates that all final
        segment domains are disjoint (class invariant).
        
        Args:
            segments (List[EuclideanTrajectorySegment]): Segments to compose
            merge_policy (TrajectorySegmentMergePolicy): Strategy for handling overlaps
                - 'average' (DEFAULT): Average y values in overlap region
                - 'left': Use left segment in overlap
                - 'right': Use right segment in overlap
                - 'stitch': Left interpolant until midpoint, then right
        
        Returns:
            EuclideanTrajectory: Composite trajectory with disjoint segment domains
        
        Raises:
            ValueError: If final domains are not disjoint (merging failed)
            NotImplementedError: If merge_policy is not 'average' (others not yet implemented)
        
        Example:
            >>> seg1 = EuclideanTrajectorySegment.from_scipy_solution(sol1, 'RK45')
            >>> seg2 = EuclideanTrajectorySegment.from_scipy_solution(sol2, 'RK45')
            >>> traj = EuclideanTrajectory.from_segments([seg1, seg2])
        """
        if not segments:
            raise ValueError("Cannot create trajectory from empty segment list")
        
        # STEP 1: Sort segments by domain start time
        # Ensures we process segments in chronological order for overlap detection
        # e.g., [seg(5,10), seg(0,3), seg(2,7)] → [seg(0,3), seg(2,7), seg(5,10)]
        sorted_segments = sorted(segments, key=lambda seg: seg.domain[0])
        
        # STEP 2: Detect and merge overlaps
        # Overlapping domains arise from:
        #   - Bidirectional integration: segments may share boundary point (tangent)
        #   - Re-solving: computing [0,1.5] and [0.5,2] separately, then composing
        #   - Patching: fixing numerical issues in specific regions
        # 
        # We process pairs sequentially, merging where overlaps exist.
        # After merging, the invariant "all domains disjoint" is enforced.
        merged_segments = []
        i = 0
        while i < len(sorted_segments):
            current = sorted_segments[i]
            
            # Check if next segment overlaps with current
            # Overlaps only occur between consecutive segments (after sorting)
            if i + 1 < len(sorted_segments):
                next_seg = sorted_segments[i + 1]
                # Returns overlap interval [a, b] if domains intersect, else None
                overlap = cls._detect_overlap(current, next_seg)
                
                if overlap is not None:
                    # Overlapping segments - merge them
                    if merge_policy == 'average':
                        merged = cls._merge_segments_average(current, next_seg, overlap)
                        merged_segments.append(merged)
                        i += 2  # Skip both segments (merged into one)
                    elif merge_policy == 'left':
                        raise NotImplementedError(
                            f"Merge policy 'left' not yet implemented. Use 'average' for now."
                        )
                    elif merge_policy == 'right':
                        raise NotImplementedError(
                            f"Merge policy 'right' not yet implemented. Use 'average' for now."
                        )
                    elif merge_policy == 'stitch':
                        raise NotImplementedError(
                            f"Merge policy 'stitch' not yet implemented. Use 'average' for now."
                        )
                    else:
                        raise ValueError(f"Unknown merge policy: {merge_policy}")
                else:
                    # Disjoint segments - keep current
                    merged_segments.append(current)
                    i += 1
            else:
                # Last segment - no next to check
                merged_segments.append(current)
                i += 1
        
        # Create trajectory instance
        trajectory = cls()
        trajectory.segments = merged_segments
        trajectory.domains = [seg.domain for seg in merged_segments]
        
        # Aggregate metadata
        trajectory.meta = {
            'all_successful': all(seg.meta['success'] for seg in merged_segments),
            'messages': [seg.meta['message'] for seg in merged_segments],
            'methods': [seg.method for seg in merged_segments],
        }
        
        # Validate disjoint domains (class invariant)
        trajectory._validate_disjoint_domains()
        
        return trajectory
    
    @staticmethod
    def _detect_overlap(
        seg1: EuclideanTrajectorySegment,
        seg2: EuclideanTrajectorySegment
    ) -> Optional[Tuple[float, float]]:
        """
        Detect if two segments have overlapping domains.
        
        Args:
            seg1, seg2: Segments to check (assumed seg1.domain[0] <= seg2.domain[0])
        
        Returns:
            Tuple[float, float]: Overlap interval [a, b] if overlap exists, else None
        """
        # seg1 ends before seg2 starts → disjoint
        if seg1.domain[1] <= seg2.domain[0]:
            return None
        
        # Overlapping: intersection is [max(starts), min(ends)]
        overlap_start = max(seg1.domain[0], seg2.domain[0])
        overlap_end = min(seg1.domain[1], seg2.domain[1])
        
        return (overlap_start, overlap_end)
    
    @staticmethod
    def _merge_segments_average(
        seg1: EuclideanTrajectorySegment,
        seg2: EuclideanTrajectorySegment,
        overlap: Tuple[float, float]
    ) -> EuclideanTrajectorySegment:
        """
        Merge two overlapping segments by averaging y values in overlap region.
        
        MATHEMATICAL CONTEXT:
        --------------------
        When we have two numerical approximations of the same trajectory x(t) over
        overlapping time intervals, we need to reconcile the competing values.
        
        Example scenario:
          Segment 1: [0, 1.5] computed with RK45, gives x(1.0) ≈ 0.5403
          Segment 2: [0.5, 2] computed with DOP853, gives x(1.0) ≈ 0.5404
        
        In overlap [0.5, 1.5], both segments provide approximations. The 'average'
        policy takes the midpoint: x_merged(1.0) = (0.5403 + 0.5404)/2 = 0.54035
        
        This is optimal when both segments have similar accuracy/trust levels.
        
        Strategy:
        1. Identify evaluation points in each region (pre-overlap, overlap, post-overlap)
        2. In overlap: average y values at shared t points (within tolerance 1e-9)
        3. Concatenate regions to form merged segment
        4. Interpolant: Set to None (averaging interpolants non-trivial, future work)
        
        Args:
            seg1, seg2: Overlapping segments (seg1.domain[0] <= seg2.domain[0] assumed)
            overlap: Overlap interval [a, b] where both segments are defined
        
        Returns:
            EuclideanTrajectorySegment: Merged segment spanning union of domains
        """
        overlap_start, overlap_end = overlap
        # Tolerance for identifying shared t values (accounts for floating-point error)
        # Two points at t1 and t2 are considered "same" if |t1 - t2| < 1e-9
        tol = 1e-9
        
        # ====================================================================
        # REGION DECOMPOSITION
        # ====================================================================
        # Split both segments into three regions:
        #   1. Pre-overlap (only in seg1)
        #   2. Overlap (in both segments)
        #   3. Post-overlap (only in seg2)
        #
        # Example: seg1=[0,1.5], seg2=[0.5,2], overlap=[0.5,1.5]
        #   seg1_before: [0, 0.5)
        #   seg1_overlap: [0.5, 1.5]
        #   seg2_overlap: [0.5, 1.5]  (competing values with seg1!)
        #   seg2_after: (1.5, 2]
        
        t1_before = seg1.t[seg1.t < overlap_start]
        y1_before = seg1.y[:, seg1.t < overlap_start]
        
        t1_overlap = seg1.t[(seg1.t >= overlap_start) & (seg1.t <= overlap_end)]
        y1_overlap = seg1.y[:, (seg1.t >= overlap_start) & (seg1.t <= overlap_end)]
        
        t2_overlap = seg2.t[(seg2.t >= overlap_start) & (seg2.t <= overlap_end)]
        y2_overlap = seg2.y[:, (seg2.t >= overlap_start) & (seg2.t <= overlap_end)]
        
        t2_after = seg2.t[seg2.t > overlap_end]
        y2_after = seg2.y[:, seg2.t > overlap_end]
        
        # ====================================================================
        # MERGE OVERLAP REGION VIA AVERAGING
        # ====================================================================
        # Merge overlap region by averaging at shared t points
        #
        # Key challenge: The two segments may have different evaluation grids!
        #   seg1 might evaluate at t = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        #   seg2 might evaluate at t = [0.5, 0.8, 1.0, 1.2, 1.5]
        #
        # Shared points (within tolerance): 0.5, 1.5
        # Unique to seg1: 0.7, 0.9, 1.1, 1.3
        # Unique to seg2: 0.8, 1.0, 1.2
        #
        # Strategy: Take union of all t values, average where both exist
        
        # Union of all t values in overlap (automatically removes duplicates)
        t_overlap_all = np.unique(np.concatenate([t1_overlap, t2_overlap]))
        n_dim = seg1.y.shape[0]  # Phase space dimension
        y_overlap_merged = np.zeros((n_dim, len(t_overlap_all)))
        
        # For each time point in the union, decide how to set the merged value
        for i, t_val in enumerate(t_overlap_all):
            # Check if this time point exists in each segment (within tolerance)
            # Using tolerance to handle floating-point comparison issues
            in_seg1 = np.any(np.abs(t1_overlap - t_val) < tol)
            in_seg2 = np.any(np.abs(t2_overlap - t_val) < tol)
            
            if in_seg1 and in_seg2:
                # CASE 1: Shared point (exists in both segments)
                # Both segments provide a value at this t → AVERAGE them
                # This is the core of the 'average' merge policy
                idx1 = np.argmin(np.abs(t1_overlap - t_val))  # Find closest point in seg1
                idx2 = np.argmin(np.abs(t2_overlap - t_val))  # Find closest point in seg2
                # Average the state vectors: x_merged = (x_seg1 + x_seg2) / 2
                y_overlap_merged[:, i] = (y1_overlap[:, idx1] + y2_overlap[:, idx2]) / 2
                
            elif in_seg1:
                # CASE 2: Only in seg1 (seg2 didn't evaluate here)
                # Keep seg1's value unchanged (no averaging needed)
                idx1 = np.argmin(np.abs(t1_overlap - t_val))
                y_overlap_merged[:, i] = y1_overlap[:, idx1]
                
            else:
                # CASE 3: Only in seg2 (seg1 didn't evaluate here)
                # Keep seg2's value unchanged (no averaging needed)
                idx2 = np.argmin(np.abs(t2_overlap - t_val))
                y_overlap_merged[:, i] = y2_overlap[:, idx2]
        
        # ====================================================================
        # FINAL ASSEMBLY
        # ====================================================================
        # Concatenate all three regions to form the complete merged segment
        # Order: [pre-overlap from seg1] + [merged overlap] + [post-overlap from seg2]
        t_merged = np.concatenate([t1_before, t_overlap_all, t2_after])
        y_merged = np.concatenate([y1_before, y_overlap_merged, y2_after], axis=1)
        
        # Create merged segment
        merged = EuclideanTrajectorySegment()
        merged.t = t_merged
        merged.y = y_merged
        merged.domain = (float(t_merged[0]), float(t_merged[-1]))
        
        # INTERPOLANT LIMITATION:
        # Averaging two interpolants is non-trivial mathematically
        # Each interpolant is a polynomial valid only on its subdomain
        # To create a merged interpolant, we'd need to:
        #   1. Sample both interpolants densely in overlap region
        #   2. Average the samples
        #   3. Fit a new spline to the averaged data
        # This is left as future work. For now, interpolant = None
        merged.interpolant = None
        
        # Method string indicates multi-method composition
        # e.g., "RK45+DOP853" shows this segment combines two different solvers
        merged.method = f"{seg1.method}+{seg2.method}"
        
        # Metadata: Aggregate success flags and messages from both segments
        merged.meta = {
            'success': seg1.meta['success'] and seg2.meta['success'],
            'message': f"Merged: {seg1.meta['message']} | {seg2.meta['message']}",
        }
        
        return merged
    
    def _validate_disjoint_domains(self) -> None:
        """
        Validate that all segment domains are disjoint (class invariant).
        
        CLASS INVARIANT:
        ---------------
        After merging, all segment domains MUST be disjoint (non-overlapping).
        This is enforced by:
          - from_segments(): Detects and merges all overlaps before validation
          - This method: Final check that merging succeeded
        
        Mathematically: For domains [a_i, b_i], we require:
          b_i <= a_{i+1} for all consecutive pairs i, i+1
        
        Tangent domains ([a,b] and [b,c]) are allowed (b_i = a_{i+1}).
        Disjoint domains ([a,b] and [c,d] with b < c) are allowed.
        Overlapping domains ([a,b] and [c,d] with c < b < d) are NOT allowed after merging.
        
        Checks that domains[i][1] <= domains[i+1][0] for all consecutive pairs.
        This ensures no overlap remains after merging.
        
        Raises:
            ValueError: If any domains overlap (indicates bug in merge logic)
        """
        for i in range(len(self.domains) - 1):
            current_end = self.domains[i][1]
            next_start = self.domains[i + 1][0]
            
            # Check for overlap: current segment extends past start of next segment
            if current_end > next_start:
                raise ValueError(
                    f"Segment domains are not disjoint: "
                    f"domain {i} ends at {current_end}, but domain {i+1} starts at {next_start}. "
                    f"Overlapping domains detected after merging. This is a bug in merge logic."
                )
    
    def in_domain(self, t: float) -> bool:
        """
        Check if time t is within trajectory domain (any segment).
        
        Args:
            t (float): Time point to check
        
        Returns:
            bool: True if t is in any segment's domain, False otherwise
        """
        return any(seg.in_domain(t) for seg in self.segments)
    
    def interpolate(self, t: float) -> NDArray[np.float64]:
        """
        Seamlessly interpolate trajectory at time t.
        
        KEY FEATURE: Unified interface across composite trajectories
        ------------------------------------------------------------
        This method provides seamless interpolation even when the trajectory is
        composed of multiple disjoint segments. The user doesn't need to know
        which segment contains t - we handle the dispatch automatically.
        
        Example use case:
          Bidirectional trajectory with domains [(-5, 0], [0, 5]]:
            traj.interpolate(-3.0) → dispatches to backward segment
            traj.interpolate(0.0)  → dispatches to forward segment (both have it, we pick first)
            traj.interpolate(3.0)  → dispatches to forward segment
            traj.interpolate(10.0) → raises ValueError (not in any domain)
        
        Implementation: O(n) linear search through segments (optimize with binary search later)
        
        Automatically finds the segment containing t and delegates to that segment's
        interpolant. Provides unified interface across potentially disjoint domains.
        
        Args:
            t (float): Time point for evaluation
        
        Returns:
            NDArray[np.float64]: State vector x(t) at time t
        
        Raises:
            ValueError: If t not in any segment domain
        
        Example:
            >>> traj = sys.trajectory(...)  # May have multiple segments
            >>> x_5 = traj.interpolate(5.0)  # Seamlessly finds right segment
        """
        # Find which segment contains this time point
        segment = self._find_segment_containing(t)
        
        if segment is None:
            # t is not in any segment's domain (e.g., in a gap or outside all domains)
            raise ValueError(
                f"Time t={t} not in any segment domain. "
                f"Available domains: {self.domains}"
            )
        
        # Delegate to the segment's interpolant
        # This may further raise ValueError if segment has no interpolant (dense_output=False)
        return segment.interpolate(t)
    
    def _find_segment_containing(self, t: float) -> Optional[EuclideanTrajectorySegment]:
        """
        Find segment whose domain contains time t.
        
        Uses linear search (optimize with binary search later if needed).
        
        Args:
            t (float): Time point to locate
        
        Returns:
            EuclideanTrajectorySegment if found, else None
        """
        for segment in self.segments:
            if segment.in_domain(t):
                return segment
        return None
    
    @property
    def t(self) -> NDArray[np.float64]:
        """
        Concatenated evaluation times from all segments.
        
        Note: May contain duplicate values at segment boundaries (tangent domains).
        
        Returns:
            NDArray[np.float64]: All evaluation times, shape (total_points,)
        """
        if not self.segments:
            return np.array([])
        return np.concatenate([seg.t for seg in self.segments])
    
    @property
    def y(self) -> NDArray[np.float64]:
        """
        Concatenated states from all segments.
        
        Returns:
            NDArray[np.float64]: All states, shape (n_dim, total_points)
        """
        if not self.segments:
            return np.array([]).reshape(0, 0)
        return np.concatenate([seg.y for seg in self.segments], axis=1)
    
    @property
    def success(self) -> bool:
        """
        Whether all segment integrations succeeded.
        
        Convenience property for backwards compatibility with SciPyIvpSolution.
        
        Returns:
            bool: True if all segments successful, False otherwise
        """
        return self.meta.get('all_successful', True)
    
    @property
    def message(self) -> str:
        """
        Aggregated messages from all segments.
        
        Convenience property for backwards compatibility with SciPyIvpSolution.
        
        Returns:
            str: Combined message from all segments
        """
        messages = self.meta.get('messages', [])
        if not messages:
            return "No messages"
        if len(messages) == 1:
            return messages[0]
        return " | ".join(messages)
