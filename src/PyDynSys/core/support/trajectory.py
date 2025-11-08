"""Trajectory representation for Euclidean dynamical systems."""

from __future__ import annotations
from typing import List, Tuple, Callable, Optional, Union, TYPE_CHECKING, Literal
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..types import SciPyIvpSolution
    
    
### Trajectory Types ###


TrajectorySegmentMergePolicy = Literal['average', 'left', 'right', 'stitch']
"""
Strategy for merging overlapping trajectory segments in EuclideanTrajectory.from_segments().

WHEN OVERLAPS OCCUR:
--------------------
Overlapping domains arise when the same physical trajectory is computed multiple
times over intersecting time intervals. Common scenarios:
  1. Re-solving for comparison: Solve [0,1.5] with RK45, then [0.5,2] with DOP853
  2. Patching numerical errors: Re-solve unstable region with tighter tolerances
  3. Composing from cache: Merging cached trajectories to avoid recomputation

MERGE POLICIES:
---------------
When two segments overlap on [a, b], we must decide:
  - Which y values to use at shared evaluation points?
  - Which interpolant to use for continuous evaluation in [a, b]?

Available policies:
- 'average' (DEFAULT): Average y values at shared evaluation points in overlap region.
                       Takes midpoint between competing numerical approximations.
                       Use case: Equal trust in both segments, want best estimate.
                       
- 'left': Prioritize left segment's values and interpolant in overlap region.
          Use case: Left segment has higher accuracy (tighter tolerance, better method).
          Status: Not yet implemented (raises NotImplementedError).
          
- 'right': Prioritize right segment's values and interpolant in overlap region.
           Use case: Right segment has higher accuracy or is more recent computation.
           Status: Not yet implemented (raises NotImplementedError).
           
- 'stitch': Use left segment's interpolant until overlap midpoint, then right's.
            Creates continuous transition across overlap region.
            Use case: Both segments equally valid, want smooth transition.
            Status: Not yet implemented (raises NotImplementedError).

TANGENT DOMAINS (Special Case):
--------------------------------
When domains touch at exactly one point (e.g., [0,1] + [1,2]), the "overlap"
is just the boundary. The average policy automatically handles this by averaging
the single shared point, which is correct for tangent segments from bidirectional
integration where both segments share x(t_0) at the tangent point.

Note: Only 'average' is implemented in current version. Others raise NotImplementedError.
"""


class TrajectorySegment: 
    """
    Represents a numerically computed segment of a trajectory on a monotone increasing evaluation space. 
    
    A segment corresponds to a single continuous solution from scipy.solve_ivp, representing
    the trajectory over a contiguous time interval with a single interpolant.
    
    Fields: 
        t (NDArray[np.float64]): Monotone increasing array of evaluation times, shape (len(t),)
        y (NDArray[np.float64]): Array of trajectory evaluations x(t), shape (n, len(t))
        domain (Tuple[float, float]): Time domain [t[0], t[-1]] where segment is defined
        interpolant (Optional[Callable]): Continuous interpolant x(t) on domain, or None
            -> NOTE: This is some iff dense_output=True flag passed to solve_ivp fn.
        method (str): ODE solver method used ('RK45', 'LSODA', etc.)
        meta (Dict[str, Any]): Metadata about numerical solution (success, message, etc.)
    
    Usage:
        - Segments are created via from_scipy_solution() factory, not direct instantiation.
        - Users primarily interact with Trajectory class, which aggregates segments. 
        - Future versions will support factories for other solvers. 
        
    """
    
    
    ### --- Factory Methods --- ###
    
    
    @classmethod
    def from_scipy_solution(
        cls, 
        sol: SciPyIvpSolution, 
        method: str
    ) -> 'TrajectorySegment':
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
            >>> result = solve_ivp(fun, t_span, y0, t_eval, method, dense_output)
            >>> wrapped = SciPyIvpSolution(raw_solution=result) # not necessary, but provides type safety
            >>> segment = EuclideanTrajectorySegment.from_scipy_solution(wrapped, 'RK45') # default method is RK45
        """
        segment = cls()
        t = sol.t # shape = (n_points,)
        y = sol.y # shape = (n_dim, n_points)
        
        # Enforce monotone increasing time convention
        if len(t) > 1 and t[0] > t[-1]:
            # Backward integration detected: reverse arrays
            t = t[::-1]
            y = y[:, ::-1]
        
        segment.t = t
        segment.y = y
        segment.domain = (float(t[0]), float(t[-1]))
        
        # Store interpolant (callable or None)
        ## NOTE: this is some iff dense_output=True flag passed to solve_ivp fn.
        segment.interpolant = sol.sol

        segment.method = method
        segment.meta = {
            'success': sol.success,
            'message': sol.message,
        }
        
        return segment
    
    
        ### --- Public Methods --- ###
        
    
    def in_domain(self, t: float) -> bool:
        """
        Check if time t is within segment domain.
        
        Args:
            t (float): Time point to check
        
        Returns:
            bool: True if t ∈ [domain[0], domain[1]], False otherwise
        """
        return self.domain[0] <= t <= self.domain[1]
    
    
    def interpolant_at_time(self, t: float) -> NDArray[np.float64]:
        """
        Evaluate interpolant at time t.
        
        Uses scipy's dense output interpolant to compute x(t) continuously within
        the segment domain. Raises errors if t is outside domain or if no interpolant
        is available (dense_output=False in original solve_ivp call).
        
        NOTE: For speed, we employ agressive programming here and assume 
            1. t is in domain 
            2. interpolant is available
        If either of these fails, an esoteric error may be incurred. This is a worthwhile 
        tradeoff as this function may be called thousands of times (e.g. when plotting a trajectory).
        
        Args:
            t (float): Time point for evaluation
        
        Returns:
            NDArray[np.float64]: State vector x(t) at time t, shape (n,)
        
        Example:
            >>> x_t = segment.interpolate(0.5)  # Evaluate at t=0.5
        """
        result = self.interpolant(t)
        if result.ndim == 2:
            return result[:, 0]  # Extract column vector as 1D
        return result
    
    
class Trajectory: 
    """
    Represents a numerically computed trajectory on a subset of the real line R.
    
    A trajectory is a composition of one or more segments with disjoint domains,
    providing seamless access to the complete solution across potentially
    non-contiguous time intervals.
    
    Key invariant: All segment domains are disjoint (validated in from_segments factory).
    
    Fields: 
        segments (List[EuclideanTrajectorySegment]): Trajectory segments in ascending domain order
        domains (List[Tuple[float, float]]): Domain intervals [t_i, t_{i+1}] in ascending order
        meta (Dict[str, Any]): Aggregate metadata from all segments
    
    Properties:
        t: Concatenated evaluation times from all segments
        y: Concatenated states from all segments
        
    Usage:
        Created via from_segments() factory. Primary user-facing class returned by
        system.trajectory() method. Provides seamless interpolation across segments.
    
    Example:
        >>> sys = AutonomousDS(...)
        >>> traj = sys.trajectory(x0, t_span=(0, 10), t_eval=np.linspace(0, 10, 100))
        >>> x_at_5 = traj.interpolate(5.0)
    """
    
    
    ### --- Factory Methods --- ###
    
    
    @classmethod
    def from_segments(
        cls,
        segments: List[TrajectorySegment],
        merge_policy: TrajectorySegmentMergePolicy = 'average'  # Default: average overlapping values
    ) -> 'Trajectory':
        """
        Factory: Create trajectory from list of segments, merging overlaps if needed.
        
        MERGE ALGORITHM: Iterative Fixed-Point Approach
        ------------------------------------------------
        This method implements an iterative fixed-point algorithm to merge overlapping
        segments until all domains are disjoint (class invariant).
        
        Problem: Cascading overlaps require multiple merge passes.
        Example: Segments [[0,1], [0.5,2], [1,3]] have:
          - Pass 1: Merge [0,1] + [0.5,2] → [0,2]
          - Pass 2: Merge [0,2] + [1,3] → [0,3]
        
        Algorithm:
          1. Sort segments by domain start time (ensures left-to-right processing)
          2. Fixed-point iteration:
             a. Scan through current segment list left-to-right
             b. For each consecutive pair, detect overlap
             c. If overlap exists, merge the pair and add to result
             d. If no overlap, add current segment to result
             e. If any merges occurred, repeat from step 2a
             f. If no merges occurred, fixed point reached → done
        
        Invariant: At each iteration, segments in the working list are sorted by start time.
        This ensures that after merging two consecutive segments, the merged segment
        cannot overlap with any previously processed segments (they all ended before
        the current segment started, by the inductive property of sorted order).
        
        Termination: Guaranteed because:
          - Each merge reduces the number of segments by 1
          - Minimum segments = 1 (fully merged trajectory)
          - Maximum iterations = n-1 (worst case: chain of n segments)
        
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
            RuntimeError: If merge algorithm fails to converge (indicates bug)
            NotImplementedError: If merge_policy is not 'average' (others not yet implemented)
        
        Example:
            >>> seg1 = TrajectorySegment.from_scipy_solution(sol1, 'RK45')
            >>> seg2 = TrajectorySegment.from_scipy_solution(sol2, 'RK45')
            >>> traj = Trajectory.from_segments([seg1, seg2])
        """
        if not segments:
            raise ValueError("Cannot create trajectory from empty segment list")
        
        # Sort segments by domain start time (critical for correct merge order)
        working_segments = sorted(segments, key=lambda seg: seg.domain[0])
        
        
        # ITERATIVE FIXED-POINT MERGE ALGORITHM #
        ## o(num_segments^2) worst case complexity
        changed = True
        iteration = 0
        max_iterations = len(segments)  # Safety limit (should never be reached)
        while changed and iteration < max_iterations:
            changed = False
            merged_segments = []
            i = 0
            
            # pass through working segments, merging consecutive overlaps
            while i < len(working_segments):
                current = working_segments[i]
                
                if i + 1 < len(working_segments):
                    next_seg = working_segments[i + 1]
                    overlap = cls._detect_overlap(current, next_seg)
                    
                    if overlap is not None:
                        # Overlapping segments detected - merge them
                        if merge_policy == 'average':
                            merged = cls._merge_segments_average(current, next_seg, overlap)
                            merged_segments.append(merged)
                            i += 2  # Skip both segments (merged into one)
                            changed = True  # Mark that we made progress
                        elif merge_policy == 'left':
                            merged = cls._merge_segments_left(current, next_seg, overlap)
                            merged_segments.append(merged)
                            i += 2  # Skip both segments (merged into one)
                            changed = True  # Mark that we made progress
                        elif merge_policy == 'right':
                            merged = cls._merge_segments_right(current, next_seg, overlap)
                            merged_segments.append(merged)
                            i += 2  # Skip both segments (merged into one)
                            changed = True  # Mark that we made progress
                        elif merge_policy == 'stitch':
                            merged = cls._merge_segments_stitch(current, next_seg, overlap)
                            merged_segments.append(merged)
                            i += 2  # Skip both segments (merged into one)
                            changed = True  # Mark that we made progress
                        else:
                            raise ValueError(f"Unknown merge policy: {merge_policy}")
                    else:
                        # Disjoint segments - keep current and move forward
                        merged_segments.append(current)
                        i += 1
                else:
                    # Last segment - no next segment to check for overlap
                    merged_segments.append(current)
                    i += 1
            
            # Update working list for next iteration (if needed)
            working_segments = merged_segments
            iteration += 1
        
        # Only raise error if we hit max_iterations WITHOUT converging (changed is still True)
        # If changed is False, we converged successfully even if iteration == max_iterations
        if changed and iteration >= max_iterations:
            # This should never happen in practice, but safety check
            raise RuntimeError(
                f"Merge algorithm failed to converge after {max_iterations} iterations. "
                f"This indicates a bug in the merge logic."
            )
        
        # Final merged segments (guaranteed disjoint by fixed-point property)
        merged_segments = working_segments
        
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
        
        
        trajectory._validate_disjoint_domains() # Validate disjoint domains (class invariant)
        return trajectory
    
    
        ### --- Public Methods --- ###
        
    
    def merge(
        self,
        other: 'Trajectory',
        merge_policy: TrajectorySegmentMergePolicy = 'average'
    ) -> 'Trajectory':
        """
        Join this trajectory with another trajectory.
        
        Combines segments from both trajectories and merges any overlaps using
        the specified merge policy. Uses the same iterative fixed-point merge
        algorithm as from_segments() to handle cascading overlaps.
        
        Args:
            other (Trajectory): The other trajectory to join with self
            merge_policy (TrajectorySegmentMergePolicy): Strategy for handling overlaps
                - 'average' (DEFAULT): Average y values in overlap region
                - 'left': Use left segment in overlap
                - 'right': Use right segment in overlap
                - 'stitch': Left interpolant until midpoint, then right
            -> NOTE: merge_policy is not yet implemented for all policies, only `average` is supported.
        
        Returns:
            Trajectory: New trajectory containing all segments from both trajectories with overlaps merged according to merge_policy
        
        Raises:
            ValueError: If final domains are not disjoint (merging failed)
            RuntimeError: If merge algorithm fails to converge (indicates bug)
            NotImplementedError: If merge_policy is not 'average' (others not yet implemented)
        
        Example:
            >>> traj1 = sys1.trajectory(x0, t_span=(0, 5))
            >>> traj2 = sys2.trajectory(x1, t_span=(3, 10))
            >>> combined = traj1.join(traj2)  # Merges overlap in [3, 5]
        """
        # Combine segments from both trajectories
        combined_segments = list(self.segments) + list(other.segments)
        
        # Use from_segments factory to merge overlaps and create new trajectory
        return Trajectory.from_segments(combined_segments, merge_policy=merge_policy)
    
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
            
        NOTE: We implement __call__ dunder by invoking interpolate method, allowing users to instead write
            >>> x = traj(5.0)  # Equivalent to traj.interpolate(5.0)
        """
        segment = self._find_segment_containing(t)
        
        if segment is None:
            raise ValueError(
                f"Time t={t} not in any segment domain. "
                f"Available domains: {self.domains}"
            )
        
        # This may further raise ValueError if segment has no interpolant (dense_output=False)
        return segment.interpolate(t)
    
    
        ### -- Properties --- ###
        
        
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
    
    
        ### --- Dunder Methods --- ### 
        
    
    def __str__(self) -> str:
        return f"Trajectory(domains={self.domains})"
    
    def __repr__(self) -> str:
        return f"Trajectory(domains={self.domains})"
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[TrajectorySegment, List[TrajectorySegment]]:
        if isinstance(index, slice):
            return self.segments[index]  
        return self.segments[index]      
    
    def __add__(self, other: 'Trajectory') -> 'Trajectory':
        return self.merge(other)
    
    def __call__(self, t: float) -> NDArray[np.float64]:
        """
        Example: 
            >>> traj = sys.trajectory(...)
            >>> x = traj(5.0)  # Equivalent to traj.interpolate(5.0), nice ;)
        """
        return self.interpolate(t)
            
    
        ### --- Private Methods --- ###
        
    
    @staticmethod
    def _detect_overlap(
        seg1: TrajectorySegment,
        seg2: TrajectorySegment
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
    def _create_merged_interpolant(
        seg1: TrajectorySegment,
        seg2: TrajectorySegment,
        overlap: Tuple[float, float],
        merged_domain: Tuple[float, float]
    ) -> Optional[Callable[[float], NDArray[np.float64]]]:
        """
        Create piecewise interpolant for merged segment.
        
        PIECEWISE DEFINITION:
        ---------------------
        For merged segment with overlap [overlap_start, overlap_end]:
        
        1. Pre-overlap region [seg1.domain[0], overlap_start):
           c(t) = seg1.interpolant(t)
        
        2. Overlap region [overlap_start, overlap_end]:
           c(t) = (seg1.interpolant(t) + seg2.interpolant(t)) / 2
        
        3. Post-overlap region (overlap_end, seg2.domain[1]]:
           c(t) = seg2.interpolant(t)
        
        This matches the discrete averaging policy: where both segments provide
        values, we average them; elsewhere, we use the single available value.
        
        Args:
            seg1, seg2: Overlapping segments (seg1.domain[0] <= seg2.domain[0] assumed)
            overlap: Overlap interval [overlap_start, overlap_end]
            merged_domain: Final domain [t_min, t_max] of merged segment
        
        Returns:
            Callable interpolant function, or None if either segment lacks interpolant
        """
        overlap_start, overlap_end = overlap
        seg1_start, seg1_end = seg1.domain
        seg2_start, seg2_end = seg2.domain
        merged_start, merged_end = merged_domain
        
        # If either segment lacks interpolant, cannot create merged interpolant
        if seg1.interpolant is None or seg2.interpolant is None:
            return None
        
        def merged_interpolant(t: float) -> NDArray[np.float64]:
            """
            Piecewise merged interpolant evaluating at time t.
            
            Handles shape normalization for scipy interpolants which may return
            shape (n_dim,) or (n_dim, 1).
            """         
            # Helper to normalize scipy interpolant output shape
            def normalize_result(result: NDArray) -> NDArray[np.float64]:
                """Normalize scipy interpolant output to shape (n_dim,)."""
                if result.ndim == 2:
                    return result[:, 0]
                return result
            
            # Region 1: Pre-overlap (only seg1)
            if seg1_start <= t < overlap_start:
                result = seg1.interpolant(t)
                return normalize_result(result)
            
            # Region 2: Overlap (average both segments)
            elif overlap_start <= t <= overlap_end:
                result1 = seg1.interpolant(t)
                result2 = seg2.interpolant(t)
                # Normalize shapes, then average
                result1 = normalize_result(result1)
                result2 = normalize_result(result2)
                return (result1 + result2) / 2
            
            # Region 3: Post-overlap (only seg2)
            elif overlap_end < t <= seg2_end:
                result = seg2.interpolant(t)
                return normalize_result(result)
        
        return merged_interpolant
    
    @staticmethod
    def _merge_segments_average(
        seg1: TrajectorySegment,
        seg2: TrajectorySegment,
        overlap: Tuple[float, float]
    ) -> TrajectorySegment:
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
        4. Interpolant: Create piecewise interpolant matching the averaging policy
        
        Args:
            seg1, seg2: Overlapping segments (seg1.domain[0] <= seg2.domain[0] assumed)
            overlap: Overlap interval [a, b] where both segments are defined
        
        Returns:
            TrajectorySegment: Merged segment spanning union of domains
        """
        overlap_start, overlap_end = overlap
        tol = 1e-4 # Two points at t1 and t2 are considered "same" if |t1 - t2| < 1e-4
        
        # Split both segments into three regions:
        #   1. Pre-overlap (only in seg1)
        #   2. Overlap (in both segments)
        #   3. Post-overlap (only in seg2)
        t1_before = seg1.t[seg1.t < overlap_start]
        y1_before = seg1.y[:, seg1.t < overlap_start]
        t1_overlap = seg1.t[(seg1.t >= overlap_start) & (seg1.t <= overlap_end)]
        y1_overlap = seg1.y[:, (seg1.t >= overlap_start) & (seg1.t <= overlap_end)]
        t2_overlap = seg2.t[(seg2.t >= overlap_start) & (seg2.t <= overlap_end)]
        y2_overlap = seg2.y[:, (seg2.t >= overlap_start) & (seg2.t <= overlap_end)]
        t2_after = seg2.t[seg2.t > overlap_end]
        y2_after = seg2.y[:, seg2.t > overlap_end]
        

        # Merge overlap region by averaging at shared t points
        ## Strategy: Take union of all t values, average where both exist
        t_overlap_all = np.unique(np.concatenate([t1_overlap, t2_overlap]))
        n_dim = seg1.y.shape[0]  # Phase space dimension
        y_overlap_merged = np.zeros((n_dim, len(t_overlap_all)))
        
        for i, t_val in enumerate(t_overlap_all):
            # Check if this time point exists in each segment (within tolerance)
            in_seg1 = np.any(np.abs(t1_overlap - t_val) < tol)
            in_seg2 = np.any(np.abs(t2_overlap - t_val) < tol)
            
            if in_seg1 and in_seg2:
                # CASE 1: Shared point (exists in both segments)
                idx1 = np.argmin(np.abs(t1_overlap - t_val))  # Find closest point in seg1
                idx2 = np.argmin(np.abs(t2_overlap - t_val))  # Find closest point in seg2
                y_overlap_merged[:, i] = (y1_overlap[:, idx1] + y2_overlap[:, idx2]) / 2
                
            elif in_seg1:
                # CASE 2: Only in seg1 (seg2 didn't evaluate here)
                idx1 = np.argmin(np.abs(t1_overlap - t_val))
                y_overlap_merged[:, i] = y1_overlap[:, idx1]
                
            else:
                # CASE 3: Only in seg2 (seg1 didn't evaluate here)
                idx2 = np.argmin(np.abs(t2_overlap - t_val))
                y_overlap_merged[:, i] = y2_overlap[:, idx2]

        # Concatenate all three regions to form the complete merged segment
        # Order: [pre-overlap from seg1] + [merged overlap] + [post-overlap from seg2]
        t_merged = np.concatenate([t1_before, t_overlap_all, t2_after])
        y_merged = np.concatenate([y1_before, y_overlap_merged, y2_after], axis=1)
        
        # Create merged segment
        merged = TrajectorySegment()
        merged.t = t_merged
        merged.y = y_merged
        merged.domain = (float(t_merged[0]), float(t_merged[-1]))
        
        # Create piecewise merged interpolant
        # Piecewise definition:
        #   - Pre-overlap: use seg1.interpolant
        #   - Overlap: average (seg1.interpolant + seg2.interpolant) / 2
        #   - Post-overlap: use seg2.interpolant
        # Returns None if either segment lacks interpolant (dense_output=False)
        merged.interpolant = Trajectory._create_merged_interpolant(
            seg1, seg2, overlap, merged.domain
        )
        
        # Method string indicates multi-method composition
        # e.g., "RK45+DOP853" shows this segment combines two different solvers
        merged.method = f"{seg1.method}+{seg2.method}"
        
        # Metadata: Aggregate success flags and messages from both segments
        merged.meta = {
            'success': seg1.meta['success'] and seg2.meta['success'],
            'message': f"Merged: {seg1.meta['message']} | {seg2.meta['message']}",
        }
        
        return merged
    
    def _merge_segments_stitch(
        self, 
        seg1: TrajectorySegment, 
        seg2: TrajectorySegment, 
        overlap: Tuple[float, float]
    ) -> TrajectorySegment:
        """
        Merge two segments by stitching their interpolants at the overlap point.
        
        Args:
            seg1 (TrajectorySegment): First segment
            seg2 (TrajectorySegment): Second segment
            overlap (Tuple[float, float]): Overlap interval [a, b]
        """
        raise NotImplementedError("Merge policy 'stitch' not yet implemented for Trajectory class")
    
    def _merge_segments_left(
        self, 
        seg1: TrajectorySegment, 
        seg2: TrajectorySegment, 
        overlap: Tuple[float, float]
    ) -> TrajectorySegment:
        """
        Merge two segments by using the left segment's interpolant in the overlap region.
        
        Args:
            seg1 (TrajectorySegment): First segment
            seg2 (TrajectorySegment): Second segment
            overlap (Tuple[float, float]): Overlap interval [a, b]
        """
        raise NotImplementedError("Merge policy 'left' not yet implemented for Trajectory class")
    
    def _merge_segments_right(
        self, 
        seg1: TrajectorySegment, 
        seg2: TrajectorySegment, 
        overlap: Tuple[float, float]
    ) -> TrajectorySegment:
        """
        Merge two segments by using the right segment's interpolant in the overlap region.
        
        Args:
            seg1 (TrajectorySegment): First segment
            seg2 (TrajectorySegment): Second segment
            overlap (Tuple[float, float]): Overlap interval [a, b]
        """
        raise NotImplementedError("Merge policy 'right' not yet implemented for Trajectory class")
        
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
    
    def _find_segment_containing(self, t: float) -> Optional[TrajectorySegment]:
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
