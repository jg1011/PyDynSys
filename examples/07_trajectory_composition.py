"""
Example 7: Trajectory Composition and Merging

Demonstrates:
- Bidirectional integration returning composite trajectory
- Manual trajectory composition from segments
- Overlapping segment merging with average policy
- Continuous interpolation across composite trajectories
- Error handling (interpolation outside domain)
"""

import numpy as np
import matplotlib.pyplot as plt
from PyDynSys.core import (
    AutonomousEuclideanDS,
    PhaseSpace,
    EuclideanTrajectory,
    EuclideanTrajectorySegment,
)


def main():
    print("=" * 70)
    print("Example 7: Trajectory Composition and Segment Merging")
    print("=" * 70)
    
    # Define simple harmonic oscillator for demonstrations
    def harmonic_oscillator(state):
        """Simple harmonic oscillator: x'' + x = 0"""
        x, y = state
        return np.array([y, -x])
    
    system = AutonomousEuclideanDS(
        dimension=2,
        vector_field=harmonic_oscillator,
        phase_space=PhaseSpace.euclidean(2)
    )
    
    x0 = np.array([1.0, 0.0])
    
    # ========================================================================
    # Part 1: Bidirectional Integration (Composite Trajectory)
    # ========================================================================
    print("\n[1] Bidirectional Integration → Composite Trajectory")
    print("    Computing flow on [-2, 2] with t₀ = 0...")
    
    t_eval_bidir = np.linspace(-2, 2, 400)
    t_span_bidir = (0.0, 2.0)  # t_0 = 0 is interior to t_eval range
    
    traj_bidir = system.trajectory(x0, t_span_bidir, t_eval_bidir, method='RK45')
    
    print(f"    ✓ Result: Single EuclideanTrajectory with {len(traj_bidir.segments)} segments")
    print(f"    ✓ Segment domains: {[seg.domain for seg in traj_bidir.segments]}")
    print(f"    ✓ Total evaluation points: {len(traj_bidir.t)}")
    
    # Verify segments are disjoint or tangent
    for i in range(len(traj_bidir.domains) - 1):
        gap = traj_bidir.domains[i+1][0] - traj_bidir.domains[i][1]
        print(f"    ✓ Gap between segments {i} and {i+1}: {gap:.2e}")
    
    # ========================================================================
    # Part 2: Manual Segment Composition (Tangent Domains)
    # ========================================================================
    print("\n[2] Manual Composition: Tangent Domains [0,1] + [1,2]")
    print("    Solving two separate IVPs and composing...")
    
    # Solve [0, 1]
    t_eval_1 = np.linspace(0, 1, 100)
    traj_1 = system.trajectory(x0, (0, 1), t_eval_1, method='RK45')
    seg_1 = traj_1.segments[0]  # Extract segment from trajectory
    
    # Solve [1, 2] (needs initial condition at t=1)
    x_at_1 = traj_1.y[:, -1]  # State at t=1
    t_eval_2 = np.linspace(1, 2, 100)
    traj_2 = system.trajectory(x_at_1, (1, 2), t_eval_2, method='DOP853')  # Different method!
    seg_2 = traj_2.segments[0]
    
    print(f"    ✓ Segment 1: domain {seg_1.domain}, method {seg_1.method}")
    print(f"    ✓ Segment 2: domain {seg_2.domain}, method {seg_2.method}")
    
    # Compose into single trajectory (tangent at t=1)
    traj_composed = EuclideanTrajectory.from_segments([seg_1, seg_2], merge_policy='average')
    
    print(f"    ✓ Composed trajectory: {len(traj_composed.segments)} segment(s)")
    print(f"    ✓ Multi-method trajectory: {[seg.method for seg in traj_composed.segments]}")
    print(f"    ✓ Total domain: [{traj_composed.domains[0][0]}, {traj_composed.domains[-1][1]}]")
    
    # ========================================================================
    # Part 3: Overlapping Segments with Average Merge Policy
    # ========================================================================
    print("\n[3] Overlapping Segments: [0, 1.5] + [0.5, 2] with Average Merge")
    print("    Solving overlapping IVPs and merging...")
    
    # Solve [0, 1.5]
    t_eval_a = np.linspace(0, 1.5, 150)
    traj_a = system.trajectory(x0, (0, 1.5), t_eval_a, method='RK45')
    seg_a = traj_a.segments[0]
    
    # Solve [0.5, 2] with slightly different initial condition to create discrepancy
    x_at_half = traj_a.interpolate(0.5)
    # Add tiny perturbation to simulate numerical error from different solve
    x_at_half_perturbed = x_at_half + np.array([1e-6, 1e-6])
    t_eval_b = np.linspace(0.5, 2, 150)
    traj_b = system.trajectory(x_at_half_perturbed, (0.5, 2), t_eval_b, method='RK45')
    seg_b = traj_b.segments[0]
    
    print(f"    ✓ Segment A: domain {seg_a.domain}")
    print(f"    ✓ Segment B: domain {seg_b.domain}")
    print(f"    ✓ Overlap region: [{max(seg_a.domain[0], seg_b.domain[0])}, "
          f"{min(seg_a.domain[1], seg_b.domain[1])}]")
    
    # Merge with average policy (default)
    traj_merged = EuclideanTrajectory.from_segments([seg_a, seg_b], merge_policy='average')
    
    print(f"    ✓ Merged trajectory: {len(traj_merged.segments)} segment(s)")
    print(f"    ✓ Final domain: [{traj_merged.domains[0][0]}, {traj_merged.domains[-1][1]}]")
    
    # ========================================================================
    # Part 4: Seamless Interpolation Across Segments
    # ========================================================================
    print("\n[4] Continuous Interpolation Across Composite Trajectory")
    print("    Testing interpolation on bidirectional trajectory...")
    
    # Interpolate at various points across both segments
    test_times = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    
    for t_test in test_times:
        if traj_bidir.in_domain(t_test):
            x_interp = traj_bidir.interpolate(t_test)
            print(f"    ✓ x({t_test:+.1f}) = [{x_interp[0]:+.4f}, {x_interp[1]:+.4f}]")
        else:
            print(f"    ✗ t = {t_test} not in trajectory domain")
    
    # ========================================================================
    # Part 5: Error Handling (Interpolation Outside Domain)
    # ========================================================================
    print("\n[5] Error Handling: Interpolation Outside Domain")
    print("    Attempting interpolation at t = 5 (outside [-2, 2])...")
    
    try:
        traj_bidir.interpolate(5.0)
        print("    ✗ ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"    ✓ Correctly raised ValueError:")
        print(f"      {e}")
    
    # ========================================================================
    # Part 6: Plotting
    # ========================================================================
    print("\n[6] Generating plots...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Bidirectional trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(traj_bidir.t, traj_bidir.y[0, :], linewidth=2, label='x(t)')
    ax1.plot(traj_bidir.t, traj_bidir.y[1, :], linewidth=2, alpha=0.7, label='y(t)')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='t₀ = 0')
    ax1.scatter([0], [x0[0]], color='red', s=100, zorder=5)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('State')
    ax1.set_title(f'Bidirectional Trajectory ({len(traj_bidir.segments)} segments)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bidirectional phase portrait
    ax2 = fig.add_subplot(gs[0, 1])
    # Color code by segment
    for i, seg in enumerate(traj_bidir.segments):
        color = 'blue' if i == 0 else 'red'
        label = f'Segment {i+1}: {seg.domain}'
        ax2.plot(seg.y[0, :], seg.y[1, :], linewidth=2, color=color, 
                alpha=0.7, label=label)
    ax2.scatter([x0[0]], [x0[1]], color='black', s=100, zorder=5, marker='*')
    ax2.set_xlabel('x (Position)')
    ax2.set_ylabel('y (Velocity)')
    ax2.set_title('Phase Portrait (Colored by Segment)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Manual composition (tangent domains)
    ax3 = fig.add_subplot(gs[1, 0])
    for i, seg in enumerate(traj_composed.segments):
        ax3.plot(seg.t, seg.y[0, :], linewidth=2, marker='o', markersize=2,
                label=f'Seg {i+1} ({seg.method})')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('x (Position)')
    ax3.set_title(f'Manual Composition: Tangent Domains ({len(traj_composed.segments)} seg)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Overlapping segments
    ax4 = fig.add_subplot(gs[1, 1])
    # Plot original segments
    ax4.plot(seg_a.t, seg_a.y[0, :], linewidth=2, alpha=0.5, 
            linestyle='--', label='Segment A [0, 1.5]', color='blue')
    ax4.plot(seg_b.t, seg_b.y[0, :], linewidth=2, alpha=0.5, 
            linestyle='--', label='Segment B [0.5, 2]', color='red')
    # Plot merged result
    ax4.plot(traj_merged.t, traj_merged.y[0, :], linewidth=2, 
            color='green', label='Merged (average)', zorder=5)
    # Highlight overlap region
    ax4.axvspan(0.5, 1.5, alpha=0.2, color='yellow', label='Overlap region')
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('x (Position)')
    ax4.set_title('Overlapping Segments: Average Merge Policy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Interpolation demonstration
    ax5 = fig.add_subplot(gs[2, :])
    # Plot trajectory
    ax5.plot(traj_bidir.t, traj_bidir.y[0, :], linewidth=2, 
            color='blue', alpha=0.5, label='Trajectory x(t)')
    
    # Interpolate at many points
    t_interp_dense = np.linspace(traj_bidir.domains[0][0], 
                                  traj_bidir.domains[-1][1], 500)
    x_interp_dense = np.array([traj_bidir.interpolate(t)[0] for t in t_interp_dense])
    
    ax5.plot(t_interp_dense, x_interp_dense, linewidth=1, 
            color='red', linestyle=':', label='Interpolated (500 pts)')
    
    # Mark evaluation points
    ax5.scatter(traj_bidir.t, traj_bidir.y[0, :], s=10, color='blue', 
               alpha=0.3, zorder=3, label=f'Eval points ({len(traj_bidir.t)})')
    
    ax5.set_xlabel('Time t')
    ax5.set_ylabel('x (Position)')
    ax5.set_title('Seamless Interpolation Across Segments')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('examples/outputs/07_trajectory_composition.png', 
                dpi=150, bbox_inches='tight')
    print("    ✓ Plot saved to examples/outputs/07_trajectory_composition.png")
    
    print("\n" + "=" * 70)
    print("Example 7 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    import os
    os.makedirs('examples/outputs', exist_ok=True)
    main()

