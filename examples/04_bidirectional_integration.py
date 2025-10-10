"""
Example 4: Bidirectional Integration

Demonstrates:
- Bidirectional integration: flow on interval I around initial time t₀
- Example: I = (-2, 2) with t₀ = 0
- How the library automatically detects and handles this case
- Useful for studying orbits and flows in both time directions
"""

import numpy as np
import matplotlib.pyplot as plt
from PyDynSys.core import AutonomousEuclideanDS, PhaseSpace


def main():
    print("=" * 70)
    print("Example 4: Bidirectional Integration")
    print("Computing flow on interval I = (-T, T) around t₀ = 0")
    print("=" * 70)
    
    # Define a damped harmonic oscillator
    gamma = 0.1  # Damping coefficient
    
    def damped_oscillator(state):
        """
        Damped harmonic oscillator: x'' + 2γx' + x = 0
        
        Args:
            state: [x, y] where x is position, y is velocity
            
        Returns:
            [dx/dt, dy/dt] = [y, -x - 2γy]
        """
        x, y = state
        return np.array([y, -x - 2*gamma*y])
    
    # Create system
    print(f"\n[1] Creating damped oscillator (γ = {gamma})...")
    system = AutonomousEuclideanDS(
        dimension=2,
        vector_field=damped_oscillator,
        phase_space=PhaseSpace.euclidean(2)
    )
    print(f"    ✓ System created")
    
    # Set up bidirectional integration
    T = 2.0  # Time horizon in each direction
    x0 = np.array([1.0, 0.5])
    t0 = 0.0
    
    # Key: t_eval spans around t₀!
    t_eval = np.linspace(-T, T, 1000)
    t_span = (t0, T)  # First element is t₀, second is just a bound
    
    print(f"\n[2] Setting up bidirectional integration...")
    print(f"    Initial time: t₀ = {t0}")
    print(f"    Time interval: I = ({-T}, {T})")
    print(f"    Initial state: x(t₀) = {x0}")
    print(f"    t_eval range: [{t_eval.min()}, {t_eval.max()}]")
    print(f"    → Library detects: t₀ = {t0} is interior to t_eval range")
    print(f"    → Automatically performs bidirectional integration!")
    
    # Compute solution - library automatically handles bidirectional case
    solution = system.trajectory(x0, t_span, t_eval, method='RK45')
    
    print(f"\n[3] Integration results:")
    print(f"    ✓ Success: {solution.success}")
    print(f"    ✓ Message: {solution.message}")
    print(f"    ✓ Solution times: {len(solution.t)} points")
    print(f"    ✓ Time range: [{solution.t.min():.2f}, {solution.t.max():.2f}]")
    
    # Verify we got both directions
    times_before_t0 = solution.t[solution.t < t0]
    times_after_t0 = solution.t[solution.t >= t0]
    
    print(f"\n[4] Bidirectional coverage:")
    print(f"    Backward (t < 0): {len(times_before_t0)} points, " +
          f"t ∈ [{times_before_t0.min():.2f}, {times_before_t0.max():.2f}]")
    print(f"    Forward (t ≥ 0):  {len(times_after_t0)} points, " +
          f"t ∈ [{times_after_t0.min():.2f}, {times_after_t0.max():.2f}]")
    
    # Compare with unidirectional forward integration
    print(f"\n[5] Comparison with standard forward integration...")
    t_eval_forward = np.linspace(0, T, 500)
    solution_forward = system.trajectory(x0, (0, T), t_eval_forward, method='RK45')
    print(f"    ✓ Forward-only solution computed")
    
    # Plot comparison
    print("\n[6] Generating plots...")
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Bidirectional time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axvline(x=t0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='t₀ = 0')
    ax1.plot(solution.t, solution.y[0, :], linewidth=2, color='blue', label='x(t) - Position')
    ax1.plot(solution.t, solution.y[1, :], linewidth=2, color='red', alpha=0.7, 
            label='y(t) - Velocity')
    ax1.scatter([t0], [x0[0]], color='green', s=150, zorder=5, marker='*', 
               label='Initial condition')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('State')
    ax1.set_title('Bidirectional Solution: Flow from t=-2 to t=2')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhspan(-3, 3, alpha=0.1, color='green')
    
    # Bidirectional phase portrait
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Color code by time
    t_normalized = (solution.t - solution.t.min()) / (solution.t.max() - solution.t.min())
    scatter = ax2.scatter(solution.y[0, :], solution.y[1, :], c=solution.t, 
                         cmap='viridis', s=10, alpha=0.6)
    ax2.scatter([x0[0]], [x0[1]], color='red', s=200, zorder=5, marker='*', 
               edgecolors='black', linewidths=2, label='t₀ = 0')
    
    # Mark endpoints
    ax2.scatter([solution.y[0, 0]], [solution.y[1, 0]], color='purple', 
               s=100, zorder=5, marker='o', label='t = -2')
    ax2.scatter([solution.y[0, -1]], [solution.y[1, -1]], color='orange', 
               s=100, zorder=5, marker='s', label='t = 2')
    
    ax2.set_xlabel('x (Position)')
    ax2.set_ylabel('y (Velocity)')
    ax2.set_title('Bidirectional Phase Portrait')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Time t')
    
    # Forward-only comparison
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(solution_forward.y[0, :], solution_forward.y[1, :], linewidth=2, 
            color='blue', alpha=0.7, label='Forward only')
    ax3.scatter([x0[0]], [x0[1]], color='red', s=200, zorder=5, marker='*', 
               edgecolors='black', linewidths=2, label='t₀ = 0')
    ax3.set_xlabel('x (Position)')
    ax3.set_ylabel('y (Velocity)')
    ax3.set_title('Forward-Only Phase Portrait (for comparison)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Zoom on behavior near t=0
    ax4 = fig.add_subplot(gs[2, :])
    t_zoom_mask = np.abs(solution.t) < 0.5
    ax4.plot(solution.t[t_zoom_mask], solution.y[0, t_zoom_mask], 
            linewidth=3, color='blue', marker='o', markersize=3, label='x(t)')
    ax4.plot(solution.t[t_zoom_mask], solution.y[1, t_zoom_mask], 
            linewidth=3, color='red', marker='o', markersize=3, label='y(t)')
    ax4.axvline(x=t0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax4.scatter([t0], [x0[0]], color='green', s=150, zorder=5, marker='*')
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('State')
    ax4.set_title('Zoom: Behavior Near t₀ = 0')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.savefig('examples/outputs/04_bidirectional_integration.png', dpi=150, bbox_inches='tight')
    print("    ✓ Plot saved to examples/outputs/04_bidirectional_integration.png")
    
    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("  1. Bidirectional integration is AUTOMATIC when t₀ is interior to t_eval")
    print("  2. Useful for studying complete orbits around a point in time")
    print("  3. Library handles backward + forward integration seamlessly")
    print("  4. Dense output not supported for bidirectional (returns None)")
    print("=" * 70)


if __name__ == "__main__":
    import os
    os.makedirs('examples/outputs', exist_ok=True)
    main()

