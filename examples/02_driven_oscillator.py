"""
Example 2: Driven Harmonic Oscillator (Non-Autonomous System)

Demonstrates:
- Creating a non-autonomous system with explicit time dependence
- Time horizon specification
- How initial time t₀ affects the solution
- Comparing trajectories from different initial times
"""

import numpy as np
import matplotlib.pyplot as plt
from PyDynSys.core import NonAutonomousEuclideanDS, PhaseSpace, TimeHorizon


def main():
    print("=" * 70)
    print("Example 2: Driven Harmonic Oscillator")
    print("System: x'' + x = sin(ωt)")
    print("First-order form: dx/dt = y, dy/dt = -x + sin(ωt)")
    print("=" * 70)
    
    # Define vector field for driven oscillator
    omega = 2.0  # Driving frequency
    
    def driven_oscillator(state, time):
        """
        Vector field for driven harmonic oscillator.
        
        Args:
            state: [x, y] where x is position, y is velocity
            time: Current time t (ESSENTIAL for non-autonomous systems)
            
        Returns:
            [dx/dt, dy/dt] = [y, -x + sin(ωt)]
        """
        x, y = state
        return np.array([y, -x + np.sin(omega * time)])
    
    # Create non-autonomous system
    print("\n[1] Creating non-autonomous system...")
    system = NonAutonomousEuclideanDS(
        dimension=2,
        vector_field=driven_oscillator,
        phase_space=PhaseSpace.euclidean(2),
        time_horizon=TimeHorizon.real_line()  # T = ℝ
    )
    print(f"    ✓ System created with dimension = {system.dimension}")
    print(f"    ✓ Phase space: X = ℝ²")
    print(f"    ✓ Time horizon: T = ℝ")
    print(f"    ✓ Driving frequency: ω = {omega}")
    
    # Demonstrate that initial time matters!
    x0 = np.array([0.0, 1.0])
    t_duration = 20.0
    t_eval_relative = np.linspace(0.0, t_duration, 500)
    
    print(f"\n[2] Computing trajectories from different initial times...")
    print(f"    Same initial state: x(t₀) = {x0}")
    print(f"    Duration: {t_duration} time units")
    
    # Three different initial times
    initial_times = [0.0, 5.0, 10.0]
    solutions = []
    
    for t0 in initial_times:
        t_span = (t0, t0 + t_duration)
        t_eval = t_eval_relative + t0
        
        print(f"\n    Starting at t₀ = {t0}:")
        sol = system.trajectory(x0, t_span, t_eval, method='RK45')
        solutions.append((t0, sol))
        print(f"        ✓ Integration successful")
        print(f"        ✓ Final state: x({t0 + t_duration}) = [{sol.y[0, -1]:.4f}, {sol.y[1, -1]:.4f}]")
    
    # Plot comparison
    print("\n[3] Generating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green']
    
    # Time series for each initial time
    for i, (t0, sol) in enumerate(solutions):
        ax = axes[0, i] if i < 2 else axes[1, 0]
        
        ax.plot(sol.t, sol.y[0, :], label='x(t) - Position', linewidth=2, color=colors[i])
        ax.plot(sol.t, sol.y[1, :], label='y(t) - Velocity', linewidth=2, 
                color=colors[i], linestyle='--', alpha=0.7)
        ax.set_xlabel('Time t')
        ax.set_ylabel('State')
        ax.set_title(f'Time Series (t₀ = {t0})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Phase portraits comparison
    ax_phase = axes[1, 1]
    for i, (t0, sol) in enumerate(solutions):
        ax_phase.plot(sol.y[0, :], sol.y[1, :], linewidth=2, 
                     color=colors[i], label=f't₀ = {t0}', alpha=0.8)
        ax_phase.scatter([x0[0]], [x0[1]], color=colors[i], s=100, zorder=5)
    
    ax_phase.set_xlabel('x (Position)')
    ax_phase.set_ylabel('y (Velocity)')
    ax_phase.set_title('Phase Portraits: Different Initial Times')
    ax_phase.legend()
    ax_phase.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/outputs/02_driven_oscillator.png', dpi=150, bbox_inches='tight')
    print("    ✓ Plot saved to examples/outputs/02_driven_oscillator.png")
    
    # Demonstrate evolve() method
    print("\n[4] Testing evolve() method (single time-step)...")
    x_start = np.array([1.0, 0.0])
    dt = 0.1
    
    # Same initial state, different times
    x_at_t0 = system.evolve(x_start, t0=0.0, dt=dt)
    x_at_t10 = system.evolve(x_start, t0=10.0, dt=dt)
    
    print(f"    Starting state: x = {x_start}")
    print(f"    Time step: dt = {dt}")
    print(f"    From t₀=0.0:  x(0.1) = [{x_at_t0[0]:.6f}, {x_at_t0[1]:.6f}]")
    print(f"    From t₀=10.0: x(10.1) = [{x_at_t10[0]:.6f}, {x_at_t10[1]:.6f}]")
    print(f"    → Different evolution due to time-dependent forcing!")
    
    print("\n" + "=" * 70)
    print("Key Takeaway: For non-autonomous systems, the initial TIME matters!")
    print("The same initial state evolves differently depending on when you start.")
    print("=" * 70)


if __name__ == "__main__":
    import os
    os.makedirs('examples/outputs', exist_ok=True)
    main()

