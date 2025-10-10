"""
Example 1: Simple Harmonic Oscillator (Autonomous System)

Demonstrates:
- Creating an autonomous system from a direct vector field
- Default phase space (ℝ²)
- Forward trajectory computation
- Plotting phase portraits
"""

import numpy as np
import matplotlib.pyplot as plt
from PyDynSys.core import AutonomousEuclideanDS, PhaseSpace


def main():
    print("=" * 70)
    print("Example 1: Simple Harmonic Oscillator")
    print("System: x'' + x = 0")
    print("First-order form: dx/dt = y, dy/dt = -x")
    print("=" * 70)
    
    # Define vector field for harmonic oscillator
    def harmonic_oscillator(state):
        """
        Vector field for undamped harmonic oscillator.
        
        Args:
            state: [x, y] where x is position, y is velocity
            
        Returns:
            [dx/dt, dy/dt] = [y, -x]
        """
        x, y = state
        return np.array([y, -x])
    
    # Create system with default phase space X = ℝ²
    print("\n[1] Creating autonomous system...")
    system = AutonomousEuclideanDS(
        dimension=2,
        vector_field=harmonic_oscillator,
        phase_space=PhaseSpace.euclidean(2)  # Explicit, but this is the default
    )
    print(f"    ✓ System created with dimension = {system.dimension}")
    print(f"    ✓ Phase space: ℝ²")
    
    # Define initial conditions and time span
    x0 = np.array([1.0, 0.0])  # Initial position x=1, velocity y=0
    t_span = (0.0, 10.0)
    t_eval = np.linspace(0.0, 10.0, 500)
    
    print(f"\n[2] Solving IVP with x(0) = {x0}")
    print(f"    Time span: {t_span}")
    
    # Compute trajectory
    solution = system.trajectory(x0, t_span, t_eval, method='RK45')
    
    print(f"    ✓ Integration successful: {solution.success}")
    print(f"    ✓ Message: {solution.message}")
    print(f"    ✓ Solution shape: {solution.y.shape} (2 states × {len(t_eval)} times)")
    
    # Verify energy conservation
    x_traj = solution.y[0, :]
    y_traj = solution.y[1, :]
    energy = 0.5 * (x_traj**2 + y_traj**2)
    energy_drift = np.std(energy)
    
    print(f"\n[3] Verifying energy conservation...")
    print(f"    Initial energy: E₀ = {energy[0]:.6f}")
    print(f"    Final energy:   E_f = {energy[-1]:.6f}")
    print(f"    Energy drift (std): {energy_drift:.2e}")
    
    # Plot results
    print("\n[4] Generating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series
    ax1.plot(solution.t, x_traj, label='x(t) - Position', linewidth=2)
    ax1.plot(solution.t, y_traj, label='y(t) - Velocity', linewidth=2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('State')
    ax1.set_title('Harmonic Oscillator: Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Phase portrait
    ax2.plot(x_traj, y_traj, linewidth=2, color='navy')
    ax2.scatter([x0[0]], [x0[1]], color='green', s=100, zorder=5, label='Initial condition')
    ax2.scatter([x_traj[-1]], [y_traj[-1]], color='red', s=100, zorder=5, label='Final state')
    ax2.set_xlabel('x (Position)')
    ax2.set_ylabel('y (Velocity)')
    ax2.set_title('Phase Portrait (Circular Orbit)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('examples/outputs/01_harmonic_oscillator.png', dpi=150, bbox_inches='tight')
    print("    ✓ Plot saved to examples/outputs/01_harmonic_oscillator.png")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Create output directory if needed
    import os
    os.makedirs('examples/outputs', exist_ok=True)
    main()

