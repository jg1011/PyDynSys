"""
Example 3: Symbolic System Construction

Demonstrates:
- Building systems from symbolic equations using sympy
- Auto-detection of autonomous vs non-autonomous
- Factory method from_symbolic()
- Parameter substitution
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from PyDynSys.core import EuclideanDS, PhaseSpace


def main():
    print("=" * 70)
    print("Example 3: Symbolic System Construction")
    print("=" * 70)
    
    # Example 3a: Autonomous system from symbolic equations
    print("\n[3a] AUTONOMOUS SYSTEM: Van der Pol Oscillator")
    print("=" * 70)
    
    # Define symbolic variables
    t = sp.symbols('t')
    x, y = sp.symbols('x y', cls=sp.Function)
    x, y = x(t), y(t)
    
    # Define parameters
    mu = sp.Symbol('mu', positive=True, real=True)
    
    # Van der Pol equations: x'' - μ(1-x²)x' + x = 0
    # First-order form:
    equations_vdp = [
        sp.diff(x, t) - y,
        sp.diff(y, t) + mu * (x**2 - 1) * y + x
    ]
    
    print("\nSymbolic equations:")
    print(f"  dx/dt = y")
    print(f"  dy/dt = -μ(x² - 1)y - x")
    
    # Substitute parameter value
    parameters = {mu: 1.0}
    
    print(f"\nParameters: μ = {parameters[mu]}")
    print("\nBuilding system via factory method...")
    
    # Use factory method - automatically detects autonomous!
    system_vdp = EuclideanDS.from_symbolic(
        equations=equations_vdp,
        variables=[x, y],
        parameters=parameters
    )
    
    print(f"    ✓ System type detected: {type(system_vdp).__name__}")
    print(f"    ✓ Is autonomous: {not hasattr(system_vdp, 'time_horizon')}")
    print(f"    ✓ Dimension: {system_vdp.dimension}")
    
    # Solve and plot
    x0 = np.array([0.1, 0.1])
    t_span = (0.0, 30.0)
    t_eval = np.linspace(0.0, 30.0, 1000)
    
    print(f"\nSolving with x(0) = {x0}...")
    solution = system_vdp.trajectory(x0, t_span, t_eval)
    print(f"    ✓ Solution computed successfully")
    
    # Example 3b: Non-autonomous system from symbolic equations
    print("\n" + "=" * 70)
    print("[3b] NON-AUTONOMOUS SYSTEM: Parametrically Driven Oscillator")
    print("=" * 70)
    
    # Redefine variables for new system
    t = sp.symbols('t')
    x, y = sp.symbols('x y', cls=sp.Function)
    x, y = x(t), y(t)
    
    # Parameters
    omega_0, omega_d, A = sp.symbols('omega_0 omega_d A', real=True, positive=True)
    
    # Parametrically driven: x'' + ω₀²x = A·sin(ω_d·t)
    equations_driven = [
        sp.diff(x, t) - y,
        sp.diff(y, t) + omega_0**2 * x - A * sp.sin(omega_d * t)
    ]
    
    print("\nSymbolic equations:")
    print(f"  dx/dt = y")
    print(f"  dy/dt = -ω₀²x + A·sin(ω_d·t)")
    
    # Parameter values
    params_driven = {
        omega_0: 1.0,
        omega_d: 1.5,
        A: 0.5
    }
    
    print(f"\nParameters: ω₀ = {params_driven[omega_0]}, " +
          f"ω_d = {params_driven[omega_d]}, A = {params_driven[A]}")
    print("\nBuilding system via factory method...")
    
    # Factory automatically detects non-autonomous!
    system_driven = EuclideanDS.from_symbolic(
        equations=equations_driven,
        variables=[x, y],
        parameters=params_driven
    )
    
    print(f"    ✓ System type detected: {type(system_driven).__name__}")
    print(f"    ✓ Is non-autonomous: {hasattr(system_driven, 'time_horizon')}")
    print(f"    ✓ Dimension: {system_driven.dimension}")
    
    # Solve
    x0_driven = np.array([1.0, 0.0])
    t_span_driven = (0.0, 50.0)
    t_eval_driven = np.linspace(0.0, 50.0, 2000)
    
    print(f"\nSolving with x(0) = {x0_driven}...")
    solution_driven = system_driven.trajectory(x0_driven, t_span_driven, t_eval_driven)
    print(f"    ✓ Solution computed successfully")
    
    # Plot both systems
    print("\n[4] Generating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Van der Pol time series
    axes[0, 0].plot(solution.t, solution.y[0, :], linewidth=2, color='blue')
    axes[0, 0].set_xlabel('Time t')
    axes[0, 0].set_ylabel('x(t)')
    axes[0, 0].set_title('Van der Pol: Time Series')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Van der Pol phase portrait (shows limit cycle)
    axes[0, 1].plot(solution.y[0, :], solution.y[1, :], linewidth=2, color='blue')
    axes[0, 1].scatter([x0[0]], [x0[1]], color='green', s=100, zorder=5)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_title('Van der Pol: Phase Portrait (Limit Cycle)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Driven oscillator time series
    axes[1, 0].plot(solution_driven.t, solution_driven.y[0, :], linewidth=2, color='red')
    axes[1, 0].set_xlabel('Time t')
    axes[1, 0].set_ylabel('x(t)')
    axes[1, 0].set_title('Driven Oscillator: Time Series')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Driven oscillator phase portrait (chaotic?)
    axes[1, 1].plot(solution_driven.y[0, :], solution_driven.y[1, :], 
                   linewidth=0.5, color='red', alpha=0.7)
    axes[1, 1].scatter([x0_driven[0]], [x0_driven[1]], color='green', s=100, zorder=5)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('Driven Oscillator: Phase Portrait')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/outputs/03_symbolic_construction.png', dpi=150, bbox_inches='tight')
    print("    ✓ Plot saved to examples/outputs/03_symbolic_construction.png")
    
    print("\n" + "=" * 70)
    print("Key Takeaway: Factory method auto-detects system type from equations!")
    print("  - No explicit time in derivatives → Autonomous")
    print("  - Explicit time in derivatives → Non-autonomous")
    print("=" * 70)


if __name__ == "__main__":
    import os
    os.makedirs('examples/outputs', exist_ok=True)
    main()

