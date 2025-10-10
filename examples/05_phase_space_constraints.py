"""
Example 5: Phase Space Constraints

Demonstrates:
- Defining phase spaces with constraints (not just R^n)
- Box constraints: X = [a_1, b_1] × [a_2, b_2] × ... × [a_n, b_n]
- State validation against phase space
- What happens when trajectories try to leave X
"""

import numpy as np
import matplotlib.pyplot as plt
from PyDynSys.core import AutonomousEuclideanDS, PhaseSpace


def main():
    print("=" * 70)
    print("Example: Phase Space Constraints")
    print("=" * 70)
    
    ### Example A: Box-constrained phase space ###
    
    print("\n[5a] BOX-CONSTRAINED PHASE SPACE")
    print("=" * 70)
    
    # Define a simple linear system
    def linear_system(state):
        """Linear system: dx/dt = Ax where A = [[0, 1], [-1, 0]]"""
        x, y = state
        return np.array([y, -x])
    
    # Create box-constrained phase space: X = [-2, 2] x [-2, 2]
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    phase_space_box = PhaseSpace.box(bounds)
    
    print(f"\nPhase space: X = [-2, 2] x [-2, 2]")
    print(f"  Dimension: {phase_space_box.dimension}")
    print(f"  Symbolic set: {phase_space_box.symbolic_set}")
    
    # Test phase space membership
    print(f"\nTesting phase space membership:")
    test_points = [
        np.array([0.0, 0.0]),    # Inside
        np.array([1.5, 1.5]),    # Inside
        np.array([3.0, 0.0]),    # Outside (x too large)
        np.array([0.0, -3.0]),   # Outside (y too small)
    ]
    
    for point in test_points:
        inside = phase_space_box.contains(point)
        print(f"  {point} ∈ X: {inside}")
    
    # Create system with box constraints
    system_box = AutonomousEuclideanDS(
        dimension=2,
        vector_field=linear_system,
        phase_space=phase_space_box
    )
    
    print(f"\n✓ System created with box-constrained phase space")
    
    # Try valid initial condition (inside box)
    x0_valid = np.array([1.0, 1.0])
    print(f"\n[1] Testing valid initial condition: x₀ = {x0_valid}")
    print(f"    Inside phase space: {phase_space_box.contains(x0_valid)}")
    
    t_span = (0.0, 10.0)
    t_eval = np.linspace(0.0, 10.0, 500)
    
    solution_valid = system_box.trajectory(x0_valid, t_span, t_eval)
    print(f"    ✓ Integration successful")
    
    # Try invalid initial condition (outside box)
    x0_invalid = np.array([3.0, 0.0])
    print(f"\n[2] Testing invalid initial condition: x₀ = {x0_invalid}")
    print(f"    Inside phase space: {phase_space_box.contains(x0_invalid)}")
    
    try:
        solution_invalid = system_box.trajectory(x0_invalid, t_span, t_eval)
        print(f"    ✗ Unexpected success!")
    except ValueError as e:
        print(f"    ✓ Correctly rejected with error:")
        print(f"      {str(e)}")
    
    ### Example b: Custom phase space (unit disk) ### 
    
    print("\n" + "=" * 70)
    print("[5b] CUSTOM PHASE SPACE: Unit Disk")
    print("=" * 70)
    
    # Define unit disk using symbolic set
    import sympy as sp
    x_sym, y_sym = sp.symbols('x y', real=True)
    unit_disk_symbolic = sp.ConditionSet(
        (x_sym, y_sym), 
        x_sym**2 + y_sym**2 < 1,
        sp.Reals**2
    )
    
    phase_space_disk = PhaseSpace.from_symbolic(unit_disk_symbolic, dimension=2)
    
    print(f"\nPhase space: Unit disk D = {{(x,y) : x² + y² < 1}}")
    print(f"  Dimension: {phase_space_disk.dimension}")
    
    # Test membership
    print(f"\nTesting phase space membership:")
    test_points_disk = [
        np.array([0.0, 0.0]),    # Center (inside)
        np.array([0.5, 0.5]),    # Inside
        np.array([0.9, 0.0]),    # Near boundary (inside)
        np.array([1.0, 0.0]),    # On boundary
        np.array([2.0, 0.0]),    # Outside
    ]
    
    for point in test_points_disk:
        inside = phase_space_disk.contains(point)
        dist = np.sqrt(point[0]**2 + point[1]**2)
        print(f"  {point} (r={dist:.2f}) ∈ D: {inside}")
    
    # Create system with disk constraint
    system_disk = AutonomousEuclideanDS(
        dimension=2,
        vector_field=linear_system,
        phase_space=phase_space_disk
    )
    
    print(f"\n✓ System created with unit disk phase space")
    
    # Test with valid point
    x0_disk = np.array([0.3, 0.3])
    print(f"\nSolving with x₀ = {x0_disk} (r = {np.linalg.norm(x0_disk):.3f})")
    
    solution_disk = system_disk.trajectory(x0_disk, (0.0, 5.0), np.linspace(0.0, 5.0, 300))
    print(f"    ✓ Integration successful")
    
    # Visualization
    print("\n[3] Generating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box constraint visualization
    ax1 = axes[0]
    
    # Draw box boundary
    box_x = [-2, 2, 2, -2, -2]
    box_y = [-2, -2, 2, 2, -2]
    ax1.plot(box_x, box_y, 'k--', linewidth=2, label='Phase space boundary')
    ax1.fill(box_x, box_y, alpha=0.1, color='green')
    
    # Plot trajectory
    ax1.plot(solution_valid.y[0, :], solution_valid.y[1, :], 
            linewidth=2, color='blue', label='Trajectory')
    ax1.scatter([x0_valid[0]], [x0_valid[1]], color='green', s=150, 
               zorder=5, marker='*', label='Initial condition')
    ax1.scatter([x0_invalid[0]], [x0_invalid[1]], color='red', s=150, 
               zorder=5, marker='x', label='Invalid IC')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Box-Constrained Phase Space')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Disk constraint visualization
    ax2 = axes[1]
    
    # Draw disk boundary
    theta = np.linspace(0, 2*np.pi, 100)
    disk_x = np.cos(theta)
    disk_y = np.sin(theta)
    ax2.plot(disk_x, disk_y, 'k--', linewidth=2, label='Phase space boundary')
    ax2.fill(disk_x, disk_y, alpha=0.1, color='green')
    
    # Plot trajectory
    ax2.plot(solution_disk.y[0, :], solution_disk.y[1, :], 
            linewidth=2, color='blue', label='Trajectory')
    ax2.scatter([x0_disk[0]], [x0_disk[1]], color='green', s=150, 
               zorder=5, marker='*', label='Initial condition')
    
    # Mark some test points
    ax2.scatter([2.0], [0.0], color='red', s=100, marker='x', 
               label='Outside disk')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Unit Disk Phase Space')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('examples/outputs/05_phase_space_constraints.png', dpi=150, bbox_inches='tight')
    print("    ✓ Plot saved to examples/outputs/05_phase_space_constraints.png")


if __name__ == "__main__":
    import os
    os.makedirs('examples/outputs', exist_ok=True)
    main()

