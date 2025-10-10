"""
Example 6: Performance Optimization with PhaseSpace

Demonstrates:
- Three ways to define phase spaces
- Performance comparison: symbolic vs callable vs both
- When to use each approach
- from_constraint() factory for maximum performance
"""

import numpy as np
import time
from PyDynSys.core import AutonomousEuclideanDS, PhaseSpace


def main():
    print("=" * 70)
    print("Example 6: PhaseSpace Performance Optimization")
    print("=" * 70)
    
    # Define a simple vector field, simple harmonic oscillator
    def simple_flow(state):
        x, y = state
        return np.array([y, -x])
    
    dimension = 2
    
    ### Approach 1: Symbolic Only (Auto-compiled) ###
    
    print("\n[1] SYMBOLIC ONLY: Auto-compilation")
    print("=" * 70)
    
    import sympy as sp
    x_sym, y_sym = sp.symbols('x y', real=True)
    unit_disk_symbolic = sp.ConditionSet(
        (x_sym, y_sym), 
        x_sym**2 + y_sym**2 < 1,
        sp.Reals**2
    )
    
    print("  Creating phase space from symbolic set...")
    start = time.time()
    phase_space_symbolic = PhaseSpace.from_symbolic(unit_disk_symbolic, dimension=2)
    creation_time = time.time() - start
    
    print(f"    Creation time: {creation_time*1000:.3f} ms")
    print(f"    Has symbolic: {phase_space_symbolic.symbolic_set is not None}")
    print(f"    Has constraint: {phase_space_symbolic.constraint is not None}")
    print(f"    → Constraint was AUTO-COMPILED from symbolic")
    
    # Benchmark membership testing
    test_points = np.random.randn(10000, 2) * 0.5  # Points around origin
    
    print("\n  Benchmarking membership tests (10,000 points)...")
    start = time.time()
    results_symbolic = [phase_space_symbolic.contains(pt) for pt in test_points]
    test_time_symbolic = time.time() - start
    print(f"    Test time: {test_time_symbolic*1000:.3f} ms")
    print(f"    Points inside: {sum(results_symbolic)}/{len(test_points)}")
    
    ### Approach 2: Callable Only (Maximum Performance) ###
    
    print("\n[2] CALLABLE ONLY: Maximum performance")
    print("=" * 70)
    
    def unit_disk_constraint(x):
        """Fast numpy-based membership test."""
        return x[0]**2 + x[1]**2 < 1.0
    
    print("  Creating phase space from constraint...")
    start = time.time()
    phase_space_callable = PhaseSpace.from_constraint(
        dimension=2, 
        constraint=unit_disk_constraint
    )
    creation_time = time.time() - start
    
    print(f"    Creation time: {creation_time*1000:.6f} ms (essentially instant!)")
    print(f"    Has symbolic: {phase_space_callable.symbolic_set is not None}")
    print(f"    Has constraint: {phase_space_callable.constraint is not None}")
    print(f"    → Constraint provided DIRECTLY, no symbolic overhead")
    
    # Benchmark membership testing
    print("\n  Benchmarking membership tests (10,000 points)...")
    start = time.time()
    results_callable = [phase_space_callable.contains(pt) for pt in test_points]
    test_time_callable = time.time() - start
    print(f"    Test time: {test_time_callable*1000:.3f} ms")
    print(f"    Points inside: {sum(results_callable)}/{len(test_points)}")
    
    ### Approach 3: Both (Best of Both Worlds) ###
    
    print("\n[3] BOTH: Symbolic + Pre-optimized Constraint")
    print("=" * 70)
    
    print("  Creating phase space with both representations...")
    start = time.time()
    phase_space_both = PhaseSpace(
        dimension=2,
        symbolic_set=unit_disk_symbolic,
        constraint=unit_disk_constraint  # Skip auto-compilation!
    )
    creation_time = time.time() - start
    
    print(f"    Creation time: {creation_time*1000:.3f} ms")
    print(f"    Has symbolic: {phase_space_both.symbolic_set is not None}")
    print(f"    Has constraint: {phase_space_both.constraint is not None}")
    print(f"    → Best of both: fast validation + symbolic operations")
    
    # Benchmark membership testing
    print("\n  Benchmarking membership tests (10,000 points)...")
    start = time.time()
    results_both = [phase_space_both.contains(pt) for pt in test_points]
    test_time_both = time.time() - start
    print(f"    Test time: {test_time_both*1000:.3f} ms")
    print(f"    Points inside: {sum(results_both)}/{len(test_points)}")
    
    ### Performance Comparison ###
    
    print("\n[4] PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"\nCreation time:")
    print(f"  Symbolic only:  {creation_time*1000:.6f} ms (baseline)")
    print(f"  Callable only:  ~instant (no compilation)")
    print(f"  Both:           {creation_time*1000:.6f} ms (same as symbolic)")
    
    print(f"\nMembership testing (10k points):")
    print(f"  Symbolic only:  {test_time_symbolic*1000:.3f} ms")
    print(f"  Callable only:  {test_time_callable*1000:.3f} ms")
    print(f"  Both:           {test_time_both*1000:.3f} ms")
    
    # Speedup
    speedup = test_time_symbolic / test_time_callable if test_time_callable > 0 else float('inf')
    print(f"\n  Speedup (symbolic → callable): {speedup:.2f}x")
    
    ### Practical Application: System with Performance-Critical Validation ###
    
    print("\n[5] PRACTICAL APPLICATION")
    print("=" * 70)
    
    # For performance-critical applications, use callable-only
    print("\n  Creating system with callable-only phase space...")
    system_fast = AutonomousEuclideanDS(
        dimension=2,
        vector_field=simple_flow,
        phase_space=phase_space_callable  # Fast validation!
    )
    
    # Test trajectory computation (validation happens at every call)
    x0 = np.array([0.3, 0.3])
    t_eval = np.linspace(0, 5, 100)
    
    print(f"  Computing trajectory with x₀ = {x0}...")
    start = time.time()
    solution = system_fast.trajectory(x0, (0, 5), t_eval)
    total_time = time.time() - start
    
    print(f"    ✓ Integration successful")
    print(f"    Total time: {total_time*1000:.3f} ms")
    print(f"    (Includes initial state validation with fast constraint)")
    
    ### Recommendations ###
    
    print("\n[6] RECOMMENDATIONS")
    print("=" * 70)
    print("""
  When to use each approach:
  
  1. SYMBOLIC ONLY (from_symbolic):
     - Need symbolic operations (intersections, closures, etc.)
     - Validation performance not critical
     - Research/exploration phase
  
  2. CALLABLE ONLY (from_constraint):
     - Performance-critical applications
     - High-frequency state validation
     - Production systems
     - Don't need symbolic operations
  
  3. BOTH (provide both arguments):
     - Need symbolic operations AND performance
     - Library development (e.g., factory methods)
     - Best of both worlds (recommended when possible)
  
  Note: Built-in factories (euclidean(), box()) automatically provide
        both representations for optimal performance!
    """)


if __name__ == "__main__":
    import os
    os.makedirs('examples/outputs', exist_ok=True)
    main()

