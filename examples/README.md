# PyFlow Examples

This directory contains comprehensive examples demonstrating the PyDynSys dynamical systems library.

## Running Examples

Each example is self-contained and can be run independently:

```bash
python examples/01_harmonic_oscillator.py
python examples/02_driven_oscillator.py
# ... etc
```

The relevant plots are saved to `examples/outputs/`.

To run all, you can use our quick bash script `run_examples.sh` or our powershell script `run_examples.ps1`. 


## Example Overview

### 01 - Harmonic Oscillator (Autonomous System)
**File**: `01_harmonic_oscillator.py`

**Demonstrates**:
- Creating an autonomous system from a direct vector field
- Default phase space (ℝ²)
- Forward trajectory computation
- Energy conservation verification
- Phase portraits

**System**: Simple harmonic oscillator `x'' + x = 0`

---

### 02 - Driven Oscillator (Non-Autonomous System)
**File**: `02_driven_oscillator.py`

**Demonstrates**:
- Creating a non-autonomous system with explicit time dependence
- Time horizon specification
- How initial time t₀ affects the solution
- Comparing trajectories from different initial times
- The `evolve()` method for single time-steps

**System**: Driven harmonic oscillator `x'' + x = sin(ωt)`

**Key Insight**: For non-autonomous systems, the same initial state evolves differently depending on when you start!

---

### 03 - Symbolic Construction
**File**: `03_symbolic_construction.py`

**Demonstrates**:
- Building systems from symbolic equations using SymPy
- Auto-detection of autonomous vs non-autonomous
- Factory method `from_symbolic()`
- Parameter substitution

**Systems**: 
- Van der Pol oscillator (autonomous, shows limit cycle)
- Parametrically driven oscillator (non-autonomous)

**Key Insight**: The factory method automatically detects system type from equations!

---

### 04 - Bidirectional Integration
**File**: `04_bidirectional_integration.py`

**Demonstrates**:
- Bidirectional integration: flow on interval I around initial time t₀
- Example: I = (-2, 2) with t₀ = 0
- Automatic detection and handling by the library
- Visualizing flow in both time directions

**System**: Damped harmonic oscillator `x'' + 2γx' + x = 0`

**Key Insight**: When t₀ is interior to the t_eval range, the library automatically performs bidirectional integration!

---

### 05 - Phase Space Constraints
**File**: `05_phase_space_constraints.py`

**Demonstrates**:
- Defining phase spaces with constraints (not just ℝⁿ)
- Box constraints: X = [a₁, b₁] × [a₂, b₂]
- Custom constraints: Unit disk
- State validation against phase space
- What happens when initial conditions violate constraints

**Systems**: Linear system with different phase space constraints

**Key Insight**: Phase spaces can be constrained. Invalid initial conditions are rejected at validation.

---

### 06 - Performance Optimization
**File**: `06_performance_optimization.py`

**Demonstrates**:
- Three ways to define phase spaces
- Performance comparison: symbolic vs callable vs both
- When to use each approach
- `from_constraint()` factory for maximum performance

**Key Insight**: PhaseSpace supports three patterns:
1. **Symbolic only**: Flexible, auto-compiles constraint (general use)
2. **Callable only**: Maximum performance (production systems)
3. **Both**: Best of both worlds (recommended when possible)

Built-in factories like `euclidean()` and `box()` automatically provide both for optimal performance!

---

## Mathematical Background

### Autonomous Systems
Systems where **dx/dt = F(x)** with F independent of time:
- Flow forms a semi-group: φₛ(φₜ(x)) = φₛ₊ₜ(x)
- Time-translation invariant
- Phase portraits are time-independent

### Non-Autonomous Systems  
Systems where **dx/dt = F(x, t)** with explicit time dependence:
- Flow depends on initial time: φₜ(x, t₀)
- Same initial state evolves differently at different times
- Useful for forced/driven systems

### Phase Space
The set X ⊆ ℝⁿ where states live:
- Default: X = ℝⁿ (entire Euclidean space)
- Constrained: Box constraints, manifolds, custom sets
- Library validates x ∈ X before integration

### Bidirectional Integration
For autonomous systems, can compute flow on intervals around t₀:
- Integrates backwards: t₀ → min(t_eval)
- Integrates forwards: t₀ → max(t_eval)
- Useful for studying complete orbits

---

## Dependencies

Examples require:
- `numpy`
- `scipy`
- `matplotlib`
- `sympy`

Install via:
```bash
pip install numpy scipy matplotlib sympy
```

---

## Next Steps

After running these examples:
1. Modify parameters to explore different behaviors
2. Try your own vector fields
3. Experiment with different phase space constraints
4. Study the source code to understand the implementation
5. Run the test suite: `pytest tests/`

