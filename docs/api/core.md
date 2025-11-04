# PyDynSys.core Module

The `PyDynSys.core` module provides the foundational classes and utilities for working with Euclidean dynamical systems.

## Submodules

- **[Core.Euclidean Module](core/euclidean.md)** - Euclidean dynamical systems and related types

## Main Classes and Types

### Dynamical Systems
- **[EuclideanDS](core/euclidean.md#euclideands)** - Abstract base class for Euclidean dynamical systems
- **[AutonomousEuclideanDS](core/euclidean.md#autonomouseuclideands)** - Systems where dx/dt = F(x)
- **[NonAutonomousEuclideanDS](core/euclidean.md#nonautonomouseuclideands)** - Systems where dx/dt = F(x, t)

### Trajectory Types
- **[EuclideanTrajectorySegment](core/euclidean.md#euclideantrajectorysegment)** - Single trajectory segment
- **[EuclideanTrajectory](core/euclidean.md#euclideantrajectory)** - Composed trajectory over multiple segments

### Phase Space and Time
- **[PhaseSpace](core/euclidean/phase_space.md)** - Phase space representation with symbolic and callable constraints
- **[TimeHorizon](core/euclidean.md#timehorizon)** - Time domain representation

### System Builder
- **[SymbolicSystemBuilder](core/euclidean.md#symbolicsystembuilder)** - Build systems from symbolic ODEs

### Type Definitions
- `AutonomousVectorField` - Vector field type for autonomous systems
- `NonAutonomousVectorField` - Vector field type for non-autonomous systems
- `VectorField` - Union type for any vector field
- `SymbolicODE` - Symbolic ODE representation
- `SystemParameters` - Parameter substitution dictionary
- `TrajectoryCacheKey` - Cache key for trajectory solutions
- `SciPyIvpSolution` - SciPy IVP solution wrapper
- `TrajectorySegmentMergePolicy` - Policy for merging trajectory segments

## Full Docs

::: PyDynSys.core
    options:
      show_root_heading: true
      show_source: true
