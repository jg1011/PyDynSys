# Core.Euclidean Module

## Overview

The `PyDynSys.core.euclidean` module provides classes for working with Euclidean dynamical systems, trajectories, phase spaces, and time horizons.

Let $X \subseteq \mathbb{R}^n$ and $T \subseteq \mathbb{R}$ be sets, let ${\bf F}: X \times T \to \mathbb{R}^n$ be a vector field, and suppose on $X \times T$ the first-order ODE $\dot{{\bf x}} = {\bf F}({\bf x}, t)$ holds. This is the setting we work within. Formally, we define a *Euclidean dynamical system* by the tuple $(X, T, {\bf F})$. As the notation suggests, ${\bf x}: T \to \mathbb{R}^n$ is a function of time. 

There is an abstract base class `DynamicalSystem`, which all types of dynamical system inherit from. There are two top-level inheritors, which we expect almost all consumers to inherit from: `AutonomousDS` and `NonAutonomousDS`, corresponding to the cases where our Euclidean dynamical system is (or isn't) autonomous, i.e. whether $\frac{\partial {\bf F}}{\partial t} \equiv 0$. In the autonomous case, where this is true, we need not specify $T$, and we opt to consider a vector field ${\bf F}: X \to \mathbb{R}^n$. All non-autonomous systems can be trivially made autonomous, though in doing so we lose some topological structure. As such, we elect to support both autonomous systems and their non-autonomous cousins.  

## Classes

### Dynamical Systems

#### [EuclideanDS](base.md#euclideands)
Abstract base class for Euclidean dynamical systems. Provides shared infrastructure for autonomous and non-autonomous systems, including solution caching, validation, and trajectory solving.

#### [AutonomousEuclideanDS](autonomous.md#autonomouseuclideands)
Systems where dx/dt = F(x) with F: R^n -> R^n independent of time. Suitable for time-invariant systems.

#### [NonAutonomousEuclideanDS](non_autonomous.md#nonautonomouseuclideands)
Systems where dx/dt = F(x, t) with F: R^n × R -> R^n dependent on time. Suitable for time-varying systems.

### Trajectory Types

#### [EuclideanTrajectorySegment](trajectory.md#euclideantrajectorysegment)
Represents a single continuous trajectory segment from a numerical integration solve.

#### [EuclideanTrajectory](trajectory.md#euclideantrajectory)
Composed trajectory that can span multiple segments, supporting bidirectional integration and trajectory composition.

### Phase Space and Time

#### [PhaseSpace](euclidean/phase_space.md)
Phase space representation with flexible symbolic and callable constraints. Supports various factory methods for common geometric shapes (boxes, hyperspheres) and custom constraints.

#### [TimeHorizon](time_horizon.md#timehorizon)
Time domain representation for dynamical systems, supporting real line, intervals, and custom time domains.

## Full Docs

::: PyDynSys.core.euclidean
    options:
      show_root_heading: true
      show_source: true
