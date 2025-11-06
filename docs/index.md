# Welcome to PyDynSys

**PyDynSys** is a powerful, intuitive Python library for the analysis of continuous-time dynamical systems. It provides a high-level, object-oriented API for defining, solving, and analyzing systems from their symbolic equations or numerical vector fields.

Developed with researchers, engineers, and students in mind, `PyDynSys` aims to make the exploration of complex systems both simple and mathematically robust.

## Key Features

- **Symbolic & Functional Support**: Initialize systems with either `Sympy` equations or a callable Python function, providing maximum flexibility.
- **Dual System Types**: First-class support for both **Autonomous** (`dx/dt = F(x)`) and **Non-Autonomous** (`dx/dt = F(x, t)`) systems.
- **Robust Solvers**: Leverages the powerful, adaptive solvers from `scipy.integrate.solve_ivp` under the hood.
- **Bidirectional Integration**: Effortlessly solve trajectories forward and backward in time from any initial condition.
- **Rich Trajectory Objects**: Work with trajectory data as first-class objects, with support for slicing, interpolation, and analysis.
- **Property Detection**: (Future) Automatically detect and compose systems with properties like linearity or Hamiltonian structure.

## Getting Started

Ready to dive in? Check out the **[Quickstart Guide](getting_started/quickstart.md)** to define and solve your first dynamical system in just a few lines of code. 