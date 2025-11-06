# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- [Hamiltonian Mixin]
- [Scalar Mixin]
- [Planar Mixin]
- [Gradient Mixin]
- [from_time_series Factory]
- [Lyapunov Exponent Computation]

## [1.0.0] - 2025-11-06

This is the first official, stable release of `PyDynSys`. All `v0.x.y` versions should be considered experimental, are undocumented, and should **not** be used.

### Added
- **Core Architecture:** Initial object-oriented framework for continuous-time dynamical systems.
- **Autonomous Systems:** `AutDynSys` class for defining and solving systems of the form `dx/dt = F(x)`.
- **Non-Autonomous Systems:** `NonAutDynSys` class for systems of the form `dx/dt = F(x, t)`.
- **Symbolic Definition:** `from_symbolic` factory method to create system instances directly from `sympy` equations.
- **Functional Definition:** `from_vector_field` factory method to create systems from any callable Python function.
- **ODE Solvers:** Integrated `scipy.integrate.solve_ivp` as the core numerical engine for solving initial value problems.
- **Bidirectional Integration:** Implemented support for solving trajectories both forward and backward in time from a single initial condition.
- **Trajectory Objects:** Created a `Trajectory` class to act as a rich container for solution data, with support for interpolation.
- **Shared Infrastructure:** Implemented a `_DynSys` mixin to provide common validation logic and a solution cache.
- **Domain Support:** Added `PhaseSpace` and `TimeHorizon` classes to define the domains of systems.
- **Testing & CI:** Set up a test suite with `pytest` and a GitHub Actions workflow for continuous integration and code coverage reporting with Codecov.
- **Documentation:** Created a `README.md`, this `CHANGELOG.md` and a docs website via `mkdocs`.
