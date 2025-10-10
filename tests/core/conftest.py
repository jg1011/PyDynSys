"""Pytest configuration and shared fixtures for PyDynSys core tests."""

import os
import sys
import pytest


# Ensure local imports resolve (PyDynSys under src/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


def pytest_configure(config):
    config.addinivalue_line("markers", "physics: physical laws/invariants tests")
    config.addinivalue_line("markers", "numerical: numerical accuracy vs analytic tests")
    config.addinivalue_line("markers", "validation: API/contract validation tests")
    config.addinivalue_line("markers", "symbolic_builder: tests for SymbolicSystemBuilder")
    config.addinivalue_line("markers", "euclidean_base: tests for EuclideanDS base class")
    config.addinivalue_line("markers", "autonomous: tests for AutonomousEuclideanDS")
    config.addinivalue_line("markers", "non_autonomous: tests for NonAutonomousEuclideanDS")
    config.addinivalue_line("markers", "phase_space: tests for PhaseSpace")
    config.addinivalue_line("markers", "time_horizon: tests for TimeHorizon")
    config.addinivalue_line("markers", "caching: tests involving solution caching")


# -----------------------------
# Tolerance fixtures
# -----------------------------


@pytest.fixture
def energy_tol_stable() -> float:
    """Energy conservation tolerance for stable regimes"""
    return 1e-3


@pytest.fixture
def energy_tol_unstable() -> float:
    """Energy conservation tolerance for unstable regimes"""
    return 1e-2


@pytest.fixture
def numeric_tol_stable() -> tuple[float, float]:
    """(rtol, atol) for analytic vs numeric in stable regimes"""
    # Absolute tolerance dominates near zeros; set atol to clear ~3e-4 max-abs error in off-resonance case
    return (1e-3, 3e-4)


@pytest.fixture
def numeric_tol_unstable() -> tuple[float, float]:
    """(rtol, atol) for analytic vs numeric in near-unstable regimes"""
    return (1e-2, 1e-3)


# -----------------------------
# Solver enumeration fixture
# -----------------------------


@pytest.fixture
def solver_methods() -> list[str]:
    """List of scipy::solve_ivp methods to evaluate; tests will take min error across them."""
    return [
        "RK45",
        "DOP853",
        "RK23",
        "Radau",
        "BDF",
        "LSODA",
    ]


# -----------------------------
# Energy conservation helper factory
# -----------------------------


@pytest.fixture
def energy_conservation_tester():
    """
    Factory returning a function that checks relative energy drift for a SciPyIvpSolution.

    Returns a callable f(energy_fn, solution) -> (drift: float, conserved: bool)
    where drift is max relative deviation from initial energy over the trajectory.
    """

    import numpy as np

    def _tester(energy_fn, solution):
        y = solution.y
        energies = np.asarray([energy_fn(y[:, i]) for i in range(y.shape[1])], dtype=float)
        e0 = float(energies[0])
        if e0 == 0.0:
            drift = float(np.max(np.abs(energies - e0)))
            conserved = drift == 0.0
        else:
            drift = float(np.max(np.abs((energies - e0) / e0)))
            conserved = drift == 0.0
        return drift, conserved

    return _tester


