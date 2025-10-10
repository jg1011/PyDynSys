import numpy as np
import pytest

from PyDynSys.core import AutonomousEuclideanDS, PhaseSpace


@pytest.mark.autonomous
@pytest.mark.validation
def test_trajectory_and_evolve_basic(solver_methods):
    # Simple linear system: x' = y, y' = -x
    def vf(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([y, -x], dtype=float)

    sys = AutonomousEuclideanDS(dimension=2, vector_field=vf, phase_space=PhaseSpace.euclidean(2))

    x0 = np.array([1.0, 0.0])
    t_span = (0.0, 1.0)
    t_eval = np.linspace(0.0, 1.0, 50)

    # trajectory succeeds for all requested methods
    for method in solver_methods:
        sol = sys.trajectory(x0, t_span, t_eval, method=method)
        assert sol.y.shape == (2, len(t_eval))

    # evolve single step
    dt = 0.05
    x1 = sys.evolve(x0, t0=0.0, dt=dt, method="RK45")
    assert x1.shape == (2,)


@pytest.mark.autonomous
@pytest.mark.validation
def test_evolve_dt_validation():
    def vf(state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -state[0]], dtype=float)

    sys = AutonomousEuclideanDS(dimension=2, vector_field=vf)
    with pytest.raises(ValueError):
        sys.evolve(np.array([1.0, 0.0]), t0=0.0, dt=0.0)
    with pytest.raises(ValueError):
        sys.evolve(np.array([1.0, 0.0]), t0=0.0, dt=-1.0)


@pytest.mark.autonomous
@pytest.mark.validation
def test_bidirectional_integration_invariants():
    # Same linear oscillator
    def vf(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([y, -x], dtype=float)

    sys = AutonomousEuclideanDS(dimension=2, vector_field=vf)

    x0 = np.array([0.5, -0.2])
    t0 = 0.0
    T = 2.0
    t_eval = np.linspace(-T, T, 401)
    # By API: set t_span first element as t0; second is a bound (forward max)
    sol = sys.trajectory(x0, (t0, T), t_eval, method="RK45")

    # Times are strictly ascending and span [-T, T]
    assert np.all(np.diff(sol.t) > 0)
    assert pytest.approx(sol.t[0], abs=1e-12) == -T
    assert pytest.approx(sol.t[-1], abs=1e-12) == T

    # No duplicate t0; continuity around t0 (left/right neighbors close)
    idx = np.searchsorted(sol.t, t0)
    assert not np.isclose(sol.t[idx-1], t0)
    assert not np.isclose(sol.t[idx], t0)
    # y is continuous (adjacent samples near t0 shouldn't jump wildly)
    jump = np.linalg.norm(sol.y[:, idx] - sol.y[:, idx-1])
    assert jump < 1e-1


