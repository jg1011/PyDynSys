import numpy as np
import pytest

from PyDynSys.core import NonAutonomousEuclideanDS, PhaseSpace, TimeHorizon


@pytest.mark.non_autonomous
@pytest.mark.validation
def test_initial_time_dependence(solver_methods):
    # Driven oscillator: x' = y, y' = -x + sin(omega t)
    omega = 2.0

    def vf(state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        return np.array([y, -x + np.sin(omega * t)], dtype=float)

    sys = NonAutonomousEuclideanDS(dimension=2, vector_field=vf, phase_space=PhaseSpace.euclidean(2), time_horizon=TimeHorizon.real_line())

    x0 = np.array([1.0, 0.0])
    T = 5.0
    t_eval_rel = np.linspace(0.0, T, 200)

    for method in solver_methods:
        sol_t0 = sys.trajectory(x0, (0.0, T), t_eval_rel, method=method)
        sol_t5 = sys.trajectory(x0, (5.0, 5.0 + T), t_eval_rel + 5.0, method=method)
        # Different initial times should yield different trajectories, as expected
        assert np.linalg.norm(sol_t0.y - sol_t5.y) > 1e-6


@pytest.mark.non_autonomous
@pytest.mark.validation
def test_time_horizon_enforcement():
    def vf(state: np.ndarray, t: float) -> np.ndarray:
        return np.array([state[1], -state[0]], dtype=float)

    horizon = TimeHorizon.interval(0.0, 10.0)
    sys = NonAutonomousEuclideanDS(dimension=2, vector_field=vf, time_horizon=horizon)

    x0 = np.array([0.0, 1.0])
    # Valid
    sys.trajectory(x0, (0.0, 1.0), np.linspace(0.0, 1.0, 10))
    # Invalid start
    with pytest.raises(ValueError):
        sys.trajectory(x0, (-1.0, 1.0), np.linspace(-1.0, 1.0, 10))
    # Invalid end
    with pytest.raises(ValueError):
        sys.trajectory(x0, (0.0, 20.0), np.linspace(0.0, 20.0, 10))


@pytest.mark.non_autonomous
@pytest.mark.validation
def test_evolve_matches_single_step():
    def vf(state: np.ndarray, t: float) -> np.ndarray:
        x, y = state
        return np.array([y, -x + 0.1 * np.sin(t)], dtype=float)

    sys = NonAutonomousEuclideanDS(dimension=2, vector_field=vf)
    x0 = np.array([0.3, -0.4])
    t0 = 1.0
    dt = 0.05

    x_next = sys.evolve(x0, t0=t0, dt=dt, method="RK45")
    sol = sys.trajectory(x0, (t0, t0 + dt), np.array([t0 + dt]), method="RK45")
    assert np.allclose(x_next, sol.y[:, -1], rtol=1e-6, atol=1e-8)


