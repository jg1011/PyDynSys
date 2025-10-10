import numpy as np
import pytest

from PyDynSys.core import AutonomousEuclideanDS, PhaseSpace


@pytest.mark.euclidean_base
@pytest.mark.validation
def test_state_validation_and_phase_space():    
    def vf(state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -state[0]], dtype=float)

    # Box-constrained space
    bounds = np.array([[-1.0, 1.0], [-2.0, 2.0]])
    X = PhaseSpace.box(bounds)
    sys = AutonomousEuclideanDS(dimension=2, vector_field=vf, phase_space=X)

    # Valid
    sys.trajectory(np.array([0.0, 0.0]), (0.0, 1.0), np.linspace(0.0, 1.0, 10))
    # Invalid: outside x bound
    with pytest.raises(ValueError):
        sys.trajectory(np.array([2.0, 0.0]), (0.0, 1.0), np.linspace(0.0, 1.0, 10))


@pytest.mark.euclidean_base
@pytest.mark.validation
def test_time_span_and_t_eval_rules():
    def vf(state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -state[0]], dtype=float)

    sys = AutonomousEuclideanDS(dimension=2, vector_field=vf)
    x0 = np.array([0.0, 1.0])

    # Zero-length interval rejected
    with pytest.raises(ValueError):
        sys.trajectory(x0, (0.0, 0.0), np.linspace(0.0, 1.0, 10))

    # Unidirectional eval outside span rejected
    with pytest.raises(ValueError):
        sys.trajectory(x0, (0.0, 1.0), np.linspace(-0.2, -0.1, 10))

    # Bidirectional allowed when t0 interior to t_eval
    t_eval = np.linspace(-1.0, 1.0, 50)
    sys.trajectory(x0, (0.0, 1.0), t_eval)


@pytest.mark.euclidean_base
@pytest.mark.caching
def test_caching_key_changes_with_method():
    def vf(state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -state[0]], dtype=float)

    sys = AutonomousEuclideanDS(dimension=2, vector_field=vf)
    x0 = np.array([1.0, 0.0])
    t_eval = np.linspace(0.0, 1.0, 20)

    sol_a = sys.trajectory(x0, (0.0, 1.0), t_eval, method="RK45")
    sol_a_cached = sys.trajectory(x0, (0.0, 1.0), t_eval, method="RK45")
    assert sol_a is sol_a_cached

    sol_b = sys.trajectory(x0, (0.0, 1.0), t_eval, method="DOP853")
    assert sol_b is not sol_a


@pytest.mark.euclidean_base
@pytest.mark.validation
def test_scipy_solution_wrapper_attrs(solver_methods):
    def vf(state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -state[0]], dtype=float)

    sys = AutonomousEuclideanDS(dimension=2, vector_field=vf)
    x0 = np.array([0.5, 0.0])
    t_eval = np.linspace(0.0, 2.0, 200)

    for method in solver_methods:
        sol = sys.trajectory(x0, (0.0, 2.0), t_eval, method=method, dense_output=True)
        assert sol.t.shape == t_eval.shape
        assert sol.y.shape == (2, len(t_eval))
        # Dense output exists for unidirectional runs
        assert sol.sol is not None


