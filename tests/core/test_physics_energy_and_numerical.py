"""
Elementary physical invariant / numerical <-> analytic accuracy tests on simple harmonic oscillator. 
    - Verifies, somewhat, validity of numerical solvers. Safety is really outsourced to scipy though, 
    this primarily serves as a sanity check. 
We usually just check some solver works, mitigating solver semantics and focusing on core logic.
"""


import numpy as np
import pytest

from PyDynSys.core import AutonomousEuclideanDS, NonAutonomousEuclideanDS, PhaseSpace

def _sho_autonomous(omega: float = 1.0) -> AutonomousEuclideanDS:
    def vf(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([y, -(omega**2) * x], dtype=float)

    return AutonomousEuclideanDS(dimension=2, vector_field=vf, phase_space=PhaseSpace.euclidean(2))


def _sho_energy_fn(omega: float):
    def energy(state: np.ndarray) -> float:
        x, y = state
        return 0.5 * (y**2 + (omega**2) * x**2)
    return energy


@pytest.mark.physics
def test_energy_conservation_sho_min_over_solvers(solver_methods, energy_conservation_tester, energy_tol_stable):
    omega = 1.0
    system = _sho_autonomous(omega)
    x0 = np.array([1.0, 0.0])
    t_eval = np.linspace(0.0, 20.0, 1000)

    drifts = []
    for method in solver_methods:
        sol = system.trajectory(x0, (0.0, t_eval[-1]), t_eval, method=method)
        drift, _ = energy_conservation_tester(_sho_energy_fn(omega), sol)
        drifts.append(drift)

    assert min(drifts) <= energy_tol_stable


@pytest.mark.numerical
def test_sho_numeric_vs_analytic(solver_methods, numeric_tol_stable):
    omega = 2.0
    system = _sho_autonomous(omega)
    x0 = np.array([1.0, 0.0])
    t_eval = np.linspace(0.0, 4 * np.pi / omega, 500)

    def analytical(t):
        return np.vstack([
            np.cos(omega * t),
            -omega * np.sin(omega * t),
        ])

    rtol, atol = numeric_tol_stable
    ok_flags = []
    errs = []
    for method in solver_methods:
        sol = system.trajectory(x0, (t_eval[0], t_eval[-1]), t_eval, method=method)
        y_true = analytical(sol.t)
        errs.append(float(np.max(np.abs(sol.y - y_true))))
        ok_flags.append(np.allclose(sol.y, y_true, rtol=rtol, atol=atol))
    if not any(ok_flags):
        raise AssertionError(f"No solver met numeric_tol_stable; max-abs errors per solver: {dict(zip(solver_methods, errs))}")


@pytest.mark.numerical
def test_driven_sho_off_vs_near_resonance(solver_methods, numeric_tol_stable, numeric_tol_unstable):
    # Driven SHO with closed-form particular solution for sinusoidal driving, zero ICs
    omega0 = 1.0
    A = 1.0

    def make_system(omega_d: float) -> NonAutonomousEuclideanDS:
        def vf(state: np.ndarray, t: float) -> np.ndarray:
            x, y = state
            return np.array([y, -(omega0**2) * x + A * np.sin(omega_d * t)], dtype=float)

        return NonAutonomousEuclideanDS(dimension=2, vector_field=vf)

    def analytical_off_res(t: np.ndarray, omega_d: float):
        # Zero ICs analytic solution for x(t): (A/(omega0^2 - omega_d^2)) * (sin(omega_d t) - (omega_d/omega0) sin(omega0 t))
        # y(t) = x'(t) 
        denom = (omega0**2 - omega_d**2)
        coef = A / denom
        x = coef * (np.sin(omega_d * t) - (omega_d / omega0) * np.sin(omega0 * t))
        y = coef * (omega_d * np.cos(omega_d * t) - omega_d * np.cos(omega0 * t))
        return np.vstack([x, y])

    x0 = np.array([0.0, 0.0])
    T = 20.0
    t_eval = np.linspace(0.0, T, 1000)

    # Off-resonance: omega_d far from omega0
    omega_d_off = 1.7
    sys_off = make_system(omega_d_off)
    rtol_s, atol_s = numeric_tol_stable
    ok_off = []
    errs_off = []
    for method in solver_methods:
        sol = sys_off.trajectory(x0, (0.0, T), t_eval, method=method)
        y_true = analytical_off_res(sol.t, omega_d_off)
        errs_off.append(float(np.max(np.abs(sol.y - y_true))))
        ok_off.append(np.allclose(sol.y, y_true, rtol=rtol_s, atol=atol_s))
    if not any(ok_off):
        raise AssertionError(f"Off-resonance: no solver met numeric_tol_stable; max-abs errors: {dict(zip(solver_methods, errs_off))}")

    # Near-resonance: omega_d close to omega0; allow looser tolerance
    omega_d_near = 1.05
    sys_near = make_system(omega_d_near)
    rtol_u, atol_u = numeric_tol_unstable
    ok_near = []
    errs_near = []
    for method in solver_methods:
        sol = sys_near.trajectory(x0, (0.0, T), t_eval, method=method)
        # Use the same closed-form (valid off exact resonance); expect larger numerical sensitivity → looser tol
        y_true = analytical_off_res(sol.t, omega_d_near)
        errs_near.append(float(np.max(np.abs(sol.y - y_true))))
        ok_near.append(np.allclose(sol.y, y_true, rtol=rtol_u, atol=atol_u))
    if not any(ok_near):
        raise AssertionError(f"Near-resonance: no solver met numeric_tol_unstable; max-abs errors: {dict(zip(solver_methods, errs_near))}")


