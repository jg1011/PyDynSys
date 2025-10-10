import numpy as np
import pytest

from PyDynSys.core import PhaseSpace, TimeHorizon


@pytest.mark.phase_space
def test_phase_space_euclidean_and_box_contains():
    Xr2 = PhaseSpace.euclidean(2)
    assert Xr2.contains(np.array([1.0, -5.0]))

    bounds = np.array([[-1.0, 1.0], [0.0, 2.0]])
    Xbox = PhaseSpace.box(bounds)
    assert Xbox.contains(np.array([0.0, 1.0]))
    assert Xbox.contains(np.array([1.0, 2.0]))
    assert not Xbox.contains(np.array([1.1, 0.5]))


@pytest.mark.phase_space
def test_phase_space_from_symbolic_and_volume_fallbacks():
    import sympy as sp
    x, y = sp.symbols('x y', real=True)
    unit_disk = sp.ConditionSet((x, y), x**2 + y**2 < 1, sp.Reals**2)
    X = PhaseSpace.from_symbolic(unit_disk, dimension=2)
    assert X.contains(np.array([0.0, 0.0]))
    assert not X.contains(np.array([2.0, 0.0]))

    # Volume: ensure bounding-box fallback works if ConvexHull missing
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]])  # 2x1 rectangle => area 2
    # If scipy is available, hull volume should equal 2; otherwise bbox also 2
    vol = PhaseSpace.euclidean(2).phase_space_volume.__self__ if False else None  # avoid lint
    # Call via dummy instance
    from PyDynSys.core import EuclideanDS
    # Create a tiny dummy system to access method (requires dimension)
    class _Dummy(EuclideanDS):  # type: ignore
        def __init__(self):
            from PyDynSys.core import PhaseSpace
            super().__init__(2, PhaseSpace.euclidean(2))
    d = _Dummy()
    area = d.phase_space_volume(pts)
    assert pytest.approx(area, rel=0, abs=1e-12) == 2.0


@pytest.mark.time_horizon
def test_time_horizon_contains_and_factories():
    R = TimeHorizon.real_line()
    assert R.contains(-1e9)
    assert R.contains(0.0)
    assert R.contains(1e9)

    I = TimeHorizon.interval(0.0, 1.0)
    assert I.contains(0.0)
    assert I.contains(0.5)
    assert I.contains(1.0)
    assert not I.contains(-1e-6)
    assert not I.contains(1.000001)

    with pytest.raises(ValueError):
        TimeHorizon.interval(1.0, 1.0)


