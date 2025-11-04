import numpy as np
import pytest
import sympy as syp

from PyDynSys.core.euclidean import TimeHorizon


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_real_line_contains_time():
    """Test contains_time for TimeHorizon.real_line() factory"""
    T = TimeHorizon.real_line()
    
    # All times should be contained in R
    assert T.contains_time(0.0)
    assert T.contains_time(1.0)
    assert T.contains_time(-5.0)
    assert T.contains_time(1e10)
    assert T.contains_time(-1e10)
    assert T.contains_time(np.pi)


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_real_line_contains_times():
    """Test contains_times for TimeHorizon.real_line() factory"""
    T = TimeHorizon.real_line()
    
    # All time sets should be contained in R
    time_set = np.array([0.0, 1.0, -5.0, 1e10, -1e10, np.pi], dtype=np.float64)
    
    assert T.contains_times(time_set)


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_closed_interval_contains_time():
    """Test contains_time for TimeHorizon.closed_interval() factory"""
    t_min = -1.0
    t_max = 2.0
    T = TimeHorizon.closed_interval(t_min, t_max)
    
    # Times inside
    assert T.contains_time(0.0)
    assert T.contains_time(1.0)
    assert T.contains_time(-0.5)
    assert T.contains_time(1.5)
    
    # Times on boundary (closed intervals)
    assert T.contains_time(t_min)  # Lower bound included
    assert T.contains_time(t_max)  # Upper bound included
    
    # Times outside
    assert not T.contains_time(t_min - 0.1)
    assert not T.contains_time(t_max + 0.1)
    assert not T.contains_time(-10.0)
    assert not T.contains_time(10.0)


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_closed_interval_contains_times():
    """Test contains_times for TimeHorizon.closed_interval() factory"""
    t_min = -1.0
    t_max = 2.0
    T = TimeHorizon.closed_interval(t_min, t_max)
    
    # All times inside
    inside_set = np.array([0.0, 1.0, -0.5, 1.5, t_min, t_max], dtype=np.float64)
    assert T.contains_times(inside_set)
    
    # Some times outside
    mixed_set = np.array([0.0, t_min - 0.1, 1.0], dtype=np.float64)
    assert not T.contains_times(mixed_set)


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_open_interval_contains_time():
    """Test contains_time for TimeHorizon.open_interval() factory"""
    t_min = -1.0
    t_max = 2.0
    T = TimeHorizon.open_interval(t_min, t_max)
    
    # Times inside
    assert T.contains_time(0.0)
    assert T.contains_time(1.0)
    assert T.contains_time(-0.5)
    assert T.contains_time(1.5)
    assert T.contains_time(t_min + 0.001)  # Just above lower bound
    assert T.contains_time(t_max - 0.001)  # Just below upper bound
    
    # Times on boundary (open interval - boundary excluded)
    assert not T.contains_time(t_min)  # Lower bound excluded
    assert not T.contains_time(t_max)  # Upper bound excluded
    
    # Times outside
    assert not T.contains_time(t_min - 0.1)
    assert not T.contains_time(t_max + 0.1)
    assert not T.contains_time(-10.0)
    assert not T.contains_time(10.0)


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_open_interval_contains_times():
    """Test contains_times for TimeHorizon.open_interval() factory"""
    t_min = -1.0
    t_max = 2.0
    T = TimeHorizon.open_interval(t_min, t_max)
    
    # All times inside (strictly)
    inside_set = np.array([0.0, 1.0, -0.5, 1.5, t_min + 0.001, t_max - 0.001], dtype=np.float64)
    assert T.contains_times(inside_set)
    
    # Some times on boundary (should fail)
    mixed_set = np.array([0.0, t_min, 1.0], dtype=np.float64)  # t_min is on boundary
    assert not T.contains_times(mixed_set)


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_custom_time_horizon_contains_time():
    """Test contains_time for custom TimeHorizon with symbolic set and constraint"""
    # Custom time horizon: T = {t : t^2 <= 4} = [-2, 2]
    # This is equivalent to a closed interval but constructed differently
    t_symbolic = syp.symbols('t', real=True)
    time_condition = t_symbolic**2 <= 4
    time_symbolic = syp.ConditionSet(t_symbolic, time_condition, syp.Reals)
    time_constraint = lambda t: t**2 <= 4
    T = TimeHorizon(symbolic_set=time_symbolic, constraint=time_constraint)
    
    # Times inside
    assert T.contains_time(0.0)
    assert T.contains_time(1.0)
    assert T.contains_time(-1.0)
    assert T.contains_time(1.5)
    assert T.contains_time(-1.5)
    
    # Times on boundary
    assert T.contains_time(2.0)
    assert T.contains_time(-2.0)
    
    # Times outside
    assert not T.contains_time(2.1)
    assert not T.contains_time(-2.1)
    assert not T.contains_time(10.0)
    assert not T.contains_time(-10.0)


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_custom_time_horizon_contains_times():
    """Test contains_times for custom TimeHorizon with symbolic set and constraint"""
    # Custom time horizon: T = {t : t^2 <= 4} = [-2, 2]
    t_symbolic = syp.symbols('t', real=True)
    time_condition = t_symbolic**2 <= 4
    time_symbolic = syp.ConditionSet(t_symbolic, time_condition, syp.Reals)
    time_constraint = lambda t: t**2 <= 4
    T = TimeHorizon(symbolic_set=time_symbolic, constraint=time_constraint)
    
    # All times inside
    inside_set = np.array([0.0, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0], dtype=np.float64)
    assert T.contains_times(inside_set)
    
    # Some times outside
    mixed_set = np.array([0.0, 2.1, 1.0], dtype=np.float64)  # 2.1 is outside
    assert not T.contains_times(mixed_set)


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_contains_times_empty_set():
    """Test contains_times with empty time set"""
    T = TimeHorizon.real_line()
    empty_set = np.array([], dtype=np.float64)
    # Empty set should be vacuously true (all times in empty set are contained)
    assert T.contains_times(empty_set)


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_closed_interval_validation():
    """Test closed_interval raises ValueError for invalid bounds"""
    with pytest.raises(ValueError):
        TimeHorizon.closed_interval(1.0, 0.0)  # t_min > t_max
    
    with pytest.raises(ValueError):
        TimeHorizon.closed_interval(1.0, 1.0)  # t_min == t_max


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_open_interval_validation():
    """Test open_interval raises ValueError for invalid bounds"""
    with pytest.raises(ValueError):
        TimeHorizon.open_interval(1.0, 0.0)  # t_min > t_max
    
    with pytest.raises(ValueError):
        TimeHorizon.open_interval(1.0, 1.0)  # t_min == t_max


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_custom_time_horizon_single_representation():
    """Test custom TimeHorizon with only symbolic_set (constraint auto-compiled)"""
    # Custom time horizon with only symbolic representation
    t_symbolic = syp.symbols('t', real=True)
    time_condition = t_symbolic >= 0  # Non-negative times
    time_symbolic = syp.ConditionSet(t_symbolic, time_condition, syp.Reals)
    T = TimeHorizon(symbolic_set=time_symbolic)
    
    # Times that should be contained (non-negative)
    assert T.contains_time(0.0)
    assert T.contains_time(1.0)
    assert T.contains_time(10.0)
    
    # Times that should not be contained (negative)
    assert not T.contains_time(-1.0)
    assert not T.contains_time(-0.1)


@pytest.mark.euclidean_time_horizon
@pytest.mark.validation
def test_custom_time_horizon_constraint_only():
    """Test custom TimeHorizon with only constraint (no symbolic set)"""
    # Custom time horizon with only constraint
    time_constraint = lambda t: 0 <= t <= 10
    T = TimeHorizon(constraint=time_constraint)
    
    # Times inside
    assert T.contains_time(0.0)
    assert T.contains_time(5.0)
    assert T.contains_time(10.0)
    
    # Times outside
    assert not T.contains_time(-1.0)
    assert not T.contains_time(11.0)

