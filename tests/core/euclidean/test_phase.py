import numpy as np
import pytest
import sympy as syp

from PyDynSys.core.euclidean import PhaseSpace


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_full_phase_space_contains_point():
    """Test contains_point for PhaseSpace.full() factory"""
    X = PhaseSpace.full(3)
    
    # All points should be contained in R^n
    assert X.contains_point(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([1.0, -5.0, 10.0], dtype=np.float64))
    assert X.contains_point(np.array([-1e10, 1e10, np.pi], dtype=np.float64))


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_full_phase_space_contains_points():
    """Test contains_points for PhaseSpace.full() factory"""
    X = PhaseSpace.full(3)
    
    # All point sets should be contained in R^n
    point_set = np.array([
        [0.0, 0.0, 0.0],
        [1.0, -5.0, 10.0],
        [-1e10, 1e10, np.pi],
        [0.5, 0.5, 0.5]
    ], dtype=np.float64)
    
    assert X.contains_points(point_set)


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_box_phase_space_contains_point():
    """Test contains_point for PhaseSpace.box() factory"""
    bounds = np.array([[-1.0, 1.0], [0.0, 2.0], [-0.5, 0.5]], dtype=np.float64)
    X = PhaseSpace.box(bounds)
    
    # Points inside
    assert X.contains_point(np.array([0.0, 1.0, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([1.0, 2.0, 0.5], dtype=np.float64))
    assert X.contains_point(np.array([-1.0, 0.0, -0.5], dtype=np.float64))
    
    # Points on boundary (closed intervals)
    assert X.contains_point(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([-1.0, 2.0, 0.5], dtype=np.float64))
    
    # Points outside
    assert not X.contains_point(np.array([1.1, 0.5, 0.0], dtype=np.float64))
    assert not X.contains_point(np.array([0.0, -0.1, 0.0], dtype=np.float64))
    assert not X.contains_point(np.array([0.0, 0.0, 0.6], dtype=np.float64))


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_box_phase_space_contains_points():
    """Test contains_points for PhaseSpace.box() factory"""
    bounds = np.array([[-1.0, 1.0], [0.0, 2.0], [-0.5, 0.5]], dtype=np.float64)
    X = PhaseSpace.box(bounds)
    
    # All points inside
    inside_set = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 2.0, 0.5],
        [-1.0, 0.0, -0.5]
    ], dtype=np.float64)
    assert X.contains_points(inside_set)
    
    # Some points outside
    mixed_set = np.array([
        [0.0, 1.0, 0.0],
        [1.1, 0.5, 0.0],  # Outside x-bound
        [0.0, 0.0, 0.0]
    ], dtype=np.float64)
    assert not X.contains_points(mixed_set)


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_closed_hypersphere_contains_point():
    """Test contains_point for PhaseSpace.closed_hypersphere() factory"""
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    radius = 1.0
    X = PhaseSpace.closed_hypersphere(center, radius)
    
    # Points inside
    assert X.contains_point(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([0.5, 0.5, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    
    # Points on boundary (closed sphere)
    assert X.contains_point(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([0.0, 1.0, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    
    # Points outside
    assert not X.contains_point(np.array([1.1, 0.0, 0.0], dtype=np.float64))
    assert not X.contains_point(np.array([0.0, 0.0, 1.001], dtype=np.float64))
    
    # Off-center sphere
    center_off = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    X_off = PhaseSpace.closed_hypersphere(center_off, radius)
    assert X_off.contains_point(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    assert X_off.contains_point(np.array([2.0, 1.0, 1.0], dtype=np.float64))
    assert not X_off.contains_point(np.array([0.0, 0.0, 0.0], dtype=np.float64))


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_closed_hypersphere_contains_points():
    """Test contains_points for PhaseSpace.closed_hypersphere() factory"""
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    radius = 1.0
    X = PhaseSpace.closed_hypersphere(center, radius)
    
    # All points inside
    inside_set = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float64)
    assert X.contains_points(inside_set)
    
    # Some points outside
    mixed_set = np.array([
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],  # Outside radius
        [0.5, 0.5, 0.0]
    ], dtype=np.float64)
    assert not X.contains_points(mixed_set)


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_open_hypersphere_contains_point():
    """Test contains_point for PhaseSpace.open_hypersphere() factory"""
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    radius = 1.0
    X = PhaseSpace.open_hypersphere(center, radius)
    
    # Points inside
    assert X.contains_point(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([0.5, 0.5, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([0.999, 0.0, 0.0], dtype=np.float64))
    
    # Points on boundary (open sphere - boundary excluded)
    assert not X.contains_point(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    assert not X.contains_point(np.array([0.0, 1.0, 0.0], dtype=np.float64))
    assert not X.contains_point(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    
    # Points outside
    assert not X.contains_point(np.array([1.1, 0.0, 0.0], dtype=np.float64))
    assert not X.contains_point(np.array([0.0, 0.0, 1.001], dtype=np.float64))

@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_open_hypersphere_contains_points():
    """Test contains_points for PhaseSpace.open_hypersphere() factory"""
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    radius = 1.0
    X = PhaseSpace.open_hypersphere(center, radius)
    
    # All points inside (strictly)
    inside_set = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.999, 0.0, 0.0]
    ], dtype=np.float64)
    assert X.contains_points(inside_set)
    
    # Some points on boundary (should fail)
    mixed_set = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],  # On boundary - excluded
        [0.5, 0.5, 0.0]
    ], dtype=np.float64)
    assert not X.contains_points(mixed_set)


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_custom_ellipsoid_contains_point():
    """Test contains_point for custom ellipsoid PhaseSpace"""
    # Custom ellipsoid: (x/2)**2 + (y/1)**2 + (z/1.5)**2 <= 1
    # Semi-axes: [2, 1, 1.5], centered at origin
    dimension = 3
    semi_axes = np.array([2.0, 1.0, 1.5], dtype=np.float64)
    R_n = syp.Reals ** dimension
    x = syp.symbols('x0 x1 x2', real=True)
    ellipsoid_condition = sum((x[i] / semi_axes[i])**2 for i in range(dimension)) <= 1
    ellipsoid_symbolic = syp.ConditionSet(syp.Tuple(*x), ellipsoid_condition, R_n)
    ellipsoid_constraint = lambda x: bool(np.sum((x / semi_axes)**2) <= 1)
    X = PhaseSpace(dimension=dimension, symbolic_set=ellipsoid_symbolic, constraint=ellipsoid_constraint)
    
    # Points inside
    assert X.contains_point(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([1.0, 0.0, 0.0], dtype=np.float64))  # Within x semi-axis
    assert X.contains_point(np.array([0.0, 0.5, 0.0], dtype=np.float64))  # Within y semi-axis
    assert X.contains_point(np.array([0.0, 0.0, 0.75], dtype=np.float64))  # Within z semi-axis
    
    # Points on boundary
    assert X.contains_point(np.array([2.0, 0.0, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([0.0, 1.0, 0.0], dtype=np.float64))
    assert X.contains_point(np.array([0.0, 0.0, 1.5], dtype=np.float64))
    
    # Points outside
    assert not X.contains_point(np.array([2.1, 0.0, 0.0], dtype=np.float64))
    assert not X.contains_point(np.array([0.0, 1.1, 0.0], dtype=np.float64))
    assert not X.contains_point(np.array([0.0, 0.0, 1.51], dtype=np.float64))
    
    # Points at corners (should satisfy ellipsoid equation)
    # Point where all coordinates are at their semi-axes: (2, 1, 1.5)
    # Check: (2/2)^2 + (1/1)^2 + (1.5/1.5)^2 = 1 + 1 + 1 = 3 > 1, so outside
    assert not X.contains_point(np.array([2.0, 1.0, 1.5], dtype=np.float64))
    
    # Point on ellipsoid surface (not at axes)
    # Choose point such that sum = 1, e.g., (1, 0.5, 0.75)
    # Check: (1/2)^2 + (0.5/1)^2 + (0.75/1.5)^2 = 0.25 + 0.25 + 0.25 = 0.75 < 1, so inside
    assert X.contains_point(np.array([1.0, 0.5, 0.75], dtype=np.float64))


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_custom_ellipsoid_contains_points():
    """Test contains_points for custom ellipsoid PhaseSpace"""
    # Custom ellipsoid: (x/2)**2 + (y/1)**2 + (z/1.5)**2 <= 1
    dimension = 3
    semi_axes = np.array([2.0, 1.0, 1.5], dtype=np.float64)
    R_n = syp.Reals ** dimension
    x = syp.symbols('x0 x1 x2', real=True)
    ellipsoid_condition = sum((x[i] / semi_axes[i])**2 for i in range(dimension)) <= 1
    ellipsoid_symbolic = syp.ConditionSet(syp.Tuple(*x), ellipsoid_condition, R_n)
    ellipsoid_constraint = lambda x: bool(np.sum((x / semi_axes)**2) <= 1)
    X = PhaseSpace(dimension=dimension, symbolic_set=ellipsoid_symbolic, constraint=ellipsoid_constraint)
    
    # All points inside
    inside_set = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [1.0, 0.5, 0.75]
    ], dtype=np.float64)
    assert X.contains_points(inside_set)
    
    # Some points outside
    mixed_set = np.array([
        [0.0, 0.0, 0.0],
        [2.1, 0.0, 0.0],  # Outside x semi-axis
        [0.0, 0.5, 0.0]
    ], dtype=np.float64)
    assert not X.contains_points(mixed_set)


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_contains_points_empty_set():
    """Test contains_points with empty point set"""
    X = PhaseSpace.full(3)
    empty_set = np.array([], dtype=np.float64).reshape(0, 3)
    # Empty set should be vacuously true (all points in empty set are contained)
    assert X.contains_points(empty_set)


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_contains_point_dimension_mismatch():
    """Test contains_point raises ValueError for dimension mismatch"""
    X = PhaseSpace.full(3)
    
    with pytest.raises(ValueError):
        X.contains_point(np.array([1.0, 2.0], dtype=np.float64))  # 2D instead of 3D
    
    with pytest.raises(ValueError):
        X.contains_point(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))  # 4D instead of 3D


@pytest.mark.euclidean_phase_space
@pytest.mark.validation
def test_contains_points_shape_validation():
    """Test contains_points validates input shape"""
    X = PhaseSpace.full(3)
    
    # Valid shape: (n_points, 3)
    valid_set = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    assert X.contains_points(valid_set)
    
    # Invalid shape: wrong dimension
    invalid_set = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        X.contains_points(invalid_set)

